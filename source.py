import os
import logging
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageNet
import wandb
import utils
from torch.autograd import Variable

from classifier import Classifier
from image_list import ImageList
from utils import (
	adjust_learning_rate,
	concat_all_gather,
	get_augmentation,
	is_master,
	per_class_accuracy,
	remove_wrap_arounds,
	save_checkpoint,
	use_wandb,
	AverageMeter,
	ProgressMeter,
	MMCE,
	MMCE_weighted,
)

from torchmetrics.classification import MulticlassCalibrationError

def get_source_optimizer(model, args):
	if args.distributed:
		model = model.module
	backbone_params, extra_params = model.get_params()
	if args.optim.name == "sgd":
		optimizer = torch.optim.SGD(
			[
				{
					"params": backbone_params,
					"lr": args.optim.lr,
					"momentum": args.optim.momentum,
					"weight_decay": args.optim.weight_decay,
					"nesterov": args.optim.nesterov,
				},
				{
					"params": extra_params,
					"lr": args.optim.lr * 10,
					"momentum": args.optim.momentum,
					"weight_decay": args.optim.weight_decay,
					"nesterov": args.optim.nesterov,
				},
			]
		)
	else:
		raise NotImplementedError(f"{args.optim.name} not implemented.")

	for param_group in optimizer.param_groups:
		param_group["lr0"] = param_group["lr"]

	return optimizer


def train_source_domain(args):
	logging.info(f"Start source training on {args.data.src_domain}...")

	model = Classifier(args.model_src).to("cuda")
	if args.distributed:
		model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
		model = DistributedDataParallel(
			model, device_ids=[args.gpu], find_unused_parameters=True
		)
	logging.info(f"1 - Created source model")

	# transforms
	pasta_args = None
	if args.data.aug_type == "pasta":
		tr_aug_type = "pasta"
		pasta_args = {"a": args.data.pasta_a, "b": args.data.pasta_b, "k": args.data.pasta_k, "prob": args.data.pasta_prob}
	else:
		tr_aug_type = "plain"
	train_transform = get_augmentation(tr_aug_type, pasta_args=pasta_args)
	val_transform = get_augmentation("test")

	# datasets
	if args.data.dataset == "imagenet-1k":
		train_dataset = ImageNet(args.data.image_root, transform=train_transform)
		val_dataset = ImageNet(
			args.data.image_root, split="val", transform=val_transform
		)
	else:
		label_file = os.path.join(
			args.data.image_root, f"{args.data.src_domain}_list.txt"
		)
		train_dataset = ImageList(
			args.data.image_root, label_file, transform=train_transform
		)
		val_dataset = ImageList(
			args.data.image_root, label_file, transform=val_transform
		)
		assert len(train_dataset) == len(val_dataset)

		# split the dataset with indices
		indices = np.random.permutation(len(train_dataset))
		num_train = int(len(train_dataset) * args.data.train_ratio)
		train_dataset = Subset(train_dataset, indices[:num_train])
		val_dataset = Subset(val_dataset, indices[num_train:])
		
		# Load target dataset for intermediate eval
		# Target domain selection specific to VISDA-C
		tgt_domain = args.data.target_domains[0]
		tgt_label_file = os.path.join(args.data.image_root, f"{tgt_domain}_list.txt")
		tgt_dataset = ImageList(args.data.image_root, tgt_label_file, val_transform)
		
	logging.info(
		f"Loaded {len(train_dataset)} samples for training "
		+ f"and {len(val_dataset)} samples for validation",
	)

	# data loaders
	train_sampler = DistributedSampler(train_dataset) if args.distributed else None
	train_loader = DataLoader(
		train_dataset,
		batch_size=args.data.batch_size,
		shuffle=(train_sampler is None),
		sampler=train_sampler,
		pin_memory=True,
		num_workers=args.data.workers,
	)
	val_sampler = DistributedSampler(val_dataset) if args.distributed else None
	val_loader = DataLoader(
		val_dataset,
		batch_size=args.data.batch_size,
		sampler=val_sampler,
		pin_memory=True,
		num_workers=args.data.workers,
	)
	
	# Target dataloader
	tgt_sampler = DistributedSampler(tgt_dataset) if args.distributed else None
	tgt_loader = DataLoader(
		tgt_dataset,
		batch_size=args.data.batch_size,
		sampler=tgt_sampler,
		pin_memory=True,
		num_workers=args.data.workers,
	)
	
	logging.info(f"2 - Created data loaders")

	optimizer = get_source_optimizer(model, args)
	args.learn.full_progress = args.learn.epochs * len(train_loader)
	logging.info(f"3 - Created optimizer")

	logging.info(f"Start training...")
	best_acc = 0.0
	for epoch in range(args.learn.start_epoch, args.learn.epochs):
		if args.distributed:
			train_sampler.set_epoch(epoch)

		# train for one epoch
		train_epoch(train_loader, model, optimizer, epoch, args)

		# evaluate
		accuracy = evaluate(val_loader, model, domain=args.data.src_domain, args=args)
		tgt_accuracy = evaluate(tgt_loader, model, domain=tgt_domain, args=args)
		if accuracy > best_acc and is_master(args):
			best_acc = accuracy
			filename = f"best_{args.data.src_domain}_{args.seed}.pth.tar"
			save_path = os.path.join(args.log_dir, filename)
			save_checkpoint(model, optimizer, epoch, save_path=save_path)
   
		# Also save intermediate checkpoints
		filename = f"intermediate_{args.data.src_domain}_{args.seed}_ep{str(epoch)}.pth.tar"
		save_path = os.path.join(args.log_dir, filename)
		save_checkpoint(model, optimizer, epoch, save_path=save_path)

	# evaluate on target before any adaptation
	for t, tgt_domain in enumerate(args.data.target_domains):
		if tgt_domain == args.data.src_domain:
			continue
		label_file = os.path.join(args.data.image_root, f"{tgt_domain}_list.txt")
		tgt_dataset = ImageList(args.data.image_root, label_file, val_transform)
		sampler = DistributedSampler(tgt_dataset) if args.distributed else None
		tgt_loader = DataLoader(
			tgt_dataset,
			batch_size=args.data.batch_size,
			sampler=sampler,
			pin_memory=True,
			num_workers=args.data.workers,
		)

		logging.info(f"Evaluate {args.data.src_domain} model on {tgt_domain}")
		evaluate(
			tgt_loader,
			model,
			domain=f"{args.data.src_domain}-{tgt_domain}",
			args=args,
			wandb_commit=(t == len(args.data.target_domains) - 1),
		)


def train_epoch(train_loader, model, optimizer, epoch, args):
	batch_time = AverageMeter("Time", ":6.3f")
	loss = AverageMeter("Loss", ":.4f")
	top1 = AverageMeter("Acc@1", ":6.2f")
	progress = ProgressMeter(
		len(train_loader), [batch_time, loss, top1], prefix="Epoch: [{}]".format(epoch),
	)

	# make sure to switch to train mode
	model.train()

	end = time.time()
	for i, data in enumerate(train_loader):
		# if i > 3:
		# 	break
		images = data[0].cuda(args.gpu, non_blocking=True)
		labels = data[1].cuda(args.gpu, non_blocking=True)

		# per-step scheduler
		step = i + epoch * len(train_loader)
		adjust_learning_rate(optimizer, step, args)

		logits = model(images)

		if args.learn.loss_fn == "smoothed_ce":
			loss_ce = smoothed_cross_entropy(
				logits,
				labels,
				num_classes=args.model_src.num_classes,
				epsilon=args.learn.epsilon,
			)
		elif args.learn.loss_fn == "focal":
			loss_ce = focal_loss(
				logits,
				labels,
				num_classes=args.model_src.num_classes,
				gamma=args.learn.gamma,
			)
		elif args.learn.loss_fn == "smoothed_ce_include_mmce":
			# print("We are using MMCE loss")
			loss_ce = smoothed_cross_entropy(
				logits,
				labels,
				num_classes=args.model_src.num_classes,
				epsilon=args.learn.epsilon,
			)
			if args.learn.mmce_mode == "vanilla":
				mmce_loss = MMCE()(logits, labels)
			else:
				mmce_loss = MMCE_weighted()(logits, labels)
			loss_ce += args.learn.mmce_wt * mmce_loss

		# train acc measure (on one GPU only)
		preds = logits.argmax(dim=1)
		acc = (preds == labels).float().mean().detach() * 100.0
		loss.update(loss_ce.item(), images.size(0))
		top1.update(acc.item(), images.size(0))

		if use_wandb(args):
			wandb.log({"Loss": loss_ce.item()}, commit=(i != len(train_loader)))

		# perform one gradient step
		optimizer.zero_grad()
		loss_ce.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % args.learn.print_freq == 0:
			progress.display(i)


def evaluate(val_loader, model, domain, args, wandb_commit=True):
	model.eval()

	logging.info(f"Evaluating...")
	gt_labels, all_preds, all_logits = [], [], []
	with torch.no_grad():
		# iterator = tqdm(val_loader) if is_master(args) else val_loader
		iterator = val_loader if is_master(args) else val_loader
		for data in iterator:
			images = data[0].cuda(args.gpu, non_blocking=True)
			labels = data[1]

			logits = model(images)
			preds = logits.argmax(dim=1).cpu()

			gt_labels.append(labels)
			all_preds.append(preds)
			all_logits.append(logits)

	all_logits = torch.cat(all_logits)
	gt_labels = torch.cat(gt_labels)
	all_preds = torch.cat(all_preds)
	all_probs = nn.Softmax(dim=1)(all_logits)

	if args.distributed:
		gt_labels = concat_all_gather(gt_labels.cuda())
		all_preds = concat_all_gather(all_preds.cuda())
		all_logits = concat_all_gather(all_logits.cuda())
		all_probs = concat_all_gather(all_probs.cuda())

		ranks = len(val_loader.dataset) % dist.get_world_size()
		gt_labels = remove_wrap_arounds(gt_labels, ranks).cpu()
		all_preds = remove_wrap_arounds(all_preds, ranks).cpu()
		all_logits = remove_wrap_arounds(all_logits, ranks).cpu()
		all_probs = remove_wrap_arounds(all_probs, ranks).cpu()

	accuracy = (all_preds == gt_labels).float().mean() * 100.0
	ent = torch.mean(utils.Entropy(all_probs))
	nll = nn.NLLLoss()(nn.LogSoftmax(dim=1)(all_logits), gt_labels)
	bsc = utils.BrierScore()(all_probs, gt_labels)
	n_bins = 10
	ece = MulticlassCalibrationError(num_classes=args.model_src.num_classes, n_bins=n_bins, norm="l1")(all_probs, gt_labels)
	mce = MulticlassCalibrationError(num_classes=args.model_src.num_classes, n_bins=n_bins, norm="max")(all_probs, gt_labels)
	
	wandb_dict = {f"{domain} Acc": accuracy}
	log_str = f"Accuracy: {accuracy:.2f}"
	log_str += f" Ent: {ent:.4f}"
	log_str += f" NLL: {nll:.4f}"
	log_str += f" BSC: {bsc:.4f}"
	log_str += f" ECE: {ece:.4f}"
	log_str += f" MCE: {mce:.4f}"
	logging.info(log_str)
	# logging.info(f"Accuracy: {accuracy:.2f}")
	# logging.info(f"Ent: {ent:.4f}")
	# logging.info(f"NLL: {nll:.4f}")
	# logging.info(f"BSC: {bsc:.4f}")
	# logging.info(f"ECE: {ece:.4f}")
	# logging.info(f"MCE: {mce:.4f}")
	if args.data.dataset == "VISDA-C":
		acc_per_class = per_class_accuracy(
			y_true=gt_labels.numpy(), y_pred=all_preds.numpy()
		)
		wandb_dict[f"{domain} Avg"] = acc_per_class.mean()
		wandb_dict[f"{domain} Per-class"] = acc_per_class
		wandb_dict[f"{domain} Ent"] = ent
		wandb_dict[f"{domain} NLL"] = nll
		wandb_dict[f"{domain} BSC"] = bsc
		wandb_dict[f"{domain} ECE"] = ece
		wandb_dict[f"{domain} MCE"] = mce

	if use_wandb(args):
		wandb.log(wandb_dict, commit=wandb_commit)

	return accuracy


def smoothed_cross_entropy(logits, labels, num_classes, epsilon=0):
	log_probs = F.log_softmax(logits, dim=1)
	with torch.no_grad():
		targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
		targets = (1 - epsilon) * targets + epsilon / num_classes
	loss = (-targets * log_probs).sum(dim=1).mean()

	return loss

def focal_loss(logits, labels, num_classes, gamma=2):
	with torch.no_grad():
		labels = labels.view(-1,1)
	logpt = F.log_softmax(logits, dim=1)
	logpt = logpt.gather(1,labels)
	logpt = logpt.view(-1)
	pt = Variable(logpt.data.exp())
	loss = -1 * (1-pt)**gamma * logpt
	return loss.mean()
	
	
	

