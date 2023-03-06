import logging
import math
import os
from PIL import Image
import yaml

from sklearn.metrics import confusion_matrix
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torchvision import transforms

import numpy as np

from moco.loader import GaussianBlur

LOG_FORMAT = "[%(levelname)s] %(asctime)s %(filename)s:%(lineno)s %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

NUM_CLASSES = {"domainnet-126": 126, "VISDA-C": 12}

def TV_Dist(all_preds, gt_labels):
    preds = all_preds.numpy()
    labels = gt_labels.numpy()
    _, pred_dist = np.unique(preds, return_counts=True)
    _, label_dist = np.unique(labels, return_counts=True)
    pred_dist = pred_dist / pred_dist.sum()
    label_dist = label_dist / label_dist.sum()
    tv_dist = 0.5*np.linalg.norm(pred_dist - label_dist, ord=1)
    return tv_dist

def Entropy(input_):
	bs = input_.size(0)
	epsilon = 1e-5
	entropy = -input_ * torch.log(input_ + epsilon)
	entropy = torch.sum(entropy, dim=1)
	return entropy

class BrierScore(nn.Module):
	def __init__(self, use_gpu=True):
		super(BrierScore, self).__init__()
		self.use_gpu = use_gpu

	def forward(self, input, target):
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
		target = target.view(-1,1)
		if self.use_gpu: target = target.cuda()
		target_one_hot = torch.FloatTensor(input.shape).to(target.get_device())
		target_one_hot.zero_()
		target_one_hot.scatter_(1, target, 1)

		pt = F.softmax(input, dim=-1)
		squared_diff = (target_one_hot - pt.to(target.get_device())) ** 2

		loss = torch.sum(squared_diff) / float(input.shape[0])
		return loss

def resize(input,
		   size=None,
		   scale_factor=None,
		   mode='nearest',
		   align_corners=None,
		   warning=True):
	if warning:
		if size is not None and align_corners:
			input_h, input_w = tuple(int(x) for x in input.shape[2:])
			output_h, output_w = tuple(int(x) for x in size)
			if output_h > input_h or output_w > output_h:
				if ((output_h > 1 and output_w > 1 and input_h > 1
					 and input_w > 1) and (output_h - 1) % (input_h - 1)
						and (output_w - 1) % (input_w - 1)):
					warnings.warn(
						f'When align_corners={align_corners}, '
						'the output would more aligned if '
						f'input size {(input_h, input_w)} is `x+1` and '
						f'out size {(output_h, output_w)} is `nx+1`')
	# print(input.shape)
	return F.interpolate(input, size, scale_factor, mode, align_corners)

class Masking:
	def __init__(
		self,
		block_size, 
		ratio,
		):
		self.block_size = block_size
		self.ratio = ratio

	def __call__(self, img):
		img = transforms.ToTensor()(img)
  
		_, H, W = img.shape
		mshape = 1, 1, round(H / self.block_size), round(W / self.block_size)
		input_mask = torch.rand(mshape, device=img.device)
		input_mask = (input_mask > self.ratio).float()
		# print(input_mask.shape)
		# print(img.shape)
		input_mask = resize(input_mask, size=(H, W))
		# print(input_mask.sum())
		masked_img = img * input_mask[0]
		masked_img = transforms.ToPILImage()(masked_img)
		return masked_img

class PASTA:
	def __init__(self, alpha: float = 3, beta: float = 0.25, k: int = 2, prob: float = 1.0):
		self.alpha = alpha
		self.beta = beta
		self.k = k
		self.prob = prob
	
	def __call__(self, img):
		use_prob = random.random()
		# print("Usage probability: ", use_prob)
		# print("Chance Correction: ", self.prob)
		if random.random() < self.prob:
			img = transforms.ToTensor()(img)
			fft_src = torch.fft.fftn(img, dim=[-2, -1])
			amp_src, pha_src = torch.abs(fft_src), torch.angle(fft_src)

			X, Y = amp_src.shape[1:]
			X_range, Y_range = None, None

			if X % 2 == 1:
				X_range = np.arange(-1 * (X // 2), (X // 2) + 1)
			else:
				X_range = np.concatenate(
					[np.arange(-1 * (X // 2) + 1, 1), np.arange(0, X // 2)]
				)

			if Y % 2 == 1:
				Y_range = np.arange(-1 * (Y // 2), (Y // 2) + 1)
			else:
				Y_range = np.concatenate(
					[np.arange(-1 * (Y // 2) + 1, 1), np.arange(0, Y // 2)]
				)

			XX, YY = np.meshgrid(Y_range, X_range)

			exp = self.k
			lin = self.alpha
			offset = self.beta

			inv = np.sqrt(np.square(XX) + np.square(YY))
			inv *= (1 / inv.max()) * lin
			inv = np.power(inv, exp)
			inv = np.tile(inv, (3, 1, 1))
			inv += offset
			prop = np.fft.fftshift(inv, axes=[-2, -1])
			amp_src = amp_src * np.random.normal(np.ones(prop.shape), prop)

			aug_img = amp_src * torch.exp(1j * pha_src)
			aug_img = torch.fft.ifftn(aug_img, dim=[-2, -1])
			aug_img = torch.real(aug_img)
			aug_img = torch.clip(aug_img, 0, 1)
			aug_img = transforms.ToPILImage()(aug_img)
		else:
			aug_img = img
		return aug_img

def configure_logger(rank, log_path=None):
	if log_path:
		log_dir = os.path.dirname(log_path)
		os.makedirs(log_dir, exist_ok=True)

	# only master process will print & write
	level = logging.INFO if rank in {-1, 0} else logging.WARNING
	handlers = [logging.StreamHandler()]
	if rank in {0, -1} and log_path:
		handlers.append(logging.FileHandler(log_path, "w"))

	logging.basicConfig(
		level=level,
		format=LOG_FORMAT,
		datefmt=LOG_DATEFMT,
		handlers=handlers,
		force=True,
	)


class UnevenBatchLoader:
	"""Loader that loads data from multiple datasets with different length."""

	def __init__(self, data_loaders, is_ddp=False):
		# register N data loaders with epoch counters.
		self.data_loaders = data_loaders
		self.epoch_counters = [0 for _ in range(len(data_loaders))]

		# set_epoch() needs to be called before creating the iterator
		self.is_ddp = is_ddp
		if is_ddp:
			for data_loader in data_loaders:
				data_loader.sampler.set_epoch(0)
		self.iterators = [iter(data_loader) for data_loader in data_loaders]

	def next_batch(self):
		"""Load the next batch by collecting from N data loaders.
		Args:
			None
		Returns:
			data: a list of N items from N data loaders. each item has the format
				output by a single data loader.
		"""
		data = []
		for i, iterator in enumerate(self.iterators):
			try:
				batch_i = next(iterator)
			except StopIteration:
				self.epoch_counters[i] += 1
				# create a new iterator
				if self.is_ddp:
					self.data_loaders[i].sampler.set_epoch(self.epoch_counters[i])
				new_iterator = iter(self.data_loaders[i])
				self.iterators[i] = new_iterator
				batch_i = next(new_iterator)
			data.append(batch_i)

		return data

	def update_loader(self, idx, loader, epoch=None):
		if self.is_ddp and isinstance(epoch, int):
			loader.sampler.set_epoch(epoch)
		self.iterators[idx] = iter(loader)


class CustomDistributedDataParallel(DistributedDataParallel):
	"""A wrapper class over DDP that relay "module" attribute."""

	def __init__(self, model, **kwargs):
		super(CustomDistributedDataParallel, self).__init__(model, **kwargs)

	def __getattr__(self, name):
		try:
			return super(CustomDistributedDataParallel, self).__getattr__(name)
		except AttributeError:
			return getattr(self.module, name)


@torch.no_grad()
def concat_all_gather(tensor):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
	dist.all_gather(tensors_gather, tensor, async_op=False)

	output = torch.cat(tensors_gather, dim=0)
	return output


@torch.no_grad()
def remove_wrap_arounds(tensor, ranks):
	if ranks == 0:
		return tensor

	world_size = dist.get_world_size()
	single_length = len(tensor) // world_size
	output = []
	for rank in range(world_size):
		sub_tensor = tensor[rank * single_length : (rank + 1) * single_length]
		if rank >= ranks:
			output.append(sub_tensor[:-1])
		else:
			output.append(sub_tensor)
	output = torch.cat(output)

	return output


def get_categories(category_file):
	"""Return a list of categories ordered by corresponding label.

	Args:
		category_file: str, path to the category file. can be .yaml or .txt

	Returns:
		categories: List[str], a list of categories ordered by label.
	"""
	if category_file.endswith(".yaml"):
		with open(category_file, "r") as fd:
			cat_mapping = yaml.load(fd, Loader=yaml.SafeLoader)
		categories = list(cat_mapping.keys())
		categories.sort(key=lambda x: cat_mapping[x])
	elif category_file.endswith(".txt"):
		with open(category_file, "r") as fd:
			categories = fd.readlines()
		categories = [cat.strip() for cat in categories if cat]
	else:
		raise NotImplementedError()

	categories = [cat.replace("_", " ") for cat in categories]
	return categories


def get_augmentation(aug_type, normalize=None, pasta_args=None):
	if not normalize:
		normalize = transforms.Normalize(
			mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
		)
	if aug_type == "moco-v2":
		return transforms.Compose(
			[
				transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
				transforms.RandomApply(
					[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
					p=0.8,  # not strengthened
				),
				transforms.RandomGrayscale(p=0.2),
				transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]
		)
	elif aug_type == "moco-v1":
		return transforms.Compose(
			[
				transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
				transforms.RandomGrayscale(p=0.2),
				transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]
		)
	elif aug_type == "plain":
		return transforms.Compose(
			[
				transforms.Resize((256, 256)),
				transforms.RandomCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]
		)
	elif aug_type == "mask_v1":
		return transforms.Compose(
			[
				Masking(block_size=64, ratio=0.7),
				transforms.Resize((256, 256)),
				transforms.RandomCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]
		)
	elif aug_type == "pasta":
		return transforms.Compose(
			[
				PASTA(alpha=pasta_args["a"], beta=pasta_args["b"], k=pasta_args["k"], prob=pasta_args["prob"]),
				transforms.Resize((256, 256)),
				transforms.RandomCrop(224),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]
		)
	elif aug_type == "pasta_mocov2_v2":
		return transforms.Compose(
			[
				PASTA(alpha=pasta_args["a"], beta=pasta_args["b"], k=pasta_args["k"], prob=pasta_args["prob"]),
				transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
				transforms.RandomApply(
					[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
					p=0.8,  # not strengthened
				),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]
		)
	elif aug_type == "pasta_mocov2":
		return transforms.Compose(
			[
				PASTA(alpha=pasta_args["a"], beta=pasta_args["b"], k=pasta_args["k"], prob=pasta_args["prob"]),
				transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
				transforms.RandomApply(
					[transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
					p=0.8,  # not strengthened
				),
				transforms.RandomGrayscale(p=0.2),
				transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.5),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				normalize,
			]
		)
	elif aug_type == "clip_inference":
		return transforms.Compose(
			[
				transforms.Resize(224, interpolation=Image.BICUBIC),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
			]
		)
	elif aug_type == "test":
		return transforms.Compose(
			[
				transforms.Resize((256, 256)),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize,
			]
		)
	return None


class AverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, name, fmt=":f"):
		self.name = name
		self.fmt = fmt
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

	def __str__(self):
		fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
		return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
	def __init__(self, num_batches, meters, prefix=""):
		self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
		self.meters = meters
		self.prefix = prefix

	def display(self, batch):
		entries = [self.prefix + self.batch_fmtstr.format(batch)]
		entries += [str(meter) for meter in self.meters]
		logging.info("\t".join(entries))

	def _get_batch_fmtstr(self, num_batches):
		num_digits = len(str(num_batches // 1))
		fmt = "{:" + str(num_digits) + "d}"
		return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def save_checkpoint(model, optimizer, epoch, save_path="checkpoint.pth.tar"):
	state = {
		"state_dict": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"epoch": epoch,
	}
	torch.save(state, save_path)


def adjust_learning_rate(optimizer, progress, args):
	"""
	Decay the learning rate based on epoch or iteration.
	"""
	if args.optim.cos:
		decay = 0.5 * (1.0 + math.cos(math.pi * progress / args.learn.full_progress))
	elif args.optim.exp:
		decay = (1 + 10 * progress / args.learn.full_progress) ** -0.75
	else:
		decay = 1.0
		for milestone in args.optim.schedule:
			decay *= args.optim.gamma if progress >= milestone else 1.0
	for param_group in optimizer.param_groups:
		param_group["lr"] = param_group["lr0"] * decay

	return decay


def per_class_accuracy(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred)
	acc_per_class = (matrix.diagonal() / matrix.sum(axis=1) * 100.0).round(2)
	logging.info(
		f"Accuracy per class: {acc_per_class}, mean: {acc_per_class.mean().round(2)}"
	)

	return acc_per_class


def get_distances(X, Y, dist_type="euclidean"):
	"""
	Args:
		X: (N, D) tensor
		Y: (M, D) tensor
	"""
	if dist_type == "euclidean":
		distances = torch.cdist(X, Y)
	elif dist_type == "cosine":
		distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
	else:
		raise NotImplementedError(f"{dist_type} distance not implemented.")

	return distances


def is_master(args):
	return args.rank % args.ngpus_per_node == 0


def use_wandb(args):
	return is_master(args) and args.use_wandb

class MMCE(nn.Module):
	"""
	Computes MMCE_m loss.
	"""
	def __init__(self):
		super(MMCE, self).__init__()
		self.device = None

	def torch_kernel(self, matrix):
		return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

	def forward(self, input, target):
		self.device = input.get_device()
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

		target = target.view(-1) #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

		predicted_probs = F.softmax(input, dim=1)
		predicted_probs, pred_labels = torch.max(predicted_probs, 1)
		correct_mask = torch.where(torch.eq(pred_labels, target),
						  torch.ones(pred_labels.shape).to(self.device),
						  torch.zeros(pred_labels.shape).to(self.device))

		c_minus_r = correct_mask - predicted_probs

		dot_product = torch.mm(c_minus_r.unsqueeze(1),
						c_minus_r.unsqueeze(0))

		prob_tiled = predicted_probs.unsqueeze(1).repeat(1, predicted_probs.shape[0]).unsqueeze(2)
		prob_pairs = torch.cat([prob_tiled, prob_tiled.permute(1, 0, 2)],
									dim=2)

		kernel_prob_pairs = self.torch_kernel(prob_pairs)

		numerator = dot_product*kernel_prob_pairs
		return torch.sum(numerator)/torch.pow(torch.tensor(correct_mask.shape[0]).type(torch.FloatTensor),2)



class MMCE_weighted(nn.Module):
	"""
	Computes MMCE_w loss.
	"""
	def __init__(self):
		super(MMCE_weighted, self).__init__()
		self.device = None

	def torch_kernel(self, matrix):
		return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(0.4))

	def get_pairs(self, tensor1, tensor2):
		correct_prob_tiled = tensor1.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)
		incorrect_prob_tiled = tensor2.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)

		correct_prob_pairs = torch.cat([correct_prob_tiled, correct_prob_tiled.permute(1, 0, 2)],
									dim=2)
		incorrect_prob_pairs = torch.cat([incorrect_prob_tiled, incorrect_prob_tiled.permute(1, 0, 2)],
									dim=2)

		correct_prob_tiled_1 = tensor1.unsqueeze(1).repeat(1, tensor2.shape[0]).unsqueeze(2)
		incorrect_prob_tiled_1 = tensor2.unsqueeze(1).repeat(1, tensor1.shape[0]).unsqueeze(2)

		correct_incorrect_pairs = torch.cat([correct_prob_tiled_1, incorrect_prob_tiled_1.permute(1, 0, 2)],
									dim=2)
		return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs

	def get_out_tensor(self, tensor1, tensor2):
		return torch.mean(tensor1*tensor2)

	def forward(self, input, target):
		self.device = input.get_device()
		if input.dim()>2:
			input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C

		target = target.view(-1)  #For CIFAR-10 and CIFAR-100, target.shape is [N] to begin with

		predicted_probs = F.softmax(input, dim=1)
		predicted_probs, predicted_labels = torch.max(predicted_probs, 1)

		correct_mask = torch.where(torch.eq(predicted_labels, target),
									torch.ones(predicted_labels.shape).to(self.device),
									torch.zeros(predicted_labels.shape).to(self.device))

		k = torch.sum(correct_mask).type(torch.int64)
		k_p = torch.sum(1.0 - correct_mask).type(torch.int64)
		cond_k = torch.where(torch.eq(k,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
		cond_k_p = torch.where(torch.eq(k_p,0),torch.tensor(0).to(self.device),torch.tensor(1).to(self.device))
		k = torch.max(k, torch.tensor(1).to(self.device))*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
		k_p = torch.max(k_p, torch.tensor(1).to(self.device))*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
											(correct_mask.shape[0] - 2))


		correct_prob, _ = torch.topk(predicted_probs*correct_mask, k)
		incorrect_prob, _ = torch.topk(predicted_probs*(1 - correct_mask), k_p)

		correct_prob_pairs, incorrect_prob_pairs,\
			   correct_incorrect_pairs = self.get_pairs(correct_prob, incorrect_prob)

		correct_kernel = self.torch_kernel(correct_prob_pairs)
		incorrect_kernel = self.torch_kernel(incorrect_prob_pairs)
		correct_incorrect_kernel = self.torch_kernel(correct_incorrect_pairs)  

		sampling_weights_correct = torch.mm((1.0 - correct_prob).unsqueeze(1), (1.0 - correct_prob).unsqueeze(0))

		correct_correct_vals = self.get_out_tensor(correct_kernel,
														  sampling_weights_correct)
		sampling_weights_incorrect = torch.mm(incorrect_prob.unsqueeze(1), incorrect_prob.unsqueeze(0))

		incorrect_incorrect_vals = self.get_out_tensor(incorrect_kernel,
														  sampling_weights_incorrect)
		sampling_correct_incorrect = torch.mm((1.0 - correct_prob).unsqueeze(1), incorrect_prob.unsqueeze(0))

		correct_incorrect_vals = self.get_out_tensor(correct_incorrect_kernel,
														  sampling_correct_incorrect)

		correct_denom = torch.sum(1.0 - correct_prob)
		incorrect_denom = torch.sum(incorrect_prob)

		m = torch.sum(correct_mask)
		n = torch.sum(1.0 - correct_mask)
		mmd_error = 1.0/(m*m + 1e-5) * torch.sum(correct_correct_vals) 
		mmd_error += 1.0/(n*n + 1e-5) * torch.sum(incorrect_incorrect_vals)
		mmd_error -= 2.0/(m*n + 1e-5) * torch.sum(correct_incorrect_vals)
		return torch.max((cond_k*cond_k_p).type(torch.FloatTensor).to(self.device).detach()*torch.sqrt(mmd_error + 1e-10), torch.tensor(0.0).to(self.device))
