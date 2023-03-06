import logging
import torch
import torch.nn as nn
import torchvision.models as models

class ViTClassifier(nn.Module):
    def __init__(self, args, checkpoint_path=None):
        super().__init__()
        self.args = args
        model = None
        self.drop_nl = 0
        self.bottleneck = nn.Identity()
        if args.drop_nl == 1:
            print("Included Dropout + NL in classifier..")
            self.dropout = nn.Dropout(p=0.5)
            self.relu = nn.ReLU()
            self.drop_nl = 1
        
        if args.arch == "vit_b_16":
            model = models.__dict__[args.arch](pretrained=True)
            rel_dim = model.heads.head.in_features
            self.encoder = model
            self.encoder.heads.head = nn.Identity()
            if not self.use_bottleneck:
                self._output_dim = rel_dim
            else:
                lin_bottleneck = nn.Linear(rel_dim, args.bottleneck_dim)
                if self.drop_nl == 1:
                    bn = nn.Sequential(
                        *[
                            self.dropout,
                            self.relu,
                            nn.BatchNorm1d(args.bottleneck_dim),
                        ]
                    )
                else:
                    bn = nn.BatchNorm1d(args.bottleneck_dim)
                self.bottleneck = nn.Sequential(lin_bottleneck, bn)
                self._output_dim = args.bottleneck_dim
        elif args.arch == "dino_vitb16":
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            modules = list(model.children())[:-1]
            rel_dim = modules[-1].normalized_shape[0]
            self.encoder = model
            if not self.use_bottleneck:
                self._output_dim = rel_dim
            else:
                lin_bottleneck = nn.Linear(rel_dim, args.bottleneck_dim)
                if self.drop_nl == 1:
                    bn = nn.Sequential(
                        *[
                            self.dropout,
                            self.relu,
                            nn.BatchNorm1d(args.bottleneck_dim),
                        ]
                    )
                else:
                    bn = nn.BatchNorm1d(args.bottleneck_dim)
                self.bottleneck = nn.Sequential(lin_bottleneck, bn)
                self._output_dim = args.bottleneck_dim
                
        self.fc = nn.Linear(self.output_dim, args.num_classes)

        if self.use_weight_norm:
            self.fc = nn.utils.weight_norm(self.fc, dim=args.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)
            
    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)
        
        if self.use_bottleneck:
            feat = self.bottleneck(feat)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        backbone_params.extend(self.encoder.parameters())
        if self.use_bottleneck:
            extra_params.extend(self.bottleneck.parameters())
        extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0
            
            
        

class Classifier(nn.Module):
    def __init__(self, args, checkpoint_path=None):
        super().__init__()
        self.args = args
        model = None
        self.drop_nl = 0
        if args.drop_nl == 1:
            print("Included Dropout + NL in classifier..")
            self.dropout = nn.Dropout(p=0.5)
            self.relu = nn.ReLU()
            self.drop_nl = 1

        # 1) ResNet backbone (up to penultimate layer)
        if not self.use_bottleneck:
            model = models.__dict__[args.arch](pretrained=True)
            modules = list(model.children())[:-1]
            self.encoder = nn.Sequential(*modules)
            self._output_dim = model.fc.in_features
        # 2) ResNet backbone + bottlenck (last fc as bottleneck)
        else:
            model = models.__dict__[args.arch](pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, args.bottleneck_dim)
            if self.drop_nl == 1:
                bn = nn.Sequential(
                    *[
                        self.dropout,
                        self.relu,
                        nn.BatchNorm1d(args.bottleneck_dim),
                    ]
                )
            else:
                bn = nn.BatchNorm1d(args.bottleneck_dim)
            # bn = nn.BatchNorm1d(args.bottleneck_dim)
            self.encoder = nn.Sequential(model, bn)
            self._output_dim = args.bottleneck_dim

        self.fc = nn.Linear(self.output_dim, args.num_classes)

        if self.use_weight_norm:
            self.fc = nn.utils.weight_norm(self.fc, dim=args.weight_norm_dim)

        if checkpoint_path:
            self.load_from_checkpoint(checkpoint_path)

    def forward(self, x, return_feats=False):
        # 1) encoder feature
        feat = self.encoder(x)
        feat = torch.flatten(feat, 1)

        logits = self.fc(feat)

        if return_feats:
            return feat, logits
        return logits

    def load_from_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = dict()
        for name, param in checkpoint["state_dict"].items():
            # get rid of 'module.' prefix brought by DDP
            name = name.replace("module.", "")
            state_dict[name] = param
        msg = self.load_state_dict(state_dict, strict=False)
        logging.info(
            f"Loaded from {checkpoint_path}; missing params: {msg.missing_keys}"
        )

    def get_bn_params(self):
        """
        Get only batchnorm parameters
        """
        bn_params = []
        for name, param in self.encoder.named_parameters():
            if "bn" in name:
                bn_params.append(param)
        bn_params = [param for param in bn_params if param.requires_grad]
        return bn_params

    def get_params(self):
        """
        Backbone parameters use 1x lr; extra parameters use 10x lr.
        """
        backbone_params = []
        extra_params = []
        # case 1)
        if not self.use_bottleneck:
            backbone_params.extend(self.encoder.parameters())
        # case 2)
        else:
            resnet = self.encoder[0]
            for module in list(resnet.children())[:-1]:
                backbone_params.extend(module.parameters())
            # bottleneck fc + (bn) + classifier fc
            extra_params.extend(resnet.fc.parameters())
            extra_params.extend(self.encoder[1].parameters())
            extra_params.extend(self.fc.parameters())

        # exclude frozen params
        backbone_params = [param for param in backbone_params if param.requires_grad]
        extra_params = [param for param in extra_params if param.requires_grad]

        return backbone_params, extra_params

    @property
    def num_classes(self):
        return self.fc.weight.shape[0]

    @property
    def output_dim(self):
        return self._output_dim

    @property
    def use_bottleneck(self):
        return self.args.bottleneck_dim > 0

    @property
    def use_weight_norm(self):
        return self.args.weight_norm_dim >= 0
