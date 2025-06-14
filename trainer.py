import datetime
import json
import os
import sys
import time
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from timm.models.vision_transformer import (
    vit_base_patch16_224,
    vit_base_patch16_384,
    vit_large_patch16_224,
)
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

import datasets
from clip import clip
from models import PeftModelFromCLIP, PeftModelFromViT, ZeroShotCLIP
from models.satmae_vit import mae_vit_large_patch16, MAEViTAdapter
from models.peft_modules import FLoRA
from utils.evaluator import Evaluator
from utils.logger import logger
from utils.losses import (
    BalancedSoftmaxLoss,
    ClassBalancedLoss,
    FocalLoss,
    GeneralizedReweightLoss,
    LADELoss,
    LDAMLoss,
    LogitAdjustedLoss,
)
from utils.meter import AverageMeter
from utils.samplers import DownSampler
from utils.templates import ZEROSHOT_TEMPLATES

CLASS_MEAN_FNAME = {
    "IN21K-ViT-B/16": "clsmean_in21k_vit_b16.pth",
    "IN21K-ViT-B/16@384px": "clsmean_in21k_vit_b16_384px.pth",
    "IN21K-ViT-L/16": "clsmean_in21k_vit_l16.pth",
    "SatMAE-ViT-B/16": "clsmean_satmae_vit_b16.pth",
    "SatMAE-ViT-L/16": "clsmean_satmae_vit_l16.pth",
}

TEXT_FEAT_FNAME = {
    "CLIP-ViT-B/16": "txtfeat_clip_vit_b16.pth",
    "CLIP-ViT-L/14": "txtfeat_clip_vit_l14.pth",
    "CLIP-ViT-L/14@336px": "txtfeat_clip_vit_l14_336px.pth",
}


def load_clip_to_cpu(backbone_name, prec):
    backbone_name = backbone_name.lstrip("CLIP-")
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu").eval()

    model = clip.build_model(state_dict or model.state_dict())

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp32" or prec == "amp":
        # CLIP's default precision is fp16
        model.float()

    return model


def load_vit_to_cpu(backbone_name, prec):
    if backbone_name == "IN21K-ViT-B/16":
        model = vit_base_patch16_224(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-B/16@384px":
        model = vit_base_patch16_384(pretrained=True).eval()
    elif backbone_name == "IN21K-ViT-L/16":
        model = vit_large_patch16_224(pretrained=True).eval()
    elif backbone_name == "SatMAE-ViT-L/16":
        # Load MAE model
        mae_model = mae_vit_large_patch16().eval()

        # Load pretrained weights
        checkpoint = torch.load("./data/fmow_pretrain.pth", map_location="cpu")
        msg = mae_model.load_state_dict(checkpoint["model"], strict=False)
        print(f"\nLoading SatMAE checkpoint: {msg}")

        # Wrap with adapter
        model = MAEViTAdapter(mae_model)

    assert prec in ["fp16", "fp32", "amp"]
    if prec == "fp16":
        model.half()

    return model


class Trainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self._writer = None
        self.device = device
        self.local_rank = dist.get_rank() if dist.is_initialized() else 1
        self.is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.debug_mode = os.environ.get("DEBUG_MODE", "").lower() == "true"

    def initialize(self):
        """Explicit initialization method to be called after constructor"""
        self.build_data_loader()
        self.build_model()
        self.evaluator = Evaluator(self.cfg, self.many_idxs, self.med_idxs, self.few_idxs)

        if not self.debug_mode and self.is_main_process:
            writer_dir = os.path.join(self.cfg.output_dir, "tensorboard")
            os.makedirs(writer_dir, exist_ok=True)
            print(f"Initialize tensorboard (log_dir={writer_dir})")
            self._writer = SummaryWriter(log_dir=writer_dir)

    def build_data_loader(self):
        cfg = self.cfg
        root = cfg.root
        resolution = cfg.resolution
        expand = cfg.expand

        if cfg.backbone.startswith("CLIP"):
            mean = [0.48145466, 0.4578275, 0.40821073]
            std = [0.26862954, 0.26130258, 0.27577711]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
        print("mean:", mean)
        print("std:", std)

        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        transform_plain = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        if cfg.tte:
            if cfg.tte_mode == "fivecrop":
                transform_test = transforms.Compose(
                    [
                        transforms.Resize(resolution + expand),
                        transforms.FiveCrop(resolution),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                        transforms.Normalize(mean, std),
                    ]
                )
            elif cfg.tte_mode == "tencrop":
                transform_test = transforms.Compose(
                    [
                        transforms.Resize(resolution + expand),
                        transforms.TenCrop(resolution),
                        transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                        transforms.Normalize(mean, std),
                    ]
                )
            elif cfg.tte_mode == "randaug":
                _resize_and_flip = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(resolution),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                    ]
                )
                transform_test = transforms.Compose(
                    [
                        transforms.Lambda(lambda image: torch.stack([_resize_and_flip(image) for _ in range(cfg.randaug_times)])),
                        transforms.Normalize(mean, std),
                    ]
                )
        else:
            transform_test = transforms.Compose(
                [
                    transforms.Resize(resolution * 8 // 7),
                    transforms.CenterCrop(resolution),
                    transforms.Lambda(lambda crop: torch.stack([transforms.ToTensor()(crop)])),
                    transforms.Normalize(mean, std),
                ]
            )

        train_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_train)
        train_init_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_plain)
        train_test_dataset = getattr(datasets, cfg.dataset)(root, train=True, transform=transform_test)
        test_dataset = getattr(datasets, cfg.dataset)(root, train=False, transform=transform_test)

        self.num_classes = train_dataset.num_classes
        self.cls_num_list = train_dataset.cls_num_list
        self.classnames = train_dataset.classnames

        if "DOTA" in cfg.dataset or "FUSRS" in cfg.dataset:
            self.many_idxs = np.array(train_dataset.many_idxs)
            self.med_idxs = np.array(train_dataset.med_idxs)
            self.few_idxs = np.array(train_dataset.few_idxs)
        else:
            if cfg.dataset in ["CIFAR100", "CIFAR100_IR10", "CIFAR100_IR50"]:
                split_cls_num_list = datasets.CIFAR100_IR100(root, train=True).cls_num_list
            else:
                split_cls_num_list = self.cls_num_list
            self.many_idxs = (np.array(split_cls_num_list) > 100).nonzero()[0]
            self.med_idxs = ((np.array(split_cls_num_list) >= 20) & (np.array(split_cls_num_list) <= 100)).nonzero()[0]
            self.few_idxs = (np.array(split_cls_num_list) < 20).nonzero()[0]

        if cfg.init_head == "1_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=1)
        elif cfg.init_head == "10_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=10)
        elif cfg.init_head == "100_shot":
            init_sampler = DownSampler(train_init_dataset, n_max=100)
        else:
            init_sampler = None

        train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None

        # Calculate effective batch size and steps
        self.accum_step = cfg.accum_step or 1
        self.eff_batch_size = cfg.batch_size  # Total logical batch size
        self.per_gpu_batch_size = self.eff_batch_size // (self.accum_step * self.world_size)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.per_gpu_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        self.train_init_loader = DataLoader(
            train_init_dataset,
            batch_size=64,
            sampler=init_sampler,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        self.train_test_loader = DataLoader(
            train_test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

        print("Total training samples:", sum(self.cls_num_list))

    def build_model(self):
        cfg = self.cfg
        classnames = self.classnames
        num_classes = len(classnames)

        print("Building model")

        # 1. Model Loading - unified for all modes
        if cfg.backbone.startswith("CLIP"):
            print(f"Loading CLIP (backbone: {cfg.backbone})")
            clip_model = load_clip_to_cpu(cfg.backbone, cfg.prec)
            if cfg.zero_shot:
                self.model = ZeroShotCLIP(clip_model)
                self.model.to(self.device)
                self.tuner = None
                self.head = None

                # CLIP specific zero-shot initialization
                template = "a photo of a {}."
                prompts = self.get_tokenized_prompts(classnames, template)
                self.model.init_text_features(prompts)
                return  # Early return for CLIP zero-shot
            else:
                self.model = PeftModelFromCLIP(cfg, clip_model, num_classes)

        elif cfg.backbone.startswith("IN21K-ViT"):
            print(f"Loading ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)

        elif cfg.backbone.startswith("SatMAE-ViT"):
            print(f"Loading SatMAE-ViT (backbone: {cfg.backbone})")
            vit_model = load_vit_to_cpu(cfg.backbone, cfg.prec)
            self.model = PeftModelFromViT(cfg, vit_model, num_classes)

        # 2. Model setup
        self.model.to(self.device)
        self.tuner = self.model.tuner
        self.head = self.model.head

        # 3. Head initialization (for both zero-shot and training)
        if cfg.init_head == "text_feat":
            if not cfg.backbone.startswith("CLIP"):
                print("Text feature initialization not implemented for non-CLIP models")
            else:
                text_feat_fname = TEXT_FEAT_FNAME.get(cfg.backbone)
                if text_feat_fname is None:
                    raise ValueError(f"No text feature filename defined for backbone {cfg.backbone}")
                full_text_feat_path = os.path.join(cfg.head_init_folder, text_feat_fname)
                self.init_head_text_feat(full_text_feat_path)

        elif cfg.init_head in ["class_mean", "1_shot", "10_shot", "100_shot"]:
            class_mean_fname = CLASS_MEAN_FNAME.get(cfg.backbone)
            if class_mean_fname is None:
                raise ValueError(f"No class mean filename defined for backbone {cfg.backbone}")
            full_class_mean_path = os.path.join(cfg.head_init_folder, class_mean_fname)
            self.init_head_class_mean(full_class_mean_path)

        elif cfg.init_head == "linear_probe":
            self.init_head_linear_probe()

        elif cfg.init_head is not None:
            print("No initialization with head")

        # 4. Training setup (skip for zero-shot)
        if not (cfg.zero_shot or cfg.test_train or cfg.test_only):
            self.build_optimizer()
            self.build_criterion()
            torch.cuda.empty_cache()

        # 5. Final model setup
        if self.world_size > 1:
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

    def build_optimizer(self):
        logger.section("Building Optimizer")
        cfg = self.cfg

        # 首先处理 head_only 模式
        if cfg.head_only:
            print("Head-only mode: Only training the classifier head")
            # 冻结所有参数
            for param in self.model.parameters():
                param.requires_grad_(False)
            # 只开启分类器头部的参数
            for param in self.head.parameters():
                param.requires_grad_(True)
            # 创建优化器，只包含分类器头部的参数
            self.optim = torch.optim.SGD(
                self.head.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
            )

        # 原有的 fine_tuning 逻辑
        elif cfg.fine_tuning:
            print("Fine-tuning: Tuning the entire model and the head")
            print("Removing the text encoder and proj after text feature extraction")
            if cfg.backbone.startswith("CLIP"):
                self.model.text_encoder = None
                self.model.image_encoder.proj = None
            print("Turning on all gradients in the model")
            for param in self.model.parameters():
                param.requires_grad_(True)

            # Optimizer for the entire model
            self.optim = torch.optim.SGD(
                self.model.parameters(),
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
            )
        # Optimizer for the tuner and head
        else:
            print("Tuner mode: Only tuning the tuner and head")
            print("Turning off gradients in the model")
            for param in self.model.parameters():
                param.requires_grad_(False)

            # Turn on gradients for tuner and head
            print("Turning on gradients in the tuner and head")
            for param in self.tuner.parameters():
                param.requires_grad_(True)
            for param in self.head.parameters():
                param.requires_grad_(True)

            # Collect parameters for optimizer
            params = []

            # FLoRA uses different learning rates for different modules
            if cfg.use_flora and hasattr(self.tuner, "flora_list"):
                flora_params, other_tuner_params = self._collect_flora_parameters()
                if other_tuner_params:
                    params.append({"params": other_tuner_params, "lr": cfg.lr})
                params.extend(flora_params)
            else:
                # If no FLoRA, use all tuner parameters at default learning rate
                params.append({"params": self.tuner.parameters(), "lr": cfg.lr})

            # Head parameters
            params.append({"params": self.head.parameters(), "lr": cfg.lr})

            # Create optimizer
            self.optim = torch.optim.SGD(
                params,
                weight_decay=cfg.weight_decay,
                momentum=cfg.momentum,
            )

        # Print parameter counts
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total params: {total_params}")
        if self.tuner is not None:
            tuned_params = sum(p.numel() for p in self.tuner.parameters() if p.requires_grad)
            print(f"Tuner params: {tuned_params}")
        head_params = sum(p.numel() for p in self.head.parameters())
        print(f"Head params: {head_params}")

        # Set up learning rate scheduler
        if cfg.scheduler == "CosineAnnealingLR":
            self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optim, cfg.num_epochs)
        elif cfg.scheduler == "ConstantLR":
            self.sched = torch.optim.lr_scheduler.ConstantLR(self.optim, factor=1.0, total_iters=0)
        else:
            raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")

        # Set up gradient scaler for mixed precision
        self.scaler = GradScaler() if cfg.prec == "amp" else None

    def _collect_flora_parameters(self):
        cfg = self.cfg.flora.optimizer
        flora_params = []
        other_tuner_params = []

        def _parse_range(range_str):
            if isinstance(range_str, int):
                return [range_str]
            if "-" in str(range_str):
                start, end = map(int, str(range_str).split("-"))
                return list(range(start, end + 1))
            return [int(range_str)]

        def _get_lr(layer_idx, module_name):
            lr = cfg.default_lr

            # Check layer-specific learning rate
            for layer_range, layer_lr in cfg.lr_config.get("layers", {}).items():
                if layer_idx in _parse_range(layer_range):
                    lr = layer_lr
                    break

            # Check module-specific learning rate
            module_lr = cfg.lr_config.get("modules", {}).get(module_name)
            if module_lr is not None:
                lr = module_lr

            # Check specific layer-module learning rate
            for spec, spec_lr in cfg.lr_config.get("specific", {}).items():
                spec_layer, spec_module = spec.split(".")
                if layer_idx in _parse_range(spec_layer) and module_name == spec_module:
                    lr = spec_lr
                    break

            return lr

        # Iterate over tuner modules
        for name, module in self.tuner.named_modules():
            if isinstance(module, FLoRA):
                parts = name.split(".")
                if len(parts) >= 3 and parts[0] == "flora_list":
                    layer_idx = int(parts[1])
                    module_name = parts[2]

                    # Get learning rate for this specific module
                    lr = _get_lr(layer_idx, module_name)

                    # Add module parameters with custom learning rate
                    flora_params.append({"params": module.parameters(), "lr": lr})
            else:
                # Add other tuner parameters
                for param in module.parameters(recurse=False):
                    if param.requires_grad:
                        other_tuner_params.append(param)

        return flora_params, other_tuner_params

    def build_criterion(self):
        logger.section("Building Criterion")
        cfg = self.cfg
        cls_num_list = torch.Tensor(self.cls_num_list).to(self.device)

        print(f"Using loss type: {cfg.loss_type}")
        if cfg.loss_type == "CE":
            print("Standard Cross Entropy Loss")
            self.criterion = nn.CrossEntropyLoss()
        elif cfg.loss_type == "Focal":  # https://arxiv.org/abs/1708.02002
            print("Focal Loss - Focusing on hard examples")
            self.criterion = FocalLoss()
        elif cfg.loss_type == "LDAM":  # https://arxiv.org/abs/1906.07413
            print(f"LDAM Loss - Label-Distribution-Aware Margin with scale={cfg.scale}")
            self.criterion = LDAMLoss(cls_num_list=cls_num_list, s=cfg.scale)
        elif cfg.loss_type == "CB":  # https://arxiv.org/abs/1901.05555
            print("Class Balanced Loss - Theoretical framework for long-tailed learning")
            self.criterion = ClassBalancedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "GRW":  # https://arxiv.org/abs/2103.16370
            print("Generalized Reweight Loss - Adaptive class reweighting")
            self.criterion = GeneralizedReweightLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "BS":  # https://arxiv.org/abs/2007.10740
            print("Balanced Softmax Loss - Implicit balanced learning")
            self.criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LA":  # https://arxiv.org/abs/2007.07314
            print("Logit Adjusted Loss - Post-hoc adjustment of logits")
            self.criterion = LogitAdjustedLoss(cls_num_list=cls_num_list)
        elif cfg.loss_type == "LADE":  # https://arxiv.org/abs/2012.00321
            print("LADE Loss - Label distribution aware margin")
            self.criterion = LADELoss(cls_num_list=cls_num_list)
        else:
            raise ValueError(f"Unknown loss type: {cfg.loss_type}")

        # Print class distribution summary
        total_samples = sum(self.cls_num_list)
        max_samples = max(self.cls_num_list)
        min_samples = min(self.cls_num_list)
        imbalance_ratio = max_samples / min_samples
        print("\nClass Distribution Summary:")
        print(f"Total samples: {total_samples}")
        print(f"Max samples per class: {max_samples}")
        print(f"Min samples per class: {min_samples}")
        print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    def get_tokenized_prompts(self, classnames, template):
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        prompts = torch.cat([clip.tokenize(p) for p in prompts])
        prompts = prompts.to(self.device)
        return prompts

    @torch.no_grad()
    def init_head_text_feat(self, text_feat_path):
        if os.path.exists(text_feat_path):
            print(f"Loading text features from {text_feat_path}")
            text_features = torch.load(text_feat_path, map_location=self.device)
        elif self.is_main_process and self.world_size == 1:
            print(f"Text features not found at {text_feat_path}.")
            print("Computing text features...")
            text_features = self.compute_text_features()
            print(f"Saving text features to {text_feat_path}")
            os.makedirs(os.path.dirname(text_feat_path), exist_ok=True)
            torch.save(text_features, text_feat_path)
        else:
            raise RuntimeError("Text feature initialization not supported in multi-GPU mode.")

        self.head.apply_weight(text_features)

    @torch.no_grad()
    def compute_text_features(self):
        cfg = self.cfg
        classnames = self.classnames

        print("Computing text features")
        if cfg.prompt == "ensemble":
            all_text_features = []
            for template in tqdm(ZEROSHOT_TEMPLATES["imagenet"]):
                prompts = self.get_tokenized_prompts(classnames, template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_text_features.append(text_features)
            all_text_features = torch.stack(all_text_features)
            text_features = all_text_features.mean(dim=0)
        elif cfg.prompt == "descriptor":
            with open("utils/descriptors_imagenet.json") as f:
                descriptors = json.load(f)
            template = "{}"
            all_class_features = []
            for cn in tqdm(classnames):
                prompts = self.get_tokenized_prompts(descriptors[cn], template)
                text_features = self.model.encode_text(prompts)
                text_features = F.normalize(text_features, dim=-1)
                all_class_features.append(text_features.mean(dim=0))
            text_features = torch.stack(all_class_features)
        elif cfg.prompt == "classname":
            template = "{}"
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)
        elif cfg.prompt == "default":
            template = "a photo of a {}."
            prompts = self.get_tokenized_prompts(classnames, template)
            text_features = self.model.encode_text(prompts)
            text_features = F.normalize(text_features, dim=-1)

        if cfg.backbone.startswith("CLIP-ViT"):
            text_features = text_features @ self.model.image_encoder.proj.t()
            text_features = F.normalize(text_features, dim=-1)

        return text_features

    @torch.no_grad()
    def init_head_class_mean(self, class_mean_path):
        if os.path.exists(class_mean_path):
            print(f"Loading class means from {class_mean_path}")
            class_means = torch.load(class_mean_path, map_location=self.device)
        elif self.is_main_process and self.world_size == 1:
            print(f"Class mean not found at {class_mean_path}.")
            print("Computing class means...")
            class_means = self.compute_class_means()
            print(f"Saving class means to {class_mean_path}")
            os.makedirs(os.path.dirname(class_mean_path), exist_ok=True)
            torch.save(class_means, class_mean_path)
        else:
            raise RuntimeError("Class mean initialization not supported in multi-GPU mode.")

        self.head.apply_weight(class_means)

    @torch.no_grad()
    def compute_class_means(self):
        all_features = []
        all_labels = []

        total_batches = len(self.train_init_loader)

        for batch in tqdm(
            self.train_init_loader,
            ascii=True,
            total=total_batches,
            miniters=int(total_batches // 10),
            maxinterval=600,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            file=sys.stdout,
        ):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        sorted_index = all_labels.argsort()
        all_features = all_features[sorted_index]
        all_labels = all_labels[sorted_index]

        unique_labels, label_counts = torch.unique(all_labels, return_counts=True)

        class_means = [None] * self.num_classes
        idx = 0
        for i, cnt in zip(unique_labels, label_counts):
            class_means[i] = all_features[idx : idx + cnt].mean(dim=0, keepdim=True)
            idx += cnt
        class_means = torch.cat(class_means, dim=0)
        class_means = F.normalize(class_means, dim=-1)

        return class_means

    @torch.no_grad()
    def init_head_linear_probe(self):
        print("Initialize head with linear probing")
        all_features = []
        all_labels = []

        total_batches = len(self.train_init_loader)

        for batch in tqdm(
            self.train_init_loader,
            ascii=True,
            total=total_batches,
            miniters=int(total_batches // 10),
            maxinterval=600,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            file=sys.stdout,
        ):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            feature = self.model(image, use_tuner=False, return_feature=True)

            all_features.append(feature)
            all_labels.append(label)

        all_features = torch.cat(all_features, dim=0).cpu()
        all_labels = torch.cat(all_labels, dim=0).cpu()

        clf = LogisticRegression(solver="lbfgs", max_iter=100, penalty="l2", class_weight="balanced").fit(all_features, all_labels)
        class_weights = torch.from_numpy(clf.coef_).to(all_features.dtype).to(self.device)
        class_weights = F.normalize(class_weights, dim=-1)

        self.head.apply_weight(class_weights)

    def train(self):
        logger.section("Training model")
        cfg = self.cfg

        # Initialize meters
        meters = {
            "batch_time": AverageMeter(),
            "data_time": AverageMeter(),
            "loss": AverageMeter(ema=True),
            "acc": AverageMeter(ema=True),
            "cls_meters": [AverageMeter(ema=True) for _ in range(self.num_classes)],
        }

        # Training start time
        time_start = time.time()

        # Calculate batch sizes and steps
        micro_batches_per_epoch = len(self.train_loader)
        _logic_batch_num = micro_batches_per_epoch // self.accum_step

        for epoch in range(cfg.num_epochs):
            if dist.is_initialized():
                self.train_loader.sampler.set_epoch(epoch)

            # Set train mode
            if self.tuner is not None:
                self.tuner.train()
            if cfg.fine_tuning:
                self.model.train()

            batch_start = time.time()
            self.optim.zero_grad()

            for batch_idx, batch in enumerate(self.train_loader):
                # Calculate logical batch index
                _logic_batch_idx = batch_idx // self.accum_step
                _is_logic_batch = (batch_idx + 1) % self.accum_step == 0

                # Measure data loading time
                meters["data_time"].update(time.time() - batch_start)

                # Move to device
                images = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                # Forward pass and loss calculation
                if cfg.prec == "amp":
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                        loss_scaled = loss / self.accum_step
                        self.scaler.scale(loss_scaled).backward()

                    if _is_logic_batch:
                        self.scaler.step(self.optim)
                        self.scaler.update()
                        self.optim.zero_grad()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss_scaled = loss / self.accum_step
                    loss_scaled.backward()

                    if _is_logic_batch:
                        self.optim.step()
                        self.optim.zero_grad()

                # Update metrics and log progress on logical batch boundary
                if _is_logic_batch:
                    self._update_metrics(outputs, labels, loss, meters)

                    # Log progress at specified intervals
                    if ((_logic_batch_idx + 1) % cfg.print_freq == 0) or (_logic_batch_idx + 1 == _logic_batch_num):
                        self._log_progress(
                            epoch,
                            _logic_batch_idx,
                            _logic_batch_num,
                            meters,
                            time_start,
                            cfg.num_epochs,
                        )

                # Measure batch time
                meters["batch_time"].update(time.time() - batch_start)
                batch_start = time.time()

            # Step scheduler after each epoch
            self.sched.step()
            torch.cuda.empty_cache()

        # Training finished - save model and evaluate
        print("Finished training")
        print("Note: Printed training accuracy is approximate. Use test_train=True for precise evaluation.")

        elapsed = str(datetime.timedelta(seconds=int(time.time() - time_start)))
        print(f"Total training time: {elapsed}")

        self.save_model(cfg.output_dir)
        self.test()

        if self._writer is not None:
            self._writer.close()

    def _update_metrics(self, outputs, labels, loss, meters):
        """Update training metrics on logical batch boundary"""
        with torch.no_grad():
            preds = outputs.argmax(dim=1)
            correct = preds.eq(labels).float()
            acc = correct.mean().mul_(100.0)

            # Update meters
            meters["loss"].update(loss.item())
            meters["acc"].update(acc.item())

            # Per-class accuracy
            for c, y in zip(correct, labels):
                meters["cls_meters"][y].update(c.mul_(100.0).item(), n=1)

    def _log_progress(
        self,
        epoch,
        _logic_batch_idx,
        _logic_batch_num,
        meters,
        time_start,
        num_epochs,
    ):
        """Log training progress"""
        # Calculate accuracies
        cls_accs = [m.avg for m in meters["cls_meters"]]
        mean_acc = np.mean(cls_accs)
        many_acc = np.mean([cls_accs[i] for i in self.many_idxs])
        med_acc = np.mean([cls_accs[i] for i in self.med_idxs])
        few_acc = np.mean([cls_accs[i] for i in self.few_idxs])

        # Get current learning rate and calculate time estimates
        current_lr = self.optim.param_groups[0]["lr"]
        elapsed_time = time.time() - time_start

        # Calculate remaining batches and ETA
        _remain = (_logic_batch_num - _logic_batch_idx - 1) + (num_epochs - epoch - 1) * _logic_batch_num
        _eta_seconds = meters["batch_time"].avg * _remain
        eta = str(datetime.timedelta(seconds=int(_eta_seconds)))

        # Print progress
        info = [
            f"epoch [{epoch + 1}/{num_epochs}]",
            f"batch [{_logic_batch_idx + 1}/{_logic_batch_num}]",
            f"time {meters['batch_time'].val:.3f} ({meters['batch_time'].avg:.3f})",
            f"data {meters['data_time'].val:.3f} ({meters['data_time'].avg:.3f})",
            f"loss {meters['loss'].val:.4f} ({meters['loss'].avg:.4f})",
            f"acc {meters['acc'].val:.4f} ({meters['acc'].avg:.4f})",
            f"(mean {mean_acc:.4f} many {many_acc:.4f} med {med_acc:.4f} few {few_acc:.4f})",
            f"lr {current_lr:.4e}",
            f"elapsed {str(datetime.timedelta(seconds=int(elapsed_time)))}",
            f"eta {eta}",
        ]
        print(" ".join(info))

        # TensorBoard logging
        if self._writer is not None:
            n_iter = epoch * _logic_batch_num + _logic_batch_idx
            self._log_tensorboard(
                n_iter,
                current_lr,
                meters,
                mean_acc,
                many_acc,
                med_acc,
                few_acc,
            )

    @torch.no_grad()
    def test(self, mode="test"):
        cfg = self.cfg
        logger.section("Evaluating model")
        if self.tuner is not None:
            self.tuner.eval()
        if self.head is not None:
            self.head.eval()
        if cfg.fine_tuning:
            self.model.eval()
        self.evaluator.reset()

        if mode == "train":
            print("Evaluate on the train set")
            data_loader = self.train_test_loader
        elif mode == "test":
            print("Evaluate on the test set")
            data_loader = self.test_loader

        total_batches = len(data_loader)

        for batch in tqdm(
            data_loader,
            ascii=True,
            total=total_batches,
            miniters=int(total_batches // 10),
            maxinterval=600,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            file=sys.stdout,
        ):
            image = batch[0]
            label = batch[1]

            image = image.to(self.device)
            label = label.to(self.device)

            _bsz, _ncrops, _c, _h, _w = image.size()
            image = image.view(_bsz * _ncrops, _c, _h, _w)

            if _ncrops <= 5:
                output = self.model(image)
                output = output.view(_bsz, _ncrops, -1).mean(dim=1)
            else:
                # CUDA out of memory
                output = []
                image = image.view(_bsz, _ncrops, _c, _h, _w)
                for k in range(_ncrops):
                    output.append(self.model(image[:, k]))
                output = torch.stack(output).mean(dim=0)

            self.evaluator.process(output, label)

        # INFO: Disable distributed evaluation for best re-producibility
        # if dist.is_initialized():
        #     results = self.evaluator.distributed_evaluate(
        #         self.model, data_loader, self.device
        #     )
        #     dist.barrier()
        # else:
        results = self.evaluator.evaluate()

        if self.is_main_process:
            print(results)
            if self._writer is not None:
                for k, v in results.items():
                    self._writer.add_scalar(f"test/{k}", v)

        return list(results.values())[0]

    def save_model(self, directory):
        cfg = self.cfg
        if self.is_main_process:
            checkpoint = {}

            if cfg.fine_tuning:
                model_dict = self.model.state_dict()
                checkpoint["model"] = model_dict
            else:
                tuner_dict = self.tuner.state_dict()
                head_dict = self.head.state_dict()
                checkpoint["tuner"] = tuner_dict
                checkpoint["head"] = head_dict

            # remove 'module.' in state_dict's keys
            for key in checkpoint.keys():
                state_dict = checkpoint[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if k.startswith("module."):
                        k = k[7:]
                    new_state_dict[k] = v
                checkpoint[key] = new_state_dict

            # save model
            save_path = os.path.join(directory, "checkpoint.pth.tar")
            torch.save(checkpoint, save_path)
            print(f"Checkpoint saved to {save_path}")

    def load_model(self, directory):
        cfg = self.cfg
        load_path = os.path.join(directory, "checkpoint.pth.tar")

        if not os.path.exists(load_path):
            raise FileNotFoundError('Checkpoint not found at "{}"'.format(load_path))

        checkpoint = torch.load(load_path, map_location=self.device)

        print("Loading weights from {}".format(load_path))

        if cfg.fine_tuning:
            if "model" in checkpoint:
                self.model.load_state_dict(checkpoint["model"], strict=False)
            else:
                print("Warning: Fine-tuning mode, but 'model' not found in checkpoint. Loading tuner and head separately.")
                if "tuner" in checkpoint:
                    self.model.tuner.load_state_dict(checkpoint["tuner"], strict=False)
                if "head" in checkpoint:
                    self.model.head.load_state_dict(checkpoint["head"], strict=False)
        else:
            if "tuner" in checkpoint:
                self.tuner.load_state_dict(checkpoint["tuner"], strict=False)
            if "head" in checkpoint and checkpoint["head"]["weight"].shape == self.head.weight.shape:
                self.head.load_state_dict(checkpoint["head"], strict=False)

        if dist.is_initialized():
            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

    @torch.no_grad()
    def distributed_evaluate(self, data_loader):
        self.model.eval()
        all_predictions = []
        all_labels = []

        for batch in data_loader:
            images, labels = batch[0].to(self.device), batch[1].to(self.device)
            outputs = self.model(images)
            predictions = outputs.argmax(dim=1)

            if dist.is_initialized():
                # Gather predictions and labels from all processes
                gathered_predictions = [torch.zeros_like(predictions) for _ in range(self.world_size)]
                gathered_labels = [torch.zeros_like(labels) for _ in range(self.world_size)]

                dist.all_gather(gathered_predictions, predictions)
                dist.all_gather(gathered_labels, labels)

                if self.is_main_process:
                    all_predictions.extend([tensor.cpu() for tensor in gathered_predictions])
                    all_labels.extend([tensor.cpu() for tensor in gathered_labels])
            else:
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())

        if self.is_main_process:
            all_predictions = torch.cat(all_predictions)
            all_labels = torch.cat(all_labels)

            # Perform evaluation metrics calculation
            accuracy = (all_predictions == all_labels).float().mean().item()
            return {"accuracy": accuracy}
        else:
            return {}

    def _log_tensorboard(self, n_iter, current_lr, meters, mean_acc, many_acc, med_acc, few_acc):
        """Log training metrics to tensorboard"""
        if self._writer is None:
            return

        # Log learning rate
        self._writer.add_scalar("train/lr", current_lr, n_iter)

        # Log loss values
        self._writer.add_scalar("train/loss.val", meters["loss"].val, n_iter)
        self._writer.add_scalar("train/loss.avg", meters["loss"].avg, n_iter)

        # Log accuracy values
        # acc.val is the accuracy of the most recent batch
        self._writer.add_scalar("train/acc.val", meters["acc"].val, n_iter)
        # acc.avg is the average accuracy of all history
        self._writer.add_scalar("train/acc.avg", meters["acc"].avg, n_iter)

        # Log split accuracies
        self._writer.add_scalar("train/mean_acc", mean_acc, n_iter)
        self._writer.add_scalar("train/many_acc", many_acc, n_iter)
        self._writer.add_scalar("train/med_acc", med_acc, n_iter)
        self._writer.add_scalar("train/few_acc", few_acc, n_iter)


class MetaTrainer(Trainer):
    def __init__(self, cfg, device):
        self.use_meta = cfg.use_meta
        self.meta_data_ratio = cfg.meta_data_ratio
        self.meta_lr = cfg.meta_lr
        self.meta_update_freq = cfg.meta_update_freq
        self.meta_inner_steps = cfg.meta_inner_steps  # New parameter
        self.meta_optim = None
        self.meta_sched = None
        self.meta_train_loader = None
        self.meta_val_loader = None  # Added validation loader

        super().__init__(cfg, device)

    def initialize(self):
        """Override the initialization process to handle meta training setup"""
        super().initialize()

        if self.use_meta:
            self._build_meta_loaders()

    def _build_meta_loaders(self):
        if not self.use_meta:
            return

        # Create meta_train_loader and meta_val_loader from training data
        train_dataset = self.train_loader.dataset
        dataset_size = len(train_dataset)

        # Create indices for splitting - use deterministic split if seed is set
        indices = list(range(dataset_size))
        if self.cfg.seed is not None:
            # Use a deterministic shuffle based on the seed
            rng = np.random.RandomState(self.cfg.seed)
            rng.shuffle(indices)
        else:
            np.random.shuffle(indices)

        # Split into main, meta_train, and meta_val
        val_ratio = self.meta_data_ratio / 2  # Half for validation
        main_split = int(dataset_size * (1 - self.meta_data_ratio))
        val_split = int(dataset_size * val_ratio)

        main_indices = indices[:main_split]
        meta_train_indices = indices[main_split : main_split + val_split]
        meta_val_indices = indices[main_split + val_split :]

        # Create subsets
        main_train_dataset = torch.utils.data.Subset(train_dataset, main_indices)
        meta_train_dataset = torch.utils.data.Subset(train_dataset, meta_train_indices)
        meta_val_dataset = torch.utils.data.Subset(train_dataset, meta_val_indices)

        # Create samplers if using distributed training
        main_sampler = meta_train_sampler = meta_val_sampler = None
        if dist.is_initialized():
            main_sampler = DistributedSampler(main_train_dataset)
            meta_train_sampler = DistributedSampler(meta_train_dataset)
            meta_val_sampler = DistributedSampler(meta_val_dataset)

        # Update the regular train loader
        self.train_loader = DataLoader(
            main_train_dataset,
            batch_size=self.per_gpu_batch_size,
            shuffle=(main_sampler is None),
            sampler=main_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        # Create meta loaders
        self.meta_train_loader = DataLoader(
            meta_train_dataset,
            batch_size=self.per_gpu_batch_size,
            shuffle=(meta_train_sampler is None),
            sampler=meta_train_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        self.meta_val_loader = DataLoader(
            meta_val_dataset,
            batch_size=self.per_gpu_batch_size,
            shuffle=(meta_val_sampler is None),
            sampler=meta_val_sampler,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

        print(f"Created meta-training dataset with {len(meta_train_dataset)} samples")
        print(f"Created meta-validation dataset with {len(meta_val_dataset)} samples")
        print(f"Main training dataset has {len(main_train_dataset)} samples")

    def build_optimizer(self):
        # First handle the regular optimization
        super().build_optimizer()

        if not self.use_meta:
            return

        # Collect meta parameters from the tuner
        meta_params = self._collect_meta_parameters()

        if not meta_params:
            logger.info("No meta parameters found, disabling meta optimization")
            self.use_meta = False
            return

        print(f"Found {len(meta_params)} meta parameters for meta-optimization")

        # Create optimizer for meta parameters
        # self.meta_optim = torch.optim.AdamW(meta_params, lr=self.meta_lr, weight_decay=self.cfg.weight_decay, betas=(0.9, 0.999))
        # SGD optimizer
        self.meta_optim = torch.optim.SGD(
            meta_params,
            lr=self.meta_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum,
        )

        # scheduler for meta optimizer
        # self.meta_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.meta_optim, T_max=self.cfg.num_epochs, eta_min=0.1 * self.meta_lr)
        # Constant LR
        self.meta_sched = torch.optim.lr_scheduler.ConstantLR(self.meta_optim, factor=1.0, total_iters=0)
        print(f"Meta optimizer using Constant LR: {self.meta_lr}")

    def _collect_meta_parameters(self):
        """Collect all meta parameters from the model"""
        meta_params = []

        # Check if tuner has meta parameters
        if self.tuner is None:
            return meta_params

        # Find all modules with meta parameters
        for name, module in self.tuner.named_modules():
            if hasattr(module, "get_meta_parameters"):
                meta_params.extend(module.get_meta_parameters())

        return meta_params

    def train(self):
        if not self.use_meta:
            return super().train()

        # Adjust number of epochs based on meta_data_ratio
        original_epochs = self.cfg.num_epochs
        if self.meta_data_ratio > 0:
            self.cfg.num_epochs = math.ceil(original_epochs / (1 - self.meta_data_ratio))
            print(
                f"Adjusting epochs from {original_epochs} to {self.cfg.num_epochs} for same iteration number, "
                f"according to meta data ratio {self.meta_data_ratio}"
            )

        # Setup for training with meta-optimization
        meta_meters = {
            "meta_train_loss": AverageMeter(ema=True),
            "meta_train_acc": AverageMeter(ema=True),
            "meta_val_loss": AverageMeter(ema=True),
            "meta_val_acc": AverageMeter(ema=True),
        }

        # Main training loop
        for epoch in range(self.cfg.num_epochs):
            # Set up epoch for all loaders
            if dist.is_initialized():
                self.train_loader.sampler.set_epoch(epoch)
                if hasattr(self.meta_train_loader, "sampler") and self.meta_train_loader.sampler is not None:
                    self.meta_train_loader.sampler.set_epoch(epoch)
                if hasattr(self.meta_val_loader, "sampler") and self.meta_val_loader.sampler is not None:
                    self.meta_val_loader.sampler.set_epoch(epoch)

            # Regular training epoch (inner loop)
            self._train_epoch(epoch)

            # Meta-optimization step (outer loop)
            if (epoch + 1) % self.meta_update_freq == 0:
                self._meta_optimization_step(meta_meters)

                # Log meta-training metrics
                if self.is_main_process:
                    print(
                        f"Meta-optimization: train_loss={meta_meters['meta_train_loss'].avg:.4f}, "
                        f"train_acc={meta_meters['meta_train_acc'].avg:.2f}%, "
                        f"val_loss={meta_meters['meta_val_loss'].avg:.4f}, "
                        f"val_acc={meta_meters['meta_val_acc'].avg:.2f}%"
                    )

                    if self._writer is not None:
                        self._writer.add_scalar("meta/train_loss", meta_meters["meta_train_loss"].avg, epoch)
                        self._writer.add_scalar("meta/train_acc", meta_meters["meta_train_acc"].avg, epoch)
                        self._writer.add_scalar("meta/val_loss", meta_meters["meta_val_loss"].avg, epoch)
                        self._writer.add_scalar("meta/val_acc", meta_meters["meta_val_acc"].avg, epoch)

            # Update schedulers
            self.sched.step()
            if self.meta_sched is not None:
                self.meta_sched.step()

        # Save final model
        self.save_model(self.cfg.output_dir)
        self.test()

    def _train_epoch(self, epoch):
        """Regular training epoch (without meta-optimization)"""
        # Set train mode
        if self.tuner is not None:
            self.tuner.train()
        if self.cfg.fine_tuning:
            self.model.train()

        # Make sure we're in normal mode (not meta mode)
        self._set_meta_mode(False)

        # Initialize meters for this epoch
        meters = {
            "batch_time": AverageMeter(),
            "data_time": AverageMeter(),
            "loss": AverageMeter(ema=True),
            "acc": AverageMeter(ema=True),
            "cls_meters": [AverageMeter(ema=True) for _ in range(self.num_classes)],
        }

        # Training start time for this epoch
        time_start = time.time()

        # Calculate batch sizes and steps
        micro_batches_per_epoch = len(self.train_loader)
        _logic_batch_num = micro_batches_per_epoch // self.accum_step

        batch_start = time.time()
        self.optim.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Calculate logical batch index
            _logic_batch_idx = batch_idx // self.accum_step
            _is_logic_batch = (batch_idx + 1) % self.accum_step == 0

            # Measure data loading time
            meters["data_time"].update(time.time() - batch_start)

            # Move to device
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            # Forward pass and loss calculation
            if self.cfg.prec == "amp":
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss_scaled = loss / self.accum_step
                    self.scaler.scale(loss_scaled).backward()

                if _is_logic_batch:
                    self.scaler.step(self.optim)
                    self.scaler.update()
                    self.optim.zero_grad()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss_scaled = loss / self.accum_step
                loss_scaled.backward()

                if _is_logic_batch:
                    self.optim.step()
                    self.optim.zero_grad()

            # Update metrics and log progress on logical batch boundary
            if _is_logic_batch:
                self._update_metrics(outputs, labels, loss, meters)

                # Log progress at specified intervals
                if ((_logic_batch_idx + 1) % self.cfg.print_freq == 0) or (_logic_batch_idx + 1 == _logic_batch_num):
                    self._log_progress(
                        epoch,
                        _logic_batch_idx,
                        _logic_batch_num,
                        meters,
                        time_start,
                        self.cfg.num_epochs,
                    )

            # Measure batch time
            meters["batch_time"].update(time.time() - batch_start)
            batch_start = time.time()

        # Don't step scheduler here - it's done in the main train method
        torch.cuda.empty_cache()

    def _meta_optimization_step(self, meters):
        """Implement bi-level optimization for meta-parameters"""
        # Backup base model parameters
        self._backup_weights()

        # Set modules to meta mode
        self._set_meta_mode(True)

        # 1. Inner loop: optimize model parameters on meta-train set
        for _ in range(self.meta_inner_steps):
            # Get a batch from meta-train
            try:
                meta_batch = next(iter(self.meta_train_loader))
            except StopIteration:
                meta_loader_iter = iter(self.meta_train_loader)
                meta_batch = next(meta_loader_iter)

            images = meta_batch[0].to(self.device)
            labels = meta_batch[1].to(self.device)

            # Forward pass and optimization
            self.model.train()
            self.optim.zero_grad()

            if self.cfg.prec == "amp":
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optim.step()

            # Update train metrics
            with torch.no_grad():
                preds = outputs.argmax(dim=1)
                acc = (preds == labels).float().mean() * 100.0

            meters["meta_train_loss"].update(loss.item())
            meters["meta_train_acc"].update(acc.item())

        # 2. Outer loop: evaluate on meta-val and update meta-parameters
        self.model.eval()
        self.meta_optim.zero_grad()

        # Get validation batch
        try:
            val_batch = next(iter(self.meta_val_loader))
        except StopIteration:
            val_loader_iter = iter(self.meta_val_loader)
            val_batch = next(val_loader_iter)

        val_images = val_batch[0].to(self.device)
        val_labels = val_batch[1].to(self.device)

        # Compute validation loss for meta-parameter update
        with torch.set_grad_enabled(True):  # Need gradients for meta-params
            if self.cfg.prec == "amp":
                with autocast():
                    val_outputs = self.model(val_images)
                    val_loss = self.criterion(val_outputs, val_labels)
                self.scaler.scale(val_loss).backward()
                self.scaler.step(self.meta_optim)
                self.scaler.update()
            else:
                val_outputs = self.model(val_images)
                val_loss = self.criterion(val_outputs, val_labels)
                val_loss.backward()
                self.meta_optim.step()

        # Update validation metrics
        with torch.no_grad():
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == val_labels).float().mean() * 100.0

        meters["meta_val_loss"].update(val_loss.item())
        meters["meta_val_acc"].update(val_acc.item())

        # Restore original parameters to continue with normal training
        self._restore_weights()

        # Set modules back to normal mode
        self._set_meta_mode(False)

    def _backup_weights(self):
        """Backup weights of LoRA modules for restoration"""
        if self.tuner is None:
            return

        for module in self.tuner.modules():
            if hasattr(module, "backup_weights"):
                module.backup_weights()

    def _restore_weights(self):
        """Restore weights of LoRA modules"""
        if self.tuner is None:
            return

        for module in self.tuner.modules():
            if hasattr(module, "restore_weights"):
                module.restore_weights()

    def _set_meta_mode(self, is_meta):
        """Set all modules with meta parameters to meta mode"""
        if self.tuner is None:
            return

        for module in self.tuner.modules():
            if hasattr(module, "set_meta_mode"):
                module.set_meta_mode(is_meta)
