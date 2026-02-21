import os
import sys
import argparse
import json
import logging
import random
import warnings
import time
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict
import copy

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR, CosineAnnealingLR
from torch.optim.swa_utils import AveragedModel, SWALR

import timm
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# Suppress warnings
warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration for SemCXR training."""
    # Data
    data_dir: str = "./data"
    csv_file: str = "data.csv"
    image_dir: str = "data/images/"
    fold: int = 0
    num_folds: int = 5
    
    # Model
    image_encoder: str = "swin_base_patch4_window7_224"  # or "densenet121"
    text_encoder: str = "emilyalsentzer/Bio_ClinicalBERT"
    embed_dim: int = 512
    num_classes: int = 3
    dropout: float = 0.3
    use_cross_attention: bool = True
    use_report_generation: bool = True
    
    # Training
    batch_size: int = 32
    num_epochs: int = 100
    warmup_epochs: int = 5
    lr: float = 1e-4
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    label_smoothing: float = 0.1
    gradient_clip: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Augmentation
    image_size: int = 224
    mixup_alpha: float = 0.4
    cutmix_alpha: float = 0.4
    mix_prob: float = 0.5
    
    # Class Imbalance
    use_weighted_sampler: bool = True
    use_focal_loss: bool = False
    focal_gamma: float = 2.0
    class_weights: Optional[List[float]] = None
    
    # Regularization
    use_swa: bool = True
    swa_start: int = 75
    stochastic_depth: float = 0.1
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "semcxr"
    save_top_k: int = 3
    atomic_save: bool = True
    auto_resume: bool = True
    
    # Hardware
    num_workers: int = 8
    pin_memory: bool = True
    precision: str = "16-mixed"  # "32", "16-mixed", "bf16-mixed"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Logging
    log_interval: int = 10
    val_check_interval: float = 1.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "Config":
        return cls(**d)


# =============================================================================
# Utilities
# =============================================================================

def setup_logging(rank: int = 0) -> logging.Logger:
    """Setup logging for distributed training."""
    logger = logging.getLogger("SemCXR")
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    
    if rank == 0:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(handler)
    
    return logger


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(local_rank: int = 0) -> torch.device:
    """Get device for training."""
    if torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cpu")


def setup_distributed(local_rank: int, world_size: int, rank: int) -> bool:
    """Setup distributed training."""
    if world_size <= 1:
        return False
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank
    )
    return True


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def smart_load_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Load state dict across DDP/non-DDP wrappers."""
    target_is_ddp = isinstance(model, DDP)
    normalized_state_dict = {}

    for key, value in state_dict.items():
        if target_is_ddp and not key.startswith("module."):
            normalized_state_dict[f"module.{key}"] = value
        elif not target_is_ddp and key.startswith("module."):
            normalized_state_dict[key.replace("module.", "", 1)] = value
        else:
            normalized_state_dict[key] = value

    model.load_state_dict(normalized_state_dict, strict=False)


# =============================================================================
# Data
# =============================================================================

class ChestXrayDataset(Dataset):
    """Chest X-ray dataset with multimodal support."""
    
    CLASS_NAMES = ["Normal", "Pneumonia", "TB"]
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        tokenizer: AutoTokenizer,
        transform: Optional[Callable] = None,
        max_length: int = 256,
        is_training: bool = True
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_length = max_length
        self.is_training = is_training
        
        # Pre-tokenize text for efficiency
        if 'Clean_Impression' in self.df.columns:
            text_series = self.df['Clean_Impression'].fillna(self.df['Impression'])
        else:
            text_series = self.df['Impression']

        self.encodings = []
        for text in text_series:
            encoding = self.tokenizer(
                str(text),
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            self.encodings.append({
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0)
            })
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load image
        image_path = self.image_dir / row['Image']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Return blank image if file not found
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Get pre-tokenized text
        encoding = self.encodings[idx]
        
        # Get label
        label = self.CLASS_NAMES.index(row['Category'])
        
        return {
            'image': image,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'label': torch.tensor(label, dtype=torch.long),
            'image_id': row['Image']
        }


class MixupCutmix:
    """Mixup and Cutmix augmentation."""
    
    def __init__(
        self,
        mixup_alpha: float = 0.4,
        cutmix_alpha: float = 0.4,
        prob: float = 0.5,
        num_classes: int = 3,
        switch_prob: float = 0.5
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
        self.switch_prob = switch_prob
    
    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() > self.prob:
            return batch
        
        images = batch['image']
        labels = batch['label']
        batch_size = images.size(0)
        
        # One-hot encode labels
        targets = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Choose mixup or cutmix
        if random.random() < self.switch_prob:
            # Mixup
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            index = torch.randperm(batch_size).to(images.device)
            
            mixed_images = lam * images + (1 - lam) * images[index]
            mixed_targets = lam * targets + (1 - lam) * targets[index]
        else:
            # Cutmix
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            index = torch.randperm(batch_size).to(images.device)
            
            # Generate random box
            _, _, h, w = images.shape
            cut_rat = np.sqrt(1. - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)
            
            cx = np.random.randint(w)
            cy = np.random.randint(h)
            
            bbx1 = np.clip(cx - cut_w // 2, 0, w)
            bby1 = np.clip(cy - cut_h // 2, 0, h)
            bbx2 = np.clip(cx + cut_w // 2, 0, w)
            bby2 = np.clip(cy + cut_h // 2, 0, h)
            
            mixed_images = images.clone()
            mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
            
            # Adjust lambda
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
            mixed_targets = lam * targets + (1 - lam) * targets[index]
        
        batch['image'] = mixed_images
        batch['target'] = mixed_targets
        batch['lam'] = lam
        
        return batch


def get_transforms(image_size: int = 224, is_training: bool = True):
    """Get image transforms."""
    from torchvision import transforms
    
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


# =============================================================================
# Model Components
# =============================================================================

class ImageEncoder(nn.Module):
    """Image encoder with SOTA architecture."""
    
    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        embed_dim: int = 512,
        pretrained: bool = True,
        drop_rate: float = 0.0
    ):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            drop_rate=drop_rate,
            drop_path_rate=0.1  # Stochastic depth
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            feat_dim = features.shape[-1]
        
        self.projection = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.projection(features)


class TextEncoder(nn.Module):
    """Text encoder using BioClinicalBERT."""
    
    def __init__(
        self,
        model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
        embed_dim: int = 512
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.projection = nn.Sequential(
            nn.LayerNorm(self.bert.config.hidden_size),
            nn.Linear(self.bert.config.hidden_size, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Freeze lower layers
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for param in self.bert.encoder.layer[:6].parameters():
            param.requires_grad = False
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0]
        return self.projection(cls_output)


class CrossModalAttention(nn.Module):
    """Cross-modal attention between image and text."""
    
    def __init__(self, embed_dim: int = 512, num_heads: int = 8):
        super().__init__()
        
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
    
    def forward(
        self,
        image_feat: torch.Tensor,
        text_feat: torch.Tensor
    ) -> torch.Tensor:
        # Image attends to text
        q = self.q_proj(self.norm1(image_feat)).unsqueeze(1)  # [B, 1, D]
        k = self.k_proj(self.norm2(text_feat)).unsqueeze(1)   # [B, 1, D]
        v = self.v_proj(self.norm2(text_feat)).unsqueeze(1)   # [B, 1, D]
        
        attn = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        out = attn @ v
        
        out = self.out_proj(out.squeeze(1))
        return self.dropout(out) + image_feat


class ReportDecoder(nn.Module):
    """Report generation decoder."""
    
    def __init__(
        self,
        embed_dim: int = 512,
        vocab_size: int = 30522,
        num_layers: int = 4
    ):
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, 256, embed_dim) * 0.02)
    
    def forward(
        self,
        memory: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        tgt_emb = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1)]
        output = self.decoder(tgt_emb, memory.unsqueeze(1))
        return self.output(output)


class SemCXR(nn.Module):
    """SemCXR: Semantic Cross-modal Alignment for CXR Classification."""
    
    def __init__(self, config: Config):
        super().__init__()
        
        self.config = config
        
        # Encoders
        self.image_encoder = ImageEncoder(
            config.image_encoder,
            config.embed_dim,
            drop_rate=config.dropout
        )
        self.text_encoder = TextEncoder(
            config.text_encoder,
            config.embed_dim
        )
        
        # Cross-modal fusion
        if config.use_cross_attention:
            self.cross_attn = CrossModalAttention(config.embed_dim)
        else:
            self.cross_attn = None
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.embed_dim * 2),
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embed_dim, config.num_classes)
        )
        
        # Report generation
        if config.use_report_generation:
            self.report_decoder = ReportDecoder(config.embed_dim)
        else:
            self.report_decoder = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Only initialize newly added layers; skip pretrained backbones
        modules_to_init = [self.image_encoder.projection, self.text_encoder.projection, self.classifier]
        if self.cross_attn is not None:
            modules_to_init.append(self.cross_attn)
        if self.report_decoder is not None:
            modules_to_init.append(self.report_decoder)

        for module in modules_to_init:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def forward(
        self,
        image: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        report_targets: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Encode
        img_feat = self.image_encoder(image)
        txt_feat = self.text_encoder(input_ids, attention_mask)
        
        # Cross-modal attention
        if self.cross_attn is not None:
            fused_feat = self.cross_attn(img_feat, txt_feat)
        else:
            fused_feat = img_feat
        
        # Concatenate features
        combined = torch.cat([fused_feat, txt_feat], dim=-1)
        
        # Classify
        logits = self.classifier(combined)
        
        outputs = {'logits': logits, 'features': combined}
        
        # Report generation (if enabled and in training)
        if self.report_decoder is not None and report_targets is not None:
            report_logits = self.report_decoder(
                fused_feat, 
                report_targets[:, :-1]
            )
            outputs['report_logits'] = report_logits
        
        return outputs


# =============================================================================
# Loss Functions
# =============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing cross entropy loss."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = F.nll_loss(logprobs, target, reduction='mean')
        smooth_loss = -logprobs.mean(dim=-1).mean()
        return confidence * nll_loss + self.smoothing * smooth_loss


# =============================================================================
# Metrics
# =============================================================================

class MetricsCalculator:
    """Calculate and track metrics."""
    
    def __init__(self, num_classes: int = 3, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or ["Normal", "Pneumonia", "TB"]
        self.reset()
    
    def reset(self):
        self.all_probs = []
        self.all_labels = []
        self.all_preds = []
    
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        self.all_probs.append(probs.cpu().numpy())
        self.all_labels.append(labels.cpu().numpy())
        self.all_preds.append(preds.cpu().numpy())
    
    def compute(self) -> Dict[str, float]:
        probs = np.concatenate(self.all_probs)
        labels = np.concatenate(self.all_labels)
        preds = np.concatenate(self.all_preds)
        
        # AUC per class (One-vs-Rest)
        aucs = {}
        for i, name in enumerate(self.class_names):
            try:
                auc = roc_auc_score(labels == i, probs[:, i])
                aucs[f"AUC_{name}"] = auc
            except ValueError:
                aucs[f"AUC_{name}"] = 0.0
        
        # Macro AUC
        macro_auc = np.mean(list(aucs.values()))
        
        # Accuracy
        accuracy = accuracy_score(labels, preds)
        
        # F1 scores
        f1_macro = f1_score(labels, preds, average='macro')
        f1_weighted = f1_score(labels, preds, average='weighted')
        
        return {
            "Macro_AUC": macro_auc,
            "Accuracy": accuracy * 100,
            "F1_Macro": f1_macro,
            "F1_Weighted": f1_weighted,
            **aucs
        }


# =============================================================================
# Checkpointing
# =============================================================================

class CheckpointManager:
    """Ultra-lightweight checkpointing for Kaggle environments."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        experiment_name: str,
        save_top_k: int = 1, # Kept for compatibility, but overridden by logic
        atomic: bool = True
    ):
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.atomic = atomic
    
    def save_checkpoint(
        self,
        state: Dict,
        epoch: int,
        score: float,
        is_best: bool = False
    ) -> str:
        """Save exactly one latest checkpoint and one best checkpoint."""
        last_path = self.checkpoint_dir / "last_model.pt"
        temp_path = str(last_path) + ".tmp"
        
        # Save to temp file first to prevent corruption if crash happens during write
        torch.save(state, temp_path)
        
        # Atomic replace
        if self.atomic:
            os.replace(temp_path, last_path)
        else:
            shutil.move(temp_path, last_path)
        
        # Copy to best_model.pt if it's the new high score
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            shutil.copy(last_path, best_path)
        
        return str(last_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict:
        """Load checkpoint."""
        return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint for auto-resume."""
        last_path = self.checkpoint_dir / "last_model.pt"
        if last_path.exists():
            return str(last_path)
        return None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get best checkpoint path."""
        best_path = self.checkpoint_dir / "best_model.pt"
        if best_path.exists():
            return str(best_path)
        return None


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """Main trainer class."""
    
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        criterion: nn.Module,
        device: torch.device,
        scaler: GradScaler,
        checkpoint_manager: CheckpointManager,
        logger: logging.Logger,
        is_distributed: bool = False,
        local_rank: int = 0,
        world_size: int = 1
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.scaler = scaler
        self.checkpoint_manager = checkpoint_manager
        self.logger = logger
        self.is_distributed = is_distributed
        self.local_rank = local_rank
        self.world_size = world_size

        # AMP precision settings
        if config.precision == "bf16-mixed":
            self.amp_enabled = True
            self.amp_dtype = torch.bfloat16
        elif config.precision == "16-mixed":
            self.amp_enabled = True
            self.amp_dtype = torch.float16
        else:  # "32"
            self.amp_enabled = False
            self.amp_dtype = torch.float32
        
        # SWA
        self.swa_model = None
        if config.use_swa:
            self.swa_model = AveragedModel(model)
            self.swa_scheduler = SWALR(optimizer, swa_lr=config.lr * 0.1)
            self.swa_start = config.swa_start
        
        # Early stopping
        self.best_score = 0.0
        self.patience_counter = 0
        self.patience = 100
        
        # Mixup/Cutmix
        self.mixup_cutmix = MixupCutmix(
            config.mixup_alpha,
            config.cutmix_alpha,
            config.mix_prob,
            config.num_classes
        )
        
        self.metrics = MetricsCalculator(config.num_classes)
        self.current_epoch = 0
    
    def train_epoch(self) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        self.metrics.reset()
        
        total_loss = 0.0
        num_batches = len(self.train_loader)

        if self.is_distributed and isinstance(self.train_loader.sampler, DistributedSampler):
            self.train_loader.sampler.set_epoch(self.current_epoch)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            for key in ['image', 'input_ids', 'attention_mask', 'label']:
                if key in batch:
                    batch[key] = batch[key].to(self.device)
            
            # Apply mixup/cutmix
            if self.config.mixup_alpha > 0 or self.config.cutmix_alpha > 0:
                batch = self.mixup_cutmix(batch)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                outputs = self.model(
                    batch['image'],
                    batch['input_ids'],
                    batch['attention_mask'],
                    report_targets=batch['input_ids'] if self.config.use_report_generation else None 
                )
                
                # Compute loss
                if 'target' in batch:
                    # Mixup/Cutmix loss
                    loss = -(batch['target'] * F.log_softmax(outputs['logits'], dim=-1)).sum(dim=-1).mean()
                else:
                    loss = self.criterion(outputs['logits'], batch['label'])
                
                # Report generation loss (if enabled)
                if 'report_logits' in outputs:
                    report_loss = F.cross_entropy(
                        outputs['report_logits'].view(-1, outputs['report_logits'].size(-1)),
                        batch['input_ids'][:, 1:].contiguous().view(-1),
                        ignore_index=0
                    )
                    loss = loss + 0.5 * report_loss

                loss_for_backward = loss / self.config.accumulate_grad_batches
            
            # Backward pass
            self.scaler.scale(loss_for_backward).backward()
            
            # Gradient accumulation
            should_step = (
                (batch_idx + 1) % self.config.accumulate_grad_batches == 0
                or (batch_idx + 1) == num_batches
            )
            if should_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Step OneCycleLR per optimizer step (not when SWA is active)
                if not (self.swa_model is not None and self.current_epoch >= self.swa_start):
                    self.scheduler.step()

            # Update metrics
            with torch.no_grad():
                self.metrics.update(outputs['logits'], batch['label'])

            total_loss += loss.item()

            # Logging
            if batch_idx % self.config.log_interval == 0 and self.local_rank == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch} [{batch_idx}/{num_batches}] "
                    f"Loss: {loss.item():.4f} LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                )

        # SWA update (per-epoch)
        if self.swa_model is not None and self.current_epoch >= self.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        
        metrics = self.metrics.compute()
        metrics['train_loss'] = total_loss / num_batches
        return metrics
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model."""
        # Use SWA model when active, otherwise base model
        if self.swa_model is not None and self.current_epoch >= self.swa_start:
            eval_model = self.swa_model
        else:
            eval_model = self.model
        eval_model.eval()
        self.metrics.reset()

        total_loss = 0.0

        for batch in self.val_loader:
            for key in ['image', 'input_ids', 'attention_mask', 'label']:
                if key in batch:
                    batch[key] = batch[key].to(self.device)

            with autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                outputs = eval_model(
                    batch['image'],
                    batch['input_ids'],
                    batch['attention_mask']
                )
                loss = self.criterion(outputs['logits'], batch['label'])

            self.metrics.update(outputs['logits'], batch['label'])
            total_loss += loss.item()

        metrics = self.metrics.compute()
        metrics['val_loss'] = total_loss / len(self.val_loader)
        return metrics
    
    def fit(self, start_epoch: int = 0, max_epochs: int = None):
        """Main training loop."""
        max_epochs = max_epochs or self.config.num_epochs
        
        for epoch in range(start_epoch, max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Logging
            if self.local_rank == 0:
                self.logger.info(
                    f"\nEpoch {epoch} Summary:\n"
                    f"  Train Loss: {train_metrics['train_loss']:.4f}\n"
                    f"  Val Loss: {val_metrics['val_loss']:.4f}\n"
                    f"  Macro AUC: {val_metrics['Macro_AUC']:.4f}\n"
                    f"  Accuracy: {val_metrics['Accuracy']:.2f}%\n"
                    f"  AUC Normal: {val_metrics.get('AUC_Normal', 0):.4f}\n"
                    f"  AUC Pneumonia: {val_metrics.get('AUC_Pneumonia', 0):.4f}\n"
                    f"  AUC TB: {val_metrics.get('AUC_TB', 0):.4f}"
                )
                
                # Checkpointing
                score = val_metrics['Macro_AUC']
                is_best = score > self.best_score
                
                if is_best:
                    self.best_score = score
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'best_score': self.best_score,
                    'config': self.config.to_dict(),
                    'metrics': val_metrics
                }
                
                if self.swa_model is not None:
                    state['swa_state_dict'] = self.swa_model.state_dict()
                
                self.checkpoint_manager.save_checkpoint(state, epoch, score, is_best)
                
                # Early stopping
                if self.patience_counter >= self.patience:
                    self.logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Final SWA validation
        if self.swa_model is not None and self.local_rank == 0:
            self.logger.info("Updating batch normalization for SWA model...")
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.device)
            final_metrics = self.validate()
            self.logger.info(f"Final SWA Metrics: Macro AUC = {final_metrics['Macro_AUC']:.4f}")


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="SemCXR Training")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--csv_file", type=str, default="data.csv")
    parser.add_argument("--image_dir", type=str, default="data/images/")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    
    # Model
    parser.add_argument("--image_encoder", type=str, default="swin_base_patch4_window7_224",
                       choices=["swin_base_patch4_window7_224", "densenet121", "resnet50"])
    parser.add_argument("--text_encoder", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--embed_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--no_cross_attn", action="store_true", help="Disable cross attention")
    parser.add_argument("--no_report_gen", action="store_true", help="Disable report generation")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    
    # Augmentation
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--cutmix_alpha", type=float, default=0.4)
    parser.add_argument("--mix_prob", type=float, default=0.5)
    
    # Class imbalance
    parser.add_argument("--no_weighted_sampler", action="store_true")
    parser.add_argument("--use_focal_loss", action="store_true")
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    
    # Regularization
    parser.add_argument("--no_swa", action="store_true", help="Disable SWA")
    parser.add_argument("--swa_start", type=int, default=75)
    
    # Checkpointing
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--experiment_name", type=str, default="semcxr")
    parser.add_argument("--save_top_k", type=int, default=3)
    parser.add_argument("--no_atomic_save", action="store_true")
    parser.add_argument("--no_auto_resume", action="store_true")
    parser.add_argument("--resume_from", type=str, default=None)
    
    # Hardware
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--precision", type=str, default="16-mixed",
                       choices=["32", "16-mixed", "bf16-mixed"])
    parser.add_argument("--local_rank", type=int, default=0)
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup distributed
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    global_rank = int(os.environ.get("RANK", local_rank))
    is_distributed = setup_distributed(local_rank, world_size, global_rank) if world_size > 1 else False
    
    # Setup logging
    logger = setup_logging(local_rank)
    
    # Set seed
    set_seed(args.seed, args.deterministic)
    
    # Create config
    config = Config(
        data_dir=args.data_dir,
        csv_file=args.csv_file,
        image_dir=args.image_dir,
        fold=args.fold,
        num_folds=args.num_folds,
        image_encoder=args.image_encoder,
        text_encoder=args.text_encoder,
        embed_dim=args.embed_dim,
        dropout=args.dropout,
        use_cross_attention=not args.no_cross_attn,
        use_report_generation=not args.no_report_gen,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_epochs=args.warmup_epochs,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        image_size=args.image_size,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
        use_weighted_sampler=not args.no_weighted_sampler,
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_swa=not args.no_swa,
        swa_start=args.swa_start,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        save_top_k=args.save_top_k,
        atomic_save=not args.no_atomic_save,
        auto_resume=not args.no_auto_resume,
        num_workers=args.num_workers,
        precision=args.precision,
        seed=args.seed,
        deterministic=args.deterministic
    )
    
    if local_rank == 0:
        logger.info(f"Configuration: {json.dumps(config.to_dict(), indent=2)}")
    
    # Load data
    df = pd.read_csv(os.path.join(config.data_dir, config.csv_file))
    
    # Stratified split
    skf = StratifiedKFold(n_splits=config.num_folds, shuffle=True, random_state=config.seed)
    splits = list(skf.split(df, df['Category']))
    train_idx, val_idx = splits[config.fold]
    
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    
    if local_rank == 0:
        logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
        logger.info(f"Train distribution: {train_df['Category'].value_counts().to_dict()}")
        logger.info(f"Val distribution: {val_df['Category'].value_counts().to_dict()}")
    
    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
    
    # Create datasets
    train_dataset = ChestXrayDataset(
        train_df,
        config.image_dir,
        tokenizer,
        get_transforms(config.image_size, is_training=True),
        is_training=True
    )
    val_dataset = ChestXrayDataset(
        val_df,
        config.image_dir,
        tokenizer,
        get_transforms(config.image_size, is_training=False),
        is_training=False
    )
    
    # Create sampler for class imbalance
    sampler = None
    if is_distributed:
        sampler = DistributedSampler(train_dataset, shuffle=True)
    elif config.use_weighted_sampler:
        class_counts = train_df['Category'].value_counts()
        class_weights = 1.0 / class_counts[train_df['Category']].values
        sampler = WeightedRandomSampler(class_weights, len(class_weights))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=(sampler is None and not is_distributed),
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create model
    device = get_device(local_rank)
    model = SemCXR(config).to(device)
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # Create scheduler
    steps_per_epoch = (len(train_loader) + config.accumulate_grad_batches - 1) // config.accumulate_grad_batches
    total_steps = max(1, steps_per_epoch * config.num_epochs)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.lr,
        total_steps=total_steps,
        pct_start=config.warmup_epochs / config.num_epochs,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=10000
    )
    
    # Create loss
    if config.use_focal_loss:
        class_counts = train_df['Category'].value_counts()
        counts_by_class = np.array([class_counts.get(name, 0) for name in ChestXrayDataset.CLASS_NAMES], dtype=np.float32)
        inv = np.where(counts_by_class > 0, 1.0 / counts_by_class, 0.0)
        alpha = torch.tensor(inv, dtype=torch.float32)
        alpha = alpha / alpha.sum()
        criterion = FocalLoss(alpha=alpha.to(device), gamma=config.focal_gamma)
    else:
        criterion = LabelSmoothingCrossEntropy(config.label_smoothing)
    
    # Create scaler (only needed for fp16, not bf16 or fp32)
    scaler = GradScaler(enabled=config.precision == "16-mixed")
    
    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        config.checkpoint_dir,
        config.experiment_name,
        config.save_top_k,
        config.atomic_save
    )
    
    # Auto resume
    start_epoch = 0
    if config.auto_resume and args.resume_from is None:
        latest_ckpt = checkpoint_manager.get_latest_checkpoint()
        if latest_ckpt:
            args.resume_from = latest_ckpt
            if local_rank == 0:
                logger.info(f"Auto-resuming from {latest_ckpt}")
    
    if args.resume_from:
        checkpoint = checkpoint_manager.load_checkpoint(args.resume_from)
        smart_load_state_dict(model, checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        if local_rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}")
    
    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        scaler=scaler,
        checkpoint_manager=checkpoint_manager,
        logger=logger,
        is_distributed=is_distributed,
        local_rank=local_rank,
        world_size=world_size
    )
    
    # Train
    trainer.fit(start_epoch=start_epoch)
    
    cleanup_distributed()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()