"""
Evaluate a trained SemCXR model checkpoint on validation/test data.

Usage:
    python evaluate.py --checkpoint checkpoints/semcxr_fold0/best.pth --fold 0
    python evaluate.py --checkpoint checkpoints/semcxr_fold0/best.pth --fold 0 --output_dir results
"""

import os
import sys
import argparse
import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from transformers import AutoTokenizer

from train import (
    Config, SemCXR, ChestXrayDataset, get_transforms,
    MetricsCalculator, set_seed
)

warnings.filterwarnings('ignore')

CLASS_NAMES = ["Normal", "Pneumonia", "TB"]


def parse_args():
    parser = argparse.ArgumentParser(description="SemCXR Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best.pth checkpoint")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--csv_file", type=str, default="data.csv")
    parser.add_argument("--image_dir", type=str, default="data/images/")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load checkpoint and reconstruct model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct config from checkpoint
    if 'config' in checkpoint:
        config = Config.from_dict(checkpoint['config'])
    else:
        config = Config()

    # Build model and load weights
    model = SemCXR(config).to(device)

    state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    # Handle DDP state_dict keys (strip 'module.' prefix)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace('module.', '', 1)] = v
    model.load_state_dict(cleaned, strict=False)

    epoch = checkpoint.get('epoch', -1)
    best_score = checkpoint.get('best_score', -1)

    return model, config, epoch, best_score


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_classes: int = 3,
) -> Dict:
    """Run inference and compute comprehensive metrics."""
    model.eval()

    all_logits = []
    all_labels = []
    all_image_ids = []

    for batch in dataloader:
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label']

        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images, input_ids, attention_mask)

        all_logits.append(outputs['logits'].cpu())
        all_labels.append(labels)
        if 'image_id' in batch:
            all_image_ids.extend(batch['image_id'])

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()
    probs = F.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()

    # --- Metrics ---
    accuracy = accuracy_score(labels, preds) * 100
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')

    # Per-class AUC (OvR)
    aucs = {}
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        try:
            aucs[f"AUC_{name}"] = roc_auc_score(labels == i, probs[:, i])
        except ValueError:
            aucs[f"AUC_{name}"] = 0.0
    macro_auc = np.mean(list(aucs.values()))

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Classification report
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES[:num_classes],
        digits=4
    )

    results = {
        "Accuracy": round(accuracy, 4),
        "Macro_AUC": round(macro_auc, 6),
        "F1_Macro": round(f1_macro, 6),
        "F1_Weighted": round(f1_weighted, 6),
        **{k: round(v, 6) for k, v in aucs.items()},
        "confusion_matrix": cm.tolist(),
        "num_samples": len(labels),
    }

    return results, report, probs, preds, labels


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config, epoch, best_score = load_checkpoint(args.checkpoint, device)
    print(f"  Loaded from epoch {epoch} | best_score={best_score:.4f}")
    print(f"  Config: embed_dim={config.embed_dim}, num_classes={config.num_classes}")

    # Load data
    csv_path = os.path.join(args.data_dir, args.csv_file)
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Reproduce the same stratified split used during training
    skf = StratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    splits = list(skf.split(df, df['Category']))
    _, val_idx = splits[args.fold]
    val_df = df.iloc[val_idx].copy()
    print(f"Fold {args.fold} validation size: {len(val_df)}")
    print(f"Distribution: {val_df['Category'].value_counts().to_dict()}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)

    # Dataset & Loader
    val_dataset = ChestXrayDataset(
        val_df,
        args.image_dir,
        tokenizer,
        get_transforms(config.image_size, is_training=False),
        is_training=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Evaluate
    print("\nRunning evaluation...")
    results, report, probs, preds, labels = evaluate(
        model, val_loader, device, config.num_classes
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Accuracy:     {results['Accuracy']:.2f}%")
    print(f"  Macro AUC:    {results['Macro_AUC']:.4f}")
    print(f"  F1 Macro:     {results['F1_Macro']:.4f}")
    print(f"  F1 Weighted:  {results['F1_Weighted']:.4f}")
    for name in CLASS_NAMES[:config.num_classes]:
        print(f"  AUC {name:10s}: {results[f'AUC_{name}']:.4f}")
    print()
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    cm = np.array(results['confusion_matrix'])
    header = "         " + "  ".join(f"{n:>10s}" for n in CLASS_NAMES[:config.num_classes])
    print(header)
    for i, name in enumerate(CLASS_NAMES[:config.num_classes]):
        row = "  ".join(f"{cm[i, j]:10d}" for j in range(config.num_classes))
        print(f"{name:>9s} {row}")
    print("=" * 60)

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result_file = output_dir / f"eval_fold{args.fold}.json"
    save_data = {
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "epoch": epoch,
        **results,
    }
    with open(result_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {result_file}")

    # Save per-sample predictions
    pred_file = output_dir / f"predictions_fold{args.fold}.csv"
    pred_df = pd.DataFrame({
        "true_label": labels,
        "pred_label": preds,
        "true_class": [CLASS_NAMES[l] for l in labels],
        "pred_class": [CLASS_NAMES[p] for p in preds],
        **{f"prob_{CLASS_NAMES[i]}": probs[:, i] for i in range(config.num_classes)},
    })
    pred_df.to_csv(pred_file, index=False)
    print(f"Per-sample predictions saved to {pred_file}")


if __name__ == "__main__":
    main()
