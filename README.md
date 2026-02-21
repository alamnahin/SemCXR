# SemCXR: Semantic Cross-modal Alignment for Chest X-ray Classification

A multimodal deep learning framework for chest X-ray classification combining image and text modalities to classify radiographs into Normal, Pneumonia, and TB categories.

## 🎯 Key Features

- **Multimodal Fusion**: Combines vision (Swin Transformer) and text (BioClinicalBERT) encoders
- **Cross-Modal Attention**: Optional attention mechanism for better image-text alignment
- **Report Generation**: Optional decoder for generating radiology reports
- **Advanced Training**: Mixed-precision training, SWA, gradient accumulation, mixup/cutmix augmentation
- **Production-Ready**: Distributed training support, automatic checkpointing, comprehensive evaluation

## 📊 Results

Evaluation results on fold 0 validation set (1,207 samples):

| Experiment | Accuracy | Macro AUC | F1 Macro | F1 Weighted |
|------------|----------|-----------|----------|-------------|
| **semcxr_full** | 93.79% | 0.9935 | 0.9122 | 0.9390 |
| semcxr_no_report | 37.20% | 0.8656 | 0.1808 | 0.2017 |
| **semcxr_no_xattn** | 92.79% | **0.9953** | 0.8976 | 0.9285 |

### Per-Class Performance (Best Model: semcxr_full)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Normal | 0.79 | 0.93 | 0.85 | 94 |
| Pneumonia | 0.99 | 0.94 | 0.97 | 664 |
| TB | 0.90 | 0.94 | 0.92 | 449 |

### Confusion Matrix (semcxr_full)

|           | Normal | Pneumonia | TB  |
|-----------|--------|-----------|-----|
| Normal    | 87     | 0         | 7   |
| Pneumonia | 0      | 622       | 42  |
| TB        | 23     | 3         | 423 |

## 🏗️ Architecture

```
┌─────────────┐         ┌──────────────┐
│   Image     │         │     Text     │
│ (X-ray CXR) │         │ (Impression) │
└──────┬──────┘         └──────┬───────┘
       │                       │
       ▼                       ▼
┌─────────────┐         ┌──────────────┐
│    Swin     │         │ BioClinical  │
│ Transformer │         │     BERT     │
└──────┬──────┘         └──────┬───────┘
       │                       │
       │    ┌──────────────┐   │
       └───►│ Cross-Modal  │◄──┘
            │  Attention   │
            └──────┬───────┘
                   │
            ┌──────▼───────┐
            │  Classifier  │
            │ (3 classes)  │
            └──────────────┘
```

### Model Components

- **Image Encoder**: `swin_base_patch4_window7_224` (pretrained on ImageNet)
- **Text Encoder**: `emilyalsentzer/Bio_ClinicalBERT` (pretrained on clinical notes)
- **Embedding Dimension**: 512
- **Cross-Modal Attention**: Multi-head attention (8 heads)
- **Classification Head**: 2-layer MLP with GELU activation

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd SemCC

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Data Preparation

Your data directory should follow this structure:

```
data/
├── data.csv          # CSV with columns: Image, Impression, Clean_Impression, Category
└── images/           # Folder containing X-ray images
    ├── img001.png
    ├── img002.png
    └── ...
```

**CSV Format:**
- `Image`: Filename of the X-ray image
- `Impression`: Raw radiology report text
- `Clean_Impression`: Cleaned/preprocessed report text (optional, falls back to Impression)
- `Category`: One of ["Normal", "Pneumonia", "TB"]

### Training

#### Single Fold Training

```bash
python train.py \
  --data_dir data \
  --csv_file data.csv \
  --image_dir data/images \
  --fold 0 \
  --num_folds 5 \
  --experiment_name semcxr_full \
  --num_epochs 100 \
  --batch_size 32 \
  --lr 1e-4
```

#### Cross-Validation Training

```bash
# Run all 5 folds with evaluation
bash run_experiments.sh

# Override parameters
FOLDS=5 NUM_EPOCHS=50 BATCH_SIZE=16 bash run_experiments.sh
```

### Evaluation

Evaluate a trained checkpoint:

```bash
python evaluate.py \
  --checkpoint experiments/semcxr_full/best_model.pt \
  --data_dir data \
  --csv_file data.csv \
  --image_dir data/images \
  --fold 0 \
  --output_dir results/semcxr_full
```

Results are saved to:
- `results/semcxr_full/eval_fold0.json` - Metrics summary
- `results/semcxr_full/predictions_fold0.csv` - Per-sample predictions

### Testing

Run unit tests:

```bash
pytest test.py -v
```

## ⚙️ Configuration

### Key Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image_encoder` | `swin_base_patch4_window7_224` | Vision backbone |
| `--text_encoder` | `emilyalsentzer/Bio_ClinicalBERT` | Text encoder |
| `--embed_dim` | 512 | Feature embedding dimension |
| `--batch_size` | 32 | Training batch size |
| `--num_epochs` | 100 | Number of training epochs |
| `--lr` | 1e-4 | Initial learning rate |
| `--weight_decay` | 0.05 | AdamW weight decay |
| `--mixup_alpha` | 0.4 | Mixup augmentation strength |
| `--cutmix_alpha` | 0.4 | CutMix augmentation strength |
| `--use_swa` | True | Use Stochastic Weight Averaging |
| `--swa_start` | 75 | Epoch to start SWA |
| `--precision` | `16-mixed` | Mixed precision (`32`, `16-mixed`, `bf16-mixed`) |

### Ablation Studies

Disable specific components:

```bash
# Without cross-modal attention
python train.py --no_cross_attn --experiment_name semcxr_no_xattn

# Without report generation
python train.py --no_report_gen --experiment_name semcxr_no_report

# Without SWA
python train.py --no_swa

# With focal loss (for class imbalance)
python train.py --use_focal_loss --focal_gamma 2.0
```

## 📁 Project Structure

```
SemCC/
├── train.py              # Main training script
├── evaluate.py           # Evaluation script
├── test.py               # Unit tests
├── run_experiments.sh    # Cross-validation runner
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── .gitignore           # Git ignore rules
├── data/                # Dataset (not tracked)
│   ├── data.csv
│   └── images/
├── experiments/         # Saved model checkpoints
│   ├── semcxr_full/
│   ├── semcxr_no_report/
│   └── semcxr_no_xattn/
└── results/             # Evaluation outputs
    ├── semcxr_full/
    ├── semcxr_no_report/
    └── semcxr_no_xattn/
```

## 🔧 Advanced Usage

### Distributed Training (Multi-GPU)

```bash
# Using torchrun
torchrun --nproc_per_node=4 train.py \
  --data_dir data \
  --csv_file data.csv \
  --image_dir data/images
```

### Resume Training

```bash
# Auto-resume from last checkpoint
python train.py --auto_resume

# Resume from specific checkpoint
python train.py --resume_from checkpoints/semcxr/last_model.pt
```

### Custom Checkpoint Directory

```bash
python train.py \
  --checkpoint_dir /path/to/checkpoints \
  --experiment_name my_experiment
```

## 📝 Training Features

### Data Augmentation
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast)
- Mixup and CutMix (configurable alpha)

### Regularization
- Label smoothing (0.1)
- Dropout (0.3)
- Weight decay (0.05)
- Stochastic Weight Averaging (SWA)
- Gradient clipping (1.0)

### Optimization
- AdamW optimizer
- OneCycleLR scheduler with warmup
- Mixed-precision training (AMP)
- Gradient accumulation support

### Class Imbalance Handling
- Weighted random sampling
- Optional focal loss
- Per-class AUC metrics

## 🐛 Known Issues

⚠️ **Important**: The following issues were identified during code review and should be addressed before production use:

1. **Distributed Training**: `rank` parameter in `dist.init_process_group` should use global rank, not local rank
2. **Data Sampling**: No `DistributedSampler` when using DDP, causing data overlap
3. **Checkpoint Loading**: Resume doesn't handle `module.` prefix differences between DDP/non-DDP
4. **LR Scheduling**: `OneCycleLR` total steps don't account for gradient accumulation
5. **Dataset Robustness**: Hardcoded column names and class list (crashes on CSV variations)
6. **Evaluation Loading**: Uses `strict=False` which can hide model mismatches

See inline comments in `train.py` for detailed fix locations.

## 📊 Experiment Insights

### Key Findings

1. **Cross-modal attention helps**: `semcxr_full` (with cross-attention) achieves highest accuracy (93.79%)
2. **Report generation ablation**: `semcxr_no_report` shows catastrophic failure (37% accuracy) - likely architecture mismatch during loading
3. **Best AUC**: `semcxr_no_xattn` achieves highest Macro AUC (0.9953) despite no cross-attention
4. **Class imbalance**: Normal class has lowest support (94 samples) but decent recall (0.93 in full model)

### Recommendations

- Use `semcxr_full` for balanced classification performance
- Use `semcxr_no_xattn` if optimizing for AUC metric
- Investigate `semcxr_no_report` checkpoint loading issue before deployment

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Fix distributed training issues
- [ ] Add TensorBoard logging
- [ ] Implement attention visualization
- [ ] Add ONNX export for deployment
- [ ] Expand dataset compatibility
- [ ] Add Grad-CAM visualization

## 📄 License

[Add your license here]

## 📧 Contact

[Add contact information]

## 🙏 Acknowledgments

- **Models**: 
  - Swin Transformer: [Microsoft Research](https://github.com/microsoft/Swin-Transformer)
  - BioClinicalBERT: [Emily Alsentzer et al.](https://arxiv.org/abs/1904.03323)
- **Frameworks**: PyTorch, Hugging Face Transformers, timm

## 📚 References

```bibtex
@article{liu2021swin,
  title={Swin transformer: Hierarchic vision transformer using shifted windows},
  author={Liu, Ze and Lin, Yutong and Cao, Yue and others},
  journal={ICCV},
  year={2021}
}

@article{alsentzer2019publicly,
  title={Publicly available clinical BERT embeddings},
  author={Alsentzer, Emily and Murphy, John R and Boag, Willie and others},
  journal={NAACL Clinical NLP},
  year={2019}
}
```
