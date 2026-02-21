#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh — Train SemCXR across K folds, save best.pth, evaluate
#
# Usage:
#   bash run_experiments.sh                           # defaults (5 folds)
#   FOLDS=3 NUM_EPOCHS=50 bash run_experiments.sh     # override via env vars
#   bash run_experiments.sh --folds 3 --epochs 50     # override via flags
# =============================================================================
set -euo pipefail

# ---- Defaults (override with env vars or flags) ----------------------------
FOLDS="${FOLDS:-5}"
DATA_DIR="${DATA_DIR:-data}"
CSV_FILE="${CSV_FILE:-data.csv}"
IMAGE_DIR="${IMAGE_DIR:-data/images/}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
IMAGE_ENCODER="${IMAGE_ENCODER:-swin_base_patch4_window7_224}"
PRECISION="${PRECISION:-16-mixed}"
NUM_WORKERS="${NUM_WORKERS:-8}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SEED="${SEED:-42}"

# ---- Parse optional CLI flags -----------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --folds)       FOLDS="$2";         shift 2 ;;
        --epochs)      NUM_EPOCHS="$2";    shift 2 ;;
        --batch_size)  BATCH_SIZE="$2";    shift 2 ;;
        --lr)          LR="$2";            shift 2 ;;
        --data_dir)    DATA_DIR="$2";      shift 2 ;;
        --csv_file)    CSV_FILE="$2";      shift 2 ;;
        --image_dir)   IMAGE_DIR="$2";     shift 2 ;;
        --encoder)     IMAGE_ENCODER="$2"; shift 2 ;;
        --precision)   PRECISION="$2";     shift 2 ;;
        --seed)        SEED="$2";          shift 2 ;;
        *)
            echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "============================================================"
echo "  SemCXR Experiment Runner"
echo "============================================================"
echo "  Folds:         $FOLDS"
echo "  Epochs:        $NUM_EPOCHS"
echo "  Batch size:    $BATCH_SIZE"
echo "  LR:            $LR"
echo "  Encoder:       $IMAGE_ENCODER"
echo "  Precision:     $PRECISION"
echo "  Seed:          $SEED"
echo "  Data dir:      $DATA_DIR"
echo "  Checkpoint dir:$CHECKPOINT_DIR"
echo "  Results dir:   $RESULTS_DIR"
echo "============================================================"
echo ""

# ---- Training loop ----------------------------------------------------------
TRAIN_FAILED=0

for fold in $(seq 0 $((FOLDS - 1))); do
    EXP_NAME="semcxr_fold${fold}"

    echo ""
    echo "------------------------------------------------------------"
    echo "  Training fold ${fold} / $((FOLDS - 1))"
    echo "------------------------------------------------------------"

    python train.py \
        --data_dir "$DATA_DIR" \
        --csv_file "$CSV_FILE" \
        --image_dir "$IMAGE_DIR" \
        --fold "$fold" \
        --num_folds "$FOLDS" \
        --experiment_name "$EXP_NAME" \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --num_epochs "$NUM_EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --lr "$LR" \
        --image_encoder "$IMAGE_ENCODER" \
        --precision "$PRECISION" \
        --seed "$SEED" \
    || { echo "ERROR: Training failed for fold $fold"; TRAIN_FAILED=1; continue; }

    # Copy best_model.pt -> best.pth (the canonical name evaluate.py expects)
    BEST_SRC="${CHECKPOINT_DIR}/${EXP_NAME}/best_model.pt"
    BEST_DST="${CHECKPOINT_DIR}/${EXP_NAME}/best.pth"

    if [ -f "$BEST_SRC" ]; then
        cp "$BEST_SRC" "$BEST_DST"
        echo "  -> Saved best.pth for fold ${fold}"
    else
        echo "  WARNING: No best_model.pt found for fold ${fold}"
    fi
done

# ---- Evaluation loop --------------------------------------------------------
echo ""
echo "============================================================"
echo "  Evaluating all folds"
echo "============================================================"

EVAL_FAILED=0

for fold in $(seq 0 $((FOLDS - 1))); do
    EXP_NAME="semcxr_fold${fold}"
    BEST_PTH="${CHECKPOINT_DIR}/${EXP_NAME}/best.pth"

    if [ -f "$BEST_PTH" ]; then
        echo ""
        echo "  Evaluating fold ${fold}..."
        python evaluate.py \
            --checkpoint "$BEST_PTH" \
            --data_dir "$DATA_DIR" \
            --csv_file "$CSV_FILE" \
            --image_dir "$IMAGE_DIR" \
            --fold "$fold" \
            --num_folds "$FOLDS" \
            --output_dir "$RESULTS_DIR" \
            --seed "$SEED" \
        || { echo "ERROR: Evaluation failed for fold $fold"; EVAL_FAILED=1; continue; }
    else
        echo "  SKIP: No best.pth for fold ${fold}"
    fi
done

# ---- Aggregate results ------------------------------------------------------
echo ""
echo "============================================================"
echo "  Per-fold Results"
echo "============================================================"

for fold in $(seq 0 $((FOLDS - 1))); do
    RESULT_FILE="${RESULTS_DIR}/eval_fold${fold}.json"
    if [ -f "$RESULT_FILE" ]; then
        echo "  Fold ${fold}:"
        python -c "
import json, sys
with open('${RESULT_FILE}') as f:
    r = json.load(f)
print(f'    Accuracy:  {r[\"Accuracy\"]:.2f}%')
print(f'    Macro AUC: {r[\"Macro_AUC\"]:.4f}')
print(f'    F1 Macro:  {r[\"F1_Macro\"]:.4f}')
"
    fi
done

# Compute cross-fold average
echo ""
echo "------------------------------------------------------------"
echo "  Cross-fold Average"
echo "------------------------------------------------------------"
python -c "
import json, glob, numpy as np
files = sorted(glob.glob('${RESULTS_DIR}/eval_fold*.json'))
if not files:
    print('  No result files found.')
else:
    accs, aucs, f1s = [], [], []
    for f in files:
        with open(f) as fp:
            r = json.load(fp)
        accs.append(r['Accuracy'])
        aucs.append(r['Macro_AUC'])
        f1s.append(r['F1_Macro'])
    print(f'  Accuracy:  {np.mean(accs):.2f} +/- {np.std(accs):.2f}')
    print(f'  Macro AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}')
    print(f'  F1 Macro:  {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}')
"

echo ""
echo "============================================================"
if [ "$TRAIN_FAILED" -eq 1 ] || [ "$EVAL_FAILED" -eq 1 ]; then
    echo "  Done (with some failures — check logs above)"
    exit 1
else
    echo "  All experiments completed successfully!"
    echo "  Results: ${RESULTS_DIR}/"
    echo "  Checkpoints: ${CHECKPOINT_DIR}/"
fi
echo "============================================================"
