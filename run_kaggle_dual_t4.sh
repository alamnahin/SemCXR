#!/usr/bin/env bash
set -euo pipefail

# =============================================
# Dual-T4 (2 GPU) launcher for Kaggle
# =============================================
# Expected paths on Kaggle:
#   /kaggle/working/SemCC
#   /kaggle/working/data/data.csv
#   /kaggle/working/data/images/

# ---- Configurable env vars ----
FOLDS="${FOLDS:-5}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LR="${LR:-1e-4}"
PRECISION="${PRECISION:-16-mixed}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_PORT="${MASTER_PORT:-29501}"
SEED="${SEED:-42}"

DATA_DIR="${DATA_DIR:-/kaggle/working/data}"
CSV_FILE="${CSV_FILE:-data.csv}"
IMAGE_DIR="${IMAGE_DIR:-/kaggle/working/data/images}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/kaggle/working/checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-/kaggle/working/results}"

mkdir -p "${CHECKPOINT_DIR}" "${RESULTS_DIR}"

# NCCL/Multi-GPU stability defaults for notebook environments
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="false"

run_exp() {
  local EXP_NAME="$1"
  local EXTRA_FLAGS="$2"

  echo "============================================================"
  echo "Running experiment: ${EXP_NAME}"
  echo "============================================================"

  for fold in $(seq 0 $((FOLDS - 1))); do
    local RUN_NAME="${EXP_NAME}_fold${fold}"

    echo "------------------------------------------------------------"
    echo "Fold ${fold}/${FOLDS} :: ${RUN_NAME}"
    echo "------------------------------------------------------------"

    torchrun \
      --standalone \
      --nnodes=1 \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      train.py \
      --data_dir "${DATA_DIR}" \
      --csv_file "${CSV_FILE}" \
      --image_dir "${IMAGE_DIR}" \
      --fold "${fold}" \
      --num_folds "${FOLDS}" \
      --experiment_name "${RUN_NAME}" \
      --checkpoint_dir "${CHECKPOINT_DIR}" \
      --num_epochs "${NUM_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}" \
      --lr "${LR}" \
      --precision "${PRECISION}" \
      --seed "${SEED}" \
      --save_fold_metrics \
      ${EXTRA_FLAGS}

    BEST_CKPT="${CHECKPOINT_DIR}/${RUN_NAME}/best_model.pt"
    if [[ -f "${BEST_CKPT}" ]]; then
      python evaluate.py \
        --checkpoint "${BEST_CKPT}" \
        --data_dir "${DATA_DIR}" \
        --csv_file "${CSV_FILE}" \
        --image_dir "${IMAGE_DIR}" \
        --fold "${fold}" \
        --num_folds "${FOLDS}" \
        --output_dir "${RESULTS_DIR}/${EXP_NAME}" \
        --seed "${SEED}"
    else
      echo "WARNING: best_model.pt missing for ${RUN_NAME}; skipping eval"
    fi
  done
}

# 1) Full model
run_exp "semcxr_full" ""

# 2) No report generation
run_exp "semcxr_no_report" "--no_report_gen"

# 3) No cross-attention
run_exp "semcxr_no_xattn" "--no_cross_attn"

echo "============================================================"
echo "All experiments finished."
echo "Checkpoints: ${CHECKPOINT_DIR}"
echo "Results:     ${RESULTS_DIR}"
echo "============================================================"
