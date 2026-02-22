#!/usr/bin/env bash
set -euo pipefail

CSV_PATH="${CSV_PATH:-data/data.csv}"
IMAGE_DIR="${IMAGE_DIR:-data/images}"
OUTPUT_DIR="${OUTPUT_DIR:-eda_reports}"
MAX_IMAGES="${MAX_IMAGES:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --csv)
      CSV_PATH="$2"
      shift 2
      ;;
    --image_dir)
      IMAGE_DIR="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --max_images)
      MAX_IMAGES="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: bash run_eda.sh [--csv PATH] [--image_dir PATH] [--output_dir PATH] [--max_images N] [--python PYTHON_BIN]"
      exit 1
      ;;
  esac
done

echo "Running EDA with:"
echo "  CSV_PATH:   ${CSV_PATH}"
echo "  IMAGE_DIR:  ${IMAGE_DIR}"
echo "  OUTPUT_DIR: ${OUTPUT_DIR}"
echo "  MAX_IMAGES: ${MAX_IMAGES}"
echo "  PYTHON_BIN: ${PYTHON_BIN}"

"${PYTHON_BIN}" eda_report.py \
  --csv "${CSV_PATH}" \
  --image_dir "${IMAGE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_images "${MAX_IMAGES}"

echo "EDA done. Artifacts saved in: ${OUTPUT_DIR}"
