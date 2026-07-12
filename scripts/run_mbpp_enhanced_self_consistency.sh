#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"

DEFAULT_PYTHON="/home/pb/assignment_A/bin/python"
if [[ -x "${DEFAULT_PYTHON}" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
else
  PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

GPU_ID="${GPU_ID:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" dpo/scripts/mbpp_infer_enhance.py \
  --mode self_consistency \
  --model_path "${MODEL_PATH:-/home/pb/assignment_A/dpo/outputs/qwen15_code_lora_grpo_v5}" \
  --train_file "${TRAIN_FILE:-/home/pb/mbpp/sanitized/train-00000-of-00001.parquet}" \
  --test_file "${TEST_FILE:-/home/pb/mbpp/sanitized/test-00000-of-00001.parquet}" \
  --output_dir "${OUTPUT_DIR:-dpo/outputs/mbpp_eval_self_consistency_8}" \
  --num_candidates "${NUM_CANDIDATES:-8}" \
  --num_fewshot "${NUM_FEWSHOT:-2}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top_p "${TOP_P:-0.95}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-256}" \
  "$@"