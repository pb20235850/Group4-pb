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

GPU_ID="${GPU_ID:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_ID}}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" dpo/scripts/mbpp_eval_dpo.py \
  --config "${MBPP_CONFIG:-sanitized}" \
  --split "${MBPP_SPLIT:-test}" \
  --prompt_mode "${PROMPT_MODE:-zero_shot}" \
  --model_path "${DPO_MODEL_PATH:-dpo/outputs/qwen15_code_full_dpo}" \
  --output_dir "${OUTPUT_DIR:-dpo/outputs/mbpp_eval_dpo_final}" \
  --batch_size "${BATCH_SIZE:-8}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  "$@"
