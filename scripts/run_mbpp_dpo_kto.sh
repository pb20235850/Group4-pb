#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${DPO_DIR}/.." && pwd)"

DEFAULT_PYTHON="/opt/conda/envs/shixun-sft/bin/python"
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

"${PYTHON_BIN}" - <<'PY'
import os
import torch

print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("torch_version=", torch.__version__)
print("torch_cuda_version=", torch.version.cuda)
print("torch.cuda.is_available()=", torch.cuda.is_available())
print("visible_device_count=", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: 当前环境没有识别到 GPU，停止评估。")

print("device0=", torch.cuda.get_device_name(0))
PY

"${PYTHON_BIN}" dpo/scripts/mbpp_eval_dpo.py \
  --config "${MBPP_CONFIG:-sanitized}" \
  --split "${MBPP_SPLIT:-test}" \
  --prompt_mode "${PROMPT_MODE:-zero_shot}" \
  --model_path "${DPO_MODEL_PATH:-dpo/outputs/qwen15_code_lora_kto}" \
  --output_dir "${OUTPUT_DIR:-dpo/outputs/mbpp_eval_kto}" \
  --batch_size "${BATCH_SIZE:-1}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  "$@"