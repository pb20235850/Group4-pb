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

"${PYTHON_BIN}" dpo/scripts/mbpp_eval_tot.py \
  --config "${MBPP_CONFIG:-sanitized}" \
  --split "${MBPP_SPLIT:-test}" \
  --prompt_mode "${PROMPT_MODE:-zero_shot}" \
  --model_path "${MODEL_PATH:-dpo/outputs/qwen15_code_lora_grpo_v5}" \
  --output_dir "${OUTPUT_DIR:-dpo/outputs/mbpp_tot_t6_b2_beam4}" \
  --num_thoughts "${NUM_THOUGHTS:-6}" \
  --branches_per_thought "${BRANCHES_PER_THOUGHT:-2}" \
  --beam_width "${BEAM_WIDTH:-4}" \
  --expand_rounds "${EXPAND_ROUNDS:-1}" \
  --temperature "${TEMPERATURE:-0.6}" \
  --top_p "${TOP_P:-0.9}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-512}" \
  "$@"