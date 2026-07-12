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

BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen1.5-0.5B-Chat}"
DPO_MODEL_PATH="${DPO_MODEL_PATH:-dpo/outputs/qwen15_code_full_dpo}"

MBPP_CONFIG="${MBPP_CONFIG:-sanitized}"
MBPP_SPLIT="${MBPP_SPLIT:-test}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
LIMIT="${LIMIT:-0}"
START_INDEX="${START_INDEX:-0}"
TEST_TIMEOUT="${TEST_TIMEOUT:-5}"
MEMORY_MB="${MEMORY_MB:-1024}"
PROMPT_TASK_IDS="${PROMPT_TASK_IDS:-2,3,4}"

BASE_GPUS="${BASE_GPUS:-0}"
DPO_GPUS="${DPO_GPUS:-0}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"

OUTPUT_ROOT="${OUTPUT_ROOT:-dpo/outputs/mbpp_base_dpo_compare}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${OUTPUT_ROOT}/base}"
DPO_OUTPUT_DIR="${DPO_OUTPUT_DIR:-${OUTPUT_ROOT}/dpo}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-${OUTPUT_ROOT}/mbpp_base_vs_dpo_metrics.json}"

export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

cd "${PROJECT_ROOT}"
mkdir -p "${BASE_OUTPUT_DIR}" "${DPO_OUTPUT_DIR}" "${OUTPUT_ROOT}"
extra_args=("$@")

common_args=(
  --config "${MBPP_CONFIG}"
  --split "${MBPP_SPLIT}"
  --prompt_mode "${PROMPT_MODE:-zero_shot}"
  --prompt_task_ids "${PROMPT_TASK_IDS}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --start_index "${START_INDEX}"
  --test_timeout "${TEST_TIMEOUT}"
  --memory_mb "${MEMORY_MB}"
)

if [[ "${LIMIT}" != "0" ]]; then
  common_args+=(--limit "${LIMIT}")
fi

if [[ "${INCLUDE_CHALLENGE_TESTS:-0}" == "1" ]]; then
  common_args+=(--include_challenge_tests)
fi

run_one() {
  local name="$1"
  local gpus="$2"
  local model_path="$3"
  local output_dir="$4"
  local log_path="${output_dir}/run.log"

  echo "[${name}] GPUs=${gpus} model=${model_path} output=${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpus}" "${PYTHON_BIN}" dpo/scripts/mbpp_eval_dpo.py \
    --model_path "${model_path}" \
    --output_dir "${output_dir}" \
    "${common_args[@]}" \
    "${extra_args[@]}" \
    >"${log_path}" 2>&1
}

if [[ "${RUN_PARALLEL}" == "1" ]]; then
  run_one "base" "${BASE_GPUS}" "${BASE_MODEL_PATH}" "${BASE_OUTPUT_DIR}" &
  base_pid=$!
  run_one "dpo" "${DPO_GPUS}" "${DPO_MODEL_PATH}" "${DPO_OUTPUT_DIR}" &
  dpo_pid=$!

  base_status=0
  dpo_status=0
  wait "${base_pid}" || base_status=$?
  wait "${dpo_pid}" || dpo_status=$?

  if [[ "${base_status}" != "0" || "${dpo_status}" != "0" ]]; then
    echo "MBPP evaluation failed. Logs:"
    echo "  base: ${BASE_OUTPUT_DIR}/run.log"
    echo "  dpo : ${DPO_OUTPUT_DIR}/run.log"
    exit 1
  fi
else
  run_one "base" "${BASE_GPUS}" "${BASE_MODEL_PATH}" "${BASE_OUTPUT_DIR}"
  run_one "dpo" "${DPO_GPUS}" "${DPO_MODEL_PATH}" "${DPO_OUTPUT_DIR}"
fi

"${PYTHON_BIN}" dpo/scripts/compare_mbpp_metrics.py \
  --base_metrics "${BASE_OUTPUT_DIR}/mbpp_metrics.json" \
  --dpo_metrics "${DPO_OUTPUT_DIR}/mbpp_metrics.json" \
  --output "${COMPARE_OUTPUT}"
