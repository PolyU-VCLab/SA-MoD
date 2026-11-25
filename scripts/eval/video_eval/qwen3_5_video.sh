#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/pycache
export PYTHONWARNINGS=ignore::SyntaxWarning

# set your own HF_HOME and LMMS_EVAL_DATASETS_CACHE for lmms-eval
export HF_HOME="${HF_HOME:-"${PROJECT_ROOT}/.cache/huggingface"}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-"${PROJECT_ROOT}/datasets/lmms_eval"}"

# Caller sets MODEL_NAME / MODEL_PATH. EVAL_ORIGINAL=1 replaces them with dense Qwen3.5.
EVAL_ORIGINAL="${EVAL_ORIGINAL:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen3.5-SAMoD-Video-R10}"
MODEL_PATH="${MODEL_PATH:-pretrained/Qwen3.5-SAMoD-Video-R10}"
LMMS_MODEL=qwen3_5_samod
if [[ "${EVAL_ORIGINAL}" == "1" ]]; then
  LMMS_MODEL=qwen3_5
  MODEL_NAME=Qwen3.5-4B
  MODEL_PATH=pretrained/Qwen3.5-4B
fi

NGPU="${1:-${NGPU:-$(python3 -c "import torch; print(torch.cuda.device_count())")}}"
TASKS="${TASKS:-egoschema_subset,mvbench}"
OUTPUT_PATH="${OUTPUT_PATH:-eval_outputs/lmms_eval}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12347}"

MAX_FRAMES="${MAX_FRAMES:-64}"
MAX_TOKENS_PER_FRAME="${MAX_TOKENS_PER_FRAME:-768}"
MIN_PIXELS="${MIN_PIXELS:-$((64 * 32 * 32))}"
MAX_PIXELS="${MAX_PIXELS:-$((MAX_TOKENS_PER_FRAME * 32 * 32))}"
FPS="${FPS:-}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
FORCE_DIRECT_ANSWER="${FORCE_DIRECT_ANSWER:-true}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
export MAX_TOKENS_PER_FRAME VIDEO_MAX_TOKEN_NUM="${MAX_TOKENS_PER_FRAME}"

if [[ -z "${MODEL_NAME}" || -z "${MODEL_PATH}" ]]; then
  echo "MODEL_NAME and MODEL_PATH must be set." >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

if [[ "${EVAL_ORIGINAL}" != "1" ]]; then
  needs_merge=0
  if [[ -d "${MODEL_PATH}/actor" ]] || [[ -f "${MODEL_PATH}/fsdp_config.json" ]] || compgen -G "${MODEL_PATH}/model_world_size_*_rank_*.pt" > /dev/null; then
    needs_merge=1
  fi
  if [[ "${needs_merge}" == "1" ]]; then
    echo "=============================================="
    echo " Merge checkpoints"
    echo " MODEL_PATH=${MODEL_PATH}"
    echo "=============================================="
    bash "${PROJECT_ROOT}/scripts/merge_checkpoint.sh" "${MODEL_PATH}"
  fi
fi

MODEL_ARGS="pretrained=${MODEL_PATH},min_pixels=${MIN_PIXELS},max_pixels=${MAX_PIXELS},max_frames=${MAX_FRAMES},attn_implementation=flash_attention_2,interleave_visuals=False,enable_thinking=${ENABLE_THINKING},force_direct_answer=${FORCE_DIRECT_ANSWER}"
if [[ -n "${FPS}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},fps=${FPS}"
fi
GEN_KWARGS=()
if [[ -n "${MAX_NEW_TOKENS}" ]]; then
  GEN_KWARGS=(--gen_kwargs "max_new_tokens=${MAX_NEW_TOKENS}")
fi

mkdir -p "${OUTPUT_PATH}"

echo "=============================================="
echo " lmms-eval video"
echo " EVAL_ORIGINAL=${EVAL_ORIGINAL}"
echo " LMMS_MODEL=${LMMS_MODEL}"
echo " MODEL_NAME=${MODEL_NAME}"
echo " MODEL_PATH=${MODEL_PATH}"
echo " TASKS=${TASKS}"
echo " MAX_FRAMES=${MAX_FRAMES} FPS=${FPS:-nframes} MAX_TOKENS_PER_FRAME=${MAX_TOKENS_PER_FRAME}"
echo " OUTPUT_PATH=${OUTPUT_PATH}"
echo " LMMS_LOG_GEN_METRICS=${LMMS_LOG_GEN_METRICS}"
echo " NGPU=${NGPU}"
echo "=============================================="

accelerate launch --num_processes="${NGPU}" --main_process_port="${MAIN_PROCESS_PORT}" \
  -m lmms_eval \
  --model "${LMMS_MODEL}" \
  --model_args "${MODEL_ARGS}" \
  --tasks "${TASKS}" \
  --batch_size "${BATCH_SIZE}" \
  --log_samples \
  --output_path "${OUTPUT_PATH}/${MODEL_NAME}" \
  "${GEN_KWARGS[@]}"
