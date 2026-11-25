#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPYCACHEPREFIX=/tmp/pycache
export PYTHONWARNINGS=ignore::SyntaxWarning

export HF_HOME="${HF_HOME:-"${PROJECT_ROOT}/.cache/huggingface"}"
export LMUData="${LMUData:-"${PROJECT_ROOT}/datasets/VLMEvalKit_Data"}"

CFG_TEMPLATE="${CFG_TEMPLATE:-${SCRIPT_DIR}/eval_cfg_qwen3_5_integrated.json}"
OUTPUT_DIR="${OUTPUT_DIR:-eval_outputs/vlmevalkit}"

# 1: dense Qwen3.5. 0: SAMoD checkpoint.
EVAL_ORIGINAL="${EVAL_ORIGINAL:-0}"
if [[ "${EVAL_ORIGINAL}" == "1" ]]; then
  MODEL_CHOICE=original
  MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B}"
  MODEL_PATH="${MODEL_PATH:-pretrained/Qwen3.5-4B}"
else
  MODEL_CHOICE=samod
  MODEL_NAME="${MODEL_NAME:-Qwen3.5-4B-SAMoD-Image-R25}"
  MODEL_PATH="${MODEL_PATH:-pretrained/Qwen3.5-4B-SAMoD-Image-R25}"
fi

NGPU="${1:-$(python3 -c "import torch; print(torch.cuda.device_count())")}"
if [[ -z "${MODEL_NAME}" || -z "${MODEL_PATH}" ]]; then
  echo "MODEL_NAME and MODEL_PATH must be set." >&2
  exit 1
fi
if [[ ! -f "${CFG_TEMPLATE}" ]]; then
  echo "Config template not found: ${CFG_TEMPLATE}" >&2
  exit 1
fi
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}/${MODEL_NAME}"
CFG_PATH="${OUTPUT_DIR}/${MODEL_NAME}/eval_cfg_runtime.json"
python3 - <<PY
import json
from pathlib import Path

template = Path(${CFG_TEMPLATE@Q})
out = Path(${CFG_PATH@Q})
model_name = ${MODEL_NAME@Q}
model_path = ${MODEL_PATH@Q}
model_choice = ${MODEL_CHOICE@Q}

cfg = json.loads(template.read_text(encoding="utf-8"))
if not isinstance(cfg.get("model"), dict) or not cfg["model"]:
    raise SystemExit(f"Invalid model section in {template}")

old_key = next(iter(cfg["model"]))
entry = dict(cfg["model"][old_key])
entry["model_path"] = model_path
entry["model_choice"] = model_choice
if model_choice == "original":
    entry.pop("model_choice_kwargs", None)
    entry["report_vision_token_retention"] = False
cfg["model"] = {model_name: entry}
out.write_text(json.dumps(cfg, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[eval] wrote runtime config: {out}")
print(f"[eval] model key    : {model_name}")
print(f"[eval] model_choice : {model_choice}")
print(f"[eval] model_path   : {model_path}")
PY

if [[ "${MODEL_CHOICE}" == "samod" ]]; then
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

echo "=============================================="
echo " VLMEvalKit eval"
echo " MODEL_CHOICE=${MODEL_CHOICE}"
echo " MODEL_NAME=${MODEL_NAME}"
echo " MODEL_PATH=${MODEL_PATH}"
echo " CFG_PATH=${CFG_PATH}"
echo " OUTPUT_DIR=${OUTPUT_DIR}"
echo " NGPU=${NGPU}"
echo "=============================================="

torchrun --nproc-per-node="$NGPU" --nnode=1 \
    VLMEvalKit/run.py \
    --config "$CFG_PATH" \
    --work-dir "$OUTPUT_DIR" \
    --judge exact_matching \
    --verbose
