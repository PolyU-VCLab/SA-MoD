#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_DIR="${BASE_DIR:-${1:-"${PROJECT_ROOT}/workbench"}}"
BASE_DIR="${BASE_DIR%/}"
ACTOR_DIR="${BASE_DIR}/actor"

# SFT checkpoints store FSDP shards at the top level. Normalize them into actor/
# so the layout matches RL/OPD and merge output stays at the top level.
if [ ! -d "${ACTOR_DIR}" ]; then
  shopt -s nullglob
  shard_files=(
    "${BASE_DIR}"/fsdp_config.json
    "${BASE_DIR}"/model_world_size_*_rank_*.pt
    "${BASE_DIR}"/optim_world_size_*_rank_*.pt
    "${BASE_DIR}"/extra_state_world_size_*_rank_*.pt
    "${BASE_DIR}"/data_*.pt
  )

  if [ -e "${BASE_DIR}/fsdp_config.json" ] || ls "${BASE_DIR}"/model_world_size_*_rank_*.pt >/dev/null 2>&1; then
    echo "Detected SFT-style checkpoint, moving FSDP artifacts to ${ACTOR_DIR}/"
    mkdir -p "${ACTOR_DIR}"

    for f in "${shard_files[@]}"; do
      if [ -e "${f}" ]; then
        echo "Moving ${f} -> ${ACTOR_DIR}/"
        mv "${f}" "${ACTOR_DIR}/"
      fi
    done

    if [ -d "${BASE_DIR}/huggingface" ]; then
      echo "Moving ${BASE_DIR}/huggingface -> ${ACTOR_DIR}/"
      mv "${BASE_DIR}/huggingface" "${ACTOR_DIR}/"
    fi
  else
    echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
    exit 1
  fi
fi

if [ ! -d "${ACTOR_DIR}" ]; then
  echo "Actor checkpoint directory not found: ${ACTOR_DIR}" >&2
  exit 1
fi

# A merged HF checkpoint lives at the top level (actor shards stay in actor/).
is_merged_ckpt() {
  local dir="$1"
  [ -f "${dir}/config.json" ] || return 1
  [ -f "${dir}/model.safetensors" ] \
    || [ -f "${dir}/model.safetensors.index.json" ] \
    || [ -f "${dir}/pytorch_model.bin" ] \
    || [ -f "${dir}/pytorch_model.bin.index.json" ] \
    || compgen -G "${dir}/model-*-of-*.safetensors" >/dev/null
}

FORCE_MERGE="${FORCE_MERGE:-0}"
if [ "${FORCE_MERGE}" != "1" ] && is_merged_ckpt "${BASE_DIR}"; then
  echo "Merged checkpoint already exists in ${BASE_DIR}, skipping merge."
  echo "Set FORCE_MERGE=1 to re-merge."
  exit 0
fi

TRUST_REMOTE_CODE_ARGS=()
if [ -f "${ACTOR_DIR}/huggingface/config.json" ] && grep -Eq '"model_type"[[:space:]]*:[[:space:]]*"(neo_chat|internvl_chat|llavaonevision1_5)"' "${ACTOR_DIR}/huggingface/config.json"; then
  # NEO, InternVL, and LLaVA-OV-1.5 are not stock Transformers architectures.
  # Qwen checkpoints do not match these model types and keep the previous loader path.
  TRUST_REMOTE_CODE_ARGS=(--trust-remote-code)
fi

echo "Merging ${ACTOR_DIR} -> ${BASE_DIR}"

# Remove previously merged top-level files so verl can overwrite the target
# directory cleanly while keeping the original actor shards intact.
find "${BASE_DIR}" -mindepth 1 -maxdepth 1 -type f -print -delete
if [ -d "${BASE_DIR}/lora_adapter" ]; then
  echo "Removing previous merge artifact: ${BASE_DIR}/lora_adapter"
  rm -rf "${BASE_DIR}/lora_adapter"
fi

python3 -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "${ACTOR_DIR}" \
  --target_dir "${BASE_DIR}" \
  "${TRUST_REMOTE_CODE_ARGS[@]}"

echo "Merge completed."
