#!/bin/bash

set -euo pipefail

export SEED=42
export PYTHONHASHSEED=$SEED
export RAY_memory_usage_threshold=0.95
export PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
export SAMOD_ROOT="${PROJECT_ROOT}/samod"

TRAINER_N_GPUS_PER_NODE=${NGPU:-8}
TRAINER_NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-${RANK:-${GROUP_RANK:-0}}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
RAY_PORT=${RAY_PORT:-6379}
NODE_IP_ADDRESS=${NODE_IP_ADDRESS:-}
RAY_HEAD_ADDRESS=${MASTER_ADDR}:${RAY_PORT}
if (( TRAINER_NNODES < 1 )); then
    echo "NNODES/TRAINER_NNODES must be >= 1, got ${TRAINER_NNODES}" >&2
    exit 1
fi
if (( NODE_RANK < 0 || NODE_RANK >= TRAINER_NNODES )); then
    echo "NODE_RANK=${NODE_RANK} out of range for NNODES=${TRAINER_NNODES}" >&2
    exit 1
fi
if (( TRAINER_NNODES > 1 )); then
    if [[ -z "${MASTER_ADDR}" || "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "localhost" ]]; then
        echo "Multi-node requires MASTER_ADDR=<head node IP/hostname reachable by workers>" >&2
        exit 1
    fi
fi

CONFIG_NAME=vopd
BASE_MODEL_PATH=${PROJECT_ROOT}/pretrained/Qwen3.5-4B
STUDENT_CONFIG_FILE=${SAMOD_ROOT}/verl/models/qwen3_5_samod/configs/samod_cfg.json
STUDENT_CONFIG_NAME=$(basename "${STUDENT_CONFIG_FILE%.*}")
STUDENT_REPO_BUILDER=${SAMOD_ROOT}/verl/models/transformers/qwen3_5_samod.py
STUDENT_MODEL_PATH=${PROJECT_ROOT}/runtime_models/$(basename "$BASE_MODEL_PATH")-${STUDENT_CONFIG_NAME}
TEACHER_MODEL_PATH=$BASE_MODEL_PATH
TEACHER_MODEL_SOURCE=fixed
TEACHER_REGULARIZATION=ema
TEACHER_UPDATE_RATE=0.0

# HF rollout uses the actual SAMoD actor with KV cache.
ROLLOUT_BACKEND=hf
TRAIN_BATCH_SIZE=32
PPO_MIMI_BATCH_SIZE=$TRAIN_BATCH_SIZE
ROLLOUT_N=4
GRAD_ACCUM_STEPS=$((TRAIN_BATCH_SIZE * ROLLOUT_N / 2 / TRAINER_N_GPUS_PER_NODE / TRAINER_NNODES))
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
ROLLOUT_HF_MICRO_BATCH_SIZE=4
ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=4
ROLLOUT_USE_GREEDY=false
LR=2e-6
ROUTER_LR=2e-6
FREEZE_VISION_TOWER=false
DONT_REPROMPT_ON_SELF_SUCCESS=true
ALPHA=0.5
MAX_PROMPT_LENGTH=48896
MAX_RESPONSE_LENGTH=256
MAX_REPROMPT_LEN=48896
TRAIN_MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$TRAIN_MAX_MODEL_LEN}"

MIN_PIXELS=$((64 * 32 * 32))
MAX_PIXELS=$((16384 * 32 * 32))
DIVIDE_MAX_PIXELS_BY_IMAGE_COUNT=false
VIDEO_FPS=2
VIDEO_NFRAMES=
VIDEO_MIN_FRAMES=4
VIDEO_MAX_FRAMES=256
VIDEO_MIN_PIXELS=$((16 * 32 * 32))
VIDEO_MAX_PIXELS=$((768 * 32 * 32))
VIDEO_TOTAL_PIXELS=$((49152 * 32 * 32))

ROLLOUT_GPU_MEMORY_UTILIZATION=0.7
ROLLOUT_MAX_NUM_SEQS=256
ACTOR_USE_DYNAMIC_BSZ=false
PPO_MAX_TOKEN_LEN_PER_GPU=$TRAIN_MAX_MODEL_LEN
# Fixed batch (dynamic_bsz=false): local mini-batch is split into GRAD_ACCUM_STEPS
# equal micro-batches. Gradients accumulate, then one optimizer step.
# Dynamic batch: each of those slices is token-packed on its own before the step.
DP_SIZE=$((TRAINER_NNODES * TRAINER_N_GPUS_PER_NODE))
GLOBAL_MINI=$((PPO_MIMI_BATCH_SIZE * ROLLOUT_N))
if (( DP_SIZE < 1 || GLOBAL_MINI % DP_SIZE != 0 )); then
    echo "global mini-batch $GLOBAL_MINI is not divisible by data-parallel size $DP_SIZE" >&2
    exit 1
fi
LOCAL_MINI=$((GLOBAL_MINI / DP_SIZE))
if (( GRAD_ACCUM_STEPS < 1 || LOCAL_MINI % GRAD_ACCUM_STEPS != 0 )); then
    echo "per-GPU mini-batch $LOCAL_MINI must be divisible by GRAD_ACCUM_STEPS ($GRAD_ACCUM_STEPS)" >&2
    exit 1
fi
# Fixed batch: one micro-batch is local_mini / GRAD_ACCUM_STEPS samples on every rank.
# Log-prob forwards use that same count so they do not token-pack independently.
if [[ "$ACTOR_USE_DYNAMIC_BSZ" == "false" ]]; then
    ACTOR_MICRO_BATCH_SIZE_PER_GPU=$((LOCAL_MINI / GRAD_ACCUM_STEPS))
    ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=$ACTOR_MICRO_BATCH_SIZE_PER_GPU
    REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU=$ACTOR_MICRO_BATCH_SIZE_PER_GPU
else
    ACTOR_MICRO_BATCH_SIZE_PER_GPU=
fi
TEACHER_MODEL_DTYPE=fp32
ACTOR_MODEL_DTYPE=fp32
ACTOR_FORWARD_DTYPE=bfloat16
ROLLOUT_DTYPE=bfloat16
ACTOR_PARAM_OFFLOAD=true
ACTOR_OPTIMIZER_OFFLOAD=true
REF_PARAM_OFFLOAD=true
TRAINER_TOTAL_EPOCHS=100 # a large epoch number set as default
TRAINER_SAVE_FREQ=100 # save intermediate checkpoints as needed
TRAINER_MAX_ACTOR_CKPT_TO_KEEP=3
ROLLOUT_AGENT_NUM_WORKERS=1
DATA_DATALOADER_NUM_WORKERS=1

# Console only. To also log to SwanLab, export these before launch and set
# TRAINER_LOGGER='["console","swanlab"]':
#   SWANLAB_API_KEY   login key (swanlab.login)
#   SWANLAB_MODE      online (default), local, or disabled
#   SWANLAB_LOG_DIR   local log directory (default swanlog)
# trainer.project_name is the SwanLab project; trainer.experiment_name is the run.
TRAINER_LOGGER='["console"]'
PROJECT_NAME=SA-MoD

CUSTOM_CHAT_TEMPLATE_FILE=${SAMOD_ROOT}/chat_templates/chat_template_update.jinja
ENABLE_THINKING=true
FORCE_DIRECT_ANSWER=false

# --- Data Paths ---
TASK_TRAIN_DIR=${PROJECT_ROOT}/datasets/SA-MoD-41K/SA-MoD-Image-25K
# TASK_TRAIN_FILE=${PROJECT_ROOT}/datasets/SA-MoD-41K/SA-MoD-Video-16K/train_video_16k.parquet
# Root directory joined with relative video paths in video training annotations (e.g., Video-R1-data).
# VIDEO_ROOT=${VIDEO_ROOT:-${PROJECT_ROOT}/datasets/Video-R1-data}
if [[ -n "${TASK_TRAIN_DIR:-}" ]]; then
    mapfile -t TASK_TRAIN_FILES < <(find -H "$TASK_TRAIN_DIR" -maxdepth 1 -name '*.parquet' | sort)
    if [[ ${#TASK_TRAIN_FILES[@]} -eq 0 ]]; then
        echo "No parquet shards found under ${TASK_TRAIN_DIR}" >&2
        exit 1
    fi
elif [[ -n "${TASK_TRAIN_FILE:-}" ]]; then
    if [[ ! -f "$TASK_TRAIN_FILE" ]]; then
        echo "Train parquet not found: ${TASK_TRAIN_FILE}" >&2
        exit 1
    fi
    TASK_TRAIN_FILES=("$TASK_TRAIN_FILE")
else
    echo "Set TASK_TRAIN_FILE (single parquet) or TASK_TRAIN_DIR (shard directory)" >&2
    exit 1
fi
TASK_TRAIN_FILES_CFG=$(printf '"%s",' "${TASK_TRAIN_FILES[@]}")
TASK_TRAIN_FILES_CFG="[${TASK_TRAIN_FILES_CFG%,}]"

MODEL_NAME="$(basename "$BASE_MODEL_PATH")"
EXPERIMENT_NAME="${MODEL_NAME}-SAMoD"

if [[ -z "${RUN_TIMESTAMP:-}" ]]; then
    RUN_TIMESTAMP="${JOB_ID:-${JOB_NAME:-${PET_JOB_NAME:-$(date +%Y%m%d_%H%M%S)}}}"
    RUN_TIMESTAMP="${RUN_TIMESTAMP//\//_}"
fi
TRAINER_WORKBENCH_DIR="${PROJECT_ROOT}/workbench/${EXPERIMENT_NAME}/${RUN_TIMESTAMP}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_WORKBENCH_DIR}/checkpoints"
TRAINER_ROLLOUT_DATA_DIR="${TRAINER_WORKBENCH_DIR}/rollouts"
TRAIN_DONE="${TRAINER_WORKBENCH_DIR}/.train_done"
STARTED_RAY=0

wait_for_tcp() {
    local host="$1"
    local port="$2"
    local timeout_s="${3:-1800}"
    echo "Waiting for TCP ${host}:${port} (timeout=${timeout_s}s)"
    python3 - "$host" "$port" "$timeout_s" <<'PY'
import socket, sys, time
host, port, timeout_s = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
deadline = time.time() + timeout_s
while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            print(f"TCP {host}:{port} is open", flush=True)
            raise SystemExit(0)
    except OSError:
        time.sleep(2)
raise SystemExit(f"Timeout waiting for TCP {host}:{port}")
PY
}

wait_for_ray_nodes() {
    local expected="$1"
    local address="${2:-${RAY_ADDRESS:-auto}}"
    EXPECTED_RAY_NODES="$expected" RAY_CONNECT_ADDRESS="$address" python3 - <<'PY'
import os, sys, time
import ray

expected = int(os.environ["EXPECTED_RAY_NODES"])
address = os.environ["RAY_CONNECT_ADDRESS"]
deadline = time.time() + 1800
last_err = None
while time.time() < deadline:
    try:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(address=address, ignore_reinit_error=True, namespace="wait-nodes")
        alive = sum(1 for n in ray.nodes() if n.get("Alive"))
        print(f"[ray] alive nodes={alive}/{expected} address={address}", flush=True)
        if alive >= expected:
            ray.shutdown()
            raise SystemExit(0)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        last_err = exc
        print(f"[ray] connect/list failed: {exc}", flush=True)
    time.sleep(2)
raise SystemExit(f"Timeout waiting for {expected} Ray nodes (last_err={last_err})")
PY
}

wait_until_local_ray_stopped() {
    local timeout_s="${1:-604800}"
    local ready_deadline=$((SECONDS + 300))
    # ray start returns before raylet is queryable; require one successful status first.
    while ! ray status >/dev/null 2>&1; do
        if (( SECONDS >= ready_deadline )); then
            echo "Local Ray did not become ready after join" >&2
            exit 1
        fi
        sleep 2
    done
    echo "Local Ray is up; waiting until head tears down the cluster"
    local stop_deadline=$((SECONDS + timeout_s))
    while ray status >/dev/null 2>&1; do
        if (( SECONDS >= stop_deadline )); then
            echo "Timeout waiting for local Ray to stop" >&2
            exit 1
        fi
        sleep 5
    done
}

ray_start_common_args=()
if [[ -n "${NODE_IP_ADDRESS}" ]]; then
    ray_start_common_args+=(--node-ip-address="${NODE_IP_ADDRESS}")
fi

cleanup_ray() {
    if (( STARTED_RAY == 1 )); then
        ray stop --force >/dev/null 2>&1 || true
    fi
}
cleanup_and_signal() {
    if [[ "${NODE_RANK}" -eq 0 ]]; then
        mkdir -p "$TRAINER_WORKBENCH_DIR"
        touch "$TRAIN_DONE" || true
    fi
    cleanup_ray
}
trap cleanup_and_signal EXIT

EXTRA_ARGS=("$@")

if (( TRAINER_NNODES == 1 )); then
    # Let verl.main_ppo create a local Ray instance; avoid attaching a stale cluster.
    unset RAY_ADDRESS || true
fi

if [[ "${NODE_RANK}" -eq 0 ]]; then
    mkdir -p "$TRAINER_ROLLOUT_DATA_DIR"
    RUN_CONFIG_DIR="${TRAINER_WORKBENCH_DIR}/run_configs"
    mkdir -p "$RUN_CONFIG_DIR"
    cp "${BASH_SOURCE[0]}" "${RUN_CONFIG_DIR}/$(basename "${BASH_SOURCE[0]}")"
    cp "$STUDENT_CONFIG_FILE" "${RUN_CONFIG_DIR}/$(basename "$STUDENT_CONFIG_FILE")"

    python3 "$STUDENT_REPO_BUILDER" \
        --build-student-repo \
        --student-config "$STUDENT_CONFIG_FILE" \
        --base-model-path "$BASE_MODEL_PATH" \
        --student-model-path "$STUDENT_MODEL_PATH" \
        --project-root "$SAMOD_ROOT" \
        --max-pixels "$MAX_PIXELS" \
        --min-pixels "$MIN_PIXELS"

    if (( TRAINER_NNODES > 1 )); then
        echo "Head node_rank=0: starting Ray head at ${RAY_HEAD_ADDRESS}"
        ray stop --force >/dev/null 2>&1 || true
        # Advertise MASTER_ADDR so workers can reach GCS (multi-NIC safe default).
        if [[ -n "${NODE_IP_ADDRESS}" ]]; then
            HEAD_IP_ARGS=(--node-ip-address="${NODE_IP_ADDRESS}")
        else
            HEAD_IP_ARGS=(--node-ip-address="${MASTER_ADDR}")
        fi
        ray start --head \
            --port="$RAY_PORT" \
            --dashboard-host=0.0.0.0 \
            --num-gpus="$TRAINER_N_GPUS_PER_NODE" \
            "${HEAD_IP_ARGS[@]}"
        STARTED_RAY=1
        # Explicit GCS address; "auto" alone is fragile across Ray versions / NICs.
        export RAY_ADDRESS="${RAY_HEAD_ADDRESS}"
        wait_for_ray_nodes "$TRAINER_NNODES" "$RAY_ADDRESS"
        echo "Head: Ray cluster ready (${TRAINER_NNODES} nodes)"
    fi
else
    if (( TRAINER_NNODES <= 1 )); then
        echo "NODE_RANK=${NODE_RANK} but NNODES=${TRAINER_NNODES}; refusing to start a worker" >&2
        exit 1
    fi
    echo "Worker node_rank=${NODE_RANK}: waiting for Ray head GCS at ${RAY_HEAD_ADDRESS}"
    wait_for_tcp "$MASTER_ADDR" "$RAY_PORT" 1800
    ray stop --force >/dev/null 2>&1 || true
    join_deadline=$((SECONDS + 1800))
    while true; do
        if ray start --address="${RAY_HEAD_ADDRESS}" \
            --num-gpus="$TRAINER_N_GPUS_PER_NODE" \
            "${ray_start_common_args[@]+"${ray_start_common_args[@]}"}"; then
            break
        fi
        if (( SECONDS >= join_deadline )); then
            echo "Worker node_rank=${NODE_RANK}: failed to join Ray at ${RAY_HEAD_ADDRESS}" >&2
            exit 1
        fi
        echo "Worker node_rank=${NODE_RANK}: join failed, retrying in 5s..."
        sleep 5
    done
    STARTED_RAY=1
    echo "Worker node_rank=${NODE_RANK}: joined Ray cluster; blocking until head stops Ray"
    # Do not depend on shared-FS TRAIN_DONE for liveness; exit when head tears down Ray.
    wait_until_local_ray_stopped 604800
    echo "Worker node_rank=${NODE_RANK}: Ray stopped, exiting"
    exit 0
fi

# =============================================================================
# ENVIRONMENT
# =============================================================================
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export USER="${USER:-$(id -un 2>/dev/null || echo root)}"
ulimit -c 0

CHAT_TEMPLATE_ARGS=()
if [[ -n "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
    if [[ ! -f "${CUSTOM_CHAT_TEMPLATE_FILE}" ]]; then
        echo "Custom chat template file not found: ${CUSTOM_CHAT_TEMPLATE_FILE}, using default template" >&2
    else
        CHAT_TEMPLATE_ARGS+=(actor_rollout_ref.model.custom_chat_template_file="$CUSTOM_CHAT_TEMPLATE_FILE")
    fi
fi

TEACHER_MODEL_ARGS=(
    actor_rollout_ref.actor.self_distillation.teacher_model_source="$TEACHER_MODEL_SOURCE"
    actor_rollout_ref.actor.self_distillation.teacher_regularization="$TEACHER_REGULARIZATION"
    actor_rollout_ref.actor.self_distillation.teacher_update_rate="$TEACHER_UPDATE_RATE"
)
if [[ "$TEACHER_MODEL_SOURCE" == "fixed" ]]; then
    TEACHER_MODEL_ARGS+=(actor_rollout_ref.actor.self_distillation.teacher_model_path="$TEACHER_MODEL_PATH")
fi

if [[ "$ROLLOUT_USE_GREEDY" == "true" ]]; then
    # HF generate ignores temperature when do_sample=false. Keep this at 1.0
    # because the same value is also used to scale training/distillation logits.
    # Setting it to zero makes logits.div_(temperature) produce NaN gradients.
    ROLLOUT_TEMPERATURE=1
    ROLLOUT_DO_SAMPLE=false
elif [[ "$ROLLOUT_USE_GREEDY" == "false" ]]; then
    ROLLOUT_TEMPERATURE=1
    ROLLOUT_DO_SAMPLE=true
else
    echo "Invalid ROLLOUT_USE_GREEDY=$ROLLOUT_USE_GREEDY (expected true or false)" >&2
    exit 1
fi

# Rollout backend args: vLLM (fast but SAMoD-lossy) vs HF (SAMoD-exact).
ROLLOUT_BACKEND_ARGS=(actor_rollout_ref.rollout.dtype=$ROLLOUT_DTYPE
    actor_rollout_ref.rollout.response_length=$MAX_RESPONSE_LENGTH
    actor_rollout_ref.rollout.temperature=$ROLLOUT_TEMPERATURE
    actor_rollout_ref.rollout.do_sample=$ROLLOUT_DO_SAMPLE
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU
    actor_rollout_ref.rollout.divide_max_pixels_by_image_count=$DIVIDE_MAX_PIXELS_BY_IMAGE_COUNT)
if [[ "$ROLLOUT_BACKEND" == "hf" ]]; then
    ROLLOUT_BACKEND_ARGS+=(actor_rollout_ref.rollout.name=hf
        actor_rollout_ref.rollout.tensor_model_parallel_size=1
        +actor_rollout_ref.rollout.micro_batch_size=$ROLLOUT_HF_MICRO_BATCH_SIZE
        actor_rollout_ref.rollout.calculate_log_probs=True)
elif [[ "$ROLLOUT_BACKEND" == "vllm" ]]; then
    ROLLOUT_BACKEND_ARGS+=(actor_rollout_ref.rollout.name=vllm
        actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE
        actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION
        actor_rollout_ref.rollout.max_num_seqs=$ROLLOUT_MAX_NUM_SEQS
        actor_rollout_ref.rollout.enforce_eager=true
        actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN
        actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN
        +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms=False
        +actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune=False
        actor_rollout_ref.rollout.calculate_log_probs=True
        actor_rollout_ref.rollout.agent.num_workers=$ROLLOUT_AGENT_NUM_WORKERS)
else
    echo "Unknown ROLLOUT_BACKEND=$ROLLOUT_BACKEND (expected 'vllm' or 'hf')" >&2
    exit 1
fi
echo "Rollout backend: $ROLLOUT_BACKEND"

echo "Running: $EXPERIMENT_NAME"
echo "Base model path: $BASE_MODEL_PATH"
echo "Student config file: $STUDENT_CONFIG_FILE"
echo "Student model path: $STUDENT_MODEL_PATH"
echo "Teacher model path: $TEACHER_MODEL_PATH"
echo "Teacher model source: $TEACHER_MODEL_SOURCE"
echo "Teacher regularization: $TEACHER_REGULARIZATION"
echo "Teacher update rate: $TEACHER_UPDATE_RATE"
echo "Enable thinking: $ENABLE_THINKING"
echo "Force direct answer: $FORCE_DIRECT_ANSWER"
echo "Actor model dtype: $ACTOR_MODEL_DTYPE"
echo "Actor FSDP forward dtype: $ACTOR_FORWARD_DTYPE"
echo "Teacher/ref model dtype: $TEACHER_MODEL_DTYPE"
echo "Rollout dtype: $ROLLOUT_DTYPE"
echo "LR: $LR"
echo "Router LR (actor.optim.indexer_lr): $ROUTER_LR"
echo "Freeze vision tower: $FREEZE_VISION_TOWER"
echo "Max pixels: $MAX_PIXELS"
echo "Min pixels: $MIN_PIXELS"
echo "Video fps: ${VIDEO_FPS:-<parquet>}  nframes: ${VIDEO_NFRAMES:-<fps mode>}  frames: ${VIDEO_MIN_FRAMES:-<parquet>}-${VIDEO_MAX_FRAMES:-<parquet>}"
echo "Video per-frame max pixels: ${VIDEO_MAX_PIXELS:-<derived>}  total pixels: ${VIDEO_TOTAL_PIXELS:-<parquet>}  min pixels: ${VIDEO_MIN_PIXELS:-<parquet>}"
echo "Video root: ${VIDEO_ROOT:-<parquet>}"
echo "Seed: $SEED"
echo "Distributed: nnodes=$TRAINER_NNODES n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE node_rank=$NODE_RANK ray_address=${RAY_ADDRESS:-<verl-local>} master=${RAY_HEAD_ADDRESS}"
echo "Grad accum: steps=$GRAD_ACCUM_STEPS dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ local_mini=$LOCAL_MINI micro_per_gpu=${ACTOR_MICRO_BATCH_SIZE_PER_GPU:-<dynamic>}"
echo "Divide max_pixels by image count: $DIVIDE_MAX_PIXELS_BY_IMAGE_COUNT"
echo "Train files (${#TASK_TRAIN_FILES[@]}): $TASK_TRAIN_FILES_CFG"

python3 -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" \
    data.train_files="$TASK_TRAIN_FILES_CFG" \
    data.val_files="[]" \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.truncation=error \
    data.shuffle=True \
    data.seed=$SEED \
    data.trust_remote_code=True \
    data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING \
    data.apply_chat_template_kwargs.force_direct_answer=$FORCE_DIRECT_ANSWER \
    data.return_multi_modal_inputs=True \
    data.image_key=images \
    data.video_root=${VIDEO_ROOT:-null} \
    data.video_fps=${VIDEO_FPS:-null} \
    data.video_nframes=${VIDEO_NFRAMES:-null} \
    data.video_min_frames=${VIDEO_MIN_FRAMES:-null} \
    data.video_max_frames=${VIDEO_MAX_FRAMES:-null} \
    data.video_min_pixels=${VIDEO_MIN_PIXELS:-null} \
    data.video_max_pixels=${VIDEO_MAX_PIXELS:-null} \
    data.video_total_pixels=${VIDEO_TOTAL_PIXELS:-null} \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.dataloader_num_workers=$DATA_DATALOADER_NUM_WORKERS \
    actor_rollout_ref.model.path=$STUDENT_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.indexer_lr=$ROUTER_LR \
    actor_rollout_ref.actor.freeze_vision_tower=$FREEZE_VISION_TOWER \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MIMI_BATCH_SIZE \
    actor_rollout_ref.actor.grad_accum_steps=$GRAD_ACCUM_STEPS \
    ${ACTOR_MICRO_BATCH_SIZE_PER_GPU:+actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_MICRO_BATCH_SIZE_PER_GPU} \
    actor_rollout_ref.actor.data_loader_seed=$SEED \
    actor_rollout_ref.actor.use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$ACTOR_USE_DYNAMIC_BSZ \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN_PER_GPU \
    actor_rollout_ref.actor.fsdp_config.model_dtype=$ACTOR_MODEL_DTYPE \
    actor_rollout_ref.actor.fsdp_config.dtype=$ACTOR_FORWARD_DTYPE \
    actor_rollout_ref.actor.fsdp_config.seed=$SEED \
    actor_rollout_ref.actor.fsdp_config.param_offload=$ACTOR_PARAM_OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$ACTOR_OPTIMIZER_OFFLOAD \
    actor_rollout_ref.actor.checkpoint.model_save_dtype=bfloat16 \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.policy_loss.loss_mode=vopd \
    actor_rollout_ref.actor.calculate_entropy=False \
    actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
    actor_rollout_ref.actor.self_distillation.max_reprompt_len=$MAX_REPROMPT_LEN \
    actor_rollout_ref.actor.self_distillation.is_clip=2.0 \
    actor_rollout_ref.actor.self_distillation.teacher_always_on=True \
    "${TEACHER_MODEL_ARGS[@]}" \
    actor_rollout_ref.actor.self_distillation.teacher_image_key=images \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS \
    actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
    actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    "${ROLLOUT_BACKEND_ARGS[@]}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.model_dtype=$TEACHER_MODEL_DTYPE \
    actor_rollout_ref.ref.fsdp_config.seed=$SEED \
    actor_rollout_ref.ref.fsdp_config.param_offload=$REF_PARAM_OFFLOAD \
    +actor_rollout_ref.ref.model.path=$TEACHER_MODEL_PATH \
    reward_model.enable=False \
    critic.model.path=$STUDENT_MODEL_PATH \
    critic.data_loader_seed=$SEED \
    critic.model.fsdp_config.seed=$SEED \
    reward_model.use_reward_loop=False \
    custom_reward_function.path=null \
    trainer.project_name=$PROJECT_NAME \
    trainer.group_name=$EXPERIMENT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS_PER_NODE \
    trainer.nnodes=$TRAINER_NNODES \
    trainer.save_freq=$TRAINER_SAVE_FREQ \
    trainer.test_freq=-1 \
    trainer.max_actor_ckpt_to_keep=$TRAINER_MAX_ACTOR_CKPT_TO_KEEP \
    trainer.total_epochs=$TRAINER_TOTAL_EPOCHS \
    trainer.val_before_train=False \
    trainer.default_local_dir=$TRAINER_DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir="$TRAINER_ROLLOUT_DATA_DIR" \
    "${CHAT_TEMPLATE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"

touch "$TRAIN_DONE"

# --- Merge Latest Checkpoint ---
TRACKER_FILE="${TRAINER_DEFAULT_LOCAL_DIR}/latest_checkpointed_iteration.txt"
if [[ -f "${TRACKER_FILE}" ]]; then
    LATEST_STEP="$(tr -d '[:space:]' < "${TRACKER_FILE}")"
    MERGE_CKPT_DIR="${TRAINER_DEFAULT_LOCAL_DIR}/global_step_${LATEST_STEP}"
else
    MERGE_CKPT_DIR="$(ls -d "${TRAINER_DEFAULT_LOCAL_DIR}"/global_step_* 2>/dev/null | sort -V | tail -n 1 || true)"
fi
if [[ -z "${MERGE_CKPT_DIR}" || ! -d "${MERGE_CKPT_DIR}" ]]; then
    echo "No checkpoint found under ${TRAINER_DEFAULT_LOCAL_DIR}, skipping merge." >&2
    exit 1
fi
echo "Merging latest checkpoint: ${MERGE_CKPT_DIR}"
bash "${PROJECT_ROOT}/scripts/merge_checkpoint.sh" "${MERGE_CKPT_DIR}"
