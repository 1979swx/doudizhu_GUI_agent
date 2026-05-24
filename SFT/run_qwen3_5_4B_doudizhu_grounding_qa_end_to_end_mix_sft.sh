#!/usr/bin/env bash
set -xeuo pipefail

GROUNDING_DATA_DIR="${GROUNDING_DATA_DIR:-data_synthesis/doudizhu_grounding_sft}"
QA_DATA_DIR="${QA_DATA_DIR:-data_synthesis/doudizhu_qa_sft}"
END_TO_END_DATA_DIR="${END_TO_END_DATA_DIR:-data_synthesis/doudizhu_end_to_end_sft}"

GROUNDING_TRAIN_FILE="${GROUNDING_TRAIN_FILE:-${GROUNDING_DATA_DIR}/train.parquet}"
GROUNDING_VAL_FILE="${GROUNDING_VAL_FILE:-${GROUNDING_DATA_DIR}/val.parquet}"
QA_TRAIN_FILE="${QA_TRAIN_FILE:-${QA_DATA_DIR}/train.parquet}"
QA_VAL_FILE="${QA_VAL_FILE:-${QA_DATA_DIR}/val.parquet}"
END_TO_END_TRAIN_FILE="${END_TO_END_TRAIN_FILE:-${END_TO_END_DATA_DIR}/train.parquet}"
END_TO_END_VAL_FILE="${END_TO_END_VAL_FILE:-${END_TO_END_DATA_DIR}/val.parquet}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix}"
PROJECT_NAME="${PROJECT_NAME:-verl_agent_sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_doudizhu_grounding_qa_end_to_end_mix_sft}"

NUM_GPUS="${NUM_GPUS:-2}"
FSDP_SHARDING="${FSDP_SHARDING:-zero2}"  # zero2, zero3, no_shard, hybrid

MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((NUM_GPUS * MICRO_BATCH_SIZE_PER_GPU))}"
MAX_LENGTH="${MAX_LENGTH:-2560}"
TRUNCATION="${TRUNCATION:-error}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
SAVE_FREQ="${SAVE_FREQ:-200}"
VAL_FREQ="${VAL_FREQ:-500}"
SAVE_FINAL="${SAVE_FINAL:-True}"
MAX_STEPS="${MAX_STEPS:--1}"

PROMPT_KEY="${PROMPT_KEY:-prompt}"
RESPONSE_KEY="${RESPONSE_KEY:-answer}"
IMAGE_KEY="${IMAGE_KEY:-images}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-False}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
CPU_OFFLOAD="${CPU_OFFLOAD:-False}"
LOGGER="${LOGGER:-wandb}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-1}"

require_file() {
  local file_path="$1"
  local hint="$2"
  if [[ ! -f "${file_path}" ]]; then
    echo "Missing ${file_path}. ${hint}" >&2
    exit 1
  fi
}

append_val_if_exists() {
  local file_path="$1"
  if [[ -f "${file_path}" ]]; then
    VAL_FILES+=("${file_path}")
  else
    echo "No validation parquet found at ${file_path}; skipping it." >&2
  fi
}

require_file "${GROUNDING_TRAIN_FILE}" "Generate grounding data first with data_synthesis/doudizhu_grounding_sft.py."
require_file "${QA_TRAIN_FILE}" "Generate QA data first with data_synthesis/doudizhu_qa_sft.py."
require_file "${END_TO_END_TRAIN_FILE}" "Generate end-to-end data first with data_synthesis/doudizhu_end_to_end_sft.py."

TRAIN_FILES=("${GROUNDING_TRAIN_FILE}" "${QA_TRAIN_FILE}" "${END_TO_END_TRAIN_FILE}")
VAL_FILES=()
append_val_if_exists "${GROUNDING_VAL_FILE}"
append_val_if_exists "${QA_VAL_FILE}"
append_val_if_exists "${END_TO_END_VAL_FILE}"

VAL_ARGS=()
if (( ${#VAL_FILES[@]} > 0 )); then
  VAL_ARGS=(--val-file "${VAL_FILES[@]}")
else
  echo "No validation parquet files found; running train-only mixed SFT." >&2
fi

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  SFT/qwen3_5_vlm_sft_trainer.py \
  --model-path "${MODEL_PATH}" \
  --train-file "${TRAIN_FILES[@]}" \
  "${VAL_ARGS[@]}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-name "${PROJECT_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --prompt-key "${PROMPT_KEY}" \
  --response-key "${RESPONSE_KEY}" \
  --image-key "${IMAGE_KEY}" \
  --max-length "${MAX_LENGTH}" \
  --truncation "${TRUNCATION}" \
  --train-batch-size "${TRAIN_BATCH_SIZE}" \
  --micro-batch-size-per-gpu "${MICRO_BATCH_SIZE_PER_GPU}" \
  --epochs "${EPOCHS}" \
  --max-steps "${MAX_STEPS}" \
  --lr "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --warmup-ratio "${WARMUP_RATIO}" \
  --grad-clip "${GRAD_CLIP}" \
  --save-freq "${SAVE_FREQ}" \
  --val-freq "${VAL_FREQ}" \
  --save-final "${SAVE_FINAL}" \
  --fsdp-sharding "${FSDP_SHARDING}" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --model-dtype "${MODEL_DTYPE}" \
  --trust-remote-code "${TRUST_REMOTE_CODE}" \
  --gradient-checkpointing "${ENABLE_GRADIENT_CHECKPOINTING}" \
  --cpu-offload "${CPU_OFFLOAD}" \
  --num-workers "${NUM_WORKERS}" \
  --logger "${LOGGER}" \
  --seed "${SEED}" \
  "$@"
