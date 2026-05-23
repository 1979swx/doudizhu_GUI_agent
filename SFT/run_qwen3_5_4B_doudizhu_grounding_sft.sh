#!/usr/bin/env bash
set -xeuo pipefail

DATA_DIR="${DATA_DIR:-data_synthesis/doudizhu_grounding_sft}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${VAL_FILE:-${DATA_DIR}/val.parquet}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft/qwen3_5_4B_doudizhu_grounding}"
PROJECT_NAME="${PROJECT_NAME:-verl_agent_sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_doudizhu_grounding_sft}"

NUM_GPUS="${NUM_GPUS:-2}"
FSDP_SHARDING="${FSDP_SHARDING:-zero2}"  # zero2, zero3, no_shard, hybrid

MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-16}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$((NUM_GPUS * MICRO_BATCH_SIZE_PER_GPU))}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
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
LOGGER="${LOGGER:-console}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

torchrun --standalone --nnodes=1 --nproc-per-node="${NUM_GPUS}" \
  SFT/qwen3_5_vlm_sft_trainer.py \
  --model-path "${MODEL_PATH}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --output-dir "${OUTPUT_DIR}" \
  --project-name "${PROJECT_NAME}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --prompt-key "${PROMPT_KEY}" \
  --response-key "${RESPONSE_KEY}" \
  --image-key "${IMAGE_KEY}" \
  --max-length "${MAX_LENGTH}" \
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
  --logger "${LOGGER}" \
  "$@"
