#!/usr/bin/env bash
set -xeuo pipefail

ENGINE="${ENGINE:-vllm}"
if [[ $# -gt 0 && "$1" != *=* && "$1" != +* ]]; then
    ENGINE="$1"
    shift
fi

MODEL_PATH="${MODEL_PATH:-/home/zhangwj/verl/models/Qwen3.5-2B}"
PROJECT_NAME="${PROJECT_NAME:-verl_agent_sokoban_qwen3_5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_5_2b_sokoban}"
NUM_GPUS="${NUM_GPUS:-2}"

num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
train_data_size="${TRAIN_DATA_SIZE:-16}"
val_data_size="${VAL_DATA_SIZE:-128}"
group_size="${GROUP_SIZE:-8}"

max_prompt_length="${MAX_PROMPT_LENGTH:-1024}"
max_response_length="${MAX_RESPONSE_LENGTH:-512}"
max_env_steps="${MAX_ENV_STEPS:-15}"

ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}"
rollout_tp_size="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.65}"
max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-8192}"
enable_gradient_checkpointing="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
ref_param_offload="${REF_PARAM_OFFLOAD:-True}"

total_epochs="${TOTAL_EPOCHS:-10}"
save_freq="${SAVE_FREQ:--1}"
test_freq="${TEST_FREQ:-5}"
logger="${LOGGER:-['console','wandb']}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

extra_args=()
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    extra_args+=(trainer.total_training_steps="${TOTAL_TRAINING_STEPS}")
fi

python3 -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --train_data_size "${train_data_size}" \
    --val_data_size "${val_data_size}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${HOME}/data/verl-agent/visual/train.parquet" \
    data.val_files="${HOME}/data/verl-agent/visual/test.parquet" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing="${enable_gradient_checkpointing}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${ppo_micro_batch_size_per_gpu}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${rollout_tp_size}" \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${gpu_memory_utilization}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="${max_num_batched_tokens}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${log_prob_micro_batch_size_per_gpu}" \
    actor_rollout_ref.ref.fsdp_config.param_offload="${ref_param_offload}" \
    actor_rollout_ref.actor.use_projection_invalid_penalty=True \
    actor_rollout_ref.actor.projection_invalid_penalty_coef=0.1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=Sokoban \
    env.seed=0 \
    env.max_steps="${max_env_steps}" \
    env.rollout.n="${group_size}" \
    env.sokoban.mode='rgb_array' \
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    trainer.critic_warmup=0 \
    trainer.logger="${logger}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${save_freq}" \
    trainer.resume_mode=auto \
    trainer.test_freq="${test_freq}" \
    trainer.total_epochs="${total_epochs}" \
    trainer.val_before_train=True \
    "${extra_args[@]}" \
    "$@"
