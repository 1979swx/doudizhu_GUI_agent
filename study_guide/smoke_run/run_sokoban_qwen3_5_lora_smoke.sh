#!/usr/bin/env bash
set -xeuo pipefail

ENGINE="${ENGINE:-vllm}"
MODEL_PATH="${MODEL_PATH:-/home/zhangwj/verl/models/Qwen3.5-2B}"
PROJECT_NAME="${PROJECT_NAME:-verl_agent_sokoban_qwen3_5}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_qwen3_5_2b_sokoban_lora_smoke}"
NUM_GPUS="${NUM_GPUS:-2}"

num_cpus_per_env_worker="${NUM_CPUS_PER_ENV_WORKER:-0.1}"
train_data_size="${TRAIN_DATA_SIZE:-2}"
val_data_size="${VAL_DATA_SIZE:-2}"
group_size="${GROUP_SIZE:-2}"

lora_rank="${LORA_RANK:-8}"
lora_alpha="${LORA_ALPHA:-16}"
lora_target_modules="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj]}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

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
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank="${lora_rank}" \
    actor_rollout_ref.model.lora_alpha="${lora_alpha}" \
    actor_rollout_ref.model.target_modules="${lora_target_modules}" \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.use_projection_invalid_penalty=True \
    actor_rollout_ref.actor.projection_invalid_penalty_coef=0.2 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    algorithm.use_kl_in_reward=False \
    env.env_name=Sokoban \
    env.seed=0 \
    env.max_steps=3 \
    env.rollout.n="${group_size}" \
    env.sokoban.mode='rgb_array' \
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NUM_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=1 \
    trainer.val_before_train=False "$@"
