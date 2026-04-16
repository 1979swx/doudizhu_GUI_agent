#!/bin/bash
set -x

# 1. 引擎配置 (已写死为 vllm)
ENGINE="vllm"
# export VLLM_ATTENTION_BACKEND=XFORMERS

# 2. 资源分配
num_cpus_per_env_worker=0.1 # 每个环境 worker 分配的 CPU 资源

# 3. 数据与组大小配置 (已根据你的要求同步修改)
# 对应 data.train_batch_size=4
train_data_size=4 
# 对应 data.val_batch_size=4
val_data_size=4 
# 对应 env.rollout.n=2
group_size=2 
mode="mean_norm" # "mean_norm" or "mean_std_norm"

# 4. 数据预处理
# 我们仅使用数据预处理来指定模态和数据大小
python3 -m examples.data_preprocess.prepare \
    --mode 'visual' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size

# 5. PPO 主训练循环 (参数已全部完成定制化替换)
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=gigpo \
    data.train_files=$HOME/data/verl-agent/visual/train.parquet \
    data.val_files=$HOME/data/verl-agent/visual/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=1024 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-VL-2B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=gym_cards/NumberLine-v0 \
    env.seed=0 \
    env.max_steps=10 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name='verl_agent_numberLine' \
    trainer.experiment_name='gigpo_qwen2_vl_2b' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    trainer.val_before_train=False