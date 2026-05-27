set -xeuo pipefail

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/verl_agent_doudizhu_qwen3_5_4b_with_SFT/grpo_qwen3_5_4b_doudizhu_zh_with_SFT/global_step_75_huggingface_model \
  --env both \
  --output-dir outputs/doudizhu_model_eval/grpo_qwen3_5_4b_doudizhu_zh_with_SFT_global_step_75 \
  --num-episodes 256 \
  --num-envs 64 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/verl_agent_doudizhu_qwen3_5_4b_with_SFT/grpo_qwen3_5_4b_doudizhu_zh_with_SFT/global_step_90_huggingface_model \
  --env both \
  --output-dir outputs/doudizhu_model_eval/grpo_qwen3_5_4b_doudizhu_zh_with_SFT_global_step_90 \
  --num-episodes 256 \
  --num-envs 64 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/verl_agent_doudizhu_qwen3_5_4b_with_SFT/grpo_qwen3_5_4b_doudizhu_zh_with_SFT/global_step_105_huggingface_model \
  --env both \
  --output-dir outputs/doudizhu_model_eval/grpo_qwen3_5_4b_doudizhu_zh_with_SFT_global_step_105 \
  --num-episodes 256 \
  --num-envs 64 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1