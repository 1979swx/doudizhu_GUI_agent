set -xeuo pipefail


# python scripts/eval_doudizhu_model.py \
#   --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_mix/global_step_300 \
#   --env both \
#   --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_mix_SFT_300_steps \
#   --num-episodes 256 \
#   --num-envs 32 \
#   --grounding-samples-per-state 8 \
#   --max-response-length 1536 \
#   --max-env-steps 30 \
#   --gpu-memory-utilization 0.9 \
#   --data-parallel-size 2 \
#   --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_mix/global_step_600 \
  --env both \
  --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_mix_SFT_600_steps \
  --num-episodes 256 \
  --num-envs 32 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_mix/global_step_900 \
  --env both \
  --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_mix_SFT_900_steps \
  --num-episodes 256 \
  --num-envs 32 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1