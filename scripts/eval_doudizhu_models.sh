set -xeuo pipefail


python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix/global_step_100 \
  --env both \
  --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_end_to_end_mix_SFT_100_steps \
  --num-episodes 256 \
  --num-envs 32 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix/global_step_200 \
  --env both \
  --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_end_to_end_mix_SFT_200_steps \
  --num-episodes 256 \
  --num-envs 32 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix/global_step_300 \
  --env both \
  --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_end_to_end_mix_SFT_300_steps \
  --num-episodes 256 \
  --num-envs 32 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix/global_step_400 \
  --env both \
  --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_end_to_end_mix_SFT_400_steps \
  --num-episodes 256 \
  --num-envs 32 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix/global_step_500 \
  --env both \
  --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_end_to_end_mix_SFT_500_steps \
  --num-episodes 256 \
  --num-envs 32 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix/global_step_600 \
  --env both \
  --output-dir outputs/doudizhu_model_eval/qwen35_4b_doudizhu_grounding_qa_end_to_end_mix_SFT_600_steps \
  --num-episodes 256 \
  --num-envs 32 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1