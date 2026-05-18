# Qwen3.5 Sokoban Smoke Run

本文档记录本仓库内 Qwen3.5 2B Sokoban one-step smoke 训练的环境配置、适配点和复用命令。适配后的代码不依赖 `latest_verl_repo/` 的运行时 import，删除该目录后仍应可运行。

## 入口脚本

一键配置环境并运行 smoke：

```bash
bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh
```

常用参数：

```bash
# 已经配置过环境时，跳过安装，只重新跑 smoke
SKIP_INSTALL=1 RUN_MINIMAL_CHECK=0 bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh

# 只配置/检查环境，不启动训练
RUN_SMOKE=0 bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh

# 删除并重建 verl-agent-bw-exp
RECREATE_ENV=1 bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh

# 指定模型路径
MODEL_PATH=/path/to/Qwen3.5-2B bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh
```

只运行训练脚本：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl-agent-bw-exp
ray stop --force || true
MODEL_PATH=/home/zhangwj/verl/models/Qwen3.5-2B \
  bash study_guide/smoke_run/run_sokoban_qwen3_5_smoke.sh
```

## 环境

新环境名为 `verl-agent-bw-exp`。默认从已经验证可运行官方 Qwen3.5 脚本的 `verl-exp` 环境 clone：

```bash
conda create -y -n verl-agent-bw-exp --clone verl-exp
conda activate verl-agent-bw-exp
```

一键脚本会使用清华 PyPI 源，并设置：

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export MAX_JOBS=32
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu129
```

关键依赖：

- `transformers==5.3.0`：识别 `model_type=qwen3_5` 和 `Qwen3_5ForConditionalGeneration`。
- `vllm==0.18.0`：提供 Qwen3.5 推理侧支持。
- `flash-linear-attention` 与 `causal-conv1d`：用于 Qwen3.5 GDN/linear-attention fast path。
- `flash-attn`：保留现有高性能 attention 依赖。
- `gym==0.26.2`、`gym-sokoban==0.0.6`：Sokoban 环境。
- `setuptools==80.10.2`：`gym-sokoban` 仍 import `pkg_resources`，`setuptools>=82` 会缺少该模块。

## 仓库适配点

本次只做 Qwen3.5 需要的最小适配，没有把本仓库升级到最新官方 verl。

- 新增 `verl/models/transformers/qwen3_5.py`，吸取官方 Qwen3.5 PPO forward、remove-padding、mrope position、vision embedding 和 linear-attention 相关逻辑。
- 更新 `verl/models/transformers/monkey_patch.py`，为 `qwen3_5` / `qwen3_5_moe` 注册 base forward、PPO forward 和 vision position 插值 patch。
- 更新 `verl/workers/fsdp_workers.py`，在 Transformers 5 环境下优先使用 `AutoModelForImageTextToText`，并兼容旧版 `AutoModelForVision2Seq`。
- 更新 `verl/utils/fsdp_utils.py`，允许 `_no_split_modules` 里存在当前 Transformers 版本没有暴露的类名，只要能解析到至少一个可 wrap 类即可继续。
- 更新 `verl/utils/flops_counter.py`，把 `qwen3_5` / `qwen3_5_moe` 映射到 Qwen 系列 FLOPs 估算路径。
- 更新 `verl/utils/vllm_utils.py`，兼容 vLLM 0.18 的 LoRA import 路径，并补充 Qwen3.5 MoE 类型判断。
- 新增 `study_guide/smoke_run/run_sokoban_qwen3_5_smoke.sh`，作为 `run_sokoban_smoke.sh` 的 Qwen3.5 对标 one-step smoke。
- 新增 `scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh`，固化环境创建、依赖安装、最小模型检查和 smoke 运行命令。

## Smoke 配置说明

训练脚本使用 visual data path，并让 Qwen3.5 直接处理 Sokoban 图像：

- `MODEL_PATH=/home/zhangwj/verl/models/Qwen3.5-2B`
- `actor_rollout_ref.rollout.name=vllm`
- `data.image_key=images`
- `data.return_raw_chat=True`
- `env.env_name=Sokoban`
- `env.sokoban.mode=rgb_array`
- `trainer.total_training_steps=1`

为了在旧版 verl-agent 训练栈内稳定通过一轮训练，smoke 默认使用较小 batch：

- `TRAIN_DATA_SIZE=2`
- `VAL_DATA_SIZE=4`
- `GROUP_SIZE=2`
- `NUM_GPUS=2`
- `env.max_steps=5`

Qwen3.5 是混合注意力模型。这里保留 linear-attention fast path：

- 一键脚本检查 `is_flash_linear_attention_available()` 和 `is_causal_conv1d_available()` 必须为真。
- full-attention 层设置 `attn_implementation=sdpa`，避免旧 FSDP 训练路径和 Qwen3.5 full-attention kernel 的兼容风险。
- `actor_rollout_ref.model.use_remove_padding=True` 开启本仓库已有高性能 padding 移除路径。
- `actor_rollout_ref.rollout.enable_chunked_prefill=True` 与 `max_num_batched_tokens=2048` 使用 vLLM chunked-prefill。
- `actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0` 避免 vLLM 0.18 多模态 cache 在本旧训练栈中触发 `Expected a cached item for mm_hash`。

## 验证结果

已在 `verl-agent-bw-exp` 环境中完成：

```bash
bash -n scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh
bash -n study_guide/smoke_run/run_sokoban_qwen3_5_smoke.sh
python -m py_compile \
  verl/models/transformers/qwen3_5.py \
  verl/models/transformers/monkey_patch.py \
  verl/workers/fsdp_workers.py \
  verl/utils/fsdp_utils.py \
  verl/utils/vllm_utils.py \
  verl/utils/flops_counter.py
SKIP_INSTALL=1 RUN_MINIMAL_CHECK=0 \
  bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh
```

成功日志：

```text
logs/qwen3_5-sokoban-smoke-20260515_162449.log
```

关键结果：

```text
Training Progress: 100%|██████████| 1/1
step:1 ... training/global_step:1.000 ... episode/length/mean:5.000
Smoke run completed: logs/qwen3_5-sokoban-smoke-20260515_162449.log
```

## 常见问题

如果遇到 `ModuleNotFoundError: No module named 'pkg_resources'`，说明 `setuptools` 版本过新：

```bash
conda activate verl-agent-bw-exp
python -m pip install setuptools==80.10.2
```

如果遇到 Qwen3.5 模型类型无法识别，优先确认当前环境不是旧的 `verl-agent-bw`：

```bash
conda activate verl-agent-bw-exp
python - <<'PY'
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained("/home/zhangwj/verl/models/Qwen3.5-2B", attn_implementation="sdpa")
print(cfg.model_type)
PY
```

输出应为：

```text
qwen3_5
```

如果 linear-attention fast path 不可用，一键脚本会失败。可先检查：

```bash
conda activate verl-agent-bw-exp
python - <<'PY'
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available
print("flash_linear_attention_available", is_flash_linear_attention_available())
print("causal_conv1d_available", is_causal_conv1d_available())
PY
```
