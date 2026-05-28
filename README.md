# Dou Dizhu GUI Agentic Post-Training

本仓库是在 `verl-agent` 基础上改造的斗地主 GUI agent 后训练项目。目标是训练一个视觉语言模型从游戏截图中读取牌局，按斗地主规则做决策，并通过归一化 GUI 坐标点击手牌、`出牌`、`不要` 等界面元素完成完整牌局。

项目主线不是文本版斗地主，也不是只预测一条抽象动作，而是端到端的 GUI 控制：

1. 从截图识别当前手牌、对手出牌、剩余牌数和按钮位置。
2. 输出结构化 XML 响应，包括策略说明、语义出牌、GUI 点击、聊天和短期记忆。
3. 由投影器解析 `left_click([x,y],...)`，在内存 GUI 环境里执行点击。
4. 环境把点击还原为斗地主动作，计算投影合法性、点击命中、规则合法性、手牌推进和胜负奖励。
5. 通过合成数据 SFT 与 GRPO 继续训练 Qwen3.5 VLM，使模型从“会点牌”过渡到“会打完整局”。

<p align="center">
  <video src="https://github.com/user-attachments/assets/a2a855bb-d3eb-484c-9abd-8d536eca81a6" controls muted autoplay loop playsinline width="80%"></video>
</p>

## 当前结果

主结果来自 `scripts/eval_doudizhu_model.py` 的在线环境评测。除 `kimi_k26_raw` 外，所有模型都在同一套本地 GUI 斗地主环境中运行；`kimi_k26_raw` 是端到端合成阶段的离线教师轨迹统计，适合作为教师数据源参考，不是严格 leaderboard 对照。

| 阶段 | 评测单位 | 胜率 | 奖励 | 投影合法率 | 规则合法动作率 | fallback 率 | 模型自主出牌推进 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-4B baseline | 128 局 | 0.00% | -0.940 | 66.95% | 5.28% | 94.72% | 1.21% |
| Qwen3.5-9B baseline | 128 局 | 0.00% | -0.945 | 59.49% | 2.85% | 97.15% | 1.25% |
| Kimi K2.6 raw teacher | 485 局离线 raw | 10.52% | -0.538 | 91.87% | 63.32% | 36.65% | 48.91% |
| Grounding-only SFT 400 | 256 局 | 0.00% | -0.868 | 93.81% | 22.76% | 77.24% | 8.54% |
| End-to-end mix SFT 600 | 256 局 | 4.69% | -0.660 | 94.85% | 68.30% | 31.70% | 42.23% |
| GRPO step 60 | 256 局 | 17.58% | -0.331 | 99.21% | 80.42% | 19.58% | 69.77% |
| GRPO step 105 | 256 局 | 16.41% | -0.399 | 93.25% | 68.04% | 31.96% | 56.19% |

对应的 command-to-click grounding 能力：

| 阶段 | target action match | non-pass match | projection valid | click valid |
|---|---:|---:|---:|---:|
| Qwen3.5-4B baseline | 12.04% | 1.22% | 73.16% | 37.50% |
| Qwen3.5-9B baseline | 15.01% | 3.31% | 87.81% | 48.79% |
| Grounding-only SFT 400 | 93.09% | 90.90% | 100.00% | 98.74% |
| End-to-end mix SFT 600 | 95.15% | 93.62% | 100.00% | 99.11% |
| GRPO step 60 | 87.84% | 84.00% | 100.00% | 97.82% |
| GRPO step 105 | 83.81% | 78.69% | 100.00% | 96.72% |

更完整的分析见 [model_report.md](./model_report.md)。简要结论：

- 原始 Qwen3.5-4B / 9B 都不会自然学会 GUI 斗地主，参数变大并不能替代任务内监督。
- Grounding-only SFT 基本解决了“给定动作后点对牌”，但不会自己选择出牌。
- 混合 grounding、QA、端到端轨迹的 SFT 让模型开始形成完整策略。
- GRPO step 60 是当前综合最优 checkpoint，显著降低 fallback 依赖，并提升规则合法动作率和自主出牌推进。
- GRPO 继续训练并非单调变好。step 105 胜率没有可靠增益，同时动作合法性、fallback 和 grounding 指标回落，应基于多指标 early stopping 选 checkpoint。

## 项目能力

- `doudizhu` 完整游戏环境：玩家 0 固定为地主，玩家 1/2 由规则 bot 控制，模型只通过 GUI 点击行动。
- `doudizhu_grounding` 指挥执行环境：规则教师给出语义动作，模型只负责把动作落到截图坐标，用于训练和评测 GUI grounding。
- 自定义渲染器：将 RLCard 风格斗地主状态渲染为 640x480 RGB GUI 图像，并维护手牌、按钮和出牌区域 hitbox。
- XML 输出协议：完整游戏使用 `<plan>`, `<action>`, `<tool_call>`, `<chat>`, `<memory>` 五个标签；grounding 只允许一个 `<tool_call>` 标签。
- 数据合成：支持视觉 QA、command-to-click grounding、Kimi/Moonshot API 教师端到端轨迹，以及 HTML review。
- SFT：独立 Qwen3.5 VLM FSDP SFT trainer，支持一个或多个 parquet 数据源混合训练。
- GRPO：接入 `verl-agent` 的多轮 rollout、视觉输入、环境分组采样和投影非法惩罚。
- 评测：同一脚本同时评估完整牌局与 grounding state，输出 JSONL、CSV、bootstrap CI。
- GUI 演示：Gradio 支持人工游玩、模型旁观、演示模式、离线轨迹回放、指挥模式和观看指挥模式。

<p align="center">
  <img src="./figures/doudizhu%20agent%20run.png" alt="Dou Dizhu agent run" width="80%">
</p>

## 代码结构

| 路径 | 用途 |
|---|---|
| `agent_system/environments/env_package/doudizhu/` | 完整 GUI 斗地主环境、渲染器、投影器和本地斗地主核心逻辑。 |
| `agent_system/environments/env_package/doudizhu_grounding/` | command-to-click grounding 环境。 |
| `agent_system/environments/prompts/doudizhu.py` | 完整游戏 prompt，含英文、中文和不同策略版本。 |
| `agent_system/environments/prompts/doudizhu_grounding.py` | grounding prompt。 |
| `data_synthesis/doudizhu_qa_sft.py` | 合成视觉规则 QA 数据。 |
| `data_synthesis/doudizhu_grounding_sft.py` | 合成 GUI 点击 grounding 数据。 |
| `data_synthesis/doudizhu_end_to_end_sft.py` | 用 API/Mock 教师采集端到端轨迹并过滤为 SFT 数据。 |
| `SFT/qwen3_5_vlm_sft_trainer.py` | Qwen3.5 VLM FSDP SFT 训练器。 |
| `SFT/run_qwen3_5_4B_doudizhu_*.sh` | 斗地主 SFT 配方。 |
| `examples/grpo_trainer/run_doudizhu_qwen3_5.sh` | 完整斗地主 GRPO 训练入口。 |
| `examples/grpo_trainer/run_doudizhu_grounding_qwen3_5.sh` | grounding GRPO 训练入口。 |
| `scripts/eval_doudizhu_model.py` | 完整游戏与 grounding 统一在线评测。 |
| `scripts/eval_kimi_doudizhu_raw.py` | Kimi raw 教师轨迹离线指标统计。 |
| `scripts/human_play_doudizhu_web.py` | Gradio 调试和演示界面。 |
| `tests/environments/test_doudizhu.py` | 环境、投影、奖励、prompt 和 grounding group 行为测试。 |

## 环境

本机主训练环境为 `verl-agent-bw-exp`。进入仓库根目录后：

```bash
conda activate verl-agent-bw-exp
pip install -e .
```

这份项目使用 Qwen3.5 VLM、vLLM、FlashAttention、Ray、Gradio、Pandas/Parquet 等依赖。不要直接把根目录 `requirements.txt` 当作 Qwen3.5 环境复现说明；其中仍保留了上游 verl/verl-agent 的旧约束。Blackwell / Qwen3.5 相关环境说明见：

- [study_guide/setup_env.md](./study_guide/setup_env.md)
- [qwen3.5_guide.md](./qwen3.5_guide.md)

建议先跑最小测试：

```bash
pytest -q tests/environments/test_doudizhu.py
pytest -q tests/data_synthesis/test_doudizhu_qa_sft.py tests/data_synthesis/test_doudizhu_end_to_end_sft.py
```

## 交互演示

使用当前最佳 GRPO checkpoint 启动中文 Gradio：

```bash
conda activate verl-agent-bw-exp

python scripts/human_play_doudizhu_web.py \
  --model-backend local \
  --model-path checkpoints/verl_agent_doudizhu_qwen3_5_4b_with_SFT/grpo_qwen3_5_4b_doudizhu_zh_with_SFT/global_step_60_huggingface_model \
  --chinese-mode \
  --server-name 127.0.0.1 \
  --server-port 7860
```

API 教师或外部 VLM 可以走 OpenAI-compatible 接口，例如 Kimi/Moonshot：

```bash
export MOONSHOT_API_KEY=...

python scripts/human_play_doudizhu_web.py \
  --model-backend api \
  --api-base-url https://api.moonshot.cn/v1 \
  --api-model kimi-k2.6 \
  --api-key-env MOONSHOT_API_KEY \
  --api-thinking disabled \
  --chinese-mode
```

Web UI 包含六个模式：人工游玩、模型旁观、演示模式、离线演示模式、指挥模式、观看指挥模式。指挥模式用于验证模型是否能把类似 `3 3`、`10 J Q K A`、`不要` 的语义动作准确转为 GUI 点击。

## 数据合成

当前仓库包含三类 SFT 数据：

- `data_synthesis/doudizhu_qa_sft/`：视觉规则 QA。当前快照为 4000/400/400 train/val/test。
- `data_synthesis/doudizhu_grounding_sft/`：command-to-click grounding。当前快照为 15000/1500/1500 train/val/test。
- `data_synthesis/doudizhu_end_to_end_sft/`：Kimi/Moonshot 教师端到端轨迹过滤后的 SFT 数据。当前 raw 统计含 485 局，过滤得到 882 个可训练 step。

重新生成 QA 数据：

```bash
conda activate verl-agent-bw-exp

python data_synthesis/doudizhu_qa_sft.py \
  --output-dir data_synthesis/doudizhu_qa_sft \
  --train-samples 4000 \
  --val-samples 400 \
  --test-samples 400 \
  --language zh
```

重新生成 grounding 数据：

```bash
python data_synthesis/doudizhu_grounding_sft.py \
  --output-dir data_synthesis/doudizhu_grounding_sft \
  --train-samples 15000 \
  --val-samples 1500 \
  --test-samples 1500 \
  --language zh \
  --jitter 0.20
```

采集端到端教师轨迹：

```bash
export MOONSHOT_API_KEY=...

python data_synthesis/doudizhu_end_to_end_sft.py \
  --output-dir data_synthesis/doudizhu_end_to_end_sft \
  --model-backend api \
  --api-base-url https://api.moonshot.cn/v1 \
  --api-model kimi-k2.6 \
  --api-key-env MOONSHOT_API_KEY \
  --api-thinking disabled \
  --temperature 0.6 \
  --max-new-tokens 1536 \
  --terminal-max-hand 2 \
  --num-workers 8 \
  --request-concurrency 8 \
  --train-samples 1000 \
  --val-samples 0 \
  --test-samples 0
```

只用已有 raw 轨迹重建 parquet：

```bash
python data_synthesis/doudizhu_end_to_end_sft.py \
  --filter-only \
  --output-dir data_synthesis/doudizhu_end_to_end_sft
```

生成 HTML review：

```bash
python data_synthesis/visualize_doudizhu_grounding_sft.py \
  --input data_synthesis/doudizhu_grounding_sft/train.parquet \
  --output data_synthesis/doudizhu_grounding_sft/review.html \
  --num-samples 40

python data_synthesis/visualize_doudizhu_end_to_end.py \
  --input data_synthesis/doudizhu_end_to_end_sft/train.parquet \
  --output data_synthesis/doudizhu_end_to_end_sft/review.html \
  --num-samples 40
```

更多细节见 [data_synthesis/README.md](./data_synthesis/README.md)。

## SFT 训练

推荐主线是先获得 grounding/QA/end-to-end 混合 SFT checkpoint，再从该 checkpoint 启动 GRPO。

```bash
conda activate verl-agent-bw-exp

NUM_GPUS=2 \
MODEL_PATH=Qwen/Qwen3.5-4B \
GROUNDING_DATA_DIR=data_synthesis/doudizhu_grounding_sft \
QA_DATA_DIR=data_synthesis/doudizhu_qa_sft \
END_TO_END_DATA_DIR=data_synthesis/doudizhu_end_to_end_sft \
bash SFT/run_qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix_sft.sh
```

默认输出：

```text
checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix/
```

常用中间实验：

```bash
# 只训练 command-to-click executor
NUM_GPUS=2 MODEL_PATH=Qwen/Qwen3.5-4B \
bash SFT/run_qwen3_5_4B_doudizhu_grounding_sft.sh

# 只训练视觉规则 QA
NUM_GPUS=2 MODEL_PATH=Qwen/Qwen3.5-4B \
bash SFT/run_qwen3_5_4B_doudizhu_qa_sft.sh

# 只训练端到端教师轨迹
NUM_GPUS=2 MODEL_PATH=Qwen/Qwen3.5-4B \
bash SFT/run_qwen3_5_4B_doudizhu_end_to_end_sft.sh
```

SFT 数据行格式：

- `prompt`：chat message list 或用户字符串，包含 `<image>`。
- `images`：`[{"bytes": PNG_BYTES}]`。
- `answer`：模型目标输出。
- grounding 只输出 `<tool_call>left_click(...)</tool_call>`。
- 完整游戏输出五标签格式。

## GRPO 训练

从混合 SFT checkpoint 启动完整斗地主 GRPO：

```bash
conda activate verl-agent-bw-exp

NUM_GPUS=2 \
MODEL_PATH=checkpoints/sft/qwen3_5_4B_doudizhu_grounding_qa_end_to_end_mix/global_step_600 \
PROJECT_NAME=verl_agent_doudizhu_qwen3_5_4b_with_SFT \
EXPERIMENT_NAME=grpo_qwen3_5_4b_doudizhu_zh_with_SFT \
GROUP_SIZE=16 \
TRAIN_DATA_SIZE=8 \
VAL_DATA_SIZE=128 \
bash examples/grpo_trainer/run_doudizhu_qwen3_5.sh
```

训练脚本会创建视觉 dummy parquet，并把真实状态交给环境 `reset()` / `step()`。关键配置：

- `env.env_name=doudizhu`
- `env.doudizhu.language=zh`
- `env.doudizhu.chinese_mode=True`
- `env.rollout.n=GROUP_SIZE`
- `actor_rollout_ref.rollout.name=vllm`
- `actor_rollout_ref.actor.use_projection_invalid_penalty=True`

grounding GRPO：

```bash
NUM_GPUS=2 \
MODEL_PATH=Qwen/Qwen3.5-4B \
GROUP_SIZE=16 \
bash examples/grpo_trainer/run_doudizhu_grounding_qwen3_5.sh
```

奖励由环境 info 写回训练：

- `projection_valid`：响应是否满足 XML 与 `left_click(...)` 解析协议。
- `click_valid_ratio`：坐标是否命中手牌或按钮 hitbox。
- `rule_action_valid`：点击还原出的动作是否是当前规则合法动作。
- `hand_depletion`：非 fallback 动作使玩家 0 减少的手牌数。
- `win/loss`：终局胜负奖励。

完整游戏环境在非法动作时会执行 fallback，以保证 episode 可继续，但 fallback 会被指标和奖励记录下来。当前核心目标之一就是降低 fallback 依赖。

## Checkpoint 转换

评测和 vLLM 推理通常需要 Hugging Face 格式权重。若训练只保存了 FSDP actor shard，先合并：

```bash
python scripts/model_merger.py merge \
  --backend fsdp \
  --local_dir checkpoints/verl_agent_doudizhu_qwen3_5_4b_with_SFT/grpo_qwen3_5_4b_doudizhu_zh_with_SFT/global_step_60/actor \
  --target_dir checkpoints/verl_agent_doudizhu_qwen3_5_4b_with_SFT/grpo_qwen3_5_4b_doudizhu_zh_with_SFT/global_step_60_huggingface_model
```

`scripts/human_play_doudizhu_web.py` 对本地 Transformers 演示有自动合并逻辑；`scripts/eval_doudizhu_model.py` 走 vLLM，建议显式传入已经合并好的目录。

## 评测

评估完整游戏与 grounding：

```bash
conda activate verl-agent-bw-exp

python scripts/eval_doudizhu_model.py \
  --model-path checkpoints/verl_agent_doudizhu_qwen3_5_4b_with_SFT/grpo_qwen3_5_4b_doudizhu_zh_with_SFT/global_step_60_huggingface_model \
  --env both \
  --output-dir outputs/doudizhu_model_eval/grpo_qwen3_5_4b_doudizhu_zh_with_SFT_global_step_60 \
  --num-episodes 256 \
  --num-envs 64 \
  --grounding-samples-per-state 8 \
  --max-response-length 1536 \
  --max-env-steps 30 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 2 \
  --tensor-model-parallel-size 1
```

输出文件：

- `samples.jsonl`：逐 step 原始响应、投影、环境 info。
- `episodes.jsonl`：完整游戏 episode 级指标。
- `grounding_state_metrics.csv`：grounding canonical state 聚合指标。
- `episode_metrics.csv`：episode/trajectory 指标。
- `summary.csv` / `summary.json`：均值、标准差和 bootstrap 95% CI。

评估 Kimi raw 教师轨迹：

```bash
python scripts/eval_kimi_doudizhu_raw.py \
  --raw-dir data_synthesis/doudizhu_end_to_end_sft/raw \
  --output-dir outputs/doudizhu_model_eval/kimi_k26_raw
```

批量评测示例见 [scripts/eval_doudizhu_models.sh](./scripts/eval_doudizhu_models.sh)。

## 模型输出协议

完整游戏响应必须严格包含五个非空 XML 标签：

```xml
<plan>根据截图简要分析当前局势。</plan>
<action>[3, 3]</action>
<tool_call>left_click([55,850],[100,860],[430,755])</tool_call>
<chat>简短聊天。</chat>
<memory>给下一回合使用的简短记忆。</memory>
```

约束：

- `<action>` 只能是 `[pass]` 或牌面列表，例如 `[3]`、`[3, 3]`、`[10, J, Q, K, A]`、`[BJ, RJ]`。
- `<tool_call>` 只能是一个 `left_click([x,y],...)` 调用。
- 坐标范围为 0 到 1000，`[0,0]` 是左上角，`[1000,1000]` 是右下角。
- 出牌动作最后一次点击必须是 `出牌` / `PLAY` 按钮；过牌动作点击 `不要` / `PASS`。
- 不要在 `<memory>` 中写入 `<image>` 或 Qwen 视觉占位符，环境管理器会清理这些 token。

grounding 响应只允许：

```xml
<tool_call>left_click([55,850],[100,860],[430,755])</tool_call>
```

## 开发注意事项

- `doudizhu` 和 `doudizhu_grounding` 都支持 `env.rollout.n` 分组采样。完整游戏会对组内环境重复同一 seed；grounding 会让组内样本共享同一个 canonical state，并由教师动作推进环境。
- GRPO 类算法不要通过 `actor_rollout_ref.rollout.n` 控制组大小，本项目使用 `env.rollout.n`。
- `doudizhu_grounding` 的环境推进由教师动作完成，模型动作只用于打分；这保证同一状态下多个样本可公平比较。
- 完整游戏里 fallback 是环境兜底，不是模型成功。报告结果时应同时看 `fallback_rate`、`rule_action_valid_rate` 和 `model_hand_depletion_rate`。
- `Kimi K2.6 raw` 是离线 raw 轨迹统计，与本地 checkpoint 的在线评测协议不同，不能当作严格同 seed 对照。
- 当前最佳模型仍不是强斗地主 AI。项目价值主要在 GUI agent 后训练闭环、指标拆解和从 SFT 到 GRPO 的能力跃迁。

## 上游与许可

本项目继承并改造了 [verl-agent](https://github.com/langfengQ/verl-agent) 和 [veRL](https://github.com/volcengine/verl) 的 agentic RL 训练框架。斗地主规则核心与动作空间参考 RLCard，相关许可见 [agent_system/environments/env_package/doudizhu/RLCARD_LICENSE.md](./agent_system/environments/env_package/doudizhu/RLCARD_LICENSE.md)。仓库整体许可见 [LICENSE](./LICENSE)。
