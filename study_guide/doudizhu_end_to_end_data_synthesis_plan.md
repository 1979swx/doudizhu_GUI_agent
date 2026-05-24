# 斗地主环境端到端数据合成方案

本文档描述一套面向 `verl-agent` 仓库中 `doudizhu` GUI 环境的端到端数据合成方案。该数据集的目标是调用商业级多模态推理模型 API，在真实斗地主环境中完成完整 episode 行动，并从最终获胜 episode 中筛选高质量 step-level 样本，用作后续斗地主 RL 训练前的冷启动 SFT 数据。

这套数据与现有两套合成数据互补：

1. `data_synthesis/doudizhu_qa_sft.py`：训练视觉读牌、规则理解、合法动作判断等基础能力。
2. `data_synthesis/doudizhu_grounding_sft.py`：训练“给定语义动作，点击正确牌和按钮”的 grounding 能力。
3. 本方案：训练“从截图和上一轮记忆出发，自主分析、选择动作、输出 GUI 点击，并最终服务于胜负目标”的端到端行动能力。

三套数据后续会混合用于 SFT，使模型在进入 RL 前已经具备较稳定的读牌、规则、点击和策略格式基础。

## 1. 目标与非目标

### 1.1 目标

本数据集主要训练以下能力：

1. 在 `doudizhu` 主环境中根据当前截图、历史出牌和上一轮 memory 自主选择动作。
2. 学习主环境的完整五标签输出格式：`<plan>`、`<action>`、`<tool_call>`、`<chat>`、`<memory>`。
3. 学习商业模型在斗地主局面中的短程策略分析、风险判断和动作选择习惯。
4. 学习动作选择和 GUI 点击的一致输出，即 `<action>` 中的语义动作、`<tool_call>` 中的点击、环境实际提交动作三者一致。
5. 为 RL 冷启动提供真实轨迹分布中的高质量 step-level 样本，减少早期 RL 的无效格式、无效点击和非法动作。
6. 在数据采集阶段统计商业模型在当前环境、prompt 和参数设置下的真实性能，用作后续训练模型对比基线。

### 1.2 非目标

本数据集不承担以下目标：

1. 不替代 QA 数据。基础读牌、牌型识别、合法动作枚举仍由 QA 数据更系统地覆盖。
2. 不替代 grounding 数据。给定目标动作的纯点击能力仍由 grounding 数据更高效地覆盖。
3. 不保证每个获胜 episode 的所有模型控制步都有效。采样成本和商业模型能力决定了这点不现实。本方案采用 episode 级胜利约束与 step 级严格过滤相结合。
4. 不强行用规则模板重写商业模型的 `<plan>`。本数据集的价值之一就是蒸馏商业模型相对高质量的自然推理过程。
5. 不把商业模型视为最优策略教师。它只是一个强冷启动教师，后续仍需要 RL 超越它。

## 2. 核心设计结论

本方案采用“两阶段数据资产”设计：

1. 先保存未过滤的 raw rollout 数据，用于性能统计、错误归因、成本核算和复现实验。
2. 再从 raw rollout 中派生 filtered SFT parquet，只保留满足训练条件的 step-level 样本。

推荐默认保留条件为：

1. 所属 episode 最终由玩家 0 获胜。
2. 当前 step `projection_valid == 1`。
3. 当前 step `click_valid_ratio == 1.0`。
4. 当前 step `rule_action_valid == 1.0`。
5. 当前 step `<action>` 解析并 canonicalize 后等于环境实际提交的 `game_action`。
6. 当前 step response 的目标模型 token 数不超过阈值，默认建议 `<= 1024`，并额外统计 `<= 512` 子集。

默认不要求整个 episode 所有模型控制步都有效。原因是：

1. 商业模型并非完美，若要求全轨迹每步有效，会显著降低可接受数据量。
2. 主环境 fallback 大多执行保守动作，非首发轮通常是 `pass`，难以单独“靠 fallback 获胜”。
3. 本数据集最终写入的是 step-level SFT 样本，而不是把整条轨迹作为一个多轮监督样本。
4. 更合理的做法是记录 `prior_invalid_steps`、`prior_fallback_steps` 等元数据，后续可切分高置信子集做 ablation。

## 3. 与现有环境和训练链路的对齐

### 3.1 环境对齐

本数据集面向主环境 `DoudizhuSingleEnv`，而不是 `DoudizhuGroundingSingleEnv`。

主环境中：

1. `agent_system/environments/prompts/doudizhu.py` 定义端到端 prompt。
2. `agent_system/environments/env_package/doudizhu/projection.py` 解析五标签输出。
3. `DoudizhuSingleEnv.step()` 根据 `clicks` 投影得到 `candidate_action`。
4. 若 `candidate_action in legal_actions`，环境执行该动作，并给出 `rule_action_valid=1.0`。
5. 若不合法，则环境执行 fallback，并给出 `rule_action_valid=0.0` 和 `fallback_used=True`。

因此，本数据集的 verifier 应复用主环境真实 step 结果，而不是另写规则裁判。

### 3.2 SFT 管线对齐

真实 SFT 管线位于 `SFT/`：

1. `SFT/qwen3_5_vlm_sft_trainer.py` 会读取 parquet 中的 `prompt`、`images`、`answer`。
2. `SyntheticSFTDataset` 会把 `prompt` 规范为 chat messages，并把 `images` 送入 processor。
3. `answer` 作为 assistant response 加入 full messages，labels 只监督 assistant 部分。
4. `SFT/run_qwen3_5_4B_doudizhu_grounding_qa_mix_sft.sh` 已支持多个 train/val parquet concat。

因此，本方案输出 parquet 应沿用现有字段：

1. `data_source`
2. `prompt`
3. `question`
4. `images`
5. `answer`
6. `ability`
7. `reward_model`
8. `extra_info`

新增端到端数据可以直接加入现有 SFT 混合脚本，或新增一个三路混合脚本。

## 4. 商业模型与 API 调用

### 4.1 默认教师模型

默认使用 Kimi/Moonshot 的 `kimi-k2.6` 或后续同级别多模态推理模型。仓库中 `scripts/human_play_doudizhu_web.py` 已验证 OpenAI-compatible chat/completions 调用方式：

1. `--api-base-url https://api.moonshot.cn/v1`
2. `--api-model kimi-k2.6`
3. `--api-key-env MOONSHOT_API_KEY`
4. 当前截图以 PNG data URL 形式传入。
5. response 仍遵循本仓库 XML 标签输出契约。

相关官方文档：

1. Kimi API 概述：`https://platform.kimi.com/docs/api/overview`
2. Kimi K2.6 模型参数与 `thinking` 参数：`https://platform.kimi.com/docs/api/models-overview`
3. Kimi thinking 使用说明：`https://platform.kimi.com/docs/guide/use-kimi-k2-thinking-model`
4. Kimi API 充值与限速：`https://platform.kimi.com/docs/pricing/limits`

端到端采集脚本应复用同类 API adapter，但需要比调试 UI 更适合批量采集：

1. 支持并发请求。
2. 支持请求超时、重试、限速、退避和断点续跑。
3. 记录原始 request/response 元数据。
4. 记录 API token、latency、finish_reason、错误类型和成本估算。
5. 显式禁用 Kimi thinking 模式，避免 `reasoning_content` 或长思考过程影响 XML 契约输出。

### 4.2 推荐生成参数

第一版建议使用与 Kimi K2.6 官方建议和现有调试脚本一致的稳定配置：

1. `max_tokens`: 1024 或略高于过滤阈值，避免过早截断导致格式坏。
2. `temperature`: 固定为 `1.0`。Kimi 当前旗舰模型对 temperature 有特定默认/约束，非默认值可能降低质量或增加格式错乱风险。
3. `top_p`: 保持默认或设为 `0.95`。若后续观察到与 `temperature=1.0` 组合产生格式波动，可优先回退为 API 默认值。
4. `thinking`: 固定为禁用，即请求体中设置 `thinking={"type":"disabled"}`。Kimi K2.6 支持思考与非思考模式，官方文档说明 K2.6 默认启用 thinking；但本任务的核心是稳定输出五个 XML 标签，thinking 会引入 `reasoning_content`、额外 token 消耗和更复杂的响应结构，因此不适合作为数据采集默认模式。
5. `timeout`: 固定为 90 秒。Kimi 多模态请求可能受图片输入、队列和网络波动影响，90 秒比调试 UI 的 60 秒更适合批量采集。

所有生成参数必须写入 metadata，保证后续评估和对比可复现。

### 4.3 API 限速与调度

Kimi API 基于账户充值等级限制并发、RPM、TPM 和 TPD。第一版工程设计暂按 Tier2 额度考虑：

| 等级 | 并发 | RPM | TPM | TPD |
| --- | ---: | ---: | ---: | ---: |
| Tier2 | 100 | 500 | 3,000,000 | Unlimited |

采集脚本不能只靠 worker 数量粗暴并发，而应有全局 rate limiter：

1. `max_inflight_requests <= 100`，建议默认先设为 32 或 64，给网络抖动和服务端临时限流留余量。
2. `max_requests_per_minute <= 500`，建议默认 450。
3. `max_tokens_per_minute <= 3_000_000`，需要按输入图片估算 token、输出 `max_tokens` 和 API 返回 usage 动态更新。
4. 遇到 HTTP 429 时指数退避，并把 episode 标记为可重试，不应直接写入失败样本。
5. 每个 API key 维护独立限速桶；如后续使用多个 key，必须避免混淆统计和成本。
6. metadata 中记录实际账户限速配置、脚本限速配置、429 次数、退避总时长和每分钟吞吐。

由于官方说明在集群负载高时可能临时调整限流，脚本应允许通过命令行覆盖这些默认值，而不是把 Tier2 常量写死。

## 5. Prompt 与输出契约

### 5.1 Prompt 来源

默认使用主环境现有中文 prompt：

1. `DOUDIZHU_VISUAL_TEMPLATE_ZH`
2. `previous_memory` 由上一轮 response 中 `<memory>` 更新。
3. 若 memory 与截图冲突，prompt 中应明确要求以截图为准。

为了配合新的 `<action>` 契约，需要在 prompt 中收紧 action 说明。`<tool_call>`、`<chat>`、`<memory>` 可以沿用现有设计。

### 5.2 五标签输出格式

模型每轮必须输出五个标签：

```xml
<plan>简要分析当前截图、上一轮出牌、手牌结构和本轮策略。</plan>
<action>[3, 3]</action>
<tool_call>left_click([55,850],[100,860],[430,755])</tool_call>
<chat>简短聊天内容。</chat>
<memory>给下一轮使用的简短记忆。</memory>
```

字段含义：

1. `<plan>`：商业模型的自然策略分析。应简洁、聚焦当前局面，不要求规则模板化。
2. `<action>`：本轮语义动作，采用 list of card 契约。
3. `<tool_call>`：一个 `left_click(...)` 调用，包含本轮所有点击。
4. `<chat>`：短聊天文本。
5. `<memory>`：下一轮 prompt 使用的短记忆。

### 5.3 `<action>` 契约

建议把 `<action>` 从自然语言动作改为 list of card：

```xml
<action>[pass]</action>
<action>[3]</action>
<action>[3, 3]</action>
<action>[9, 10, J, Q, K]</action>
<action>[BJ, RJ]</action>
```

规范如下：

1. 过牌固定输出 `[pass]`，不要在 `<action>` 中输出 `[不要]`、`[过]`、`pass` 以外别名。
2. 普通牌使用 `3,4,5,6,7,8,9,10,J,Q,K,A,2,BJ,RJ`。
3. 内部环境表示仍为 `T/B/R`，解析器负责把 `10 -> T`、`BJ -> B`、`RJ -> R`。
4. parser 可宽松接受引号，例如 `["3", "3"]`，但数据写入时应规范化成无引号形式或另存 canonical 字段。
5. `<action>` 里不允许坐标、解释文本、代码块或牌型名称。

### 5.4 `<action>` canonicalization

为了和环境动作对齐，需要实现一个 action canonicalizer。设计要求：

1. 抽取 `<action>` 内容。
2. 检查是否是 `[...]` 形式。
3. 按逗号切分元素，strip 空白和可选引号。
4. 将 `pass/不要/过牌/不出` 等别名映射到 `pass`，但写入训练答案前建议统一为 `[pass]`。
5. 将 `10/BJ/RJ` 映射为 `T/B/R`。
6. 检查所有牌值合法。
7. 对非 pass 动作输出内部 action string。

比较策略建议采用两级：

1. `action_raw_from_tag == info["game_action"]` 作为默认硬过滤。
2. 对附带牌型和顺序歧义，可额外记录 multiset match，例如 `Counter(action_raw_from_tag) == Counter(info["game_action"])`。第一版建议仍用 raw string match，避免把语义顺序不一致的样本写入高置信集。

注意：环境实际 `game_action` 来自点击选中的手牌索引顺序。对于三带一、飞机带牌、四带二等动作，环境 raw string 可能与模型自然书写顺序不同。若 raw string match 导致过度拒绝，可在 metadata 中分析 rejected action-multiset-match 样本，再决定是否放宽。

### 5.5 Response 长度

长度过滤必须基于目标 SFT/RL 模型 tokenizer，而不是字符数。

建议记录：

1. `response_chars`
2. `response_tokens_teacher_api`，如果 API 返回 usage。
3. `response_tokens_target_model`
4. `full_sequence_tokens_target_model`

默认硬过滤：

1. `response_tokens_target_model <= 1024`

建议额外打标：

1. `response_tokens_target_model <= 512`
2. `full_sequence_tokens_target_model <= SFT_MAX_LENGTH`

后续 RL 计划使用 `max_prompt_length=1536`、`max_response_length=1024`，因此端到端 SFT 默认 `SFT_MAX_LENGTH` 应设置为 `2560`。过滤时需要同时记录 response token 和 full sequence token，避免 response 合格但图文 prompt 加 response 后超过 SFT/RL 预算。

## 6. 数据采集与处理总体架构

### 6.1 模块划分

建议新增脚本：

```text
data_synthesis/doudizhu_end_to_end_sft.py
```

可按以下模块组织：

| 模块 | 职责 |
| --- | --- |
| `EnvRunner` | 管理 `DoudizhuSingleEnv` reset/step、seed、episode 状态、memory |
| `ApiClient` | 调用 Kimi/OpenAI-compatible API，处理图片、参数、重试、限速 |
| `ResponseProjector` | 调用 `doudizhu_projection` 解析五标签和点击 |
| `ActionCanonicalizer` | 解析 `<action>` list of card，并与 `game_action` 对齐 |
| `StepRecorder` | 写 raw step JSONL，包含 prompt、response、info、API 元数据 |
| `EpisodeRecorder` | 写 episode 级 JSONL，包含胜负、长度、累计指标 |
| `StatsAggregator` | 统计未过滤性能和过滤 yield |
| `SftFilter` | 从 raw rollout 中筛选 accepted step |
| `ParquetWriter` | 写 `train.parquet`、`val.parquet`、`test.parquet` |
| `MetadataWriter` | 写 `metadata.json`、统计摘要和配置 |
| `ReviewExporter` | 可选生成 HTML review 页面 |

第一版可以写在一个脚本中，但逻辑上应保持这些边界，避免后续维护困难。

### 6.2 数据流

完整数据流如下：

```text
episode seed
  -> env.reset(seed)
  -> render observation image
  -> build prompt(previous_memory)
  -> call commercial VLM API
  -> raw response
  -> projection parser
  -> env.step(projected action)
  -> update memory
  -> record raw step
  -> repeat until done/max_steps/bot limit/API abort
  -> record raw episode
  -> after raw collection, filter winning episode steps
  -> write SFT parquet and metadata
```

强烈建议 raw rollout 和 filtered parquet 分离：

1. raw rollout 是不可变审计资产。
2. filtered parquet 是可多次重建的数据产品。
3. 修改过滤条件时不需要重新花 API 成本采样。
4. 可从同一 raw 数据派生多个版本，例如 `strict`、`balanced`、`short512`。

### 6.3 输出目录结构

建议输出目录：

```text
data_synthesis/doudizhu_end_to_end_sft/
  raw/
    train_steps.jsonl
    train_episodes.jsonl
    val_steps.jsonl
    val_episodes.jsonl
    test_steps.jsonl
    test_episodes.jsonl
  filtered/
    train.parquet
    val.parquet
    test.parquet
  reports/
    train_stats.json
    val_stats.json
    test_stats.json
    kimi_baseline_summary.json
    review.html
  metadata.json
```

也可以把 parquet 直接放在根目录以对齐现有两套数据：

```text
data_synthesis/doudizhu_end_to_end_sft/train.parquet
data_synthesis/doudizhu_end_to_end_sft/val.parquet
data_synthesis/doudizhu_end_to_end_sft/test.parquet
```

但 raw 数据应单独保留，不能只写最终 parquet。

## 7. Episode 采样策略

### 7.1 Seed 与 split

建议沿用现有数据生成脚本的 split seed 偏移方式：

1. train: `seed_start`
2. val: `seed_start + 10_000_000`
3. test: `seed_start + 20_000_000`

目标不是固定采样多少 raw episode，而是达到 accepted step 目标或 raw budget 上限：

1. `--train-samples`
2. `--val-samples`
3. `--test-samples`
4. `--max-raw-episodes`
5. `--max-api-calls`
6. `--max-cost`

### 7.2 Episode 终止条件

每局终止条件：

1. 环境 `done=True`。
2. 达到 `max_env_steps`。
3. API 连续失败超过阈值。
4. bot turn limit reached。
5. 人工指定采样预算耗尽。

若 episode 非正常终止，应记录 `terminated_reason`，并默认不从该 episode 产生 SFT 样本。

### 7.3 Memory 更新

与主环境 RL 逻辑保持一致：

1. 初始 memory 使用中文模式默认初始记忆。
2. 当前 step prompt 使用上一轮 memory。
3. 若 response 中 `<memory>` 非空，则 sanitize 后截断到 `max_memory_chars`。
4. 若 `<memory>` 为空或 projection invalid，则沿用上一轮 memory 或置空，由配置决定。

建议 raw step 同时记录：

1. `prompt_memory_before`
2. `response_memory_raw`
3. `memory_after_sanitized`

filtered SFT 样本必须使用当时真实传给 Kimi 的 prompt，而不是事后重建不同 memory 的 prompt。

## 8. Step 级过滤设计

### 8.1 默认硬过滤

一个 step 写入 SFT parquet 必须同时满足：

1. `episode_won == true`
2. `episode_normal_end == true`
3. `step_projection_valid == 1`
4. `step_click_valid_ratio == 1.0`
5. `step_rule_action_valid == 1.0`
6. `step_fallback_used == false`
7. `action_tag_parse_ok == true`
8. `action_tag_raw == game_action`
9. `response_tokens_target_model <= max_response_tokens`
10. `full_sequence_tokens_target_model <= sft_max_length`，默认 `sft_max_length=2560`，与后续 RL 的 `max_prompt_length=1536` 和 `max_response_length=1024` 对齐。

其中第 6 点理论上已由第 5 点隐含，但建议显式保存并作为硬过滤，方便读代码和统计。

### 8.2 不作为默认硬过滤的条件

以下条件只统计，不默认过滤：

1. `prior_invalid_steps == 0`
2. `prior_fallback_steps == 0`
3. `selected_before_pass == false`
4. `click_count == expected_min_click_count`
5. 五标签恰好出现一次、无额外文本

原因：

1. 全 episode 无瑕疵会大幅降低数据量。
2. 本数据是 step-level SFT，不是整轨迹监督。
3. 商业模型 plan 的价值在于自然策略推理，不应被过强规则模板化过滤。
4. 对格式过严可能丢掉实际上可解析、可执行、可训练的高价值样本。

但上述可观测字段必须写入 `extra_info.verifier` 和报告中，便于后续做高置信子集实验。

### 8.3 Pass 点击特殊检查

环境层面的 `click_valid_ratio=1.0` 和 `rule_action_valid=1.0` 已经足以保证动作有效，但为了识别不干净示范，建议记录：

1. `submit_kind`
2. `selected_cards`
3. `selected_indices`
4. `click_count`
5. `selected_before_pass = action_tag_raw == "pass" and selected_cards != ""`

第一版不默认过滤 `selected_before_pass`，因为最终环境执行动作仍是 pass，且样本可能具有训练价值。但后续可比较是否过滤该类样本会提升 grounding 稳定性。

## 9. 数据 Schema

### 9.1 Parquet 行结构

每个 accepted step 写一行：

```text
data_source: "doudizhu_end_to_end_sft"
prompt: [{"role": "user", "content": "...<image>..."}]
question: prompt text
images: [{"bytes": PNG_BYTES}]
answer: assistant response string
ability: "doudizhu_end_to_end"
reward_model: {"style": "rule", "ground_truth": game_action}
extra_info: {...}
```

`answer` 应使用可训练 response。建议：

1. 默认保留 Kimi 原始 response。
2. 若 `<action>` 非标准但可解析，则生成一个 `normalized_answer`，仅规范化 `<action>` 标签。
3. parquet 中 `answer` 使用 `normalized_answer`，`extra_info.raw_response` 保存原始 response。

这样可以把训练契约收敛到统一格式，同时保留审计信息。

### 9.2 `extra_info` 字段

建议至少包含：

```json
{
  "sample_id": "doudizhu_end_to_end_sft:train:seed:step:uuid",
  "split": "train",
  "sample_index": 0,
  "episode_seed": 20260524,
  "episode_index": 0,
  "step_index": 3,
  "source_model": "kimi-k2.6",
  "source_backend": "moonshot_openai_compatible",
  "language": "zh",
  "prompt_memory_before": "...",
  "memory_after_sanitized": "...",
  "game_action": "334455",
  "action_tag_raw_text": "[3, 3, 4, 4, 5, 5]",
  "action_tag_raw": "334455",
  "action_tag_cards": ["3", "3", "4", "4", "5", "5"],
  "action_match": true,
  "selected_cards": "334455",
  "selected_indices": [0, 1, 2, 3, 4, 5],
  "submit_kind": "play",
  "legal_actions": ["pass", "334455"],
  "current_hand": "...",
  "num_cards_left": [12, 8, 6],
  "trace_tail": [{"player": 1, "action": "99"}],
  "episode": {
    "won": true,
    "episode_length": 12,
    "winner_id": 0,
    "payoffs": [1, -1, -1],
    "total_reward": 0.0,
    "prior_invalid_steps": 1,
    "prior_fallback_steps": 1
  },
  "verifier": {
    "projection_valid": 1,
    "click_valid_ratio": 1.0,
    "rule_action_valid": 1.0,
    "fallback_used": false,
    "action_tag_parse_ok": true,
    "action_match": true,
    "response_tokens_target_model": 438,
    "full_sequence_tokens_target_model": 1512,
    "selected_before_pass": false
  },
  "api": {
    "request_id": "...",
    "latency_sec": 2.31,
    "finish_reason": "stop",
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "retry_count": 0
  }
}
```

字段可以按实际需要精简，但上述信息要么用于过滤，要么用于复现，要么用于后续论文/报告中的统计。

### 9.3 Raw step JSONL

raw step 应包含 rejected 样本也需要的信息：

1. prompt text。
2. image bytes 的路径或 hash，不建议在 JSONL 中直接嵌入大 base64。
3. raw response。
4. projected action。
5. env info。
6. action parser result。
7. tokenizer length result。
8. API request metadata。
9. rejection reasons。

raw JSONL 可以引用图片文件路径，也可以只在 filtered parquet 中内嵌图片 bytes。若为简化第一版实现，也可 raw step 中不保存图片，只保存 seed/step，并依赖环境复现。但考虑到后续审计和 UI 变化，建议保存截图 hash 或图片文件。

## 10. 未过滤统计与商业模型基线

### 10.1 为什么必须统计未过滤数据

未过滤统计有两个作用：

1. 判断 Kimi 在当前 `doudizhu` 环境中的真实能力，发现主要失败模式。
2. 后续训练模型可在相同环境和 seeds 上与 Kimi 对比，论证模型经过 SFT/RL 后超过商业模型。

如果只报告 filtered 数据，会严重高估 Kimi 能力，因为 filtered 数据已经条件在“获胜且单步有效”上。

### 10.2 Episode 级指标

每个 split 和 overall 都应统计：

1. `num_episodes`
2. `num_completed_episodes`
3. `num_won_episodes`
4. `win_rate`
5. `avg_episode_length`
6. `median_episode_length`
7. `truncated_rate`
8. `bot_limit_rate`
9. `api_abort_rate`
10. `avg_total_reward`
11. `avg_payoff_player0`
12. `winner_id_distribution`

### 10.3 Step 级指标

统计全部 raw steps：

1. `num_steps`
2. `projection_valid_rate`
3. `click_valid_ratio_mean`
4. `click_valid_all_rate`
5. `rule_action_valid_rate`
6. `fallback_rate`
7. `action_tag_parse_rate`
8. `action_match_rate`
9. `legal_given_projection_rate`
10. `accepted_step_rate`
11. `accepted_from_winning_episode_rate`

### 10.4 错误分解

至少记录以下 rejection/error reason：

1. `api_timeout`
2. `api_http_error`
3. `api_invalid_response`
4. `missing_required_tag`
5. `tool_call_parse_fail`
6. `tool_call_empty`
7. `tool_call_out_of_range`
8. `click_invalid`
9. `no_submit`
10. `rule_invalid`
11. `fallback_used`
12. `action_tag_parse_fail`
13. `action_mismatch`
14. `episode_lost`
15. `episode_truncated`
16. `response_too_long`
17. `full_sequence_too_long`
18. `plan_hidden_info`

这些原因应允许多标签统计，因为一个 step 可能同时有多个问题。

### 10.5 策略和状态分布

为避免数据只来自简单局面，应统计：

1. action category：`pass`、`solo`、`pair`、`trio`、`chain`、`bomb_rocket`、`other`。
2. turn type：`lead`、`respond`。
3. current hand length bucket。
4. legal action count bucket。
5. opponent min card count bucket。
6. whether pass is legal。
7. whether bomb/rocket is available。
8. game phase：early/mid/late，可用玩家剩余牌数粗略划分。

若 raw 数据分布严重偏斜，filtered parquet 应增加 quota 或采样权重。

### 10.6 成本与吞吐

记录：

1. API calls。
2. 成功/失败/重试次数。
3. prompt tokens。
4. completion tokens。
5. 总 token。
6. 平均 latency。
7. p50/p90/p99 latency。
8. 估算成本。
9. accepted samples per 1000 API calls。
10. accepted samples per dollar。

这对后续决定是否扩大采样规模很关键。

## 11. Filtered SFT 数据均衡策略

### 11.1 为什么需要均衡

如果只按获胜 episode 和有效 step 过滤，数据可能偏向：

1. 首发顺利的强牌局。
2. 短局。
3. pass 或低风险单牌。
4. Kimi 容易处理的清晰截图状态。

这些样本虽然质量高，但不一定覆盖 RL 冷启动最需要的难点。

### 11.2 推荐 quota 维度

第一版建议至少按 action category 做 quota：

| 类别 | 说明 |
| --- | --- |
| `pass` | 过牌 |
| `solo` | 单张 |
| `pair` | 对子 |
| `trio` | 三张、三带一、三带二 |
| `chain` | 顺子、连对、飞机 |
| `bomb_rocket` | 炸弹、王炸 |
| `other` | 四带二、复杂飞机带牌等 |

可参考 grounding 数据的类别权重，但端到端数据应更尊重真实轨迹分布。建议先不强行 oversample 稀有类，只在 metadata 中报告分布；当某类严重不足时再加 quota。

### 11.3 子集版本

从同一 raw rollout 派生多个 filtered 版本：

1. `default`：按本方案默认条件。
2. `short512`：额外要求 response tokens `<= 512`。
3. `clean_pass`：额外过滤 `selected_before_pass`。
4. `no_prior_fallback`：只保留此前没有 fallback 的 episode prefix 中的 step。
5. `balanced_action`：按动作类别重采样。

第一版只需要产出 `default`，但 raw/metadata 设计要支持后续派生。

## 12. 与评估基线的关系

### 12.1 数据采集统计不是最终公平评估

raw rollout 统计可以作为 Kimi teacher 的采样性能报告，但不能直接作为最终“模型超过 Kimi”的唯一证据，因为：

1. 采集数据会被用于 SFT。
2. filtered 数据存在选择偏差。
3. train split 的 seeds 已进入训练过程。

### 12.2 Held-out Kimi baseline

需要保留一套未用于 SFT/RL 的 held-out seeds，使用完全相同环境配置和 prompt 评估：

1. Kimi teacher。
2. QA + grounding SFT 后模型。
3. QA + grounding + end-to-end SFT 后模型。
4. RL 后模型。

评估必须固定：

1. env seed 列表。
2. image size。
3. language。
4. max steps。
5. max clicks。
6. bot policy。
7. prompt 模板。
8. generation 参数。

报告指标：

1. win rate。
2. projection valid rate。
3. click valid ratio。
4. rule action valid rate。
5. fallback rate。
6. average episode length。
7. hand depletion rate。
8. response token length。
9. latency，如果比较部署效率。

建议使用 paired bootstrap confidence interval，因为同一 seed 下不同模型的结果可以配对比较。

## 13. 工程配置建议

### 13.1 命令行参数

端到端合成脚本建议支持：

```text
--output-dir
--train-samples
--val-samples
--test-samples
--seed
--language
--image-width
--image-height
--max-clicks
--max-env-steps
--max-bot-turns
--max-raw-episodes
--max-api-calls
--model-backend api
--api-base-url
--api-model
--api-key-env
--api-timeout
--api-thinking
--temperature
--top-p
--max-new-tokens
--target-tokenizer-path
--max-response-tokens
--max-full-sequence-tokens
--num-workers
--request-concurrency
--request-rpm
--request-tpm
--rate-limit-tier
--retry-max-attempts
--retry-backoff-min-seconds
--retry-backoff-max-seconds
--resume
--log-every
--write-raw
--filter-only
```

其中 `--filter-only` 用于从已有 raw rollout 重建 parquet，不再调用 API。

### 13.2 断点续跑

由于 API 采集成本高，必须支持断点续跑：

1. raw step JSONL append-only。
2. episode 结束后写 episode JSONL。
3. metadata 中记录 last completed episode seed。
4. resume 时跳过已完成 episode。
5. 对未正常结束的 episode，可选择丢弃或重新采集。

### 13.3 并发与限速策略

环境 episode 是有状态的，单个 episode 内不能简单并行 step。但可以并行多个 episode：

1. 每个 worker 持有一个独立 `DoudizhuSingleEnv`。
2. worker 内顺序推进 episode。
3. API client 层做全局限速和并发控制，默认按 Tier2 额度的安全折扣运行。
4. 请求调度需要同时满足 inflight、RPM 和 TPM 三个约束；任一约束达到上限时暂停发起新请求。
5. 写盘通过主进程聚合，或每 worker 写 shard 后合并。

第一版可先单进程顺序采集，验证质量后再加并发。

## 14. 验证与测试计划

### 14.1 单元测试

建议新增测试覆盖：

1. `<action>` list parser：
   - `[pass]`
   - `[3, 3]`
   - `[9, 10, J, Q, K]`
   - `[BJ, RJ]`
   - 带引号、空格、中文别名。
2. action canonicalization：
   - `10 -> T`
   - `BJ -> B`
   - `RJ -> R`
   - 非法牌值拒绝。
3. response normalizer：
   - 只规范化 `<action>`，不改其它标签。
4. filter reason：
   - projection invalid。
   - action mismatch。
   - episode lost。
   - token too long。

### 14.2 小规模 dry run

先跑小规模：

1. raw episodes: 20 到 50。
2. accepted samples: 50 到 200。
3. 生成 review HTML。
4. 人工抽查 plan/action/tool_call/image 对齐。
5. 检查 metadata 中各项统计是否合理。

### 14.3 SFT smoke test

用生成的小数据跑 `SFT/qwen3_5_vlm_sft_trainer.py`：

1. `max_steps=1` 或极小步数。
2. 确认 parquet schema 可读。
3. 确认 image processor 可处理。
4. 确认 full sequence 不超过 `max_length`。
5. 确认 loss 非 NaN。

### 14.4 环境回放验证

可选实现 replay checker：

1. 从 parquet 中读取 `episode_seed`、`step_index`、`answer`。
2. 重建 episode 到该 step。
3. 重新 projection 和 env scoring。
4. 确认 verifier 结果一致。

这可以发现环境版本变化、prompt 变化或数据写入 bug。

## 15. 风险与缓解

### 15.1 商业模型策略偏差

风险：Kimi 赢的局可能偏简单，策略并非最优。

缓解：

1. 保留 QA 和 grounding 数据占比。
2. 不把端到端 SFT 作为唯一训练数据。
3. 后续通过 RL 优化胜率。
4. 用 held-out seeds 对比不同训练阶段。

### 15.2 Plan 噪声

风险：动作有效不代表 plan 每句话都正确。

缓解：

1. 控制 response token 长度。
2. 不对 `<plan>` 做自动内容检查或强规则过滤，避免把商业模型的自然策略推理模板化。
3. 人工 review 抽查。
4. 后续比较保留 plan、截短 plan、弱化 plan loss 的训练效果。

### 15.3 数据选择偏差

风险：只保留获胜 episode step 会偏向优势牌局。

缓解：

1. raw 统计报告分布。
2. 保存 rejected 数据用于分析。
3. 后续按状态难度和动作类别重采样。
4. held-out evaluation 不使用过滤样本。

### 15.4 成本不可控

风险：accepted yield 低，API 成本高。

缓解：

1. 先小规模估算 accepted samples per dollar。
2. 支持 max cost 和 max calls。
3. 支持 filter-only 重建。
4. 支持并发但必须限速。
5. 默认按 Tier2 限速设计，遇到 429 自动退避，并允许通过配置降低并发/RPM/TPM。

### 15.5 契约漂移

风险：prompt、projection、环境点击逻辑或 SFT tokenizer 变化导致旧数据不可比。

缓解：

1. metadata 记录 git commit、prompt hash、env config、model config。
2. raw response 保留。
3. replay checker。
4. 数据版本号。

## 16. 建议实施里程碑

里程碑按工程风险从高到低推进：先完成端到端代码闭环，再用极小规模真实数据验证链路，最后才进行大规模采集。不要在第一阶段追求样本数量；这一阶段的核心价值是让 API 调用、环境交互、动作解析、过滤、统计、落盘和 SFT 读取这些契约全部闭合。

### Milestone 1：实现全套完整代码逻辑

目标：

1. 实现完整 rollout 主流程：reset 环境、构造 prompt、调用 Kimi API、解析 response、projection、执行环境 step、记录 step 和 episode。
2. 实现 Kimi API adapter：`temperature=1.0`、thinking disabled、90 秒 timeout、retry、429 backoff、Tier2 限速、断点续跑。
3. 实现 `<action>` list-of-card 契约、action canonicalizer 和环境实际提交动作的 canonical match。
4. 实现 raw rollout JSONL、episode summary、stats report、filtered SFT parquet、metadata 的完整写入。
5. 实现 filter-only 模式，可以从 raw rollout 重新生成 filtered parquet 和统计报告。
6. 实现 review exporter 和 `SFT/` dataloader smoke 所需的数据 schema。

验收：

1. 可以用 mock API 或极少量真实 API 跑通完整 episode。
2. 每个 step 都能记录 prompt、image、raw response、normalized answer、projection info、env info、filter reasons。
3. 未过滤 Kimi 表现统计可生成，包括 win rate、projection valid rate、click valid rate、rule valid rate、fallback rate、action match rate、response token 分布、accepted yield。
4. parser、canonicalizer、filter predicate、token length、schema writer、resume 至少有最小单元测试。
5. filtered parquet 的字段能被 `SFT/` 当前训练管线识别。

### Milestone 2：用极小规模数据集测试

目标：

1. 使用真实 Kimi API 跑 20 到 100 个 episode，或运行到获得 20 到 50 条 accepted step。
2. 保留全部 raw/rejected 数据，不只保存 accepted 数据。
3. 用小数据验证真实链路稳定性、数据质量和过滤损耗来源。
4. 用 review HTML 人工抽查 accepted 和 rejected case。
5. 用 `SFT/` 进行极小步数 smoke test。

验收：

1. 统计报告能解释主要损耗来源，例如 episode lost、projection invalid、click invalid、rule invalid、action mismatch、token too long、API timeout、429 retry。
2. accepted 样本能通过 replay 或 verifier 重新验证。
3. `prompt_tokens <= 1536`、`response_tokens <= 1024`、`prompt_tokens + response_tokens <= 2560` 的长度约束可统计并可过滤。
4. SFT dataloader 能正常读取 parquet 和图片，极小步数训练 loss 非 NaN。
5. 根据真实失败分布完成一次 prompt、契约、parser、限速或 retry 参数校准，并记录 prompt/config 版本。

### Milestone 3：收集大规模数据集

目标：

1. 在确认小规模链路稳定后，按目标规模扩大 raw episode 采集。
2. 按 Tier2 或实际账号限速稳定运行，支持多进程/多 worker 但共享全局 RPM、TPM 和 inflight 限流。
3. 生成正式 train/val/test parquet。
4. 生成完整 metadata、质量报告、成本报告和 Kimi 未过滤表现统计。
5. 在 held-out seeds 上固定 Kimi baseline，供后续训练模型对比。

验收：

1. 数据达到目标样本量，且每个 split 的动作类别、角色、局面阶段、response token 长度分布有统计。
2. 所有产物记录 model id、prompt hash、git commit、env config、filter config、tokenizer、API 参数、限速参数和采集时间。
3. 采集过程可断点续跑，失败 episode 不污染 accepted 数据。
4. held-out baseline seeds、Kimi 评估结果和统计脚本可被后续模型复用。
5. 最终数据能和 QA SFT、grounding SFT 一起进入冷启动混合 SFT。

## 17. 推荐默认配置

第一版建议：

```text
language: zh
image_width: 640
image_height: 480
max_clicks: 20
max_env_steps: 30
max_bot_turns: 256
api_model: kimi-k2.6
max_new_tokens: 1024
max_response_tokens: 1024
sft_max_length: 2560
rl_max_prompt_length: 1536
rl_max_response_length: 1024
temperature: 1.0
top_p: 0.95
api_thinking: disabled
api_timeout: 90
rate_limit_tier: Tier2
request_concurrency: 64
request_rpm: 450
request_tpm: 2700000
train_samples: 5000 起步
val_samples: 500
test_samples: 500
```

建议先用小规模数据验证 yield 和质量，再扩大到数万级样本。端到端数据由于成本更高、策略偏差更强，不一定需要和 QA/grounding 数据同量级；它更适合作为高价值策略冷启动补充数据。

## 18. 最终产物

本方案完成后应产生：

1. `data_synthesis/doudizhu_end_to_end_sft.py`
2. `data_synthesis/doudizhu_end_to_end_sft/train.parquet`
3. `data_synthesis/doudizhu_end_to_end_sft/val.parquet`
4. `data_synthesis/doudizhu_end_to_end_sft/test.parquet`
5. `data_synthesis/doudizhu_end_to_end_sft/metadata.json`
6. raw rollout JSONL。
7. Kimi 未过滤性能统计报告。
8. filtered SFT yield 报告。
9. review HTML。
10. 可复用的 held-out Kimi baseline seeds 和评估结果。

这些产物将与 QA SFT、grounding SFT 一起构成斗地主 RL 冷启动数据体系。
