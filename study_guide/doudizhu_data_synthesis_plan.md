# 面向 doudizhu_grounding 和 doudizhu 的 Kimi K2.6 数据合成方案

日期：2026-05-20

本文设计一套用于冷启动 RL 的数据合成方案，目标环境是仓库中的 `doudizhu_grounding` 和完整 `doudizhu`。方案重点是数据生产、质量控制、规模规划和风险控制，不涉及具体代码实现。

核心结论：

- `doudizhu_grounding` 应优先做，且主数据源应是“环境命令 + 程序化点击标签 + Kimi 生成简短 plan”的高精度数据。Kimi 可以少量参与端到端候选生成，但必须经过环境 verifier 严格过滤。
- 完整 `doudizhu` 可行，但不应让 Kimi 决定出牌。推荐使用 DouZero 或同等级策略模型生成语义出牌，用环境 renderer 生成 GUI 点击，再让 Kimi 只补全 `<plan>`、`<chat>`、`<memory>`。最终 response 由系统拼接并回放验证。
- 两个环境都应把“能被环境确定性验证的部分”放在标签生成的核心位置，把 Kimi 限制在自然语言增强、风格多样化和少量候选/审查上。

## 外部依据和本仓库约束

外部依据：

- Kimi 官方文档显示 `kimi-k2.6` 支持文本、图像、视频输入，适合对话、视觉理解和 Agent 任务，模型上下文窗口为 256k。官方 Chat Completions API 兼容 OpenAI SDK，支持 `image_url` 多模态输入、`response_format` JSON/JSON Schema、`thinking` 开关和 Batch API。
- Kimi Batch API 适合大规模、低实时性任务，官方文档说明其支持 `kimi-k2.6` 和 `kimi-k2.5`，并可降低批量推理成本。批处理模型参数受限，不能依赖温度采样制造多样性，应通过 prompt 模板、状态采样和任务分层制造多样性。
- DouZero 论文把斗地主描述为含竞争/协作、不完全信息、大状态空间和大动作空间的困难任务，并报告 DouZero 通过自博弈强化学习取得强策略表现。它适合作为完整 `doudizhu` 的 teacher policy 候选。
- RLCard 论文也强调 Dou Dizhu 具有巨大信息集和动作组合空间，随机或弱规则数据很难覆盖关键策略状态。因此完整对局数据应依赖强 teacher，并按局面类型做分层采样。

本仓库约束：

- `doudizhu_grounding` 的目标是执行给定 `target_action`，奖励能直接验证 `projection_valid`、`click_valid_ratio`、`submit_correct`、`target_action_match`。
- 完整 `doudizhu` 的环境真正执行的是点击投影后的 `game_action`。只要点击形成的 `candidate_action` 在当前 `legal_actions` 中，就不会触发 fallback。
- Renderer 已能根据当前手牌和选中状态返回手牌、PLAY、PASS 的 hitbox。点击标签应优先由 hitbox 生成，而不是由 Kimi 猜坐标。
- 底层 state 包含 `others_hand`，这是对手真实手牌，不应暴露给 DouZero wrapper、Kimi prompt 或任何 teacher 生成过程。否则会产生学生模型从截图无法推断的信息泄漏。
- 完整 `doudizhu` 训练 prompt 在中文模式下要求五个标签：`<plan>`、`<action>`、`<tool_call>`、`<chat>`、`<memory>`。`doudizhu_grounding` 要求两个标签：`<plan>`、`<tool_call>`。

参考来源：

- Kimi Model List: https://platform.kimi.ai/docs/models
- Kimi K2.6 Quickstart: https://platform.kimi.ai/docs/guide/kimi-k2-6-quickstart
- Kimi Chat Completion API: https://platform.kimi.ai/docs/api/chat
- Kimi API Overview: https://platform.kimi.ai/docs/api/overview
- Kimi JSON Mode: https://platform.kimi.ai/docs/guide/use-json-mode-feature-of-kimi-api
- Kimi Batch API: https://platform.kimi.ai/docs/guide/use-batch-api
- DouZero paper: https://arxiv.org/abs/2106.06135
- RLCard paper: https://dczha.com/files/rlcard-a-toolkit.pdf

---

# 第一部分：doudizhu_grounding 数据合成方案

## 1. 任务定位

`doudizhu_grounding` 是“指令到点击”的视觉 grounding 任务。输入是截图和一个被指挥的出牌动作，输出只需要说明点击计划并给出一个 `left_click(...)` 调用。

这个环境不需要模型学习斗地主策略。它只需要学会：

- 从截图中定位底部手牌和按钮；
- 按指令选择正确的同 rank 手牌；
- 点击每张目标牌后点击“出牌”；
- 指令为“不要/pass”时只点击“不要/PASS”按钮；
- 严格遵守 XML 和 `left_click([x,y],...)` 格式。

推荐优先级：最高。它是完整 `doudizhu` 冷启动的视觉点击基础，如果 grounding 不稳，完整 RL 会大量浪费在格式和坐标探索上。

## 2. 数据生产路线

推荐采用双流数据，但训练主集以高精度流为主。

### 2.1 主流：程序化金标点击 + Kimi 生成 plan

这是主推荐路线。

生成流程：

1. 用 `doudizhu_grounding` 环境采样状态，获得截图和当前 `target_action`。
2. 用 renderer hitbox 生成 gold click sequence。
3. 将截图、指挥动作、gold click sequence 的语义说明发送给 Kimi，让 Kimi 只生成简短 `<plan>` 内容。
4. 系统拼接最终 response：`<plan>...</plan><tool_call>left_click(...)</tool_call>`。
5. 将 response 回放到环境，保留 verifier 全通过的样本。

优点：

- 坐标标签精度最高；
- Kimi 失误不会污染动作标签；
- 数据成本低，因为 Kimi 输出很短；
- 容易扩展到数万或十万级 step。

注意：Kimi 的输入可以包含截图，但不应让 Kimi 自行决定最终点击。截图的作用是让 plan 的表述更自然，例如“点击底部两张 3，再点击出牌”。

### 2.2 辅流：Kimi 端到端候选 + 环境过滤

这是补充路线，不建议作为主集。

生成流程：

1. 直接给 Kimi 截图和指挥动作。
2. 要求 Kimi 输出 `<plan>` 和 `<tool_call>`。
3. 用 `doudizhu_grounding_projection` 和环境 scorer 过滤。
4. 只保留 `target_action_match=1` 的样本。

用途：

- 估计 Kimi 在该 UI 上的自然 grounding 能力；
- 生成少量更接近“模型自己看图输出”的样本；
- 作为自动评测集或鲁棒性分析材料。

限制：

- 接受率可能随牌数、牌距、中文字体、长顺子等因素波动；
- 一旦只靠 Kimi 生成坐标，失败样本会显著增加；
- 即使通过环境，也可能存在 plan 描述和点击不完全一致的问题，需要额外文本一致性检查。

建议比例：训练主集中 80% 到 95% 使用主流金标点击，5% 到 20% 使用 Kimi 端到端候选且严格过滤后的样本。冷启动早期可以完全不用辅流。

## 3. Kimi API 使用策略

推荐使用 `kimi-k2.6`，原因是它支持图像输入、长上下文和 Agent 类任务。调用策略如下：

- 对大规模 plan 生成，优先使用 Batch API。
- 使用 JSON Mode 或 JSON Schema，让 Kimi 输出结构化字段，例如 `plan`、`quality_flags`，再由系统组装 XML。
- 对主流金标数据，建议关闭 thinking，输出控制在 80 到 160 中文字以内。
- 对端到端候选，可打开更严格的自检要求，但仍应限制输出长度，避免模型在 `<plan>` 中产生冗长推理。
- 不依赖 temperature 制造多样性。Batch API 下 K2.6 的部分采样参数不可调，多样性应来自 prompt 模板、状态采样和动作类别采样。

## 4. 推荐数据规模

建议分三阶段推进。

| 阶段 | 目标 | 推荐规模 |
| --- | --- | --- |
| Pilot | 验证流水线、prompt、过滤器、接受率 | 1k 到 3k accepted steps |
| 冷启动主集 | 让模型稳定学会 UI 点击和格式 | 30k 到 80k accepted steps |
| 增强集 | 面向更大模型或更强泛化 | 100k 到 200k accepted steps |

验证/测试集建议：

- validation：5k steps；
- heldout test：5k steps；
- 所有 split 按 seed 和 episode 级别切分，避免同一局游戏的相邻状态同时出现在 train 和 test。

如果预算有限，优先保证 30k 高质量 grounding step，而不是追求更大但含噪的数据。

## 5. 数据多样性设计

`doudizhu_grounding` 的多样性重点是视觉定位和点击组合，不是策略。

### 5.1 动作类别分布

推荐 accepted 数据中的动作分布大致如下：

| 类别 | 建议比例 | 目的 |
| --- | ---: | --- |
| pass/不要 | 15% 到 20% | 学会只点 PASS/不要 |
| 单张 | 20% 到 25% | 覆盖所有 rank 和不同手牌位置 |
| 对子 | 15% 到 20% | 学会点同 rank 多张 |
| 三张、三带一、三带二 | 10% 到 15% | 覆盖组合点击 |
| 顺子、连对、飞机 | 10% 到 15% | 覆盖长点击序列 |
| 炸弹、王炸 | 3% 到 5% | 稀有动作过采样 |
| 近终局动作 | 5% 到 10% | 手牌很少、按钮/牌间距变化 |

不建议完全按环境自然分布采样，因为炸弹、王炸、长顺子等稀有动作在训练中太少，模型会在关键场景失稳。

### 5.2 视觉和状态覆盖

应覆盖：

- 手牌数量从 1 到 20；
- 目标牌位于最左、中间、最右；
- 相同 rank 的多张牌相邻和非目标牌相邻；
- 王、2、10 等视觉标签较特殊的牌；
- 中文模式下“出牌/不要”按钮；
- 英文模式可作为少量补充，但若训练脚本主要用中文，中文数据应占 80% 以上；
- 不同局面下的桌面已出牌区域，避免模型只记住空桌布局。

图片分辨率默认应保持训练环境一致。只有当后续 RL 也会改变 `image_width` 或 `image_height` 时，才引入多分辨率数据；否则多分辨率会增加不必要分布差异。

## 6. 验证与过滤

每条样本必须通过两级验证。

### 6.1 格式验证

硬性条件：

- 只包含 `<plan>` 和 `<tool_call>` 两个必需标签；
- 两个标签非空；
- `<tool_call>` 只含一个 `left_click(...)`；
- 坐标均在 0 到 1000；
- 点击数不超过环境 `max_clicks`；
- 无代码块、额外 XML 标签、自然语言混入 `<tool_call>`。

### 6.2 环境验证

硬性条件：

- `projection_valid = 1`；
- `click_valid_ratio = 1.0`；
- `submit_correct = 1.0`；
- `target_action_match = 1.0`；
- `predicted_action == target_action`；
- 无提前 done、无异常、无 bot limit。

任何一项不满足都不进入训练主集。

### 6.3 文本一致性检查

对 Kimi 生成的 `<plan>` 做轻量检查：

- plan 中描述的动作必须和 `target_action` 一致；
- 不允许说“换一手牌”“选择更优出牌”等违背 grounding 指令的话；
- 不允许提到截图中不可见的信息；
- 不允许生成很长的战术推理，目标是简短定位说明。

可以采用规则检查加少量 Kimi judge 抽检。若抽检发现超过 2% 的 plan 有明显不一致，应回滚 prompt 模板。

## 7. 验收指标

数据集级别指标：

- accepted 样本的 `target_action_match` 必须为 100%；
- accepted 样本的 `click_valid_ratio` 必须为 100%；
- 格式投影成功率必须为 100%；
- 端到端候选流的接受率单独统计，不影响主流金标质量；
- action type 分布不能严重塌缩，pass 不应超过 25%，单张不应超过 35%。

模型冷启动后的验证目标：

- `doudizhu_grounding_target_action_match` 明显高于未 SFT 基线；
- `doudizhu_grounding_click_valid_ratio` 接近 1；
- `doudizhu_grounding_submit_correct` 接近 1；
- 低频动作类别上单独报告准确率，尤其是连对、飞机、炸弹、王炸。

---

# 第二部分：完整 doudizhu 数据合成方案

## 1. 任务定位

完整 `doudizhu` 是多轮视觉 GUI 对局。模型需要同时学会：

- 看懂截图；
- 根据当前局面选择合法且尽量强的语义出牌；
- 将语义出牌转成 GUI 点击；
- 维护上一轮 `<memory>`；
- 输出五个标签；
- 在多轮对局中争取获胜。

这里的关键判断是：Kimi K2.6 可用于生成自然语言解释和陪玩风格文本，但不应作为主策略来源。完整斗地主策略涉及不完全信息、长程规划和巨大动作空间，Kimi 偶发失误会污染 SFT 数据。强策略 teacher 加环境验证更稳。

推荐方案：DouZero 类 teacher 决定 `<action>`，renderer 决定 `<tool_call>`，Kimi 决定 `<plan>`、`<chat>`、`<memory>`，系统拼接完整 response，并回放过滤。

## 2. 数据生产路线

### 2.1 主流：强 teacher 策略 + 程序化点击 + Kimi 文本增强

生成流程：

1. 采样完整 `doudizhu` episode seed。
2. 每次轮到 Player 0 时，构造 teacher 可见状态。
3. 用 DouZero 或同等级 teacher 从当前合法动作中选择 `target_action`。
4. 确认 `target_action` 属于环境 `legal_actions`。
5. 用 renderer hitbox 生成对应点击序列。
6. 将截图、公开局面摘要、上一轮 memory、强制 `target_action`、最终点击语义发送给 Kimi。
7. Kimi 只输出结构化字段：`plan`、`chat`、`memory`，不得修改动作。
8. 系统拼接：
   `<plan>...</plan><action>...</action><tool_call>...</tool_call><chat>...</chat><memory>...</memory>`
9. 回放环境 step，验证实际 `game_action` 等于 `target_action` 且无 fallback。
10. 对整局 trajectory 做终局质量评估和去重。

这个路线的优点是将完整任务拆成三类可控问题：

- 策略质量由 DouZero 类模型保证；
- GUI 坐标由环境 renderer 保证；
- 自然语言和 memory 风格由 Kimi 保证；
- 最终质量由环境 verifier 保证。

### 2.2 辅流：Kimi 策略候选

不建议在冷启动主集中大量使用，但可以小规模探索。

可用于：

- 比较 Kimi 和 DouZero teacher 的动作差异；
- 发现某些规则 teacher 过于机械的场景；
- 生成少量“人类风格但经强过滤”的策略样本。

过滤要求应更严格：

- Kimi 的动作必须在 `legal_actions` 中；
- 点击必须由代码重算或环境验证；
- 只保留终局表现不差于 teacher baseline 的 trajectory；
- 不允许 Kimi 看到隐藏手牌；
- 占训练主集比例不超过 5% 到 10%。

## 3. Teacher 策略要求

### 3.1 防止信息泄漏

这是完整 `doudizhu` 数据合成的第一风险。

底层 state 中存在 `others_hand`，这是对手真实手牌。任何 teacher wrapper、Kimi prompt、日志导出、plan 生成都不应使用该字段。

teacher 只能使用：

- 当前玩家手牌；
- 当前合法动作；
- 公开出牌历史；
- 地主身份和底牌公开信息；
- 对手剩余手牌数量；
- 轮次和当前需要压制的上一手牌；
- 自己上一轮 memory 或由公开 trace 生成的摘要。

如果使用 DouZero 的开源实现或改写版，应单独审计输入特征，确认它在推理时不读取真实对手手牌。若无法确认，应将该 teacher 降级为研究参考，不进入主数据管线。

### 3.2 策略多样性

只用一个 greedy teacher 可能让学生模型过度模仿单一风格。推荐：

- 80% 到 90% 样本使用主 teacher greedy action；
- 5% 到 15% 使用 teacher 高置信候选中的次优但合理动作；
- 5% 左右使用规则 teacher 或环境 fallback-free baseline 作为风格补充；
- 所有非 greedy 动作必须合法、可回放，并通过终局或局部质量筛选。

如果 teacher 能输出动作价值或置信度，应保存到 metadata，用于后续加权训练或过滤低置信状态。

## 4. Kimi API 使用策略

完整 `doudizhu` 中，Kimi 的职责是“把已确定动作写成可训练 response 的自然语言部分”，而不是玩牌。

推荐设置：

- 使用 `kimi-k2.6`。
- 大规模任务用 Batch API。
- 使用 JSON Mode 或 JSON Schema 输出字段：`plan`、`chat`、`memory`、`consistency_check`。
- 系统再把 JSON 字段拼成 XML response，避免 Kimi 直接输出 XML 时混入多余文本。
- `thinking` 默认关闭，除非用于少量高难局面的文本审查。
- 单步输出长度建议：
  - `plan`：1 到 3 句，80 到 180 中文字；
  - `chat`：一句短话，5 到 25 中文字；
  - `memory`：最多 80 到 160 中文字，且最终进入环境前仍受 `max_memory_chars` 限制。

Kimi prompt 应明确：

- `target_action` 是强制动作，不得改变；
- 不要输出坐标；
- 不要声称知道对手隐藏手牌；
- 只能根据截图、公开出牌历史、手牌和给定动作解释；
- memory 只能记录公开信息和下一步计划；
- 若无法解释，也应给出保守、短的可见依据，不得编造。

## 5. 推荐数据规模

完整 `doudizhu` 的合成成本和噪声风险都高于 grounding，应从小到大推进。

| 阶段 | 推荐规模 | 用途 |
| --- | ---: | --- |
| Pilot | 200 到 500 accepted episodes，约 2k 到 6k Player 0 decision steps | 验证 teacher 接入、Kimi 文本、回放过滤 |
| 冷启动主集 | 3k 到 5k accepted episodes，约 30k 到 70k decision steps | 训练模型稳定完成完整 response 和合法出牌 |
| 增强集 | 8k 到 15k accepted episodes，约 100k 到 200k decision steps | 面向更大模型、更多局面覆盖和更强策略模仿 |

若只能做一版，建议目标是：

- `doudizhu_grounding`：至少 30k accepted steps；
- 完整 `doudizhu`：至少 30k accepted decision steps；
- 两者先按 1:1 或 2:1 混合做 SFT，之后进入在线 RL。

完整对局不建议一开始追求百万级 step。过大的 teacher imitation 数据会让模型过度贴近 teacher 策略，减少后续 RL 改善空间，也会放大潜在 text hallucination 的累计影响。

## 6. 数据多样性设计

### 6.1 Episode 级多样性

按 seed 分层采样，覆盖：

- 初始手牌强弱：强牌、中等牌、弱牌；
- 是否持有王炸、炸弹、多个 2；
- 手牌结构：单张多、对子多、顺子多、三张多、牌型碎；
- 对手出牌速度：某个农民很快剩 1 到 3 张；
- 长局和短局；
- landlord 从头顺风、被压制、终局反超等轨迹。

### 6.2 Step 级多样性

每个 decision step 标注局面类别，并在采样中控制比例。

| 局面类别 | 建议覆盖 |
| --- | --- |
| 首发出牌 | 覆盖低单、对子、顺子、三带等开局策略 |
| 跟牌压制 | 覆盖用最小可压动作、是否保留炸弹 |
| 合理 pass | 覆盖不能压、能压但不值得压、队形保留 |
| 对手近终局 | 对手剩 1 到 3 张时的拦截动作 |
| 自己近终局 | 自己剩 1 到 5 张时的收尾动作 |
| 稀有大牌 | 炸弹、王炸、多个 2 的使用和保留 |
| 长组合 | 顺子、连对、飞机、四带二等多点击动作 |

自然分布通常会低估稀有动作，因此应进行过采样，但不要让稀有动作占比失真。炸弹/王炸在训练集中可提高到 3% 到 6%，但不宜更高。

### 6.3 语言和风格多样性

如果后续训练环境主要使用中文模式，完整 `doudizhu` 合成数据也应以中文为主。

建议：

- 中文样本：80% 到 95%；
- 英文样本：0% 到 20%，仅在需要 bilingual 泛化时加入；
- plan 风格分为简短战术型、保守解释型、陪玩轻语气型；
- chat 必须短，不应包含战术泄漏或冗长解释；
- memory 应保持紧凑，优先记录公开出牌和下一轮策略，不要写“对手一定有某牌”这类确定性隐藏信息。

## 7. 验证与过滤

完整 `doudizhu` 必须做 step 级、trajectory 级和文本级三层过滤。

### 7.1 Step 级硬过滤

每个 accepted step 必须满足：

- XML 五标签齐全且非空；
- `<tool_call>` 可被 `doudizhu_projection` 解析；
- `projection_valid = 1`；
- `click_valid_ratio = 1.0`；
- `rule_action_valid = 1.0`；
- `fallback_used = False`；
- `game_action == target_action`；
- `selected_cards` 与目标动作一致；
- 点击数不超过 `max_clicks`；
- `<action>` 语义与 `target_action` 一致，中文 pass 用“不要”或约定的统一写法；
- `<memory>` 经 sanitization 后不为空且不超过长度限制。

只要某一步失败，默认丢弃该 step。若要保留完整 trajectory，则整局任一步失败都应丢弃整局。

### 7.2 Trajectory 级过滤

整局 accepted trajectory 应满足：

- 从同一 seed 回放可完全复现；
- 全局 fallback 次数为 0；
- bot turn limit 未触发；
- episode 正常终止；
- 记录最终 `won`、`payoffs`、总 reward、步数、动作类型分布；
- 重复 seed、重复轨迹、同质化极高的轨迹去重。

是否只保留赢局需要谨慎。

推荐策略：

- 主训练集 70% 到 85% 来自 teacher 赢局；
- 15% 到 30% 可来自无 fallback、teacher 高置信但最终输掉或困难的局；
- 若目标是极致稳定冷启动而不是策略多样性，可第一版只保留赢局，但后续应补充困难局面，避免模型在逆风状态完全不会处理。

### 7.3 文本一致性过滤

Kimi 生成的自然语言部分必须与动作和公开信息一致。

规则检查：

- plan 中的牌型必须和 `<action>` 一致；
- chat 不应包含坐标、内部字段、teacher 名称；
- memory 不应包含 `<image>`、`<video>`、`<|vision_start|>` 等多模态 token；
- memory 不应说出未公开的对手手牌；
- plan 不应说“我选择另一手”或“改成 pass”等违背 target action 的内容；
- 不应出现长篇思维链、代码块、JSON 残留。

Kimi judge 抽检：

- 对全量样本做规则检查；
- 对 5% 到 10% 样本做 Kimi judge 或人工抽检；
- 对高风险样本加大抽检比例，例如炸弹、王炸、对手近终局、teacher 输局；
- 若某个 prompt 模板的问题率超过 2%，该模板产物整体重跑或下线。

## 8. 数据集组织和 metadata

每条样本都应保存足够 metadata，便于追踪、过滤和后续 ablation。

建议字段：

- `env_name`：`doudizhu`；
- `language`：`zh` 或 `en`；
- `episode_id`、`seed`、`turn_index`；
- `prompt_version`、`response_version`；
- `kimi_model_id`、`kimi_prompt_template_id`；
- `teacher_policy`、`teacher_version`；
- `target_action`、`action_type`、`teacher_confidence`；
- `legal_actions_count`；
- `click_sequence`；
- `verification`：projection、click、rule、fallback、game_action、reward；
- `terminal`：won、payoffs、episode_len；
- `hidden_info_used`：必须为 false；
- `split`：train/val/test。

图片可保存为路径、对象存储 key 或 parquet 中的二进制字段，但 split 必须按 episode seed 切分。

## 9. 训练混合建议

冷启动 SFT 可以按阶段混合：

1. 第一阶段：只用 `doudizhu_grounding`，让模型学格式和点击。
2. 第二阶段：加入完整 `doudizhu` step，比例可从 grounding:full = 2:1 过渡到 1:2。
3. 第三阶段：只用完整高质量 trajectory 做短暂收敛，再进入在线 RL。

一个实用初始配比：

- 40% `doudizhu_grounding` 金标；
- 50% 完整 `doudizhu` teacher trajectory；
- 10% 稀有动作/困难局面过采样。

后续如果模型在完整环境中仍有坐标错误，应提高 grounding 比例；如果坐标稳定但策略差，应提高完整 teacher trajectory 或直接进入 RL。

## 10. 验收指标

完整 `doudizhu` 数据集验收：

- accepted step 的 `projection_valid`、`click_valid_ratio`、`rule_action_valid` 均为 100%；
- accepted step 的 `fallback_used` 为 0；
- accepted step 的 `game_action == target_action` 为 100%；
- 完整 trajectory 可复现率为 100%；
- 文本抽检隐藏信息泄漏率低于 1%，动作文本不一致率低于 1%；
- train/val/test 的 seed 无交叉；
- action type 分布满足预设覆盖，不被 pass 或单张主导。

模型冷启动后的目标：

- 格式投影成功率显著提升；
- click valid ratio 接近 1；
- fallback rate 明显低于未 SFT 模型；
- rule action valid rate 明显提升；
- 在固定验证 seeds 上胜率高于原始模型，并接近 teacher 的可模仿下界；
- 进入 RL 后，早期 rollout 不再主要浪费在 XML 和坐标错误上。

## 11. 主要风险和缓解

| 风险 | 表现 | 缓解 |
| --- | --- | --- |
| hidden state 泄漏 | plan 或 action 使用对手真实手牌 | teacher/Kimi 输入白名单，禁止 `others_hand` |
| Kimi 改动作 | plan 说的动作和 target 不一致 | Kimi 只输出 JSON 字段，系统拼 action/tool_call，文本一致性过滤 |
| 坐标污染 | Kimi 自行坐标偶发点偏 | 主集使用 renderer gold click，Kimi 坐标候选只做辅流 |
| teacher 过拟合 | 学生只会 DouZero 风格 | 少量多 teacher/高置信次优动作，后续用 RL 继续优化 |
| 只保留赢局导致分布窄 | 逆风状态处理差 | 保留一部分高质量困难局和输局 |
| plan 过长 | SFT 学到冗长输出，RL 响应成本高 | 限制 plan/chat/memory 长度，过滤长思维链 |
| 稀有动作不足 | 炸弹、王炸、飞机等失败 | 按 action type 分层过采样 |

## 最终推荐落地顺序

1. 做 `doudizhu_grounding` 1k 到 3k pilot，验证 hitbox 标签、Kimi plan、环境过滤。
2. 扩到 30k 到 80k grounding accepted steps，先训练一个 grounding 冷启动模型。
3. 接入 DouZero 类完整 teacher，先做 200 到 500 局完整 `doudizhu` pilot。
4. 审计 teacher 输入，确保没有 `others_hand` 等隐藏信息。
5. 扩到 30k 到 70k 完整 decision steps，做主冷启动 SFT。
6. 用固定 validation seeds 比较：原始模型、grounding-only SFT、full SFT、full SFT + RL。
7. 根据失败模式补数据：坐标错补 grounding，策略错补完整 teacher，记忆幻觉补 public trace 约束和文本过滤。

总体判断：该方案可行且推荐。最重要的工程原则是“动作和坐标由可验证系统产生，Kimi 只做语言增强和少量可验证候选”。这样能最大化 Kimi K2.6 的多模态和语言能力，同时避免把不可控策略错误蒸馏进冷启动模型。
