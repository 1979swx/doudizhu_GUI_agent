# 斗地主 (Dou Dizhu) 环境指南：大语言模型 Agentic RL 训练

本指南系统性地介绍了代码库中 `Doudizhu` 视觉环境的架构和运行机制。该环境专门为大视觉语言模型 (VLM/LLM) 的多模态、多轮次、Agentic 强化学习训练（如 PPO/GRPO）而设计。

---

## 1. 核心定位与训练目标

在传统的强化学习中，斗地主通常被建模为带有离散动作空间（如出牌组合的枚举）和状态向量的 MDP 过程。然而，本代码库的目的是**训练能够直接操作 GUI 的智能体 (GUI Agent)**。
*   **角色设定**: Agent 固定扮演**地主 (Player 0)**。
*   **对手设定**: 农民（Player 1 和 Player 2）由内置的规则模型 (`DouDizhuRuleAgentV1`) 控制。
*   **交互方式**: 纯视觉输入（截屏）+ 纯坐标点击（归一化坐标输出）。模型必须学会看懂手牌、桌面已出的牌、对手剩余牌数，并学会点击特定卡牌区域再点击“出牌 (PLAY)”或“不出 (PASS)”按钮。

这种设计使得斗地主不仅是一个策略博弈环境，更是一个**视觉定位、指令遵循、UI 交互与长程规划**的综合 Agentic 训练场。

---

## 2. 环境系统架构

斗地主环境的代码位于 `agent_system/environments/env_package/doudizhu` 目录下，主要由以下层级构成：

1.  **Core (底层引擎)**: `core/` 目录下实现了斗地主的游戏逻辑、发牌 (`dealer.py`)、规则裁判 (`judger.py`) 和轮次管理 (`round.py`)。它维护了每一局游戏的硬性规则和状态更新。
2.  **Renderer (视觉渲染)**: `renderer.py` 根据 Core 提供的状态，将卡牌、按钮绘制成图像。该图像是 Agent 的主要观测输入（默认分辨率 640x480）。它同时维护了 UI 元素的包围盒 (Hitbox)，用于计算点击的有效性。
3.  **Env Wrapper (环境封装)**: `envs.py` 中的 `DoudizhuSingleEnv` 提供了标准的 `reset` 和 `step` 接口。对于批量强化学习，`DoudizhuVectorEnv` 利用 Ray 实现了多进程的向量化环境加速数据采样。
4.  **Env Manager (环境管理器)**: `agent_system/environments/env_manager.py` 中的 `DoudizhuEnvironmentManager` 负责在模型 Rollout 阶段与底层环境对接，处理多轮对话的上下文、图像组装和奖励统计。

---

## 3. 观测空间 (Observation Space)

Agent 在每一个回合 (Turn) 都会接收到多模态的观测输入：

### 3.1 视觉输入 (Image)
*   **截屏 (`<image>`)**: 包含 Agent 当前手牌（底部排列）、中心区域的上一手出牌、两位农民的剩余牌数标识，以及右下角的 "PLAY" 和 "PASS" 交互按钮。

### 3.2 文本输入与系统 Prompt
文本输入的定义在 `agent_system/environments/prompts/doudizhu.py` 中。系统使用 `DOUDIZHU_VISUAL_TEMPLATE` 指导模型如何行动：
*   **明确任务目标**: "You are Player 0 and the landlord. Win the current Dou Dizhu game..."
*   **空间坐标系**: 强制模型使用 1-1000 的归一化坐标（[1, 1] 为左上角，[1000, 1000] 为右下角）。
*   **长期记忆 (Previous Memory)**: 每一轮的 Prompt 会注入 `{previous_memory}`，即上一轮模型自己生成的 `<memory>` 标签的内容。这允许模型跨步传递战术思考、记牌信息，克服部分可观察马尔可夫决策过程 (POMDP) 的局限。

---

## 4. 动作空间与 XML 投影 (Action & Projection)

为了防止 LLM 在自由输出中偏离任务，系统对模型的回复进行了严格的结构化约束。

### 4.1 四标签强制格式
模型必须在其回复中精确包含以下四个 XML 标签，且顺序和内容必须合法（解析逻辑位于 `projection.py` 中的 `doudizhu_projection`）：
1.  **`<think>`**: 模型的 Chain-of-Thought (CoT) 区域。要求模型在这里分析当前桌面局势、猜测对手手牌并制定出牌策略。
2.  **`<action>`**: 动作执行区。只能是一个 JSON 风格的二维浮点数/整数数组，代表要点击的屏幕归一化坐标。例如：`[[140, 850], [210, 850]]`。模型需要点击选中的牌，最后加上点击 "PLAY" 或 "PASS" 按钮的坐标。
3.  **`<chat>`**: 提供拟人化的交互，输出一句简短的桌边聊天。
4.  **`<memory>`**: 隐式状态传递。模型需要在这里记录对下一轮有用的紧凑信息，如重点牌型的出牌记录。此字段将被裁剪（默认不超过 512 字符）并放入下一回合的 Prompt 中。

### 4.2 动作投影与 Fallback 机制
*   **点击映射 (Hit Test)**: `DoudizhuSingleEnv._project_clicks_to_game_action` 会将模型的 `<action>` 坐标还原为实际屏幕像素坐标，并检测命中了哪些 UI 元素。
*   **合法性判定**: 如果点击转换成的出牌组合在当前游戏规则下不合法，或者解析 XML 完全失败（如缺少标签、坐标越界），系统会触发 **Fallback 机制** (`_fallback_action`)：通常表现为被迫点击 "PASS" 或者打出最小的合法牌以保证游戏能够继续，同时给予模型奖励惩罚。

---

## 5. 奖励机制设计 (Reward Shaping for RL)

Agentic RL 中，特别是在视觉-语言模型 (VLM) 的训练初期，纯粹的稀疏奖励（赢或输）很难驱动模型学会如何正确遵循 XML 格式并精确点击 UI。因此，代码库设计了**稠密子奖励与稀疏终点奖励结合**的机制。

在 `DoudizhuSingleEnv` 的初始化及 `step` 函数中，单步奖励 $R_t$ 由以下几个部分组成：

1.  **格式合规奖励 (`reward_projection`，默认 0.05)**: 只要模型严格遵循了四个 XML 标签格式并输出了合法的坐标数组，即给予此奖励。
2.  **精准点击奖励 (`reward_click`，默认 0.05)**: 乘以 `click_valid_ratio` (命中有效 UI 元素的点击数 / 总点击数)。鼓励模型不要去点击背景黑边或无效区域。
3.  **规则合法奖励 (`reward_rule_action`，默认 0.10)**: 当模型组成的点击序列形成了一手合法的斗地主出牌（即未触发 Fallback 机制）时给予该奖励。这促使模型去真正理解斗地主规则。
4.  **游戏胜负奖励 (Terminal Reward)**: 仅在环境触发 `done=True` 时结算。胜利（赢）获得 `reward_win` (默认 1.0)，失败（输）获得 `reward_loss` (默认 -1.0)。

总奖励公式大致为：
$Reward = (R_{proj} \times \mathbb{I}_{valid\_proj}) + (R_{click} \times Ratio_{valid\_click}) + (R_{rule} \times \mathbb{I}_{valid\_rule}) + R_{terminal}$

此外，在训练脚本中（如 GRPO），还会开启 `actor_rollout_ref.actor.use_projection_invalid_penalty=True`，对严重违规进行额外的 Loss 级惩罚。

---

## 6. RL 训练脚本集成 (GRPO 实例)

本环境被深度集成在 `verl` 的 PPO/GRPO 训练管线中。通过 `examples/grpo_trainer/run_doudizhu_qwen3vl.sh` 可以看出如何驱动这一环境：

*   **模型选型**: 示例使用的是 `Qwen/Qwen3-VL-4B-Instruct`，它原生支持图像输入和坐标理解，非常适合这种基于 GUI 的强化学习任务。
*   **环境配置**:
    *   `env.env_name=doudizhu`
    *   `env.rollout.n=$group_size` (对于 GRPO，每个 prompt / 初始状态会采集中一定数量的并发 rollout 轨迹以计算基线和 Advantage)。
    *   `data.max_prompt_length=1024`，`data.max_response_length=1024`：保证有足够长度的上下文去容纳 `<think>` 过程和多步 `<action>` 坐标。
*   **引擎加速**: `actor_rollout_ref.rollout.name=$ENGINE` (通常为 vLLM) 用于加速大视觉模型的在线生成速度，而 Ray 负责在 CPU 上并行跑多个斗地主环境以消除环境推演的等待。

### 训练的侧重点建议：
如果您的目标是改进基于此环境的 Agentic RL：
1.  **观察 `<think>` 质量**：早期模型可能会输出随机坐标，如果 `<think>` 逻辑正确但 `<action>` 点击错位，说明模型的空间坐标 Grounding 能力薄弱；如果连规则都分析错，则说明需要更多的 SFT 预热。
2.  **调整奖励权重 (Reward Shaping)**：如果在训练中发现模型倾向于“每次都 PASS 来逃避错误点击惩罚”，可能需要提高赢牌的权重 (`reward_win`) 或者适当降低非法点击的局部惩罚，鼓励其探索进攻性动作。
3.  **Memory 的利用率**：监控多轮次下模型是否真的读取了上一轮自己写入的 `<memory>`，可以通过在评测中注入特定伪造 memory 观察其行为变化来评估 Agent 长程一致性。

---

## 7. 环境潜在优化方向 (Optimization Opportunities)

在当前架构基础上，如果您想进一步提升 Agentic RL 的上限，可以从以下几个维度对环境进行深度优化：

### 7.1 观测与状态追踪 (POMDP 到 MDP 的过渡)
*   **痛点**: 目前的 Prompt 仅依赖 `<image>`（只显示最后一手牌）和由大模型自回归生成的 `{previous_memory}` 来维持上下文。这导致模型容易产生记忆幻觉 (Hallucination)，忘记早期出过的关键牌。
*   **优化建议**: 在环境中显式维护一个 `Textual Game Trace` (历史出牌记录文本)，并将其作为 Prompt 的一部分结构化地注入到观测中，以此减轻大模型的记忆负担，使其能够将算力集中在战术推理 (Reasoning) 上。

### 7.2 动作空间与 Grounding 机制
*   **痛点**: 模型目前需要输出如 `[140, 850]` 这样的绝对归一化坐标。对于不具备极强细粒度 Grounding 能力的视觉模型（如较小尺寸的模型），极其容易点偏，从而触发 Fallback。
*   **优化建议**: 
    *   **相对坐标或区域抽象**: 将屏幕划分为离散的网格或预定义的 UI 区域（例如 `<HandCard_3>`, `<PassButton>`），让模型输出更抽象的动作意图。
    *   **Fallback 机制改良**: 目前的 Fallback (`_fallback_action`) 逻辑是“如果非法则强制出最小合法牌或 Pass”。这可能会导致模型在探索初期陷入“随意输出乱码坐标以换取系统代打 (且大概率 Pass)”的局部最优。可以考虑让 Fallback 直接导致当前回合的严重失败，以强制模型学会精准出牌。

### 7.3 细粒度奖励设计 (Dense Reward Shaping)
*   **痛点**: 虽然目前有 `reward_click` 和 `reward_rule_action`，但在策略层面，只有终点的 `reward_win` 和 `reward_loss`。这对于长程游戏来说过于稀疏。
*   **优化建议**:
    *   **手牌衰减奖励 (Hand-depletion Reward)**: 每次成功合法出牌并减少手牌数量时，给予一个正向奖励 $r = + \alpha \times \Delta N_{cards}$。
    *   **夺得牌权奖励 (Tempo Control)**: 成功压制对手，赢得本轮出牌权，可给予一个小的正奖励。
    *   **高牌留存惩罚**: 在游戏结束时，如果手牌中还保留有大牌（如王、2）且输掉了游戏，可以增加额外的负向惩罚。

### 7.4 渲染与采样效率 (Sampling Efficiency)
*   **痛点**: 强化学习非常吃采样吞吐量。目前每个 Step 都需要调用 `renderer.render()` 实时生成新图片，对于大规模并发 Rollout 可能是 CPU/内存 的瓶颈。
*   **优化建议**: 
    *   在预训练阶段 (SFT/早期 RL) 可提供纯文本版本的斗地主环境（直接输出手牌与桌面的文本表示），以极高的吞吐量先教会大模型斗地主的逻辑规则。
    *   在后期再切换到视觉版本对齐 GUI Grounding，或者引入渲染缓存池机制。

### 7.5 对手智能与多智能体博弈 (MARL)
*   **痛点**: 当前的农民对手使用的是静态规则引擎 `DouDizhuRuleAgentV1`。规则模型通常存在固定套路，限制了 RL Agent 探索更高阶策略（如算牌、诱导）的空间。
*   **优化建议**: 扩展环境使其支持多 Agent (Multi-Agent RL) 共同推演，让三个玩家均为 LLM 并引入自对弈 (Self-Play) 或 PBT (Population-Based Training)，以此形成策略的自动课程学习 (Auto-Curriculum)。