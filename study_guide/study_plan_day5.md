# Day 5 保姆级落地指南

这份文档不是一份新的学习计划，而是对 [study_plan.md](/home/zhangwj/science/verl-agent/study_guide/study_plan.md:523) 里 Day 5 的逐步展开版。今天严格只做原计划里的两件事：

1. 把你对 `verl-agent` 的理解收敛成“我自己的 agent 接入模板”。
2. 完成 1 个最小但真实的项目预演，也就是 `dummy_gui_agent_mvp`。

今天的宗旨也只保留一句话：

**不要继续扩阅读面，不要继续追更高级算法，不要试图直接做斗地主。今天只做一个最小闭环，把你前 4 天学到的东西，压缩成一套以后可以复用的工程骨架。**

---

## 今天的最终目标，先钉死

今天结束时，你必须在本地拿到下面 4 个硬结果：

1. 一个最小自定义 GUI 风格 env/projection/reward 原型。
2. 一份你自己总结出来的新环境接入 checklist。
3. 一次真正跑通的短训练或至少完整 smoke test。
4. 一份能直接拿去讲给面试官听的 Day 5 讲稿提纲。

如果今天推进顺利，理想状态是：

1. 你已经在仓库里新增了一个 `dummy_gui_agent` 风格的最小环境包。
2. 模型可以接收“文本 + 图片”观测。
3. 模型输出必须包含：
   `<think>...</think><action>...</action><chat>...</chat><memory>...</memory>`
4. `projection` 只负责把 `<action>` 映射为环境动作，但会顺手检查四段结构是否齐全。
5. reward 至少包含：
   终局奖励 + 格式奖励 + 动作合法性奖励。
6. 下一轮 prompt 里，至少能带回最近 2 到 3 步的 `action/chat/memory` 历史。

如果 8 小时内没有把训练完整跑起来，也不要慌。今天的最低可接受线不是“收敛出好成绩”，而是下面这个闭环：

`自定义环境能 reset/step -> projection 能解析结构化输出 -> reward 能按你的设计记分 -> memory 能把最近历史塞回 prompt`

只要这四件事都亲手打通，你就已经从“会跑仓库”跨到了“会接自己的 agent 项目”。

---

## 为什么 Day 5 不直接做斗地主

这件事你一定要在脑子里想明白，不然今天很容易走偏。

今天不直接做斗地主 GUI，不是因为那个目标不对，而是因为它对 Day 5 来说信息量太大了：

1. 真 GUI 斗地主会把视觉解析、窗口控制、动作时序、规则引擎、多人对局状态、聊天上下文一次性捆在一起。
2. 一旦第一版就做真环境，你很难分辨到底是环境坏了、projection 坏了、reward 坏了，还是 rollout 链路没接对。
3. 你的真实目标不是“今天做完斗地主”，而是“从明天开始，我能以这套框架为底座，稳定推进自己的项目”。

所以今天只做“局部精心设计”，不做“宏观完美蓝图”。

这正对应你原始 prompt 里的那条哲学：

**先把一个局部闭环设计到可运行、可解释、可复用；更大的系统，会在后续这些局部闭环的基础上自然长出来。**

---

## 今天的唯一主线：`dummy_gui_agent_mvp`

为了避免你今天陷入“设计环境”本身，我们直接把原型任务定义死。

### 任务定义

你今天要做的 `dummy_gui_agent_mvp`，不是一个聪明环境，而是一个**足够像 GUI agent、但足够小**的环境。

我建议你把它实现成一个 3 步有限状态机：

1. `state_0 = lobby`
   正确动作是 `click[start_match]`
2. `state_1 = ready_room`
   正确动作是 `click[ready]`
3. `state_2 = turn_page`
   正确动作是 `click[end_turn]`

成功轨迹就是：

`click[start_match] -> click[ready] -> click[end_turn]`

环境结束条件：

1. 三步都做对，`done=True`，给 `env_reward=+1.0`
2. 达到 `max_steps` 还没完成，`done=True`，给 `env_reward=0.0` 或 `-1.0`

我更推荐你 Day 5 用：

1. 成功 `+1.0`
2. 失败 `0.0`
3. 中间过程都给 `0.0`

原因很简单：

1. 环境 reward 越简单，你越容易看清格式奖励和合法性奖励到底有没有工作。
2. 失败给 `0.0`，而不是大负数，可以避免 Day 5 第一版 reward 过于抖动。
3. 你后面做真项目时，再把失败惩罚和更细的 step reward 慢慢加进来。

### 观测定义

每一步环境都返回两部分：

1. 文本观测
2. 图片观测

文本观测建议包含：

1. 当前 screen 名称
2. 当前任务目标
3. 当前可用动作列表
4. 最近 2 到 3 步历史摘要

图片观测今天不要引入任何真实截图素材，直接返回不同颜色或不同图案的 `numpy uint8` 假图片即可。比如：

1. `lobby` 返回蓝色图
2. `ready_room` 返回绿色图
3. `turn_page` 返回橙色图

这样做完全够用，因为今天你要验证的是多模态链路接没接通，而不是视觉语义理解能力。

### 输出定义

今天强制模型输出这个结构：

```text
<think>...</think><action>...</action><chat>...</chat><memory>...</memory>
```

例子：

```text
<think>先从大厅进入对局，再准备并结束回合。</think><action>click[start_match]</action><chat>这把我先开，后面我继续配合你。</chat><memory>当前处于大厅，我准备点击开始匹配。</memory>
```

注意：

1. `projection` 只看 `<action>`，并校验四段标签是否存在。
2. `chat` 和 `memory` 不直接决定环境动作。
3. `memory` 今天不是“聪明记忆”，只是为下一轮 prompt 留下最小策略痕迹。

---

## 今天为什么要拿 `Sokoban` 当模板，而不是别的环境

这一步你必须明确，不要一上来乱抄目录。

我建议你以 [run_sokoban.sh](/home/zhangwj/science/verl-agent/examples/grpo_trainer/run_sokoban.sh:1)、[SokobanEnvironmentManager](/home/zhangwj/science/verl-agent/agent_system/environments/env_manager.py:245)、[sokoban_projection](/home/zhangwj/science/verl-agent/agent_system/environments/env_package/sokoban/projection.py:22) 为模板，而不是从 `AlfWorld` 或 `Webshop` 开始。

理由有 5 个：

1. `Sokoban` 本身就是视觉输入，更接近你未来 GUI agent 的“图片 + 文本”形态。
2. `Sokoban` 的动作空间离散、简单，很适合你先练“结构化输出 -> projection -> 合法动作”的链路。
3. 它的 [env_manager 分支](/home/zhangwj/science/verl-agent/agent_system/environments/env_manager.py:649) 很短，Day 5 更容易看全。
4. 它已经有一个与你 Day 5 非常接近的参考改法： [sokoban_projection_my_modification](/home/zhangwj/science/verl-agent/agent_system/environments/env_package/sokoban/projection.py:95)。
5. 它的训练脚本已经帮你把视觉模型、双卡、并行参数基本配好了，你不用再从零折腾一遍 trainer 命令。

一句话总结：

**今天你不是在学“环境多样性”，而是在学“如何把自己的环境接进这条主链”。Sokoban 是最省力的模板。**

---

## 今天的时间总表

严格按这个节奏推进。不要一上来就开始大改代码。

| 时间块 | 时长 | 目标 |
|---|---:|---|
| 0. 开场定界 | 20 分钟 | 明确今天只做什么，不做什么，锁定模板与文件边界 |
| 1. 上午第一段 | 70 分钟 | 选模板，画出 `dummy_gui_agent_mvp` 的最小接口草图 |
| 2. 上午第二段 | 90 分钟 | 搭自定义 env 骨架，先让 `reset/step` 跑通 |
| 3. 下午第一段 | 80 分钟 | 写 projection，并让结构化输出解析与动作映射跑通 |
| 4. 下午第二段 | 70 分钟 | 把最近 2 到 3 步 `action/chat/memory` 塞回 prompt |
| 5. 傍晚 | 70 分钟 | 在 reward 路径里加入格式奖励 + 合法性奖励 + 终局奖励 |
| 6. 晚上第一段 | 60 分钟 | 做 smoke test，先验证 env/projection/reward/memory 四件事 |
| 7. 晚上第二段 | 60 分钟 | 跑一次最小短训练，并记录现象 |
| 8. 收尾 | 20 分钟 | 整理交付物、checklist、面试讲稿与硬核拷问 |

总计：约 8 小时

---

## 开场 20 分钟：先把边界画清楚

### 动作 1：先重读 Day 5 原计划，只看这 4 段

预计用时：5 分钟

只看 [study_plan.md 的 Day 5](/home/zhangwj/science/verl-agent/study_guide/study_plan.md:523) 这四个位置：

1. “当日目标”
2. “最推荐的最终实战任务”
3. “今天建议你改的最小模块”
4. “今天的最终交付物”

你现在要用自己的话，在纸上写一句话：

`Day 5 不是为了读更多代码，而是为了沉淀一个最小自定义 agent 模板。`

如果这句话没写出来，今天很容易再次滑回“看源码模式”。

### 动作 2：把今天允许改动的文件先列出来

预计用时：5 分钟

我建议你今天只允许自己碰下面这些文件：

1. [agent_system/environments/env_manager.py](/home/zhangwj/science/verl-agent/agent_system/environments/env_manager.py:245)
2. [agent_system/reward_manager/episode.py](/home/zhangwj/science/verl-agent/agent_system/reward_manager/episode.py:20)
3. [agent_system/multi_turn_rollout/rollout_loop.py](/home/zhangwj/science/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:314)
4. [agent_system/memory/memory.py](/home/zhangwj/science/verl-agent/agent_system/memory/memory.py:19)
5. `agent_system/environments/env_package/dummy_gui_agent/*`

今天刻意不要碰：

1. `verl/models/*`
2. `critic` / `value` / `gae`
3. `verl/third_party/*`
4. `megatron`

这一步是在给你“防走神”。

### 动作 3：确定今天的实现策略

预计用时：10 分钟

你今天应该选下面这条最小路径，不要自己再重新发明路线：

1. 复制 `Sokoban` 的 env package 结构做模板
2. 自己写一个极小的 `DummyGUIEnv`
3. 在 `env_manager.py` 里新增一个 `DummyGUIEnvironmentManager`
4. 在 `make_envs()` 里注册 `dummy_gui_agent`
5. 写一个新的 `projection.py`
6. 在 `EpisodeRewardManager` 里加入复合奖励
7. 先用现有 visual 数据占位，不引入新数据集

你今天**不应该**做的 4 件事：

1. 不要上来先新建一堆 prompt 模板文件
2. 不要先做真截图、真 GUI 控制
3. 不要先追“奖励设计很优雅”
4. 不要先纠结“是不是该直接上 GiGPO”

---

## 上午第一段 70 分钟：把接口草图画出来

这一段不写代码。先把接口想清楚。

### Step 1：顺着真实主链，再过一遍今天会走到哪些接口

预计用时：20 分钟

你只看下面 5 个位置：

1. [prepare.py 的注释](/home/zhangwj/science/verl-agent/examples/data_preprocess/prepare.py:36)
2. [rollout_loop.py 中 `uid/traj_uid/is_projection_valid` 的注入位置](/home/zhangwj/science/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:314)
3. [EpisodeRewardManager.__call__](/home/zhangwj/science/verl-agent/agent_system/reward_manager/episode.py:29)
4. [apply_projection_invalid_penalty](/home/zhangwj/science/verl-agent/verl/trainer/ppo/ray_trainer.py:200)
5. [make_envs()](/home/zhangwj/science/verl-agent/agent_system/environments/env_manager.py:602)

你今天必须重新确认 4 个事实：

1. 数据集只是占位，不是今天的监督核心。
2. `env.rollout.n` 决定 group，和你的自定义环境直接相关。
3. `is_projection_valid` 会被 rollout loop 塞进 `non_tensor_batch`。
4. reward manager 看到的是整条 episode 聚合后的 `episode_rewards`。

如果这 4 件事没重新想清楚，你后面改 reward 和 projection 很容易手抖。

### Step 2：画一张 Day 5 专属草图

预计用时：20 分钟

你现在在纸上画这样一条链：

```text
占位 parquet
-> main_ppo
-> make_envs(dummy_gui_agent)
-> DummyGUIEnvironmentManager.reset()
-> TrajectoryCollector.multi_turn_loop()
-> actor_rollout_wg.generate_sequences()
-> dummy_gui_projection()
-> DummyGUIEnv.step()
-> EpisodeRewardManager()
-> GRPO advantage
-> update_actor
```

然后你在每个节点旁边各写一句：

1. 这个节点的输入是什么
2. 这个节点的输出是什么
3. 这里最可能翻车的点是什么

比如：

1. `projection`
   输入：模型原始字符串
   输出：环境动作 id + 合法性标记
   风险：四段标签缺失、`<action>` 提取失败、动作名拼写漂移
2. `reward manager`
   输入：`response_str`、`episode_rewards`、`is_projection_valid`
   输出：最后一个 response token 上的 score
   风险：重复计罚、格式解析与 projection 标准不一致

### Step 3：把今天的环境协议写死

预计用时：30 分钟

你现在必须把协议写成一页清单，不能边写边想。

建议你直接写成下面这版：

#### 3.1 环境动作集合

```text
click[start_match]
click[ready]
click[end_turn]
noop
```

#### 3.2 环境状态转移

1. `lobby` 收到 `click[start_match]` -> `ready_room`
2. `ready_room` 收到 `click[ready]` -> `turn_page`
3. `turn_page` 收到 `click[end_turn]` -> 成功结束
4. 其它动作 -> 状态不变

#### 3.3 模型输出协议

必须同时出现：

1. `<think>...</think>`
2. `<action>...</action>`
3. `<chat>...</chat>`
4. `<memory>...</memory>`

#### 3.4 历史记忆协议

只保留最近 3 步，每步只保存：

1. 动作
2. 聊天
3. memory 字段

#### 3.5 奖励协议

1. `env_reward`
   成功 `+1.0`
   失败 `0.0`
2. `format_reward`
   四段标签齐全 `+0.05`
   否则 `0.0`
3. `legality_reward`
   合法 `+0.05`
   非法 `-0.05`

这张协议纸非常重要。它是你今天后续所有代码动作的统一标准。

---

## 上午第二段 90 分钟：先把自定义 env 骨架搭起来

这段开始写代码，但你先记住一句话：

**先让环境自己跑起来，再考虑训练；先让 reset/step 正确，再考虑 reward 漂不漂亮。**

### Step 4：先创建最小 env package

预计用时：20 分钟

你今天建议新增的目录和文件是：

1. `agent_system/environments/env_package/dummy_gui_agent/__init__.py`
2. `agent_system/environments/env_package/dummy_gui_agent/envs.py`
3. `agent_system/environments/env_package/dummy_gui_agent/projection.py`

你现在就去对照 [sokoban/__init__.py](/home/zhangwj/science/verl-agent/agent_system/environments/env_package/sokoban/__init__.py:1) 和 [sokoban/envs.py](/home/zhangwj/science/verl-agent/agent_system/environments/env_package/sokoban/envs.py:1)。

今天最小要求是：

1. `__init__.py` 能导出 `build_dummy_gui_envs` 和 `dummy_gui_projection`
2. `envs.py` 里能构造并行环境
3. `projection.py` 里能解析结构化字符串

不要在这一步加别的文件。

### Step 5：在 `envs.py` 里先做一个极简 `DummyGUIEnv`

预计用时：35 分钟

这一段的目标，不是把环境写得“像正式产品一样完美”，而是做出一个**内核清晰、接口干净、后面真能扩成斗地主 GUI 环境**的最小版本。

这里我建议你采用一条很稳的路线：

1. `DummyGUIEnv` 写成“轻量标准 env 内核”
2. `Ray worker` 负责并行化
3. `DummyGUIMultiProcessEnv` 负责批量 `reset/step`

也就是说，**最里面的环境尽量像 `gym.Env`，最外面继续服从 `verl-agent` 当前的并行包装方式。**

这样做有两个好处：

1. 对 Day 5 来说，最小可行，接仓库省力。
2. 对你后面的斗地主 MVP 来说，这个环境内核以后还能继续复用，不会只是一段“一次性调通代码”。

#### 5.0 先把这一层的职责划清楚

在动手之前，你先记住这 3 层分工：

1. `DummyGUIEnv`
   只负责单个环境实例自己的状态转移。
2. `DummyGUIWorker`
   只负责在 Ray actor 里持有一个 `DummyGUIEnv`。
3. `DummyGUIMultiProcessEnv`
   只负责把很多 worker 组织成批量接口。

今天最容易犯的错，就是把这 3 层写混。

如果你把所有逻辑都堆进并行包装器里，短期也许能跑，但后面你要调试单环境、做规则 agent、做离线回放时会非常痛苦。

#### 5.1 `DummyGUIEnv` 最好写成什么样

我的建议是：

**写成“接近标准 `gym.Env` 的普通 Python 类”，但不要为了追求 gym 教科书规范而额外花太多时间。**

Day 5 你至少做到这些就够了：

1. 类内部状态明确
2. 有 `reset()`
3. 有 `step()`
4. 最好再有一个 `render()`，专门返回假图像

如果你愿意，可以顺手定义：

1. `action_space`
2. `observation_space`

但这不是 Day 5 的硬要求。  
当前 `verl-agent` 真正依赖的，还是你 `reset()` 和 `step()` 的返回值形状，而不是 gym 全家桶。

#### 5.2 这个基础环境类应该有哪些成员

今天这个 `DummyGUIEnv`，建议最少保留下面 4 个核心成员：

1. `self.state_id`
2. `self.step_count`
3. `self.max_steps`
4. `self.script`

你也可以加一个很有用的成员：

5. `self.available_actions`

这样后面 `reset()` 和 `step()` 里就不用重复硬编码动作列表。

其中 `self.script` 可以直接写死成：

```python
[
    {"screen": "lobby", "target_action": "click[start_match]"},
    {"screen": "ready_room", "target_action": "click[ready]"},
    {"screen": "turn_page", "target_action": "click[end_turn]"},
]
```

这个 `script` 的含义非常简单：

1. 它不是“大型任务配置系统”
2. 它不是“复杂 GUI 状态机”
3. 它只是今天这条成功轨迹的唯一真值来源

所以你今天后面写的所有逻辑，都应该围绕它来展开。

#### 5.3 为什么今天要把环境写成“有限状态机”

因为 Day 5 的首要任务不是“拟真”，而是“可解释”。

写成有限状态机有 4 个直接好处：

1. 你知道当前页面到底是哪一页。
2. 你知道当前正确动作到底是什么。
3. 你一眼就能判断 step 转移是否正确。
4. 你后面接 reward 时，不会搞不清环境 reward 为什么是这个数。

这就是 Day 5 的黄金甜蜜点：

1. 足够像 GUI agent
2. 足够小，方便调试

#### 5.4 `reset()` 这一步到底要做什么

你今天的 `reset()`，逻辑上只做下面 5 件事：

1. 把 `state_id` 归零
2. 把 `step_count` 归零
3. 根据 `state_id=0` 生成当前 screen 的假图像
4. 根据 `state_id=0` 生成当前 screen 的文本描述
5. 返回 `obs, info`

这里最重要的是，你要先想清楚 **`obs` 长什么样**，再去写代码。

我建议 Day 5 的单环境 `obs` 就设计成：

1. 一个图像对象
2. 一个文本描述

但为了兼容你后面在 `EnvironmentManager` 里自己拼装 `text/image/anchor`，这里的 `DummyGUIEnv` 本体可以先保持简单：

1. `reset()` 返回原始图像数组 `obs`
2. 文本描述放进 `info`

也可以反过来：

1. `reset()` 返回一个字典 `{"image": ..., "text": ...}`

两种都行。  
Day 5 更重要的是：**你自己要统一，不要一会儿返回数组，一会儿返回 dict。**

`info` 我建议最少包含这 3 个 key：

1. `won`
2. `available_actions`
3. `screen_name`

如果你想让后面 `EnvironmentManager` 更省事，我还建议多给一个：

4. `text_obs`

也就是把当前页面对应的自然语言描述直接塞进 `info["text_obs"]`。

这样你后面在 manager 里更容易拼 prompt。

其中 `available_actions` 最好先固定成：

```python
["click[start_match]", "click[ready]", "click[end_turn]", "noop"]
```

Day 5 不要追求“不同页面显示不同动作集合”。  
统一动作集合更利于你排查：

1. projection 是否稳定
2. 合法性判定是否稳定
3. reward 是否稳定

#### 5.5 `reset()` 里文本描述应该怎么写

不要把这件事想复杂。

你今天的文本描述只要能清楚回答 3 个问题就够了：

1. 当前在哪个 screen
2. 当前目标是什么
3. 当前允许什么动作

比如 `lobby` 可以写成：

```text
Current screen: lobby.
Goal: start the match by clicking the start button.
Available actions: click[start_match], click[ready], click[end_turn], noop.
```

你不需要一开始就把 prompt 写得像最终产品那样花哨。  
Day 5 只要做到“语义清楚、结构稳定”。

#### 5.6 `step(action)` 到底要做什么

`step()` 也只做下面 6 件事：

1. `step_count += 1`
2. 取出当前状态对应的 `target_action`
3. 判断传入动作是否等于 `target_action`
4. 如果正确且不是最后一步，就推进到下一个状态
5. 如果正确且已经是最后一步，就成功结束
6. 如果步数用尽还没成功，就失败结束

把它翻译成更具体的判断规则，就是：

1. 当前在 `lobby`，收到 `click[start_match]`
   进入 `ready_room`
2. 当前在 `ready_room`，收到 `click[ready]`
   进入 `turn_page`
3. 当前在 `turn_page`，收到 `click[end_turn]`
   成功结束，给 `reward=1.0`
4. 其它动作
   不推进状态，给 `reward=0.0`
5. 如果 `step_count >= max_steps` 且还没成功
   失败结束，给 `reward=0.0`

这里请注意一个非常重要的边界：

**今天不要在环境里写“非法动作惩罚”。**

原因是：

1. 环境本体只负责状态转移和终局 reward
2. 合法性奖励和格式奖励应该放在 reward manager 里统一做
3. 这样以后你看分数时，归因才清楚

否则你很容易出现：

1. 环境里扣一遍
2. reward manager 再扣一遍
3. 结果自己都不知道到底是哪层在起作用

#### 5.7 `step()` 返回的 `info` 至少要带什么

和 `reset()` 一样，我建议你最少保持下面这些字段：

1. `won`
2. `available_actions`
3. `screen_name`
4. `text_obs`

其中：

1. `won`
   成功时为 `1.0`
   否则为 `0.0`
2. `screen_name`
   方便你在外层观测里知道现在页面切到哪了
3. `text_obs`
   方便 `EnvironmentManager` 下一轮直接读

你甚至还可以多放一个：

5. `target_action`

不过这个字段更多是 debug 用。  
如果你怕“泄题感”太强，也可以不放。

#### 5.8 假图像怎么做最合理

最简单做法仍然是：

1. `lobby` 返回蓝底图
2. `ready_room` 返回绿底图
3. `turn_page` 返回橙底图

尺寸统一成例如 `96 x 96 x 3` 的 `np.uint8` 即可。

你今天完全不需要：

1. 真截图
2. OCR
3. 图标素材
4. GUI 控件检测

假图像在 Day 5 的意义只有一个：

**确认“视觉输入 -> processor -> 模型 -> rollout”这条多模态路径是通的。**

也就是说，今天的图像不是为了“让模型真正看懂画面”，而是为了确认这个系统从接口上已经具备了接视觉任务的能力。

#### 5.9 一个很关键的工程提醒

今天请把 `DummyGUIEnv` 当成一个“单环境内核”，而不是直接当成“训练环境总入口”。

你现在写它的时候，脑子里要一直分清两件事：

1. 单环境状态转移逻辑
2. 多环境并行采样逻辑

前者属于 `DummyGUIEnv`，后者属于下一步的 `DummyGUIWorker` 和 `DummyGUIMultiProcessEnv`。

这层边界一旦守住，你后面做斗地主 GUI 环境时就会很舒服，因为：

1. 你可以单独调一个环境实例
2. 你可以单独验证规则逻辑
3. 你可以先写 rule-based agent 和它交互
4. 等单环境稳定后，再并行化

#### 5.10 这一小步的过关标准

这一小步结束时，你应该能不看代码，直接口头回答下面 4 个问题：

1. 这个环境当前一共有几个状态，它们分别是什么？
2. 成功轨迹的动作序列是什么？
3. `reset()` 和 `step()` 各自会返回哪些信息？
4. 为什么 Day 5 要把非法动作惩罚放到 reward manager，而不是先写进环境？

如果这 4 个问题你都能说清楚，说明这一步你不是“抄了一个环境”，而是真的把环境内核想明白了。

### Step 6：复用 Sokoban 的并行包装方式

预计用时：35 分钟

你不需要重新设计并行框架，直接沿用 [SokobanMultiProcessEnv](/home/zhangwj/science/verl-agent/agent_system/environments/env_package/sokoban/envs.py:38) 这套模式：

1. 每个 Ray worker 持有一个独立 `DummyGUIEnv`
2. `build_dummy_gui_envs(seed, env_num, group_n, ...)` 返回并行环境对象
3. `reset()` 时同一个 group 内复用同一 seed

你今天必须保住的接口形状：

1. `reset()` 返回 `obs_list, info_list`
2. `step(actions)` 返回 `obs_list, reward_list, done_list, info_list`

只要这个形状对了，`EnvironmentManager` 那层就能接上。

#### 6.1 为什么这里一定要保留 `group_n`

因为 [make_envs()](/home/zhangwj/science/verl-agent/agent_system/environments/env_manager.py:609) 会把 `env.rollout.n` 传下去，而 [rollout_loop.py](/home/zhangwj/science/verl-agent/agent_system/multi_turn_rollout/rollout_loop.py:314) 是按这个 group 构造 `uid` 的。

你今天这个 dummy env 也必须遵守同一组协议，否则你学到的是“另一个私有玩具系统”，不是可迁移的 `verl-agent` 接入方法。

#### 6.2 这一段最容易犯的错

1. 返回 shape 不一致
2. `obs_list` 长度和 batch 对不上
3. `reset()` 忘了为每个 worker 都返回独立 `info`
4. `done=True` 之后还继续错误推进状态

这一步的心法是：

**别急着接训练，先把环境对象当成一个普通 Python 模块单独验证。**

---

## 下午第一段 80 分钟：写 projection，把结构化输出真正钉住

今天的 `projection` 是 Day 5 的核心之一。

你要一直提醒自己：

**projection 的职责不是“理解整个回复”，而是“从结构化回复里稳定、可解释地抽出动作”。**

### Step 7：明确 projection 的边界

预计用时：10 分钟

先对照 [sokoban_projection](/home/zhangwj/science/verl-agent/agent_system/environments/env_package/sokoban/projection.py:22) 和 [sokoban_projection_my_modification](/home/zhangwj/science/verl-agent/agent_system/environments/env_package/sokoban/projection.py:95)。

然后你把今天自己的 projection 边界写成一句话：

`projection 只输出 env action id 和 valid 标记，不负责记忆管理，不负责奖励计算。`

### Step 8：给 `dummy_gui_projection` 定协议

预计用时：25 分钟

你今天的 `projection.py` 最少要有两个函数：

1. `extract_tag(text, tag)`
2. `dummy_gui_projection(actions: List[str])`

更推荐你加第三个函数：

3. `parse_structured_response(text)`

这样做的好处是：

1. `projection` 自己能用
2. `EnvironmentManager` 也能复用同一个解析逻辑去拼 memory

#### 8.1 `parse_structured_response(text)` 该返回什么

建议返回一个字典：

```python
{
    "think": "...",
    "action": "...",
    "chat": "...",
    "memory": "...",
    "has_all_tags": True or False,
}
```

#### 8.2 `dummy_gui_projection()` 该做什么

只做这 4 步：

1. 解析四段标签
2. 检查四段是否齐全
3. 把 `<action>` 映射到动作 id
4. 产出 `valids`

动作池建议直接写成：

```python
{
    "click[start_match]": 0,
    "click[ready]": 1,
    "click[end_turn]": 2,
    "noop": 3,
}
```

#### 8.3 何时判 `valid=1`

我建议 Day 5 用下面这个标准，不要含糊：

同时满足才算合法：

1. `<think>` 存在
2. `<action>` 存在
3. `<chat>` 存在
4. `<memory>` 存在
5. `<action>` 的内容在动作池中

只要有任何一项不满足，就：

1. `valid=0`
2. 环境动作强制映射为 `noop`

这样好处很大：

1. 规则单一
2. 排错容易
3. 可以稳定统计 `projection_valid_ratio`

### Step 9：不要把 projection 写得过度宽容

预计用时：15 分钟

今天常见的错误是：

1. 想“先让它都过掉”
2. 结果判定规则太松，最后你根本不知道模型到底学会了什么

Day 5 的正确策略是：

1. 对结构严格
2. 对动作池严格
3. 对额外文本宽容

具体来说：

1. 允许 `<think>` 很长
2. 允许 `<chat>` 很长
3. 允许 `<memory>` 很长
4. 但 `<action>` 必须精确落在小动作池内

这和你未来 GUI 斗地主项目的逻辑是一致的：

**自由表达可以宽，真正驱动 GUI 的动作接口必须窄。**

### Step 10：这一段的过关标准

预计用时：30 分钟

这一段结束时，你至少要手工拿 6 条字符串去喂 projection：

1. 四段都齐 + 合法动作
2. 缺 `<think>`
3. 缺 `<chat>`
4. 缺 `<memory>`
5. `<action>` 不在动作池
6. 四段都有，但顺序略有变化

你要明确写出每一条期望得到的：

1. `parsed action`
2. `env action id`
3. `valid`

只有当这 6 条都符合预期，才进入下一段。

---

## 下午第二段 70 分钟：把最近 2 到 3 步历史塞回 prompt

这一段非常重要，因为它决定你是不是已经从“单步 RL”走到了“真正的 agent loop”。

### Step 11：先看清现在默认 memory 的能力边界

预计用时：15 分钟

重新看一遍 [SimpleMemory](/home/zhangwj/science/verl-agent/agent_system/memory/memory.py:19)。

你今天必须看清 3 件事：

1. 它本质上只是“按 env 存储历史记录”
2. `store()` 能存任意 key
3. `fetch()` 默认只会拼 `obs_key` 和 `action_key`

也就是说，今天你完全没必要一开始就大改 memory 框架。

### Step 12：Day 5 最推荐的最小做法

预计用时：25 分钟

为了最小化改动，我建议你**先不要重写整个 memory 类**，而是直接复用 `SimpleMemory`，但把 `action/chat/memory` 打包成一个字符串存进去。

比如在 `DummyGUIEnvironmentManager.step()` 中，你可以把每一条历史写成：

```text
ACTION=click[ready] | CHAT=这回合我先稳一下 | MEMORY=我已经进入 ready_room
```

然后继续调用 `self.memory.store({"text_obs": ..., "action": packed_history})`

这样会有 3 个好处：

1. 你不用破坏现有 `SimpleMemory` 接口
2. 你不用为了 Day 5 再加一整套新 memory 类
3. 你已经足够验证“最近历史拼回 prompt”这个核心能力

如果你非常想做得更工整，可以在 [memory.py](/home/zhangwj/science/verl-agent/agent_system/memory/memory.py:19) 里追加一个 `DummyGUIMemory`，但这不是 Day 5 必需项。

### Step 13：在 `EnvironmentManager` 里构造下一轮文字观测

预计用时：30 分钟

你现在参考 [SokobanEnvironmentManager.build_text_obs()](/home/zhangwj/science/verl-agent/agent_system/environments/env_manager.py:308) 的写法，新增一个 `DummyGUIEnvironmentManager`。

这个 manager 至少要有 3 个方法：

1. `reset()`
2. `step()`
3. `build_text_obs()`

#### 13.1 `reset()` 做什么

1. 调 env `reset()`
2. 初始化 `SimpleMemory`
3. 构造第一轮 prompt 文本
4. 返回 `{"text": ..., "image": ..., "anchor": ...}, infos`

#### 13.2 `step()` 做什么

1. 先调用 `dummy_gui_projection(text_actions)`
2. 再把动作送进环境
3. 再把每个样本的 `is_projection_valid` 塞到 `infos`
4. 再把当前回复里的 `action/chat/memory` 打包进 `SimpleMemory`
5. 再调用 `build_text_obs()` 生成下一轮文本观测

#### 13.3 `build_text_obs()` 里应该放什么

我建议你就放下面 5 块，不要再发散：

1. 当前 screen 名称
2. 当前目标
3. 当前可用动作列表
4. 最近 2 到 3 步历史
5. 结构化输出格式要求

你今天可以把 prompt 模板直接写在 `build_text_obs()` 里，示意如下：

```text
你正在一个模拟 GUI 环境中执行任务。

Current screen: {screen_name}
Goal: 按正确顺序完成 start_match -> ready -> end_turn
Available actions:
- click[start_match]
- click[ready]
- click[end_turn]
- noop

Recent history:
{history}

请严格输出：
<think>...</think><action>...</action><chat>...</chat><memory>...</memory>
```

Day 5 不需要为了“代码优雅”再拆一个 prompts 文件。今天先保证闭环。

#### 13.4 这一段真正的目标

不是“memory 写得多高级”，而是：

**下一轮 prompt 里真的出现了上一轮的行为痕迹。**

这件事一旦通了，你就已经有了未来 GUI 斗地主项目里“最近几步操作、聊天、记牌思路回灌到上下文”的原型能力。

---

## 傍晚 70 分钟：把 reward 路径改成你自己的复合奖励

这一段是 Day 5 的另一核心。

你今天一定要抓住一句话：

**格式奖励和合法性奖励不是“点缀”，而是你未来 GUI agent 早期训练里最重要的稠密信号。**

### Step 14：先搞清楚 reward 现在长什么样

预计用时：10 分钟

重新看 [EpisodeRewardManager.__call__](/home/zhangwj/science/verl-agent/agent_system/reward_manager/episode.py:29)。

你要确认它现在在做什么：

1. decode `response_str`
2. 取 `episode_rewards`
3. 把分数写到最后一个 response token 上

你还要再看一眼 [apply_projection_invalid_penalty](/home/zhangwj/science/verl-agent/verl/trainer/ppo/ray_trainer.py:200)。

因为你今天如果自己又写了合法性奖励，同时又保留默认 invalid penalty，就会出现**双重计分**。

### Step 15：Day 5 的推荐做法

预计用时：10 分钟

我建议你今天直接采用这套策略：

1. 你自己的 reward manager 负责：
   `env_reward + format_reward + legality_reward`
2. 同时把默认 invalid penalty 关掉

也就是说，训练命令里要显式加：

```bash
actor_rollout_ref.actor.use_projection_invalid_penalty=False
```

这样做的工程意义很大：

1. 归因单一
2. 现象可解释
3. 日志更容易读

### Step 16：把 Day 5 reward 公式写死

预计用时：15 分钟

你今天就用最简单的一版：

```text
final_reward = env_reward + format_reward + legality_reward
```

其中：

1. `env_reward`
   直接取环境给的 `episode_rewards`
2. `format_reward`
   四段标签都在 -> `+0.05`
   否则 -> `0.0`
3. `legality_reward`
   `is_projection_valid=True` -> `+0.05`
   否则 -> `-0.05`

为什么今天不要把 `format_reward` 设成很大：

1. 如果设太大，模型可能只学会“凑格式”
2. 你就很难观察终局奖励有没有影响
3. Day 5 目标是闭环，不是追求最优 reward 系数

### Step 17：把 reward 分解信息打出来

预计用时：20 分钟

今天你不能只算 `final_reward`，你必须把分解项留下来。

你应该让 `return_dict=True` 时至少返回这些 key：

1. `env_reward`
2. `format_reward`
3. `legality_reward`
4. `final_reward`
5. `has_all_tags`
6. `is_projection_valid`

这是因为 [ray_trainer.py](/home/zhangwj/science/verl-agent/verl/trainer/ppo/ray_trainer.py:1199) 会把 `reward_extra_infos_dict` 打到训练日志里。

如果你不把分解项留下，后面你就只能看总分瞎猜。

### Step 18：这一段的过关标准

预计用时：15 分钟

这一段结束时，你至少要能回答：

1. 现在一个样本的总分由哪三部分组成
2. 合法性奖励是不是只在你自己的 reward manager 里算了一次
3. 为什么最终还是写回最后一个 response token

如果你答不出来，就先不要进训练。

---

## 晚上第一段 60 分钟：先做 smoke test，不要一上来跑训练

这是 Day 5 非常关键的纪律。

**动态调试优先。先 smoke test，后训练。**

### Step 19：先做环境级 smoke test

预计用时：20 分钟

你今天应该先验证下面 4 件事，而不是直接跑 `main_ppo`：

1. `build_dummy_gui_envs()` 能正常构造对象
2. `reset()` 返回的 `obs_list` 和 `info_list` 长度正确
3. `step()` 能接动作 id 并返回 reward/done/info
4. 正确动作序列会得到成功终局

你完全可以写一个极小测试脚本或临时交互命令，检查：

1. `obs[0]` 的图像 shape
2. `info[0]["available_actions"]`
3. 3 步正确动作后的 `done=True`
4. 成功时 `info["won"] == 1.0`

这一小步通过后，你已经证明环境本体没有大问题。

### Step 20：再做 projection 级 smoke test

预计用时：15 分钟

现在拿下面两条字符串分别测试：

```text
<think>先开始匹配</think><action>click[start_match]</action><chat>我先开局</chat><memory>大厅准备开始</memory>
```

```text
<think>先开始匹配</think><action>bad_action</action><chat>我先开局</chat><memory>大厅准备开始</memory>
```

你必须明确看到：

1. 第一条 `valid=1`
2. 第二条 `valid=0`
3. 第二条会被映射成 `noop`

### Step 21：再做 reward 级 smoke test

预计用时：25 分钟

你现在至少手工构造两类回复，观察 reward 组件：

1. 四段齐全 + 合法动作 + 成功终局
2. 缺字段或动作非法 + 失败终局

你要检查的不是“分高不高”，而是**分解项是否符合你自己的规则**。

最笨但最可靠的检查方法：

1. 在 reward manager 里临时打印少量样本
2. 把 `response_str`、`env_reward`、`format_reward`、`legality_reward`、`final_reward` 一起打印
3. 手工对照

如果这一步对不上，不要进入训练。

---

## 晚上第二段 60 分钟：跑一次真正的最小短训练

只有在上面三个 smoke test 都过了之后，你才开始这一步。

### Step 22：继续复用 visual 占位数据，不要新造数据

预计用时：10 分钟

你今天的数据依然可以复用 [prepare.py](/home/zhangwj/science/verl-agent/examples/data_preprocess/prepare.py:24) 生成的 visual parquet。

原因就是它在 [注释里已经说得很清楚](/home/zhangwj/science/verl-agent/examples/data_preprocess/prepare.py:36)：

1. 数据本身不是核心
2. 它只是告诉系统“这是视觉任务”和“batch 有多大”

如果你本地还没有一个足够小的 visual 数据，可以先执行：

```bash
conda activate verl-agent-bw
cd /home/zhangwj/science/verl-agent

python3 -m examples.data_preprocess.prepare \
  --mode visual \
  --train_data_size 8 \
  --val_data_size 16
```

### Step 23：不要急着新建脚本，先直接用一条临时命令

预计用时：10 分钟

Day 5 更推荐你先直接抄 [run_sokoban.sh](/home/zhangwj/science/verl-agent/examples/grpo_trainer/run_sokoban.sh:17) 的核心参数，先用临时命令把第一轮跑起来。

建议你先用下面这组“小步快跑”配置：

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=$HOME/data/verl-agent/visual/train.parquet \
  data.val_files=$HOME/data/verl-agent/visual/test.parquet \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  data.max_prompt_length=1024 \
  data.max_response_length=160 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.image_key=images \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.use_projection_invalid_penalty=False \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  algorithm.use_kl_in_reward=False \
  env.env_name=dummy_gui_agent \
  env.seed=0 \
  env.max_steps=4 \
  env.rollout.n=4 \
  env.resources_per_worker.num_cpus=0.1 \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=day5_dummy_gui \
  trainer.experiment_name=day5_dummy_gui_mvp \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 \
  trainer.val_before_train=True
```

### Step 24：这轮训练你只观察 6 个现象

预计用时：20 分钟

不要被海量日志淹没。你只盯下面 6 件事：

1. 程序有没有在 `make_envs(dummy_gui_agent)` 处成功进入你的环境分支
2. 有没有报 `obs size` 和 batch 对不上
3. `episode/projection_valid_ratio` 是否不是恒为 0
4. 你新增的 `reward_extra_info` 是否真的被打印或记录
5. 终局成功率是否不是完全随机死水
6. 训练有没有因为多模态输入或字符串解析而直接挂掉

### Step 25：如果挂了，优先排查顺序

预计用时：20 分钟

这是今天最关键的工程习惯之一。

如果训练挂了，按下面顺序排查，不要乱跳：

1. `make_envs()` 有没有注册到你的 `dummy_gui_agent`
2. `build_dummy_gui_envs()` 的 `reset/step` 返回值长度是否正确
3. `EnvironmentManager.reset()` 输出的 `text/image/anchor` 三个字段是否齐全
4. `projection` 是否返回了“动作列表 + valids”
5. `is_projection_valid` 是否被正确放进 `infos`
6. reward manager 里 decode 的 `response_str` 是否真的是你以为的那种格式

这是 Day 5 你最应该养成的定位习惯。

---

## 收尾 20 分钟：把今天的成果变成以后能复用的资产

今天最后 20 分钟，必须完成，不可省略。

### Step 26：写出你的“新环境接入 checklist”

预计用时：8 分钟

我建议你直接写成下面这 9 条：

1. 先定义动作协议和成功轨迹，不要先写代码
2. 先写极小环境本体，单独验证 `reset/step`
3. 再写并行包装，保证输出 shape 和长度正确
4. 再写 `projection`，只做结构解析和动作映射
5. 再写 `EnvironmentManager`，负责观测拼装、memory 注入、`is_projection_valid` 写回
6. 再在 `make_envs()` 里注册
7. 再在 reward manager 里加规则奖励
8. 先做 env/projection/reward 三层 smoke test
9. 最后才跑最小短训练

你以后接真 GUI 斗地主环境，基本就是照这个 checklist 再走一遍。

### Step 27：整理今天的 4 份交付物

预计用时：6 分钟

今天你应该明确保存下面 4 份东西：

1. 一张你自己画的 `verl-agent` Day 5 主链图
2. 一份 `dummy_gui_agent_mvp` 的环境协议纸
3. 一份新环境接入 checklist
4. 一份面试讲稿提纲

### Step 28：把今天的工作翻译成面试语言

预计用时：6 分钟

你今天结束时，至少要练熟下面 4 段话：

1. `verl-agent` 不是另起炉灶，而是在 `verl` 的 trainer 上插入环境、多轮 rollout、memory、projection 和 reward。
2. 我没有直接做真 GUI 斗地主，而是先做了一个最小 dummy GUI 环境，目的是把结构化输出、合法动作判定、多模态输入和规则奖励闭环先打通。
3. 对早期 GUI agent 训练来说，格式奖励和合法性奖励往往比胜负奖励更早产生价值，因为它们更稠密、credit assignment 更短。
4. 我已经把未来项目最关键的 5 个改动点识别出来了：
   `env_manager`、`projection`、`memory`、`reward_manager`、`rollout_loop`

---

## Day 5 最终验收标准

如果你今天做完之后，下面 8 条里能稳定满足 6 条以上，Day 5 就算成功：

1. 你能清楚解释为什么今天选 `Sokoban` 当模板
2. 你能清楚解释为什么 dummy env 要先做成有限状态机
3. 你能说清 `projection`、`memory`、`reward` 各自的边界
4. 你能说清为什么 Day 5 要先关掉默认 invalid penalty
5. 你能在纸上画出自定义环境接入主链
6. 你已经留下一个最小 env/projection/reward 原型
7. 你至少做过完整 smoke test
8. 你能把今天的 dummy GUI 原型映射到未来的 GUI 斗地主项目

---

## Day 5 面试硬核拷问

下面这些问题，不是让你“背答案”，而是检查你今天是不是真的学透了。

### 1. 如果让你把一个全新的 GUI 环境接入 `verl-agent`，你会按什么顺序动手？

你回答时至少要覆盖：

1. 先定义动作协议和成功标准
2. 再做环境 `reset/step`
3. 再做 projection
4. 再做 memory 与 prompt 拼装
5. 再做 reward
6. 最后才是训练

### 2. 为什么你会优先先做“结构输出稳定 + 动作合法性奖励”，而不是直接追求高胜率？

你回答时至少要覆盖：

1. 胜负奖励稀疏
2. 早期模型连协议都不稳定
3. 合法动作奖励更稠密
4. 对 GUI agent，动作合法通常是胜率的前置条件

### 3. 对你的项目，`memory` 最好放在 prompt 层、latent 层、还是外部状态层？为什么？

Day 5 版本你至少要答出：

1. MVP 阶段先放 prompt 层和外部状态层最现实
2. 先靠外部可控 memory 打通闭环
3. latent memory 不是 Day 5 的工程优先级

### 4. 你会如何定义斗地主 GUI agent 的 group，才能让 GRPO 真正有意义？

你回答时至少要覆盖：

1. group 里的样本必须共享同一个初始局面
2. 但 rollout 过程要允许策略分化
3. 如果 group 内 reward 全同，GRPO 学习信号会很弱

### 5. 如果训练不收敛，你会优先排查 prompt、projection、reward、group 构造、还是模型容量？为什么？

Day 5 的推荐答法顺序是：

1. 先查 projection
2. 再查 reward
3. 再查 group
4. 再查 prompt
5. 最后才查模型容量

因为前面几项更基础，也更常见。

---

## 最后一句提醒

今天最有价值的成果，不是你把 `dummy_gui_agent_mvp` 做得多漂亮，而是你亲手完成了下面这个动作：

**我定义了一个最小 agent 任务协议 -> 我把它接入 `verl-agent` 的环境、多轮 rollout、reward 链路 -> 我验证了它真的能跑。**

这就是你从“会使用 RL 框架”走向“能做自己的 agentic RL 项目”的分水岭。
