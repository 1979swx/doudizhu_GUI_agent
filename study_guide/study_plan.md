# verl-agent 5 天高强度学习计划

面向对象：已经会用 `verl` 做基础 RL 训练，但对 agentic RL 工程链路还不熟；目标是在 5 天内真正吃透 `verl-agent` 的主干，并且能自然过渡到你后续的“多模态 GUI 斗地主陪玩 Agent”项目。

---

## 先讲结论：这 5 天该抓什么，不该抓什么

### 这个仓库最值得你学的主干

你现在最该学的是下面这条真实执行链，而不是按目录扫源码：

`examples/grpo_trainer/run_*.sh`
-> `verl.trainer.main_ppo`
-> `agent_system.environments.make_envs`
-> `agent_system.multi_turn_rollout.TrajectoryCollector.multi_turn_loop`
-> `actor_rollout_wg.generate_sequences`
-> `envs.step`
-> `agent_system.reward_manager.EpisodeRewardManager`
-> `verl.trainer.ppo.ray_trainer.compute_advantage`
-> `actor_rollout_wg.update_actor`

这条链里最关键的 6 个认知点：

1. 这个仓库的 agentic RL 不是“另起炉灶”，而是在 `verl` 的 PPO/GRPO trainer 上，插入了环境、多轮 rollout、记忆、动作投影、回合奖励。
2. 真正的 GRPO 分组不是用 `actor_rollout_ref.rollout.n`，而是用 `env.rollout.n`。
3. 数据集在 agent 训练里几乎只是“占位符”，主要作用是告诉系统模态和 batch size；真实监督信号主要来自环境 rollout。
4. 合法动作约束的第一道闸门不在 loss，而在 `projection.py` 和 `env_manager.py`。
5. 奖励默认是 episode-level reward，再回填到每个 step 的最后一个 token 上；这会直接影响 GRPO advantage 的统计形态。
6. 你未来做 GUI agent，真正要魔改的主要不是 `verl/models`，而是 `env_manager`、`projection`、`memory`、`reward_manager`、`rollout_loop` 这 5 个点。

### 明确不深究的内容

这 5 天刻意不深究：

1. `critic` / `value model` / GAE。
2. `megatron` 路线。
3. `verl/third_party/vllm`、`sglang`、内核 patch 细节。
4. 大规模数学 DAPO 32B 复现链路。
5. 模型结构细节，比如 Qwen-VL 的内部层实现。

你要达到的“黄金甜蜜点”不是“知道所有底层”，而是：

1. 能自己从训练脚本一路追到 reward 和 advantage。
2. 能定位 rollout 慢、reward 怪、动作格式崩、显存炸、group 无差异这些真实工程问题。
3. 能在 1 到 2 天内把一个自定义环境接进来并跑通最小闭环。

---

## 你这 5 天的总策略

### 学习哲学

1. 先跑出可观测现象，再回到代码解释现象。
2. 每天都要产出“可验证结果”，不是只做笔记。
3. 每天至少留 2 小时做局部魔改，不做纯阅读日。
4. 只围绕你未来项目相关的路径深挖：多模态输入、结构化输出、动作合法性、记忆、规则奖励、长时程多轮交互。

### 每天固定工作流

每天都重复这一套，不要变：

1. 先跑一个最小实验，观察日志、吞吐、成功率、合法动作率、reward。
2. 再追 1 到 2 条代码链，只解释刚才观察到的现象。
3. 再做一个最小改动，验证你是否真的理解。
4. 最后用“面试硬核拷问”自测。

### 强烈建议的常用命令模板

所有命令默认先：

```bash
conda activate verl-agent-bw
cd /home/zhangwj/science/verl-agent
```

建议你在学习期间优先用“小步快跑”的 override，而不是直接跑默认长训练：

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.rollout.n=4 \
  env.max_steps=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
```

### 结合你硬件的建议

你的机器非常强，足够把“快速迭代”和“中等规模验证”分开：

1. Day 1 到 Day 3：优先用 `Qwen2.5-1.5B-Instruct` 和 `Qwen2.5-VL-3B-Instruct`，目标是高频调试。
2. Day 4：在流程稳定后，尝试 7B 文本模型做一次中型验证，但不是必须。
3. Ray CPU 建议手动限制，例如先试 `ray_init.num_cpus=64`，避免默认吃满系统导致调度噪声。
4. `env.resources_per_worker.num_cpus=0.1` 在这个仓库里是很重要的吞吐开关，因为 group rollout 会拉起大量 Ray env actor。
5. 双卡场景优先沿用仓库里已经验证过的 `tensor_model_parallel_size=2`。

---

## 先建立一张心智地图

### 你必须优先熟悉的文件

第一优先级：

1. `examples/grpo_trainer/run_alfworld.sh`
2. `examples/grpo_trainer/run_webshop.sh`
3. `examples/grpo_trainer/run_sokoban.sh`
4. `verl/trainer/main_ppo.py`
5. `verl/trainer/ppo/ray_trainer.py`
6. `verl/trainer/ppo/core_algos.py`
7. `agent_system/multi_turn_rollout/rollout_loop.py`
8. `agent_system/multi_turn_rollout/utils.py`
9. `agent_system/environments/env_manager.py`
10. `agent_system/environments/base.py`
11. `agent_system/reward_manager/episode.py`
12. `agent_system/memory/memory.py`

第二优先级：

1. `verl/utils/dataset/rl_dataset.py`
2. `verl/workers/fsdp_workers.py`
3. `agent_system/environments/env_package/*/projection.py`
4. `recipe/dapo/README.md`
5. `recipe/hgpo/README.md`

### 你必须牢牢记住的几个实现事实

1. `main_ppo.py` 里明确断言：在 `verl-agent` 里，`actor_rollout_ref.rollout.n` 必须等于 1，GRPO 靠 `env.rollout.n` 实现。
2. `make_envs()` 会把同一个初始样本按 `group_n = env.rollout.n` 扩成多个环境实例；`AlfWorld`、`Webshop`、`Sokoban` 都是这么做的。
3. `TrajectoryCollector.multi_turn_loop()` 在训练时会先把 prompt repeat 成 `env.rollout.n` 份，再进入 agent-environment loop。
4. `uid` 是 GRPO group id，`traj_uid` 是单条轨迹 id。
5. `EpisodeRewardManager` 默认只在每步 response 的最后一个 token 上打分，但分数来自整条 episode。
6. `apply_projection_invalid_penalty()` 会根据 `is_projection_valid` 对 reward 再减惩罚。
7. `compute_grpo_outcome_advantage()` 是按 `uid` 分组做 GRPO，而不是按样本 index 直接做。

---

## Day 1：先把主干跑通，并建立“现象 -> 代码”的第一闭环

### 当日目标

不要急着懂算法细枝末节。Day 1 的目标只有 3 个：

1. 亲手跑通一个最小 GRPO agentic 训练闭环。
2. 把“数据、环境、rollout、reward、advantage、actor update”这一条链路在脑中串起来。
3. 搞清楚这个仓库和原版 `verl` 的本质差异到底在哪。

### 时间分配

上午 3 小时：跑最小实验，拿到第一手日志。

下午 3 小时：追主链文件，只解释你刚看到的现象。

晚上 2 小时：整理一页自己的“主链路草图”，并回答硬核问题。

### 今天必须跑的实验

优先选 `AlfWorld` 文本版本，因为最能体现 agent loop 的完整性：

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.rollout.n=4 \
  env.max_steps=6 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
```

如果 AlfWorld 环境当天有安装问题，再退而求其次跑：

```bash
bash examples/grpo_trainer/run_sokoban.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.rollout.n=4 \
  env.max_steps=8 \
  ray_init.num_cpus=64
```

### 今天必须读懂的文件

按这个顺序读：

1. `examples/grpo_trainer/run_alfworld.sh`
2. `examples/data_preprocess/prepare.py`
3. `verl/trainer/main_ppo.py`
4. `agent_system/environments/env_manager.py`
5. `agent_system/multi_turn_rollout/rollout_loop.py`
6. `agent_system/reward_manager/episode.py`
7. `verl/trainer/ppo/ray_trainer.py`

### 今天必须搞清楚的关键问题

1. 为什么脚本写的是 `main_ppo`，但训练出来的是 agentic RL？
2. 为什么数据准备脚本明明读了 `geometry3k`，但实际却“不用里面的数据内容”？
3. 为什么 `env.rollout.n` 才是 agent 任务里的 group size？
4. 为什么 reward manager 里拿到的是 `episode_rewards`，而不是环境一步一步的细粒度奖励？
5. `TrajectoryCollector` 到底是在“采样 token”，还是在“采样轨迹”？

### 今日实践任务

1. 手画一张单页图，图上必须标出：
   `run_*.sh`、`main_ppo.py`、`make_envs()`、`multi_turn_loop()`、`reward_fn()`、`compute_advantage()`、`update_actor()`
2. 在你自己的笔记里写一句话解释：
   “为什么这个仓库里 dataset 只是 agent rollout 的起点，而不是主要监督来源”
3. 记录 5 个真实日志字段，并写出它们分别对应哪段代码产出。

### Day 1 产出标准

今天结束时，你必须能不看代码口头说出：

1. 一个 batch 是怎么从 parquet 样本变成一组环境轨迹的。
2. group 是在哪里复制出来的。
3. reward 是在哪里回填到 token 上的。

### Day 1 面试硬核拷问

1. 为什么 `verl-agent` 明明调用的是 `verl.trainer.main_ppo`，却能做 agentic RL？
2. `actor_rollout_ref.rollout.n` 和 `env.rollout.n` 在这个仓库里的职责差异是什么？
3. 如果你把 `env.rollout.n` 从 8 改成 1，训练行为会发生什么根本变化？
4. 为什么这里的 dataset 不需要包含标准答案？
5. `EpisodeRewardManager` 为什么只在 response 最后一个 token 位置写 reward？

---

## Day 2：把环境、动作合法性、记忆拼接这三层吃透

### 当日目标

今天不碰“更高级算法”，只攻工程落地里最容易翻车的三层：

1. 环境批量构建与 group reset。
2. 动作投影与合法性判定。
3. 历史记忆如何拼回 prompt。

这三层正是你未来做 GUI 斗地主 Agent 时最需要复用和魔改的地方。

### 时间分配

上午 2.5 小时：看 `env_manager` 和 env package 的 reset/step/group 逻辑。

下午 3 小时：看 `projection.py` 和 `memory.py`，再做一次小改动验证。

晚上 2.5 小时：整理“如何接入一个新环境”的模板。

### 今天重点文件

1. `agent_system/environments/base.py`
2. `agent_system/environments/env_manager.py`
3. `agent_system/memory/memory.py`
4. `agent_system/environments/env_package/alfworld/envs.py`
5. `agent_system/environments/env_package/webshop/envs.py`
6. `agent_system/environments/env_package/sokoban/envs.py`
7. `agent_system/environments/env_package/alfworld/projection.py`
8. `agent_system/environments/env_package/webshop/projection.py`
9. `agent_system/environments/env_package/sokoban/projection.py`
10. `agent_system/environments/README.md`

### 今天必须抓住的实现细节

1. `Webshop` 和 `Sokoban` 都会在 reset 时把同一个初始样本或 seed 重复 `group_n` 次，这是 GRPO 的环境侧基础。
2. `projection.py` 通常负责从模型原始文本里抽出 `<action>` 段，并给出 `valids`。
3. `is_projection_valid` 不是环境天然给你的，而是 projection 和 env manager 联合构造出来的。
4. `SimpleMemory` 其实非常朴素，本质就是把过去若干步 `(obs, action)` 重新拼成 prompt 文本。
5. `env_manager.py` 里真正影响 agent 表现的，不只是环境 API，还有 prompt 模板的组织方式。

### 今天必须做的调试实验

选一个最容易修改的环境，优先 `Sokoban`：

1. 故意让模型输出缺少 `<think>` 或 `<action>`。
2. 观察 `projection.py` 如何把它判成 invalid。
3. 观察 `episode/projection_valid_ratio` 和 reward 的变化。

可选做法：

1. 直接改 `sokoban_projection.py`，把合法动作从仅接受 `up/down/left/right/still` 改成只接受 `up/down/left/right`。
2. 跑 1 个短实验，观察非法动作率是否飙升。

### 今日微型魔改任务 1

把一个现有环境的输出协议临时改成更接近你未来项目的形式：

目标格式：

```text
<think>...</think>
<action>...</action>
<chat>...</chat>
<memory>...</memory>
```

今天只做“抽取与容错”，不做真正使用：

1. 在某个 `projection.py` 中允许存在 `<chat>` 和 `<memory>`，但仍只抽出 `<action>`。
2. 确保不因为多余字段导致合法动作解析失败。

这一步是为 Day 4 和 Day 5 铺路。

### Day 2 产出标准

你必须能回答：

1. 新增一个环境时，最少要改哪些文件。
2. 想做 GUI agent 的动作合法性奖励，应该放在 projection、env、reward manager 哪一层，为什么。
3. 想做“记牌/策略记忆”，仓库当前 memory 机制能承载什么，不能承载什么。

### Day 2 面试硬核拷问

1. 为什么动作合法性最好先在 `projection.py` 做第一层约束，而不是完全丢给环境？
2. 如果你把 `history_length` 调大，最可能先炸的是哪里：显存、吞吐、还是 reward 稳定性？为什么？
3. 为什么 `WebshopEnvironmentManager` 会在 prompt 中显式注入 `available_actions`？
4. `SimpleMemory` 和真正长期记忆系统的本质差距是什么？
5. 如果你做斗地主 GUI agent，`projection.py` 最合理的职责边界是什么？

---

## Day 3：把 GRPO 真正学透，但只学和 agentic RL 落地相关的部分

### 当日目标

今天只研究一件事：

“这套 agent rollout 最后是如何变成 GRPO 更新信号的？”

重点不是背公式，而是把实现和工程影响说清楚。

### 时间分配

上午 3 小时：精读 `ray_trainer.py` 的 reward/adv/update 主链。

下午 3 小时：精读 `core_algos.py` 的 GRPO advantage 实现。

晚上 2 小时：做一次参数扫，观察 group size、invalid penalty、KL loss 的实际影响。

### 今天重点文件

1. `verl/trainer/ppo/ray_trainer.py`
2. `verl/trainer/ppo/core_algos.py`
3. `agent_system/reward_manager/episode.py`
4. `agent_system/multi_turn_rollout/rollout_loop.py`
5. `agent_system/multi_turn_rollout/utils.py`
6. `verl/workers/fsdp_workers.py`

### 今天必须彻底搞懂的关键实现

1. `TrajectoryCollector.gather_rollout_data()` 会把每个 active step 都保留下来，并把整条 episode 的统计量塞回每个 step。
2. `EpisodeRewardManager` 默认把 `episode_rewards` 当作 step sample 的 outcome reward。
3. `apply_projection_invalid_penalty()` 会再从 reward 中减去非法动作惩罚。
4. `compute_grpo_outcome_advantage()` 是按 `uid` 聚组做 mean/std 归一化。
5. 这里的 `uid` 是“同一个初始问题下的一组轨迹”，`traj_uid` 是“单条轨迹身份”。
6. 当前实现里，step 展开后的样本会参与 advantage 统计，这意味着“长轨迹”和“短轨迹”的统计权重不一定相同。

### 这是今天最重要的洞察

这个仓库默认的 reward/advantage 设计，对长时程 agent 有一个很值得面试展开的点：

1. 轨迹级 reward 被广播回多个 step sample。
2. GRPO 分组统计可能因此受到轨迹长度分布影响。
3. 所以 agentic RL 的难点不只是“有没有 reward”，而是“reward 经过 rollout 展开之后，统计上变成了什么”。

如果你把这个点讲清楚，面试官会明显感觉你不只是会跑脚本。

### 今天必须做的实验

做一个小参数对照：

1. `env.rollout.n=2` 跑一次。
2. `env.rollout.n=8` 跑一次。
3. `actor_rollout_ref.actor.projection_invalid_penalty_coef=0.0` 跑一次。
4. `actor_rollout_ref.actor.projection_invalid_penalty_coef=0.1` 跑一次。

观察：

1. `episode/projection_valid_ratio`
2. `val/*/test_score`
3. 训练是否更容易出现 group 内全同 reward
4. 生成吞吐和 batch 对齐是否有变化

### 今日微型魔改任务 2

做一个“最小 reward 分解”实验：

目标：把最终 reward 拆成 3 项并打日志

1. 环境原始 reward
2. 非法动作 penalty
3. 最终送入 GRPO 的 token-level score

你不一定要永久改代码，但至少要能在本地临时打印出来。

这一步是为了训练你未来做“格式奖励 + GUI 合法性奖励 + 胜负奖励”的感觉。

### Day 3 产出标准

你必须能独立解释：

1. 这个仓库的 GRPO group 是怎么构造出来的。
2. 为什么 reward manager、invalid penalty、GRPO advantage 是三个不同层次。
3. 为什么“能跑通”和“reward 统计正确”完全不是一回事。

### Day 3 面试硬核拷问

1. 为什么 `verl-agent` 里 GRPO 分组必须绑定环境 reset，而不是只在 token 采样阶段做 `n` 次生成？
2. `uid` 和 `traj_uid` 分别解决什么问题？
3. 这个仓库的 outcome reward 在 step 展开后会带来什么统计偏差风险？
4. 非法动作 penalty 放在 reward 之后而不是 projection 里直接丢弃，有什么好处？
5. 如果 group 内 reward 全一样，GRPO 会发生什么？这和 DAPO 的 dynamic sampling 有什么关系？

---

## Day 4：聚焦工业级壁垒，攻克“训练能跑”到“训练有效”的真实问题

### 当日目标

今天开始从“读懂代码”切换到“解决工程问题”。

重点攻 4 类工业级壁垒：

1. 吞吐和资源利用。
2. group 退化问题。
3. 长上下文和记忆膨胀。
4. 奖励设计与调试闭环。

### 时间分配

上午 2.5 小时：做吞吐和显存诊断。

下午 3 小时：读 DAPO / GiGPO / HGPO 的高信号部分，理解它们分别在补什么洞。

晚上 2.5 小时：做一个真实魔改任务。

### 今天重点文件

1. `recipe/dapo/README.md`
2. `recipe/hgpo/README.md`
3. `agent_system/multi_turn_rollout/utils.py`
4. `verl/workers/fsdp_workers.py`
5. `examples/gigpo_trainer/run_*.sh`
6. `examples/dapo_trainer/run_*.sh`

### 今天要带着问题去读

1. 为什么 DAPO 要做 group filtering？
2. 为什么 agentic RL 里 group 内 reward 很容易全同？
3. 为什么 GiGPO / HGPO 会关注 step-level 或 hierarchical grouping？
4. 你的斗地主 GUI agent 更像需要 DAPO、GiGPO 还是 HGPO？

### 我给你的判断

按你未来项目的 MVP 目标，优先级建议是：

1. 先吃透 GRPO baseline。
2. 再学 DAPO dynamic sampling。
3. 再看 GiGPO 的 step-level grouping 思路。
4. HGPO 可以知道思想，但不必在这 5 天里深实现。

原因：

1. 你的 MVP 首先要解决的是格式稳定、动作合法、基本朝目标推进。
2. 这类问题最先遇到的是 group reward 无差异、稀疏奖励、长轨迹 credit assignment 粗糙。
3. DAPO 和 GiGPO 对你比 HGPO 更直接。

### 今日微型魔改任务 3

任务：在一个现有环境里模拟你的未来奖励设计

建议选 `Sokoban` 或 `Webshop`，加一个临时的复合奖励：

1. 格式奖励：输出同时包含 `<think>` 和 `<action>` 时给小正奖励。
2. 合法性奖励：动作合法额外加小正奖励，非法减小负奖励。
3. 结果奖励：保留原有 `won` 或环境终局奖励。

你不必把它做成优雅的最终版，但必须完成：

1. 改 reward 路径。
2. 跑一次短训练。
3. 看日志验证这 3 项是否真的在起作用。

### 结合你未来 GUI 斗地主项目的映射

今天你要开始做这个思维映射：

1. `Sokoban` 的离散动作合法性，映射到 GUI 点击合法性。
2. `Webshop` 的 `available_actions`，映射到 GUI 上当前可点击区域。
3. `SimpleMemory` 的历史拼接，映射到“记牌 + 局势摘要 + 对话记忆”。
4. `EpisodeRewardManager` 的 episode reward，映射到胜负奖励。
5. `projection.py` 的结构抽取，映射到 `<think><action><chat><memory>` 协议解析。

### Day 4 产出标准

你必须能讲出一个成熟的工程判断：

“如果我下周开始接斗地主 GUI 环境，我会先复用哪些模块，重写哪些模块，为什么”

### Day 4 面试硬核拷问

1. 为什么很多 agentic RL 任务早期会出现 group 内 reward 全同？
2. DAPO 的 dynamic sampling 本质上在解决什么统计问题？
3. 对 GUI agent 而言，为什么动作合法性奖励常常比胜负奖励更早产生训练价值？
4. 什么时候应该继续用 episode-level reward，什么时候必须引入 step-level reward？
5. 你的项目为什么更像“多轮、稀疏、结构化、环境约束强”的 agentic RL，而不是普通 single-turn RLHF？

---

## Day 5：做面向你项目的最小切入，把学习成果变成可复用工程资产

### 当日目标

最后一天不要再扩阅读面。只做两件事：

1. 把你对这个仓库的理解收敛成“自己的 agent 接入模板”。
2. 完成 1 个最小但真实的项目预演。

### 最推荐的最终实战任务

在仓库里做一个“斗地主 GUI Agent 的最小前置雏形”，但不需要真的接斗地主：

任务名：`dummy_gui_agent_mvp`

要求：

1. 观察输入同时包含文本和图片占位。
2. 模型必须输出：
   `<think>...</think><action>...</action><chat>...</chat><memory>...</memory>`
3. `projection` 只把 `<action>` 映射成环境动作，但会验证 4 个字段是否齐全。
4. 奖励至少包含：
   格式奖励 + 动作合法性奖励 + 终局奖励
5. memory 至少能把过去 2 到 3 步的 `action/chat/memory` 摘进下一轮 prompt

如果 1 天做不完完整环境，退一步做成“假环境”也可以：

1. reset 返回固定图片和局面文本
2. step 根据 action 是否合法返回规则奖励
3. done 条件可以很简单

### 今天建议你改的最小模块

1. 复制一个最简单的 env package 作为模板。
2. 新建你自己的 `projection.py`。
3. 在 `env_manager.py` 里仿照现有环境接入。
4. 临时扩展 `SimpleMemory` 或新建一个轻量 memory 类。
5. 在 reward 路径里加入格式奖励和合法性奖励。

### 今天的最终交付物

你应该在本地留下 4 份真正有价值的资产：

1. 一张你自己画的 `verl-agent` 主链架构图。
2. 一份“新环境接入 checklist”。
3. 一个最小自定义 env/projection/reward 原型。
4. 一份面试讲稿提纲。

### 你的面试讲稿提纲建议

请把下面 4 段练熟：

1. 这个仓库如何在 `verl` 的 PPO/GRPO trainer 上扩成 agentic RL。
2. 我如何理解环境分组、reward 回填、GRPO advantage 的关系。
3. 我实际踩过哪些工程坑，怎么定位。
4. 我会如何把这套框架迁移到多模态 GUI 斗地主陪玩 Agent。

### Day 5 面试硬核拷问

1. 如果让你把一个全新的 GUI 环境接入 `verl-agent`，你会按什么顺序动手？
2. 为什么你会优先先做“结构输出稳定 + 动作合法性奖励”，而不是直接追求高胜率？
3. 对你的项目，`memory` 最好放在 prompt 层、latent 层、还是外部状态层？为什么？
4. 你会如何定义斗地主 GUI agent 的 group，才能让 GRPO 真正有意义？
5. 如果训练不收敛，你会优先排查 prompt、projection、reward、group 构造、还是模型容量？为什么？

---

## 5 天结束后，你应该达到的能力标准

### 算法理解层

1. 你能解释 agentic RL 里 GRPO 的 group 是怎么从环境构造出来的。
2. 你能解释为什么 reward 设计和 rollout 展开方式会改变 advantage 的统计性质。
3. 你知道 DAPO / GiGPO / HGPO 分别想解决什么问题。

### 工程落地层

1. 你能独立接入一个新环境。
2. 你能设计结构化输出协议并做动作合法性校验。
3. 你能做最小规则奖励系统。
4. 你能在双卡机上做小到中等规模的快速迭代。

### 面试表达层

1. 你能把 agentic RL 讲成一个完整的数据流，而不是几个散概念。
2. 你能讲出具体工程细节，而不是泛泛而谈“奖励设计很重要”。
3. 你能把 `verl-agent` 的经验自然迁移到自己的 GUI 多模态 agent 项目。

---

## 最后给你的执行建议

1. 每天结束必须写“今天我验证了什么，不是我看了什么”。
2. 如果某个源码段落你读了 30 分钟还没感觉，立刻回去跑实验，用现象逼理解。
3. 你的目标不是 5 天后“会背代码”，而是 5 天后“敢改代码、敢接新环境、敢跟面试官聊工程细节”。
4. 真正决定你后续项目上限的，不是你今天懂了多少算法名词，而是你有没有把 reward、action contract、memory、group 构造这几个局部问题一个个做扎实。
