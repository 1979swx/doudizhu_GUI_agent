# Day 4 保姆级实战指南

这份文档不是一份新的学习计划，而是对 [study_plan.md](/home/zhangwj/science/verl-agent/study_guide/study_plan.md) 里 Day 4 的逐步展开版。今天只做原计划里的四件事：

1. 吞吐和资源利用。
2. group 退化问题。
3. 长上下文和记忆膨胀。
4. 奖励设计与调试闭环。

今天的宗旨只有一句话：**不追求“又读了很多”，只追求“我能解释一个真实现象，并且用一个小改动把它改出我想要的行为”。**

---

## 先定主线：今天全程尽量围绕 `Sokoban + Qwen2.5-VL-3B`

原因不是它最强，而是它最适合你今天的目标：

1. 它是视觉输入，更接近你未来的 GUI 斗地主项目。
2. 动作空间离散，合法性问题非常清楚，最适合练“格式奖励 + 合法性奖励 + 终局奖励”。
3. 环境简单，出问题时更容易判断是环境侧、projection 侧、reward 侧，还是 trainer 侧。

除非你在 Day 1 到 Day 3 一直主跑 `AlfWorld`，否则今天不要把主线切回文本环境。Day 4 最怕的是同时换环境、换模型、换算法，最后什么都说不清。

---

## 今天的总产出

今天结束时，你必须拿到下面 5 个硬产出：

1. 一份你自己写的“吞吐瓶颈判断单”。
2. 一份“为什么会出现 group 内 reward 全同”的工程解释。
3. 一份“GRPO / DAPO / GiGPO / HGPO 对我未来 GUI 斗地主项目的选型判断”。
4. 一次真实跑通的复合奖励短实验。
5. 一段成熟表述：
   “如果我下周开始接斗地主 GUI 环境，我会先复用哪些模块，重写哪些模块，为什么。”

---

## 时间总表

严格按这个节奏推进。不要一上来读一堆源码。

| 时间块 | 时长 | 任务 |
|---|---:|---|
| 0. 热身准备 | 15 分钟 | 打开监控面板，固定 baseline 命令，建立记录模板 |
| 1. 上午主任务 | 2 小时 30 分钟 | 吞吐与资源诊断，定位瓶颈在 env / rollout / actor / batch 对齐 的哪一层 |
| 2. 下午主任务 | 3 小时 | 读 DAPO / GiGPO / HGPO 的高信号部分，并把它们映射到你的 GUI 项目 |
| 3. 晚上主任务 | 2 小时 30 分钟 | 做复合奖励微型魔改：格式奖励 + 合法性奖励 + 结果奖励 |
| 4. 收尾 | 15 分钟 | 写出工程判断与面试回答 |

---

## 开始前 15 分钟：把观测面板搭好

### 动作 1：固定工作目录与环境

预计用时：3 分钟

在你的训练终端里执行：

```bash
conda activate verl-agent-bw
cd /home/zhangwj/science/verl-agent
```

### 动作 2：再开两个终端，只做观测

预计用时：5 分钟

终端 B 看 GPU：

```bash
watch -n 1 nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

终端 C 看 CPU / Ray actor 压力。你可以用 `htop`，如果没装就用：

```bash
top
```

你今天不是为了“盯着数字好看”，而是为了建立下面这组因果感：

1. GPU 利用率很低，同时 CPU 很忙，通常说明瓶颈更可能在环境侧或 Ray 调度侧。
2. GPU 显存接近顶满、训练进程容易卡住或 OOM，通常说明是 actor/ref/logprob micro-batch 太大。
3. GPU 显存不高、利用率也低，通常说明 batch 太小、group 太小、或者 rollout 端没有喂饱训练端。

### 动作 3：提前建一张记录表

预计用时：7 分钟

今天所有实验都记这 8 列：

| 实验名 | 模型 | `env.rollout.n` | `train_batch_size` | 主要改动 | GPU 利用率 | 现象 | 结论 |
|---|---|---:|---:|---|---|---|---|

强制要求自己每次实验后写一句结论。哪怕只写：

`A_run：GPU 只有 25%-35%，CPU 很满，怀疑 env actor 太多且每个 actor 分到 CPU 太少。`

这一步看起来很“笨”，但它恰好是面试里最值钱的部分。面试官不缺会背公式的人，缺的是能把现象、代码和调参动作串起来的人。

---

## 第一部分：上午 2.5 小时，吞吐和资源利用诊断

今天上午的目标不是“把吞吐调到极致”，而是**学会判断瓶颈到底在哪一层**。

---

### Step 1：先跑一个足够小、但结构完整的 baseline

预计用时：35 分钟

请先跑下面这个命令，不要一开始就用脚本默认的 `train_batch_size=32, group_size=8`。默认配置太大，会把多个问题混在一起。

```bash
bash examples/grpo_trainer/run_sokoban.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.project_name=day4_debug \
  trainer.experiment_name=day4_sokoban_grpo_baseline \
  data.train_batch_size=8 \
  data.val_batch_size=16 \
  env.rollout.n=4 \
  env.max_steps=8 \
  env.resources_per_worker.num_cpus=0.1 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
```

为什么今天先用这组数字：

1. `train_batch_size=8, env.rollout.n=4`，训练环境数就是 `8 * 4 = 32`，验证环境数是 `16`，总 actor 数规模合适，便于定位问题。
2. 这组设置下，有效轨迹数是 32，足够看到 group 行为，又不至于把 Ray actor 一次性拉太多。
3. 这组 batch 更容易与 `adjust_batch()` 的整除要求对齐，减少“为了凑整被复制样本”的干扰。

你在这一次运行里只盯 6 件事：

1. GPU 利用率大致区间。
2. GPU 显存占用大致区间。
3. CPU 是否明显吃满。
4. 控制台里训练是卡在 rollout 前后，还是卡在 update actor 前后。
5. `episode/projection_valid_ratio` 是否明显低。
6. `val/*/test_score` 是否几乎没信号。

#### 这一轮跑完后，你必须立刻回答 3 个问题

1. 现在瓶颈更像环境侧，还是模型侧？
2. 现在更像“吃不满 GPU”，还是“显存先到顶”？
3. 现在的 reward 更像“有区分度”，还是“大家都差不多”？

如果你答不出来，不要进入下一步读代码。先回去再看 5 分钟监控面板。

---

### Step 2：用代码解释你刚才看到的现象

预计用时：30 分钟

只读下面 5 个点，不要扩散：

1. `agent_system/multi_turn_rollout/rollout_loop.py`
2. `agent_system/multi_turn_rollout/utils.py`
3. `verl/trainer/main_ppo.py`
4. `verl/workers/fsdp_workers.py`
5. `agent_system/environments/env_manager.py`

请按这个顺序理解：

#### 2.1 先看 `rollout_loop.py`

你要确认两件事：

1. `multi_turn_loop()` 在训练时会先把 `gen_batch` 按 `env.rollout.n` 重复。
2. 环境 step 得到的 `rewards` 会先累积成 `episode_rewards`，最后才交给 reward manager 写回 token。

这意味着：

1. 你一旦把 `env.rollout.n` 从 4 调到 8，环境侧压力会直接翻倍。
2. 但 actor 侧的很多 batch 配置并不会因为 `env.rollout.n` 自动为你重新设计。

这正是很多人第一次做 agentic RL 会踩的坑：**以为自己只是把 group size 调大了，实际上是把整条 rollout 链路的环境开销、采样开销、日志开销、后续 advantage 统计都一起放大了。**

#### 2.2 再看 `main_ppo.py`

这里有一句你今天必须牢牢记住的断言：

`actor_rollout_ref.rollout.n == 1`

在这个仓库里，agentic RL 的 group 不是靠传统 `rollout.n` 做的，而是靠 `env.rollout.n` 在环境侧实现。

这会带来一个很关键的工程结论：

**你调大 `env.rollout.n` 后，不能指望 FSDP worker 自动帮你把 actor 的 mini-batch 和 micro-batch 一起调顺。**

#### 2.3 再看 `fsdp_workers.py`

重点只看一件事：

`ActorRolloutRefWorker` 里对 `ppo_mini_batch_size` 的归一化，是基于 `config.rollout.n` 做的。

而在 agentic RL 主线里，这个 `config.rollout.n` 被固定成 1。

所以今天的实战意义非常大：

1. `env.rollout.n` 增大，真实收集到的轨迹会变多。
2. 但 `ppo_mini_batch_size` 不会因为你把 group 变大就自动等比例变化。
3. 因此你要自己判断：现在是该加大 actor batch 吃满 GPU，还是该先保守一点避免 update 端过慢。

#### 2.4 再看 `utils.adjust_batch()`

这个函数今天一定要看懂。

它会为了让 batch 对齐 `rollout/ref/actor` 的 micro-batch 整除要求，做两种事之一：

1. 删除部分样本。
2. 或复制部分样本。

默认常见的是 `copy`。

这会带来一个很容易被忽略的后果：

**你在 debug group 统计时，如果只按 batch index 看，很容易被补齐出来的样本骗到。**

今天你要养成一个习惯：**凡是谈 group，就优先看 `uid` / `traj_uid`，不要只看“第几个样本”。**

#### 2.5 最后看 `env_manager.py`

今天只抓住一个事实：

环境不是“返回 observation 就完了”，环境管理器还负责：

1. prompt 组织；
2. memory 拼接；
3. action projection；
4. 合法性标记 `is_projection_valid`；
5. 某些环境里的 `available_actions` 注入。

所以当训练“能跑但没效果”时，不要只盯 loss。很多时候第一层问题就在这里。

---

### Step 3：做一次有目的的二次实验，只改最可能的瓶颈

预计用时：55 分钟

这一轮不要盲调。按下面的决策树走。

#### 情况 A：GPU 利用率低，CPU 很忙

这通常更像环境/Ray 侧瓶颈。

优先改这两个方向之一：

1. 适度提高 `env.resources_per_worker.num_cpus`，比如从 `0.1` 提到 `0.2`。
2. 如果 group 太大，先把 `env.rollout.n` 保守下来，不要急着追大 group。

建议命令：

```bash
bash examples/grpo_trainer/run_sokoban.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.project_name=day4_debug \
  trainer.experiment_name=day4_sokoban_cpu_fix \
  data.train_batch_size=8 \
  data.val_batch_size=16 \
  env.rollout.n=4 \
  env.max_steps=8 \
  env.resources_per_worker.num_cpus=0.2 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
```

你要看的是：

1. GPU 利用率有没有抬起来。
2. 每轮训练是不是更平滑，不那么“等环境”。
3. CPU 是否反而被更快打满。

#### 情况 B：GPU 显存很高，甚至接近 OOM

这更像 actor/ref/logprob micro-batch 太大。

优先降这几个：

1. `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
2. `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
3. `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`

注意，不要优先乱降 `train_batch_size`。那会把吞吐问题和统计问题一起搅乱。

#### 情况 C：GPU 显存很安全，但利用率也不高

这通常说明你可以进一步增大 actor 侧吞吐，或者说明 rollout 太慢喂不饱训练。

今天只做一个保守动作：把 `ppo_micro_batch_size_per_gpu` 从 2 提到 4，其他不动，看看 update 端是否明显变快。

#### 情况 D：训练速度还行，但 reward / projection_valid_ratio 看起来像死水

这说明你上午最重要的瓶颈不是吞吐，而是统计有效性。那就不要继续卷资源调优，下午尽快进入 DAPO / GiGPO / 奖励设计。

---

### Step 4：写一张“上午判断单”

预计用时：10 分钟

请你亲手写 4 句话，缺一不可：

1. 当前主瓶颈在什么层。
2. 为什么你这样判断。
3. 下一个最值得调的旋钮是什么。
4. 哪个旋钮今天暂时不该动，为什么。

你未来做 GUI 斗地主项目时，这张“判断单”的能力，比你多记住十个配置项更值钱。

---

## 第二部分：下午 3 小时，读 DAPO / GiGPO / HGPO 的高信号部分

下午不是“文献阅读时间”，而是“带着你上午看到的问题去读算法补丁”。

读的顺序非常重要：

1. 先读 DAPO，因为它最直接解决 group 全同导致的训练无效。
2. 再读 GiGPO，因为你的未来任务是多轮、长轨迹、动作受环境约束，step-level 信息非常关键。
3. 最后读 HGPO，只吃思想，不深实现。

---

### Step 5：先把 DAPO 的真正价值读出来

预计用时：70 分钟

#### 5.1 只读 `recipe/dapo/README.md` 的这两段

预计用时：15 分钟

只看：

1. `Dynamic Sampling (with Group Filtering)`
2. `Separated Clip Epsilons`

别的先略过。

你要抓住的不是论文名词，而是这个朴素工程事实：

**如果一个 group 里 8 条轨迹的 reward 全都一样，那么 GRPO 的组内比较几乎没有学习信号。**

在 agentic RL 里，这种事情太常见了，因为：

1. 奖励经常很稀疏，很多轨迹都拿 0。
2. 早期模型很弱，group 内经常全错。
3. 某些简单任务里，group 内又可能全对。
4. episode-level reward 很粗，很多不同过程最后都被压成一个同样分数。

#### 5.2 立刻回到代码，看它在仓库里怎么落地

预计用时：25 分钟

只看这三个点：

1. `agent_system/multi_turn_rollout/utils.py` 里的 `filter_group_data()`
2. `agent_system/multi_turn_rollout/rollout_loop.py` 里的 `dynamic_multi_turn_loop()`
3. `verl/trainer/ppo/core_algos.py` 里的 `compute_grpo_outcome_advantage()`

你要顺着这条链理解：

1. 先 oversample；
2. 再把 group 内 reward 全同的组过滤掉；
3. 只保留真正有组内差异的 group；
4. 再拿这些 group 去做 GRPO advantage。

这就是 DAPO 的高价值之处：**它不是在 loss 上“耍花活”，而是在数据进入 advantage 之前，先提高有效样本的密度。**

#### 5.3 写出你自己的 DAPO 解释

预计用时：10 分钟

请你强制写下这段话，不能照抄文档：

`DAPO 的 dynamic sampling 本质上不是“让模型更聪明”，而是先把那些没有相对比较价值的 group 去掉，避免 GRPO 用一堆没有方差的组做无效更新。`

#### 5.4 如果你上午已经明显看到 group 无差异，做一个可选对照

预计用时：20 分钟，可选

跑一次动态 GiGPO 或 DAPO 的短实验，只为感受“filter_groups 开了以后，训练 batch 可能要多次生成才凑够有效 group”。

可选命令：

```bash
bash examples/gigpo_dynamic_trainer/run_sokoban.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.project_name=day4_debug \
  trainer.experiment_name=day4_sokoban_gigpo_dynamic \
  data.train_batch_size=8 \
  data.val_batch_size=16 \
  env.rollout.n=4 \
  env.max_steps=8 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
```

这一步不是为了追曲线，而是为了亲手感受：

`有效训练样本数` 和 `原始生成样本数` 在 agentic RL 里并不是一回事。

---

### Step 6：读 GiGPO，只读与你未来 GUI 项目最相关的部分

预计用时：65 分钟

#### 6.1 先看脚本层差异

预计用时：10 分钟

对照看：

1. `examples/grpo_trainer/run_sokoban.sh`
2. `examples/gigpo_trainer/run_sokoban.sh`
3. `examples/gigpo_dynamic_trainer/run_sokoban.sh`

你只关心这几个配置差异：

1. `algorithm.adv_estimator=grpo` 还是 `gigpo`
2. 是否有 `algorithm.gigpo.step_advantage_w`
3. 是否有 `algorithm.filter_groups.enable`

#### 6.2 再看 trainer 里 GiGPO 比 GRPO 多了什么

预计用时：25 分钟

只读：

1. `verl/trainer/ppo/ray_trainer.py` 里 `GiGPO` 分支
2. 同文件里 `step_rewards` 的注入位置

你要抓住两个实现事实：

1. GiGPO 不只看最终 `token_level_rewards`，还会看 `step_rewards`。
2. 它还会用 `anchor_obs` 这类中间状态信息做 step-level grouping。

这和纯 GRPO 的区别非常关键：

1. GRPO 主要比较“整条轨迹最后拿了多少分”。
2. GiGPO 试图把“中间每一步是否在往好的方向走”也变成可利用信号。

这对你的 GUI 斗地主 MVP 为什么重要：

1. 你未来的早期 reward 很可能先是“格式合法”和“点击合法”，不是“最终胜率”。
2. 多轮 GUI 任务中，不同轨迹可能最后都输，但中间有的轨迹明显更合理。
3. 如果只看 episode-level outcome，这些差别容易被吞掉。

#### 6.3 你现在就要形成一个判断

预计用时：10 分钟

请你写下：

`对我的 GUI 斗地主 MVP，GiGPO 的价值不在“更先进”，而在“当大量样本终局 reward 还没有拉开时，step-level 信号更早能工作”。`

#### 6.4 但今天为什么仍然不主攻 GiGPO 实现

预计用时：20 分钟

原因很现实：

1. 你现在最先要解决的是输出协议稳定、动作合法、奖励闭环跑通。
2. 这些问题在 GRPO baseline 上就已经足够暴露。
3. 过早切到 GiGPO，容易把“reward 设计不清楚”和“advantage 形式更复杂”混在一起。

所以今天的正确策略是：

1. 懂 GiGPO 的价值；
2. 但主实验仍然先用 GRPO baseline 做 reward 路径魔改；
3. 等你自己的 GUI 环境里开始出现“都输但输法不一样”的大量样本，再认真上 GiGPO。

---

### Step 7：把“长上下文和记忆膨胀”这件事读明白

预计用时：35 分钟

这一部分非常重要，因为它直接决定你以后会不会把 GUI agent 做成一个上下文爆炸、吞吐崩盘、行为越来越迟钝的系统。

#### 7.1 只看 `agent_system/memory/memory.py`

预计用时：10 分钟

你会发现 `SimpleMemory` 本质非常朴素：

1. 每一步存 `(obs, action)`；
2. 下一轮按 `history_length` 取最近几步；
3. 再重新拼成 prompt 文本。

这意味着什么：

**默认 memory 不是“聪明记忆”，而是“原样拼接历史”。**

这对你未来 GUI 斗地主项目的警告非常重：

1. 如果你把截图描述、聊天记录、行动记录、记牌内容都原样拼回去，context 会非常快地膨胀。
2. 一旦 context 膨胀，首先坏掉的不是“模型理解力”，而是吞吐、显存、响应时间和有效 batch。

#### 7.2 再看 `agent_system/environments/env_manager.py`

预计用时：15 分钟

重点找两类逻辑：

1. `history_length` 如何控制拼接多少步历史；
2. WebShop 那种 prompt 过长时的兜底逻辑。

你要立刻形成一个工程判断：

`在 GUI agent 里，memory 模块绝不能长期停留在“简单拼接历史”的阶段。Day 5 可以先用它跑通 MVP，但很快就要把它升级为摘要式 memory。`

#### 7.3 现在就做一次映射

预计用时：10 分钟

把你未来 GUI 斗地主项目的历史信息分成三层：

1. 必须逐步保留的：最近几步 GUI 操作与结果。
2. 应该摘要保留的：记牌、局势判断、长期策略。
3. 不应该长期原样保留的：冗长聊天、重复截图描述、低价值无效动作。

只要你今天把这件事想清楚，Day 5 搭最小雏形时就会顺很多。

---

### Step 8：最后看 HGPO，但只吃思想，不深实现

预计用时：30 分钟

只读：

1. `recipe/hgpo/README.md`
2. 它对 `history_length`、`base_group`、`weight_type` 的说明

你只需要理解一句话：

**HGPO 试图在长时程任务里，不只按“同一个初始状态”分组，还考虑“走到当前这一步之前的历史上下文”去做更细的 grouping。**

这对你的 GUI 项目未来有没有价值？有。

但为什么今天不深做？

1. 你现在连自己的 GUI 环境都还没接进来。
2. 你的首要问题还是结构输出、合法动作、基础 reward 是否闭环。
3. HGPO 更像你项目第二阶段以后，处理超长时程和隐式状态时再上的武器。

今天你只需要得到一个成熟判断：

1. `GRPO`：先跑通 baseline。
2. `DAPO`：解决 group 无效更新。
3. `GiGPO`：解决多步过程信号太粗。
4. `HGPO`：解决超长历史上下文下更精细的 grouping。

---

## 第三部分：晚上 2.5 小时，做真实微型魔改任务 3

今天晚上的任务是整个 Day 4 的核心。

目标不是把代码写得多优雅，而是让你**亲手走完一次“我设计一个 reward -> 我把它接进训练链路 -> 我验证它真的生效”** 的完整闭环。

---

### 先定策略：今晚一律用 `Sokoban`

原因：

1. projection 简单。
2. 动作合法性简单。
3. 环境本身已有终局奖励。
4. 最接近你未来 GUI 场景里的“结构输出 + 动作约束 + 结果奖励”三层结构。

今晚建议你直接在下面两个文件里动手：

1. `agent_system/reward_manager/episode.py`
2. `agent_system/environments/env_package/sokoban/projection.py`

如果你在 Day 2 已经把 projection 扩成支持 `<chat>` / `<memory>` 了，今晚直接复用。
如果没有，今晚不要分心补太多协议字段，先围绕 `<think>` 和 `<action>` 跑通复合奖励即可。

---

### Step 9：先避免一个常见大坑，别把合法性惩罚算两遍

预计用时：10 分钟

默认训练链路里，`apply_projection_invalid_penalty()` 会在 reward manager 之后，基于 `is_projection_valid` 再减一次 penalty。

如果你今晚自己又在 `EpisodeRewardManager` 里额外写了“合法加分 / 非法减分”，而又忘了关默认 penalty，就会出现双重计分。

所以今晚建议你：

1. **把合法性奖励整合进你自己的复合 reward。**
2. **临时关闭内置 invalid penalty。**

今晚训练命令里记得加：

```bash
actor_rollout_ref.actor.use_projection_invalid_penalty=False
```

这是非常典型的工业级调试意识：**做实验时先让 reward 归因单一、可解释，不要同时开两套重叠机制。**

---

### Step 10：在 `EpisodeRewardManager` 里做最小复合奖励

预计用时：40 分钟

今晚的 reward 设计就用最简单、最直白、最好验证的一版：

1. `env_reward`：环境原始终局 reward，也就是现在的 `episode_rewards`
2. `format_reward`：同时包含 `<think>` 和 `<action>` 给一个小正奖励
3. `legality_reward`：动作合法给小正奖励，不合法给小负奖励
4. `final_score = env_reward + format_reward + legality_reward`

建议你先用下面这组系数，不要一开始就纠结最优值：

1. `format_reward = +0.05`
2. `legality_reward = +0.05` 或 `-0.05`
3. `env_reward` 保持原样

这组数的意义不是最优，而是：

1. 足够小，不会一下把环境 reward 完全淹没；
2. 足够明显，短实验里你看得出来它是否在起作用。

#### 10.1 改法原则

你要改的不是整条 trainer，而只是 `EpisodeRewardManager.__call__()` 里每条样本的 `score` 计算。

你需要拿到这 4 样东西：

1. `response_str`
2. `episode_rewards`
3. `is_projection_valid`
4. 你自己算出的 `format_reward`

其中：

1. `response_str` 已经在 `EpisodeRewardManager` 里 decode 了。
2. `episode_rewards` 已经在 `data_item.non_tensor_batch['episode_rewards']` 里。
3. `is_projection_valid` 已经在 rollout loop 里塞进 `non_tensor_batch` 了。

#### 10.2 你可以按这个思路改

下面不是完整补丁，只是你今晚应该写出的核心逻辑：

```python
has_think = ("<think>" in response_str) and ("</think>" in response_str)
has_action = ("<action>" in response_str) and ("</action>" in response_str)
format_reward = 0.05 if (has_think and has_action) else 0.0

is_projection_valid = bool(data_item.non_tensor_batch["is_projection_valid"])
legality_reward = 0.05 if is_projection_valid else -0.05

env_reward = float(episode_rewards)
score = env_reward + format_reward + legality_reward
```

然后把 `score` 写回最后一个 response token 上，保持和原始 reward manager 一样的写法。

#### 10.3 同时把 reward 分解信息打出来

这一步非常重要。

`EpisodeRewardManager` 可以在 `return_dict=True` 时返回：

1. `reward_tensor`
2. `reward_extra_info`

你今晚应该把 `reward_extra_info` 改成至少包含：

1. `env_reward`
2. `format_reward`
3. `legality_reward`
4. `final_reward`
5. `has_think_action`
6. `is_projection_valid`

这样训练主循环会拿到这些字段，你至少能在本地看到：

```text
list(reward_extra_infos_dict.keys())=['env_reward', 'format_reward', 'legality_reward', 'final_reward', ...]
```

如果你想更直观一点，可以临时在 `EpisodeRewardManager` 里随机打印少量样本：

```python
print({
    "env_reward": env_reward,
    "format_reward": format_reward,
    "legality_reward": legality_reward,
    "final_reward": score,
    "is_projection_valid": is_projection_valid,
})
```

今晚允许你打印得“土一点”。因为今天的目标不是优雅，是验证。

---

### Step 11：如果需要，让 projection 对结构化输出更宽容

预计用时：20 分钟

如果你已经做过 Day 2 的协议扩展，今晚检查两件事：

1. 多出来的 `<chat>` / `<memory>` 不会导致 `<action>` 解析失败；
2. projection 仍然只把 `<action>` 映射成环境动作。

`agent_system/environments/env_package/sokoban/projection.py` 里已经有一个现成的参考：

`sokoban_projection_my_modification`

它说明了一件很重要的工程思想：

**输出协议变复杂，不代表环境动作映射就要一起变复杂。**

未来 GUI 斗地主项目里，你也应该坚持这个边界：

1. `projection` 负责把结构化模型输出解释成环境动作；
2. `chat` 和 `memory` 可以被校验、被奖励，但不必直接影响动作映射函数本身。

如果你今晚还没做 Day 2 那步扩展，就先不动这里。只要 `<think>` 和 `<action>` 能稳定解析，你今晚的主要任务就成立。

---

### Step 12：跑一次真正的短训练，验证复合奖励是否生效

预计用时：35 分钟

建议命令：

```bash
bash examples/grpo_trainer/run_sokoban.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  trainer.project_name=day4_debug \
  trainer.experiment_name=day4_sokoban_composite_reward \
  data.train_batch_size=8 \
  data.val_batch_size=16 \
  env.rollout.n=4 \
  env.max_steps=8 \
  env.resources_per_worker.num_cpus=0.1 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.use_projection_invalid_penalty=False
```

这一轮你要验证的不是分数高低，而是下面 4 个判断：

1. 训练确实没有因为你加 reward 分解而跑挂。
2. 控制台里能看到你新增的 reward component。
3. `final_reward` 会随着 `is_projection_valid` 和输出格式变化而变化。
4. 你能明确说出“最终送入 GRPO 的分数到底是什么”。

---

### Step 13：用一个最笨但最可靠的方法验证奖励真的在起作用

预计用时：25 分钟

请你做下面这个核对动作：

1. 随机抓 3 到 5 条样本输出。
2. 手工判断它们是否有 `<think>` 和 `<action>`。
3. 手工判断它们是否合法。
4. 对照你打印的 `env_reward / format_reward / legality_reward / final_reward`。

如果这 4 步能对上，说明你今晚真的打通了 reward 设计闭环。

如果对不上，不要继续训练。立刻回去查这三层：

1. `projection` 有没有把 action 判错。
2. `EpisodeRewardManager` 用的 `response_str` 是不是你想的那个字符串。
3. 有没有忘记关闭内置 invalid penalty，导致双重计分。

---

### Step 14：把今晚的实验上升为你未来 GUI 项目的雏形判断

预计用时：20 分钟

现在请你把今晚的复合奖励映射到你的 GUI 斗地主 MVP：

1. `format_reward`
   映射为：是否稳定输出 `<think><action><chat><memory>` 这四段结构。
2. `legality_reward`
   映射为：点击区域是否合法、操作是否符合当前回合规则、出牌是否合法。
3. `env_reward`
   映射为：本回合收益、局面改善、最终胜负。

然后再写下这句非常关键的话：

`我的 GUI 斗地主 MVP 的早期训练价值，主要来自格式奖励和合法性奖励；胜负奖励是重要但更慢、更稀疏的长期信号。`

这句话如果你讲顺了，已经非常像一个真正做过 agentic RL 工程的人了。

---

## 今天一定要记住的 8 个工程判断

1. `env.rollout.n` 变大，影响的不只是 advantage，而是整条环境交互链的吞吐和调度成本。
2. 在这个仓库里，agentic RL 的 group 是环境侧构造的，不是传统 `rollout.n`。
3. `ppo_mini_batch_size` 不会因为你调大 `env.rollout.n` 就自动帮你适配。
4. `adjust_batch()` 为了凑整可能复制样本，debug group 时必须优先看 `uid` / `traj_uid`。
5. group 内 reward 全同，是 agentic RL 早期最常见的“训练看起来在跑，其实更新几乎没信息”的原因之一。
6. DAPO 的 dynamic sampling 先解决“有效组太少”的统计问题。
7. GiGPO 的价值在于把多步过程信号从“被终局吞掉”里救出来。
8. 你的 GUI agent 很快就会遇到 memory 膨胀，所以 `SimpleMemory` 只能是起点，不能是终点。

---

## 今天结束前，你必须能回答的最终问题

请你不用看代码，直接口头回答下面这组问题。

### A. 吞吐与资源

1. 为什么 `env.rollout.n` 一变大，吞吐问题会立刻变复杂？
2. 为什么这个仓库里 actor 侧 batch 设置不能完全“跟着 group size 自动走”？
3. 什么时候该优先调 `env.resources_per_worker.num_cpus`，什么时候该优先调 `ppo_micro_batch_size_per_gpu`？

### B. 统计有效性

1. 为什么很多 agentic RL 任务早期会出现 group 内 reward 全同？
2. 为什么这种情况下，训练“还在跑”不等于训练“有效”？
3. DAPO 的 dynamic sampling 到底解决了什么问题？

### C. 长轨迹与 memory

1. 为什么 `SimpleMemory` 对 MVP 很好用，但对长期项目不够？
2. 你的 GUI 斗地主 agent 里，哪些信息应该原样保留，哪些应该摘要？

### D. 奖励闭环

1. 为什么动作合法性奖励往往比胜负奖励更早产生训练价值？
2. 为什么今天做复合奖励时，要先关掉默认 invalid penalty？
3. 你今晚真正送进 GRPO 的最终 score 是什么？

---

## Day 4 面试硬核拷问

1. 为什么很多 agentic RL 任务早期会出现 group 内 reward 全同？请从环境稀疏奖励、模型早期策略塌缩、episode-level reward 压缩这三个角度一起回答。
2. DAPO 的 dynamic sampling 本质上在解决什么统计问题？为什么它不是简单的“多采样”？
3. 对 GUI agent 而言，为什么动作合法性奖励常常比胜负奖励更早产生训练价值？请从 reward 稠密度和 credit assignment 两个角度回答。
4. 什么时候应该继续用 episode-level reward，什么时候必须引入 step-level reward？请用“斗地主 GUI agent 早期 MVP”和“更成熟版本”各举一个阶段说明。
5. 你的项目为什么更像“多轮、稀疏、结构化、环境约束强”的 agentic RL，而不是普通 single-turn RLHF？

---

## 额外提醒：今天什么都不要做

1. 不要今天就深挖 critic / value model / GAE。
2. 不要一边改 reward，一边又改环境逻辑、一边又换算法。
3. 不要把 7B、大 group、长历史、复杂 reward 一次性全开。
4. 不要看到训练在跑就误以为“今天学会了”。

今天真正的胜利标准只有两个：

1. 你能解释一个真实工程现象。
2. 你能做一个小改动，让训练行为按你的解释发生变化。
