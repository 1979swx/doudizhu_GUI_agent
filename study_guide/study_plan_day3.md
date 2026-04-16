# verl-agent Day 3 超详细学习指南

这份文档只展开 [study_guide/study_plan.md](./study_plan.md) 里的 Day 3，不重写整份 5 天计划。

今天只研究一件事，而且只研究到对你未来做 agentic RL 真正有用的深度：

`多轮 agent rollout`
-> `episode reward`
-> `非法动作惩罚`
-> `GRPO advantage`
-> `actor update`

你今天故意不碰 critic / value model / GAE，不补 PPO 大一统教材，不背空公式。你的目标是把一个最关键的工程事实彻底打透：

“这套 agent rollout 最后到底是怎样变成可训练的 GRPO 更新信号的？”

如果 Day 3 学透，后面你做 GUI 斗地主 Agent 时，至少会提前避开 3 个大坑：

1. 明明 reward 写了，但统计形态已经变掉了。
2. 明明 group size 配了，但 group 的构造位置理解错了。
3. 明明训练在跑，但 actor 真正吃到的学习信号和你脑中以为的不是一回事。

---

## 0. Day 3 的作战原则

今天请一直遵守这 8 条，不然很容易把时间耗在“看起来懂了，实际上没抓住关键”的地方：

1. 先看真实调用顺序，再看公式。今天先从 `rollout_loop.py`、`episode.py`、`ray_trainer.py` 入手，`core_algos.py` 放到下午再精读。
2. 只围绕 Day 3 原计划已有内容推进，不扩散到 critic、megatron、底层 kernel。
3. 任何结论都必须有两种证据：
   一次真实运行现象，外加一段对应源码。
4. 今天主战场优先选 `AlfWorld` 文本版本。
   原因不是它更“高级”，而是它更容易把 reward/adv 统计链看清楚。
5. 如果 `AlfWorld` 因环境依赖卡住超过 20 分钟，立刻切 `Sokoban`，但 Day 3 的学习结构不变。
6. 今天每次记笔记都要写“这一层的职责边界”。
   例如：reward manager 负责什么，不负责什么；invalid penalty 负责什么，不负责什么。
7. 今天最重要的不是“记住函数名”，而是抓住“统计口径”。
   尤其是：一个轨迹被展开成多少个 step sample，每个 sample 带着什么 reward，group mean/std 是对谁算的。
8. 晚上的实验不是为了追求最优分数，而是为了制造差异，让你亲眼看到 group size、invalid penalty、KL loss 对训练信号的影响。

---

## 1. 今日总目标与时间安排

总时长按 8 小时设计。

### 你今天结束时必须达到的状态

你必须能不看代码，直接口头讲清楚下面 6 件事：

1. `env.rollout.n` 是怎样在 rollout 阶段真正变成 GRPO group 的。
2. `uid` 和 `traj_uid` 分别是什么，它们为什么都必须存在。
3. `EpisodeRewardManager` 是怎样把 episode-level reward 写回 token-level tensor 的。
4. `apply_invalid_action_penalty()` 究竟改了什么，没改什么。
5. `compute_grpo_outcome_advantage()` 到底是按“轨迹”统计，还是按“step sample”统计。
6. `update_actor()` 真正消费的字段有哪些，它已经看不到哪些 rollout 细节了。

### 时间切块

1. `09:00 - 09:30`：开工准备，只搭今天的记录区和实验底板。
2. `09:30 - 12:00`：上午主任务，先把 `rollout -> reward -> invalid penalty -> advantage -> actor update` 主链钉死。
3. `14:00 - 17:00`：下午主任务，精读 `core_algos.py`，把 GRPO 分组统计与长度偏差风险讲透。
4. `20:00 - 21:20`：做 4 个极短对照实验，观察 `env.rollout.n`、`invalid penalty`、KL loss 的影响。
5. `21:20 - 22:00`：做微型魔改任务 2，临时把 reward 拆成 3 项打印出来，再回答面试硬核拷问。

如果你作息不同，可以整体平移时间，不影响结构。

---

## 2. 开工前 30 分钟：只做最必要的准备

这一段不要急着跑长训练，也不要先读 `core_algos.py`。先把今天的“观察区”和“验证区”搭好。

### Step 1：进入正确环境

执行：

```bash
conda activate verl-agent-bw
cd /home/zhangwj/science/verl-agent
```

今天后续所有命令默认都在这个环境和仓库根目录下执行。

### Step 2：创建 Day 3 工作记录区

执行：

```bash
mkdir -p study_guide/day3_notes
mkdir -p study_guide/day3_runs
```

建议今天只用下面 4 个文件记录，不要额外开很多碎文件：

1. `study_guide/day3_notes/reward_adv_chain.md`
2. `study_guide/day3_notes/experiment_compare.md`
3. `study_guide/day3_notes/length_bias.md`
4. `study_guide/day3_notes/interview_q.md`

这 4 个文件各自只干一件事：

1. `reward_adv_chain.md`：记录“训练信号主链”的顺序、变量名、职责边界。
2. `experiment_compare.md`：记录晚上 4 个短实验的命令、日志字段、你的解释。
3. `length_bias.md`：专门记录“step 展开后为什么会引入统计权重偏差”。
4. `interview_q.md`：晚上做硬核拷问时使用。

### Step 3：只重读原计划里的 Day 3，不超过 10 分钟

只看原文件 [study_plan.md](./study_plan.md) 的 Day 3 这 8 块：

1. `当日目标`
2. `时间分配`
3. `今天重点文件`
4. `今天必须彻底搞懂的关键实现`
5. `这是今天最重要的洞察`
6. `今天必须做的实验`
7. `今日微型魔改任务 2`
8. `Day 3 产出标准`

要求：

1. 不做源码阅读。
2. 不做扩展联想。
3. 只在纸上写 3 句话：
   - 今天主战场是哪一个环境。
   - 今天只追哪一条训练信号链。
   - 今天下班时必须能讲透哪个统计陷阱。

### Step 4：今天先选定主战场环境

今天优先用 `AlfWorld`，原因很简单：

1. Day 3 的核心是 reward 和 advantage 统计，不是多模态处理。
2. `Qwen2.5-1.5B-Instruct` 调试更轻，更利于你高频重复短跑。
3. 文本环境更容易肉眼看 prompt、response、reward 的对应关系。

如果 `AlfWorld` 跑不通，再用 `Sokoban`，但你今天依然只盯同样的链：

`TrajectoryCollector.multi_turn_loop`
-> `EpisodeRewardManager`
-> `apply_invalid_action_penalty`
-> `compute_advantage`
-> `update_actor`

### Step 5：今天开工前先记住这 3 个关键事实

在开始读代码前，先把这 3 句话写进 `reward_adv_chain.md` 顶部：

1. 这个仓库的 `GRPO group` 不是在 token 采样侧用 `actor_rollout_ref.rollout.n` 复制出来的，而是在环境 rollout 侧通过 `env.rollout.n` 构造出来的。
2. 这个仓库当前默认脚本里 `algorithm.use_kl_in_reward=False`，但 `actor_rollout_ref.actor.use_kl_loss=True`。
   这意味着：KL 不是进 advantage 的 reward，而是在后面的 actor loss 里出现。
3. `EpisodeRewardManager` 返回的是 token-level tensor，但里面装的其实是 episode-level outcome reward 的“广播结果”。

这 3 句就是 Day 3 最容易讲错的地方。

---

## 3. 正式开始前，先搭 Day 3 的三条心智骨架

在正式读代码前，你先把今天要吃透的 3 条链抄下来。后面所有现象，都往这 3 条链上挂。

### 链 1：从 group rollout 到 step sample

```text
multi_turn_loop(is_train=True)
-> gen_batch.repeat(env.rollout.n, interleave=True)
-> vanilla_multi_turn_loop()
-> uid_batch 按 group_n 赋同一个 uid
-> traj_uid 每条轨迹单独一个 uuid
-> 每个 step 都 append 到 total_batch_list[i]
-> gather_rollout_data()
-> episode 统计量被塞回每个 active step sample
```

这条链回答的问题是：

“为什么一条完整轨迹最后会变成多个训练样本，而且这些样本共享同一个 episode reward？”

### 链 2：从 episode reward 到最终 token-level rewards

```text
EpisodeRewardManager()
-> reward_tensor 初始全 0
-> 每个 step sample 的最后一个 response token 写入 episode_rewards
-> ray_trainer: batch.batch["token_level_scores"] = reward_tensor
-> apply_invalid_action_penalty()
-> 如果 use_kl_in_reward=False:
   token_level_rewards = token_level_scores
```

这条链回答的问题是：

“环境 reward、非法动作 penalty、最终送去算 advantage 的 reward，三者到底是什么关系？”

### 链 3：从 uid / traj_uid 到 advantage，再到 actor update

```text
compute_advantage()
-> compute_grpo_outcome_advantage(
     token_level_rewards,
     response_mask or loss_mask,
     index=uid,
     traj_index=traj_uid
   )
-> advantages / returns 写回 batch
-> update_actor()
-> actor.update_policy() 只消费 advantages 等训练字段
```

这条链回答的问题是：

“actor 真正拿到的学习信号是什么，它还看不看得到 group、trajectory、environment 这些概念？”

---

## 4. 上午主任务：先把 reward / adv / update 主链钉死

上午最重要的原则是：

先建立清晰的调用顺序，再去理解统计意义。不要一上来陷进 `core_algos.py` 的细节里。

---

## 4.1 先读训练脚本，但只读跟 Day 3 直接相关的开关

预计用时：`15 分钟`

执行：

```bash
sed -n '1,220p' examples/grpo_trainer/run_alfworld.sh
```

你今天只盯下面这些配置：

1. `algorithm.adv_estimator=grpo`
2. `actor_rollout_ref.actor.use_kl_loss=True`
3. `actor_rollout_ref.actor.kl_loss_coef=0.01`
4. `actor_rollout_ref.actor.use_invalid_action_penalty=True`
5. `actor_rollout_ref.actor.invalid_action_penalty_coef=0.1`
6. `algorithm.use_kl_in_reward=False`
7. `env.rollout.n=$group_size`
8. `env.max_steps=50`
9. `trainer.critic_warmup=0`

你此时只需要得出下面 5 个结论：

1. 训练目标是 `GRPO`，不是 `GAE`。
2. 这里同时启用了 `invalid action penalty` 和 `KL loss`。
3. 但这里没有启用“把 KL 直接加进 reward”。
4. `group size` 来自 `env.rollout.n`。
5. 今天看 reward 链路时，必须同时把“reward 分支”和“loss 分支”区分开。

现在立刻把下面这句写进 `reward_adv_chain.md`：

```text
Day 3 最关键的辨析：
invalid penalty 改的是 token_level_scores / token_level_rewards；
actor.use_kl_loss 改的是后续 policy loss；
algorithm.use_kl_in_reward=False 意味着 KL 不进入 advantage 的 reward 端。
```

---

## 4.2 先读 rollout 端：搞清楚 step sample 到底是怎样长出来的

预计用时：`45 分钟`

先执行：

```bash
rg -n "def multi_turn_loop|def vanilla_multi_turn_loop|def gather_rollout_data|uid_batch|traj_uid|episode_rewards|episode_lengths" agent_system/multi_turn_rollout/rollout_loop.py
sed -n '230,560p' agent_system/multi_turn_rollout/rollout_loop.py
```

你不要平均阅读。今天重点只看 6 处：

1. `multi_turn_loop()` 里训练时的 `gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)`
2. `vanilla_multi_turn_loop()` 里 `uid_batch` 的构造
3. `traj_uid` 的初始化
4. `episode_rewards[active_masks] += ...`
5. `total_batch_list[i].append(batch_list[i])`
6. `gather_rollout_data()` 里把 `episode_rewards`、`episode_lengths` 塞回每个 step sample

### 你要怎样读这段代码

请按下面顺序，一步一步读，不要跳：

#### Step A：先看 `multi_turn_loop()` 的入口

你要抓住一句：

```python
gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
```

这句话的意义不是“让模型多采样几次 token”，而是：

1. 先把原始 prompt batch 在进入环境前就复制成 `group_n` 份。
2. 后续每一份副本会进入独立环境实例，长成独立轨迹。
3. 所以 Day 3 的 group 是“轨迹组”，不是“同一条 response 的 n 次重采样”。

请你现在在纸上写一个最小例子：

1. `data.train_batch_size=2`
2. `env.rollout.n=4`

然后强迫自己口述：

“原始只有 2 个 prompt，但训练 rollout 阶段实际会进入 8 个环境实例。”

#### Step B：再看 `uid_batch` 是怎样生成的

你要重点看这段逻辑：

```python
for i in range(batch_size):
    if i % self.config.env.rollout.n == 0:
        uid = str(uuid.uuid4())
    uid_batch.append(uid)
```

你要明确：

1. `uid` 不是“每条轨迹一个”。
2. `uid` 是“一组 sibling trajectories 共用一个”。
3. 每 `group_n` 条轨迹共享同一个 `uid`。

如果 `batch_size=8` 且 `group_n=4`，那么典型模式就是：

```text
index:    0 1 2 3 4 5 6 7
uid:      A A A A B B B B
traj_uid: t1 t2 t3 t4 t5 t6 t7 t8
```

这张图必须写进你的笔记里。

#### Step C：再看 `traj_uid`

你要抓住这句：

```python
traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
```

`traj_uid` 的职责是：

1. 标识单条轨迹身份。
2. 区分同组内的不同 sibling trajectory。
3. 在后面的 advantage 统计中，用于区分“跨 step 的重复样本”与“真正不同轨迹”。

现在你必须能用一句话区分：

```text
uid = group id
traj_uid = trajectory id
```

如果这句你说不顺，Day 3 后面都会乱。

#### Step D：看每个 step 是怎样被存起来的

重点看：

```python
batch_list: list[dict] = to_list_of_dict(batch)
for i in range(batch_size):
    total_batch_list[i].append(batch_list[i])
```

这段的真正含义是：

1. 每走一步环境交互，就把当前这一步的模型输入输出、reward、validity、uid、traj_uid 都存下来。
2. `total_batch_list[i]` 存的是第 `i` 条轨迹跨多个 step 的数据列表。
3. 所以后面 `gather_rollout_data()` 看到的不是“每条轨迹一个样本”，而是“每条轨迹一个 step 列表”。

#### Step E：看 `episode_rewards` 的累积方式

重点看：

```python
episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
episode_lengths[active_masks] += 1
```

你要理解：

1. 环境每一步都有即时 `rewards`。
2. 但 rollout 端先把这些 step reward 累加成 episode total。
3. 后面 reward manager 默认用的不是逐步 reward 序列，而是累加后的 `episode_rewards`。

#### Step F：最后看 `gather_rollout_data()`

重点看：

```python
data['episode_rewards'] = episode_rewards[bs]
data['episode_lengths'] = episode_lengths[bs]
...
effective_batch.append(data)
```

这是 Day 3 最关键的代码之一。你必须亲自把它翻译成人话：

1. 对于一条长度为 `L` 的轨迹，最后不会只生成 1 个训练样本。
2. 它会生成 `L` 个有效 step sample。
3. 这 `L` 个 step sample 都会带上同一个 `episode_rewards`。
4. 因此“episode reward 被广播回多个 step sample”这一事实，是后面所有统计现象的起点。

### 上午此刻你必须写下来的结论

把下面这段原封不动写进 `reward_adv_chain.md`：

```text
在 verl-agent 里，一条轨迹不是直接对应一个 GRPO 样本。
rollout 先把每个 step 都保留成一个 sample，
再把整条轨迹的 episode_rewards / episode_lengths 塞回每个 step sample。
所以后续 advantage 统计面对的是“step 展开后的样本集合”，不是“原始轨迹集合”。
```

---

## 4.3 再读 reward manager：只看 episode reward 怎样写到 token 上

预计用时：`30 分钟`

执行：

```bash
sed -n '1,220p' agent_system/reward_manager/episode.py
```

今天只盯下面 4 处：

1. `reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)`
2. `episode_rewards = data_item.non_tensor_batch['episode_rewards']`
3. `score = episode_rewards / episode_lengths` 或 `score = episode_rewards`
4. `reward_tensor[i, valid_response_length - 1] = ...`

### 你要怎样理解这段代码

#### Step A：先看 reward tensor 的形状

`reward_tensor` 的 shape 跟 `responses` 一样。

这意味着：

1. trainer 后面希望拿到 token-level reward tensor。
2. 但当前 reward manager 手上只有 scalar episode reward。
3. 所以它采取的是“只在每个 response 的最后一个有效 token 上放分数，其余位置为 0”的做法。

#### Step B：再看 reward 的来源

这里的 `episode_rewards` 根本不是 reward manager 自己算出来的。

它是：

1. rollout 阶段已经累计好的 episode total reward。
2. 在 `gather_rollout_data()` 中被附着到每个 step sample 上。
3. reward manager 只是把这个 episode-level scalar 转成 token-level tensor。

所以 reward manager 的职责边界是：

1. 它负责“重排 reward 的表示形式”。
2. 它不负责决定 group。
3. 它不负责判断动作是否合法。
4. 它也不负责算 KL。

#### Step C：注意它默认是 outcome reward，不是逐 token reward

当前默认逻辑：

1. 一条 step sample 的 response 序列，大部分 token reward 都是 0。
2. 只有最后一个 token 位置拿到分数。
3. 这个分数来自整条 episode，而不是当前这一步的局部表现。

这件事非常重要，因为它意味着：

1. token 级 tensor 的形式不代表 reward 本质上就是 token-level supervision。
2. 当前实现本质上还是 outcome reward。

#### Step D：你现在要做一个 2 分钟口头复述

对着空气说下面这段话，直到你说顺为止：

```text
EpisodeRewardManager 并没有发明新的 reward。
它拿到的是 rollout 端已经累计好的 episode_rewards，
然后把这个 episode-level scalar 写到每个 step sample 最后一个有效 token 上，
从而适配后续 trainer 统一的 token-level reward 接口。
```

如果你此刻说不顺，后面就很容易把“episode reward”和“token-level tensor”混为一谈。

---

## 4.4 回到 `ray_trainer.py`：把训练信号顺序彻底钉死

预计用时：`60 分钟`

执行：

```bash
sed -n '244,360p' verl/trainer/ppo/ray_trainer.py
sed -n '1180,1260p' verl/trainer/ppo/ray_trainer.py
```

今天只看下面这个顺序，不要被别的分支吸走注意力：

1. `batch.batch["token_level_scores"] = reward_tensor`
2. `apply_invalid_action_penalty(...)`
3. `apply_kl_penalty(...)` 或 `batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]`
4. `compute_advantage(...)`
5. `update_actor(...)`

### 你必须亲手写出这条顺序的“中文翻译版”

把下面这 5 句写进 `reward_adv_chain.md`：

1. `token_level_scores` 是 reward manager 刚吐出来的“基础分数张量”。
2. `apply_invalid_action_penalty()` 会直接原地修改 `token_level_scores`。
3. 如果 `algorithm.use_kl_in_reward=False`，那 `token_level_rewards` 就只是修改后的 `token_level_scores`。
4. `compute_advantage()` 用的是 `token_level_rewards`，不是原始 `episode_rewards`。
5. `update_actor()` 吃的是 `advantages`，不是 `uid` 或 `episode_rewards` 本身。

### 你现在要特别警惕的 3 个误解

#### 误解 1：`invalid penalty` 发生在环境里

不对。

当前仓库里：

1. 动作合法性标签 `is_action_valid` 由环境 / projection / env manager 链路产出。
2. 真正把“非法动作惩罚”减到 reward 上，是在 `ray_trainer.py` 的 `apply_invalid_action_penalty()`。

#### 误解 2：脚本里打开了 KL，所以 advantage 里一定已经包含 KL

不对。

你今天必须区分两件事：

1. `actor_rollout_ref.actor.use_kl_loss=True`
   这是在 actor loss 里加 KL 正则。
2. `algorithm.use_kl_in_reward=False`
   这表示 advantage 的 reward 端不含 KL 惩罚。

所以当前默认训练脚本下：

```text
advantage 主要由 outcome reward + invalid penalty 决定；
KL 主要在 actor loss 阶段影响更新。
```

#### 误解 3：`update_actor()` 仍然知道 group 的细节

也不对。

`uid` 和 `traj_uid` 的主要价值已经在 `compute_advantage()` 阶段被消化掉了。到了 actor 这边，真正留下来的核心训练字段已经是：

1. `responses`
2. `input_ids`
3. `attention_mask`
4. `position_ids`
5. `old_log_probs`
6. `advantages`
7. 多轮时额外的 `loss_mask`
8. 若启用 KL loss，则还需要 `ref_log_prob`

---

## 4.5 上午收口：做第一次“只讲主链不讲公式”的复述

预计用时：`30 分钟`

现在不要看任何代码，对自己做 5 分钟复述，内容必须包含下面这 6 句：

1. `gen_batch.repeat(env.rollout.n)` 发生在 environment rollout 前。
2. `uid` 是 group id，`traj_uid` 是 trajectory id。
3. rollout 会保留每个 step sample，而不是只保留完整轨迹终点。
4. `EpisodeRewardManager` 把整条 episode 的 reward 写到每个 step sample 最后一个 response token。
5. `apply_invalid_action_penalty()` 修改的是 token-level reward tensor。
6. `compute_advantage()` 之后，actor 真正看到的是 `advantages` 而不是原始 episode reward。

如果这 6 句你还说不顺，下午不要急着进 `core_algos.py`。
先把上午这些代码再过一遍。

---

## 5. 下午主任务：精读 `core_algos.py`，把 GRPO 统计意义讲透

下午的目标不是背公式，而是弄清楚这套实现到底在“对谁求平均、对谁求方差、谁被重复计权了”。

---

## 5.1 先看 `compute_grpo_outcome_advantage()` 的输入输出映射

预计用时：`20 分钟`

执行：

```bash
sed -n '100,180p' verl/trainer/ppo/core_algos.py
```

先不要逐行读，你只做变量映射：

1. `token_level_rewards`：已经过 reward manager 和 invalid penalty 处理后的 token 级 reward。
2. `response_mask`：哪些 response token 参与 advantage。
3. `index`：这里传进来的是 `uid`。
4. `traj_index`：这里传进来的是 `traj_uid`。
5. `scores = token_level_rewards.sum(dim=-1)`：每个 step sample 先压成一个 scalar score。

你此时必须得出一句非常关键的话：

```text
GRPO advantage 不是直接拿 episode_rewards 算的，
而是先拿每个 step sample 的 token_level_rewards 做 sum，
再按 uid 分组做归一化。
```

---

## 5.2 一行一行拆 `compute_grpo_outcome_advantage()`

预计用时：`60 分钟`

现在重新打开同一段代码，一行一行读。

### Step A：先看 `scores = token_level_rewards.sum(dim=-1)`

你要立即联想到上午的事实：

1. 对于一个 step sample，reward manager 只在最后一个有效 token 上放了 episode reward。
2. 所以这里的 `sum(dim=-1)` 本质上会把那个 scalar reward 再拿出来。
3. 如果同一条轨迹有多个 step sample，那么这些 step sample 的 `scores` 会是重复的 episode reward 值。

这就是“step 展开后的重复计数”起点。

### Step B：再看 `id2score`

你要重点盯住：

```python
id2score[index[i]].append(scores[i])
```

这意味着：

1. 这里只按 `uid` 聚组。
2. 只要 `uid` 相同，这个 sample 的 score 就会进入同一个 group 的均值和方差统计。
3. 默认情况下，不管这个 sample 是同一条轨迹的第 1 步还是第 7 步，只要它是一个有效 step sample，它就会参与统计。

### Step C：理解 `seen_pairs` 和 `compute_mean_std_cross_steps`

这是 Day 3 最需要你细嚼慢咽的逻辑。

代码里有：

```python
seen_pairs = set()
...
if (index[i], traj_index[i]) in seen_pairs:
    continue
...
if not compute_mean_std_cross_steps:
    seen_pairs.add((index[i], traj_index[i]))
```

你必须明确：

1. 当 `compute_mean_std_cross_steps=True` 时，`seen_pairs` 根本不会起去重作用。
2. 这意味着同一条轨迹跨多个 step 的重复 sample，都会进入 group mean/std 统计。
3. 当前 `compute_grpo_outcome_advantage()` 的默认参数就是 `compute_mean_std_cross_steps=True`。
4. 而 `ray_trainer.py` 调这个函数时，没有覆盖这个参数。

所以当前默认实现的真实语义是：

```text
GRPO 的 mean/std 默认是在“step 展开后的样本级别”统计，
而不是严格按“每条轨迹一票”统计。
```

这句必须写进 `length_bias.md` 顶部。

### Step D：再看 mean / std 的用途

代码本质在做：

```text
adv = (score - group_mean) / group_std
```

或者在不按 std 归一化时：

```text
adv = score - group_mean
```

这说明：

1. `GRPO` 的核心不是“奖励越高越直接更新”，而是“相对组内基线更高或更低”。
2. 所以 group 内 reward 如果全一样，优势就会塌掉。
3. 这也是为什么 Day 3 原计划里专门提到“group 内全同 reward”和 DAPO dynamic sampling 的关系。

### Step E：最后看返回值

```python
scores = scores.unsqueeze(-1) * response_mask
return scores, scores
```

你要理解：

1. 这里返回的 `advantages` 和 `returns` 都是把 scalar group-normalized score 重新铺回 token 维度。
2. 只有被 `response_mask` 选中的 token 位置保留这个 advantage。
3. 对当前 outcome reward 场景而言，这是一种统一接口适配，而不是说 reward 天然就是每个 token 都有不同值。

---

## 5.3 手算一个最小例子：亲手看见“长度偏差”是怎么出现的

预计用时：`40 分钟`

这一段不要偷懒。你如果不手算，Day 3 最重要的洞察很难真正变成你的直觉。

### 例子设定

假设同一个 `uid=A` 下面有两条轨迹：

1. `traj_uid=t1`
   - 总 reward = `1.0`
   - 轨迹长度 = `2`
   - 所以 rollout 展开后，会产生 `2` 个 step sample
   - 这 `2` 个 step sample 的 `score` 都是 `1.0`
2. `traj_uid=t2`
   - 总 reward = `0.0`
   - 轨迹长度 = `5`
   - 所以 rollout 展开后，会产生 `5` 个 step sample
   - 这 `5` 个 step sample 的 `score` 都是 `0.0`

### 如果你按“轨迹级”理解

很多人脑中会下意识觉得：

```text
group_mean = (1.0 + 0.0) / 2 = 0.5
```

但这不是当前默认实现的口径。

### 当前默认实现的真实口径

实际进入 `id2score[A]` 的，是：

```text
[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
```

所以：

```text
group_mean = 2 / 7 ≈ 0.286
```

你现在应该立刻意识到：

1. 长轨迹 `t2` 因为 step sample 更多，对 group 统计的权重更大。
2. 即使两条轨迹在“轨迹数”层面是一比一，它们在“当前 advantage 统计”里也不是一比一。
3. 这就是 Day 3 原计划里说的：
   “step 展开后的样本会参与 advantage 统计，这意味着长轨迹和短轨迹的统计权重不一定相同。”

### 这件事为什么对你未来项目重要

你未来做 GUI 斗地主 Agent 时，非常可能出现：

1. 一些局面很快结束。
2. 一些局面会拖很长。
3. 如果你继续沿用这种“step 展开后再按 group 统计”的做法，长局面可能天然带来更大统计权重。

这不一定绝对错误，但你必须知道它在发生。

### 现在把这 4 句话写进 `length_bias.md`

1. 当前默认实现更接近“sample-level group normalization”，不是纯轨迹级 normalization。
2. 轨迹长度分布会影响 group mean/std。
3. reward 设计正确，不等于 reward 统计口径正确。
4. 面试时要讲的是“这个偏差来自 rollout 展开与 group 归一化的组合”，不是笼统说“长轨迹不好”。

---

## 5.4 最后看 actor 端：确认 actor 真正吃到了什么

预计用时：`30 分钟`

执行：

```bash
sed -n '600,660p' verl/workers/fsdp_workers.py
sed -n '317,445p' verl/workers/actor/dp_actor.py
```

今天你只盯两个函数：

1. `FSDPWorker.update_actor()`
2. `DataParallelPPOActor.update_policy()`

### 你读这两段代码时，只回答两个问题

#### 问题 1：actor update 真正消费哪些字段？

你要抓住这句：

```python
select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "old_log_probs", "advantages"]
```

多轮时再加：

```python
loss_mask
```

若启用 KL loss，再加：

```python
ref_log_prob
```

所以你必须明确：

1. `uid` 没进 actor update。
2. `traj_uid` 没进 actor update。
3. `episode_rewards` 也没直接进 actor update。
4. 这些 rollout 语义已经在 advantage 计算阶段被折叠掉了。

#### 问题 2：KL 在哪一步影响 actor？

你要抓住：

```python
if self.config.use_kl_loss:
    ...
    policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
```

所以 Day 3 你必须能把两条支路分开讲：

1. reward 支路：
   `EpisodeRewardManager -> invalid penalty -> compute_advantage`
2. loss 支路：
   `old_log_probs / ref_log_prob -> policy loss + KL loss -> optimizer step`

这两个分支混讲，是很多人面试时最容易暴露“其实没真懂实现”的地方。

---

## 5.5 下午收口：写一页“训练信号主链图”

预计用时：`30 分钟`

现在你必须在纸上或笔记里画一张单页图，图里至少要有下面这些节点：

1. `gen_batch.repeat(env.rollout.n)`
2. `uid`
3. `traj_uid`
4. `episode_rewards`
5. `EpisodeRewardManager`
6. `token_level_scores`
7. `apply_invalid_action_penalty`
8. `token_level_rewards`
9. `compute_grpo_outcome_advantage`
10. `advantages`
11. `update_actor`
12. `KL loss`

要求：

1. 用箭头把 reward 支路和 loss 支路分开。
2. 在 `compute_grpo_outcome_advantage` 旁边写一句：
   `默认按 step 展开后的 sample 做 group 统计`
3. 在 `uid` 旁边写：
   `group id`
4. 在 `traj_uid` 旁边写：
   `trajectory id`

如果你能把这张图画准确，Day 3 已经完成了一半。

---

## 6. 晚上主任务：做 4 个极短对照实验，让差异自己冒出来

晚上这 4 个实验，不是为了刷分，而是为了让你亲眼看到：

1. group size 变了，训练信号和吞吐会怎么变。
2. invalid penalty 变了，reward 形态和合法动作率会怎么变。
3. KL 虽然不进 reward，但仍然会在 actor loss 里留下痕迹。

### 实验总原则

1. 所有实验都只跑 `1 epoch`。
2. 所有实验都用同一套基础命令，只改一个关键变量。
3. 所有实验都把输出重定向到 `study_guide/day3_runs/` 里。
4. 所有实验都至少记录 4 个字段：
   - `episode/valid_action_ratio`
   - `val/*/test_score`
   - `actor/kl_loss` 或 `actor/ppo_kl`
   - 你对“group 内 reward 是否更容易全同”的主观判断

### 建议的基础命令

预计每次运行：`12 到 20 分钟`

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.max_steps=8 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4
```

如果 `AlfWorld` 当晚不稳定，就把命令里的脚本换成 `run_sokoban.sh`，其它 override 保持同样逻辑。

### 实验 1：只改 group size 为 2

执行：

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.max_steps=8 \
  env.rollout.n=2 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  > study_guide/day3_runs/alf_n2.log 2>&1
```

你要观察：

1. `val/*/test_score`
2. `episode/valid_action_ratio`
3. 每个 batch 的实际吞吐是否更轻快
4. group 内是否更容易“信息不足”

你要怎么解释：

1. `n=2` 时，同一问题下只有 2 条 sibling trajectories。
2. group baseline 更粗糙，组内比较信息更少。
3. 但环境并发更低，单次 rollout 压力更小。

### 实验 2：只改 group size 为 8

执行：

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.max_steps=8 \
  env.rollout.n=8 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  > study_guide/day3_runs/alf_n8.log 2>&1
```

你要观察：

1. 吞吐是否下降
2. `val/*/test_score` 是否更稳定
3. group 内 reward 是否更不容易完全相同
4. 资源调度是否更紧

你要怎么解释：

1. `n=8` 提供了更丰富的组内比较。
2. 但环境数、rollout 数、log_prob 计算压力都会上升。
3. 这就是工业落地里常见的 trade-off：
   “统计更稳”与“吞吐更高”很少同时免费拿到。

### 实验 3：固定 `env.rollout.n=4`，把 invalid penalty 设为 0

执行：

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.max_steps=8 \
  env.rollout.n=4 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  > study_guide/day3_runs/alf_penalty0.log 2>&1
```

你要观察：

1. `episode/valid_action_ratio`
2. reward 是否变得更“宽松”
3. 非法动作是否不再被 reward 直接拉低
4. `actor/kl_loss` 是否仍然存在

你要怎么解释：

1. penalty=0 只影响 reward 支路。
2. 它不会让 `is_action_valid` 指标本身神奇消失。
3. 它也不会把 KL loss 关掉。

### 实验 4：固定 `env.rollout.n=4`，把 invalid penalty 设为 0.1

执行：

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.max_steps=8 \
  env.rollout.n=4 \
  ray_init.num_cpus=64 \
  actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  > study_guide/day3_runs/alf_penalty01.log 2>&1
```

你要观察：

1. 非法动作样本的最终 reward 是否被明显压低
2. `episode/valid_action_ratio` 的后续变化趋势
3. `val/*/test_score` 是否有任何明显响应

### 做完 4 个实验后，统一抽日志

执行：

```bash
rg -n "valid_action_ratio|test_score|kl_loss|ppo_kl" study_guide/day3_runs/*.log
```

然后把每个实验用下面的 4 行格式写进 `experiment_compare.md`：

```text
实验名：
我看到的现象：
对应代码位置：
我的解释：
```

### 晚上实验时你必须保持的清醒

1. 1 个 epoch 的短跑，不足以证明最终性能优劣。
2. 但足以暴露“训练信号形状”与“工程代价”的差异。
3. 你今天追求的是“定性理解”，不是“统计显著性”。

---

## 7. 微型魔改任务 2：把 reward 拆成 3 项，并临时打日志

这是 Day 3 最关键的动手部分。

今天不是让你设计一个完美奖励系统，而是让你亲手看见：

1. 环境原始 reward
2. 非法动作 penalty
3. 最终送入 GRPO 的 token-level score

这 3 项在代码里是怎样拼起来的。

---

## 7.1 先决定改哪一层

预计用时：`5 分钟`

今天最合适的落点是 [ray_trainer.py](../verl/trainer/ppo/ray_trainer.py)，不是 `episode.py`。

原因：

1. `episode.py` 只能看到 reward manager 产出的基础 reward tensor。
2. `invalid penalty` 发生在 `ray_trainer.py`。
3. 最终送入 GRPO 的 `token_level_rewards` 也是在 `ray_trainer.py` 这一段定稿。

所以 Day 3 的临时日志，最适合插在 `with _timer("adv", timing_raw):` 这段里。

---

## 7.2 你要加什么日志

预计用时：`15 分钟`

打开：

```bash
sed -n '1188,1248p' verl/trainer/ppo/ray_trainer.py
```

你要临时打印下面这 6 样东西：

1. `uid`
2. `traj_uid`
3. `episode_lengths`
4. 环境原始 score
5. invalid penalty 大小
6. 最终送入 advantage 的 score

### 推荐的临时代码块

把下面这段逻辑插在 reward / invalid penalty / final reward 都处理完之后、`compute_advantage()` 之前。

你不需要一字不差照抄，但语义要一致：

```python
raw_scores_before_invalid = batch.batch["token_level_scores"].sum(-1).detach().cpu().numpy().copy()

invalid_mask = 1 - batch.non_tensor_batch["is_action_valid"].astype(np.float32)
invalid_penalty = self.config.actor_rollout_ref.actor.invalid_action_penalty_coef * invalid_mask

if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
    batch, invalid_metrics = apply_invalid_action_penalty(
        batch,
        invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
    )
    metrics.update(invalid_metrics)

if self.config.algorithm.use_kl_in_reward:
    batch, kl_metrics = apply_kl_penalty(
        batch,
        kl_ctrl=self.kl_ctrl_in_reward,
        kl_penalty=self.config.algorithm.kl_penalty,
    )
    metrics.update(kl_metrics)
else:
    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

final_scores_for_grpo = batch.batch["token_level_rewards"].sum(-1).detach().cpu().numpy().copy()

debug_show = min(12, len(raw_scores_before_invalid))
print("[reward_debug] idx uid traj_uid ep_len is_valid raw_score invalid_penalty final_score")
for i in range(debug_show):
    print(
        "[reward_debug]",
        i,
        batch.non_tensor_batch["uid"][i],
        batch.non_tensor_batch["traj_uid"][i],
        batch.non_tensor_batch["episode_lengths"][i],
        batch.non_tensor_batch["is_action_valid"][i],
        raw_scores_before_invalid[i],
        invalid_penalty[i],
        final_scores_for_grpo[i],
    )
```

### 这段日志最重要的观察点是什么

不是“看数字漂不漂亮”，而是看下面 3 件事：

1. 同一个 `traj_uid` 是否会连续出现多次。
   如果会，说明你真的看到了 step 展开。
2. 同一个 `traj_uid` 的 `raw_score` 是否会重复。
   如果会，说明你真的看到了 episode reward 被广播给多个 step sample。
3. `final_score` 是否恰好等于 `raw_score - invalid_penalty`。
   在当前默认 `use_kl_in_reward=False` 的脚本下，应该接近这个关系。

---

## 7.3 跑一个最小验证实验

预计用时：`15 到 20 分钟`

插完临时代码后，跑一个最短实验：

```bash
bash examples/grpo_trainer/run_alfworld.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=2 \
  data.val_batch_size=4 \
  env.rollout.n=4 \
  env.max_steps=6 \
  ray_init.num_cpus=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
```

你只看 `reward_debug` 那几行，不看别的。

### 你应该期待看到什么

最理想的观测是：

1. 你会看到同一个 `uid` 下有多条不同 `traj_uid`。
2. 你会看到同一个 `traj_uid` 重复出现多次。
3. 这些重复行的 `episode_lengths` 相同，`raw_score` 也相同。
4. 如果该 step sample 对应非法动作，`final_score` 会比 `raw_score` 少一截 penalty。

如果你看到了这 4 点，说明 Day 3 最关键的统计链你已经真正“看见”了，不只是抽象理解。

### 做完后一定要撤掉临时代码

因为这是 Day 3 的学习仪器，不是长期特性。

撤掉时你要做的不是“无脑 revert”，而是：

1. 看一遍你自己加的日志块。
2. 确认自己已经明白每个变量的来源。
3. 再把这段临时代码手动删掉。

这样 Day 3 的动手就不只是“打印了一堆数”，而是一次真正的认知闭环。

---

## 8. 今天最后 40 分钟：整理产出，并回答硬核拷问

现在不要再开新代码文件。你只做“收口”。

### 你今天必须交付给自己的 5 个产出

1. 一张“训练信号主链图”。
2. 一个 `reward_adv_chain.md`，里面用你自己的话写清每一层职责。
3. 一个 `length_bias.md`，里面把“step 展开导致的统计权重偏差”讲透。
4. 一个 `experiment_compare.md`，里面至少有 4 个短实验的现象与解释。
5. 一个你能口头讲顺的答案：
   “为什么 reward 设计正确，不等于 reward 统计口径正确？”

### Day 3 产出标准

今天结束时，你必须能脱稿解释：

1. 这个仓库的 GRPO group 是怎么构造出来的。
2. 为什么 reward manager、invalid penalty、GRPO advantage 是三个不同层次。
3. 为什么“能跑通”和“reward 统计正确”完全不是一回事。

如果这 3 条你还讲不顺，Day 4 不要急着攻吞吐和工程壁垒。
先把 Day 3 补扎实。

---

## 9. Day 3 面试硬核拷问

下面这 5 题就是今晚的自测题。不要只写结论，要强迫自己把“代码位置 + 现象 + 工程含义”一起说出来。

### 题 1

为什么 `verl-agent` 里 GRPO 分组必须绑定环境 reset，而不是只在 token 采样阶段做 `n` 次生成？

一个过关答案至少要覆盖：

1. `env.rollout.n` 对应的是多条独立环境轨迹。
2. agentic RL 需要环境交互、动作合法性、记忆状态、终止条件共同演化。
3. 单纯 token 侧 `n` 次采样，不会自然得到多条真正独立的 agent-environment trajectory。

### 题 2

`uid` 和 `traj_uid` 分别解决什么问题？

一个过关答案至少要覆盖：

1. `uid` 是 group id，用于 GRPO 组内比较。
2. `traj_uid` 是单条轨迹 id，用于区分同组内不同轨迹，以及识别跨 step 的重复 sample。
3. 如果没有 `traj_uid`，你很难讨论“同一条轨迹在 step 展开后被重复统计”这件事。

### 题 3

这个仓库的 outcome reward 在 step 展开后会带来什么统计偏差风险？

一个过关答案至少要覆盖：

1. episode reward 会被广播到同一轨迹的多个 step sample。
2. 默认 `compute_mean_std_cross_steps=True` 时，group mean/std 是对这些 step sample 统计。
3. 因而长轨迹可能天然拥有更大的统计权重。

### 题 4

非法动作 penalty 放在 reward 之后而不是 projection 里直接丢弃，有什么好处？

一个过关答案至少要覆盖：

1. projection 负责解析和判合法，不宜直接承担最终 reward 设计。
2. 保留样本再施加 penalty，可以让模型看到“非法动作的负反馈”，而不是把失败样本完全从训练中抹掉。
3. 这对你未来做“格式奖励 + GUI 合法性奖励 + 胜负奖励”的分层设计非常重要。

### 题 5

如果 group 内 reward 全一样，GRPO 会发生什么？这和 DAPO 的 dynamic sampling 有什么关系？

一个过关答案至少要覆盖：

1. group 内 reward 全同，组内差分信号会塌掉。
2. advantage 会接近零或缺乏区分度。
3. DAPO 风格的 dynamic sampling 本质上是在尽量过滤掉“整组无差异”的 group。

---

## 10. 结束 Day 3 前，最后再提醒你一次今天真正该带走什么

今天的真正收获，不是“我看过了 `core_algos.py`”，而是下面这 4 句话已经变成了你的直觉：

1. agentic RL 的 reward 问题，本质上经常是“统计口径问题”，不只是“奖励函数定义问题”。
2. `uid`、`traj_uid`、step 展开、episode reward 广播，这 4 件事一旦串起来，你才真正看懂了这套 GRPO 实现。
3. 当前默认脚本下，invalid penalty 在 reward 端，KL 在 actor loss 端，它们不是一回事。
4. 你未来做 GUI 斗地主 Agent 时，最该复用的不只是环境接入框架，更是今天学会的“训练信号拆解方式”。

如果 Day 3 结束时你能把这 4 句讲得自然、具体、带代码依据，那么这一天就算真正学到了。
