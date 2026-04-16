# verl-agent Day 1 超详细学习指南

这份文档只展开 [study_guide/study_plan.md](./study_plan.md) 中的 Day 1，不重写整份 5 天计划。

你的 Day 1 只有一个核心任务：

先亲手跑通一个最小 agentic GRPO 闭环，再用代码解释你刚刚亲眼看到的现象，完成第一轮“现象 -> 代码 -> 再回到现象”的闭环。

今天故意不追求“大而全”。你不需要吃透 critic、value model、GAE，也不需要按目录扫源码。你只需要把下面这条主干跑顺：

`run_alfworld.sh`
-> `examples.data_preprocess.prepare`
-> `verl.trainer.main_ppo`
-> `agent_system.environments.make_envs`
-> `TrajectoryCollector.multi_turn_loop`
-> `envs.step`
-> `EpisodeRewardManager`
-> `compute_advantage`
-> `update_actor`

---

## 0. Day 1 的作战原则

今天请一直遵守这 6 条，不然很容易学偏：

1. 先跑，再看代码。没有日志现象支撑的源码阅读，效率会很低。
2. 只围绕主链读代码。凡是偏离“数据 -> 环境 -> rollout -> reward -> advantage -> actor update”的，今天先压住。
3. 不碰 critic / value model 细节。它们今天只作为“trainer 里存在的背景模块”。
4. 每读完一个文件，必须回答一句话：“它对主链贡献了什么？”
5. 每 60 到 90 分钟必须产出一个可验证结果，比如一条日志解释、一张链路图、一条命令输出、一段自己的口头复述。
6. 一旦 AlfWorld 卡环境，不要死磕半天，立即切 Sokoban 跑最小闭环，下午的主链阅读依然照常推进。

---

## 1. 今日总目标与时间安排

总时长按 8 小时设计。

### 你今天结束时必须达到的状态

你必须能不看代码，直接口述下面 3 件事：

1. 一个 batch 是怎样从 parquet 样本变成多条环境轨迹的。
2. GRPO 的 group 是在哪里被复制出来的，为什么这里不是靠 `actor_rollout_ref.rollout.n`。
3. episode reward 是怎样被回填到 token 级别，然后再进入 advantage 计算的。

### 时间切块

1. 09:00 - 09:30：开工准备，确认环境、目录、目标。
2. 09:30 - 11:30：跑最小实验，收集第一手日志与现象。
3. 11:30 - 12:00：做第一次口头复盘，只用“现象语言”不看代码。
4. 14:00 - 16:30：按主链顺序读 7 个关键文件，只解释上午观察到的现象。
5. 16:30 - 18:00：把“日志字段 -> 代码位置 -> 机制解释”连起来。
6. 20:00 - 22:00：整理单页主链路草图，回答硬核拷问，补齐 Day 1 产出。

如果你习惯晚起，可以整体平移时间，不影响结构。

---

## 2. 开工前 30 分钟：只做最必要的准备

这一段不要沉迷“配置优化”。今天的目标不是把环境打磨到完美，而是尽快看到闭环。

### Step 1：进入正确环境

执行：

```bash
conda activate verl-agent-bw
cd /home/zhangwj/science/verl-agent
```

目的：

1. 你后面所有命令都默认以这个 conda 环境和仓库根目录为基准。
2. 原仓库说明里已经明确，这个环境里安装了主训练依赖和 gym-cards、sokoban 等环境。

### Step 2：确认目标文件存在

执行：

```bash
ls study_guide
ls examples/grpo_trainer
```

你现在只需要肉眼确认这几件事：

1. `study_guide/study_plan.md` 在。
2. `study_guide/study_plan_day1.md` 在。
3. `examples/grpo_trainer/run_alfworld.sh` 在。
4. `examples/grpo_trainer/run_sokoban.sh` 在。

不要继续乱翻目录。确认完就停。

### Step 3：先读一遍 Day 1 的原计划，不超过 10 分钟

只看原文件里的 Day 1 这几段：

1. `当日目标`
2. `时间分配`
3. `今天必须跑的实验`
4. `今天必须读懂的文件`
5. `今天必须搞清楚的关键问题`
6. `今日实践任务`
7. `Day 1 产出标准`
8. `Day 1 面试硬核拷问`

要求：

1. 不做摘抄。
2. 不做扩展联想。
3. 只在纸上写 3 句话：
   - 今天先跑什么。
   - 今天只读哪几段主链代码。
   - 今天下班时要能说清楚什么。

### Step 4：创建一个 Day 1 工作记录区

执行：

```bash
mkdir -p study_guide/day1_notes
```

建议今天临时用下面 3 个文件记录：

1. `study_guide/day1_notes/obs.md`
2. `study_guide/day1_notes/code_map.md`
3. `study_guide/day1_notes/interview_q.md`

作用分别是：

1. `obs.md`：只记现象，不解释。
2. `code_map.md`：只记“哪个现象对应哪段代码”。
3. `interview_q.md`：晚上自测用。

---

## 3. 上午主任务：跑出第一个最小闭环

这一段是今天最重要的部分。你不是为了“训练出效果”，而是为了看到整个 agentic RL 管线真的动起来。

## 3.1 先读训练脚本，但只读 15 分钟

先打开：

```bash
sed -n '1,220p' examples/grpo_trainer/run_alfworld.sh
```

你只看这几个东西，不要逐行深究：

1. `ENGINE=${1:-vllm}`
2. `train_data_size=16`
3. `val_data_size=128`
4. `group_size=8`
5. `python3 -m examples.data_preprocess.prepare`
6. `python3 -m verl.trainer.main_ppo`
7. `algorithm.adv_estimator=grpo`
8. `env.env_name=alfworld/AlfredTWEnv`
9. `env.rollout.n=$group_size`
10. `trainer.n_gpus_per_node=2`

你现在只需要得出 4 个结论：

1. 它先造一个很小的 parquet 数据集，再启动 trainer。
2. 这里显式用的是 `main_ppo`，不是某个单独叫 `main_grpo` 或 `main_agent` 的入口。
3. 这里把 group size 交给了 `env.rollout.n`。
4. 这个脚本天然就是双卡配置，和你的硬件匹配。

把这 4 句话写进 `obs.md`。

## 3.2 先跑最小版 AlfWorld

执行：

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

### 为什么今天一定要用这个“小步快跑”版本

1. `trainer.total_epochs=1`：今天不是练模型，是做链路认知验证。
2. `data.train_batch_size=4`：方便你用脑子跟住 batch 的流动。
3. `env.rollout.n=4`：恰好能形成最小 GRPO group，对现象最清楚。
4. `env.max_steps=6`：能看到多轮交互，但不会拖太长。
5. `ray_init.num_cpus=64`：避免 Ray 过度吃满机器，减少调度噪音。
6. micro batch 调小：为了尽快启动，降低你在 Day 1 遇到显存/吞吐问题的概率。

### 跑的时候你只盯 4 类输出

第一类：数据准备阶段

你大概率会看到类似信息：

1. `processing data for mode: text`
2. `dataset len: ...`
3. `filter dataset len: ...`

你此时要意识到：

1. 训练脚本确实先走了 `examples.data_preprocess.prepare`。
2. 数据集是被显式构造成 parquet 的。
3. 这一步的意义更像“给 trainer 喂一个模态正确、大小正确的起点”，不是给标准答案监督。

第二类：Ray / Hydra / 配置打印

你大概率会看到：

1. `ray init kwargs: ...`
2. 一大段 resolved config

你此时不要害怕配置巨长。今天只查 6 个键是否真的生效：

1. `algorithm.adv_estimator=grpo`
2. `data.train_batch_size=4`
3. `data.val_batch_size=8`
4. `env.rollout.n=4`
5. `env.max_steps=6`
6. `actor_rollout_ref.rollout.n=1`

第 6 个最关键。因为它直接验证“这里的 group 不是靠 actor rollout 的 n”。

第三类：训练过程日志

你今天重点盯下面 8 个键，看到 5 个以上就够了：

1. `training/global_step`
2. `episode/reward/mean`
3. `episode/length/mean`
4. `episode/valid_action_ratio`
5. `critic/score/mean`
6. `critic/advantages/mean`
7. `timing_s/gen`
8. `perf/throughput`

第四类：验证日志

如果验证顺利跑到，常见关注点是：

1. `val/.../test_score`
2. `val/...success_rate...`

Day 1 只把它们当成“验证环节确实存在”的证据，不需要深究指标优化。

## 3.3 一旦卡住，按这个顺序排障

你今天最忌讳的事，是遇到报错后开始盲修一小时。请按下面顺序判断。

### 情况 A：一开始就卡在环境导入、依赖、路径

典型表现：

1. AlfWorld import error
2. 找不到环境资源
3. 某个第三方组件没装好

你的动作：

1. 先把报错原文粘到 `obs.md`。
2. 标记为“环境问题，不是主链机制问题”。
3. 立刻切换到 Sokoban 最小实验。

执行：

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

为什么这样做是对的：

1. 你 Day 1 的核心目标是吃透主链，不是解决某个具体环境生态问题。
2. Sokoban 依然保留了“环境 + rollout + reward + GRPO group”这条链，足够完成今天任务。

### 情况 B：能启动，但显存紧张或启动特别慢

先做这 4 个判断：

1. 是否已经把 `ppo_micro_batch_size_per_gpu`、`log_prob_micro_batch_size_per_gpu` 调小。
2. 是否已经把 `data.train_batch_size` 压到 4。
3. 是否把 `env.max_steps` 压到 6。
4. 是否把 `env.rollout.n` 压到 4 而不是 8。

今天不要先做复杂性能调优。只要能稳定跑到一个 step，你就赢了。

### 情况 C：训练跑了，但你看不懂日志

不要立刻去翻所有 logger 代码。

你现在只做：

1. 找到 5 个日志字段。
2. 每个字段后写一句“它描述的是训练中的哪一层现象”。
3. 下午再回到代码查产地。

## 3.4 上午结束前，必须做一次纯“现象复盘”

11:30 左右停下来，暂时不要看源码。

打开 `obs.md`，强迫自己只用自然语言回答下面 6 句：

1. 这个训练脚本大体分成哪两大阶段。
2. 为什么我说它不是普通单轮 RLHF 采样。
3. 一个 prompt 为什么最后会变成多条轨迹。
4. reward 看起来更像按 episode 来算，证据是什么。
5. 我今天看到的最关键日志字段是哪 3 个。
6. 如果让我给别人解释这个仓库和原版 verl 的差别，我会先说什么。

如果你答不出来，不要进入下午。先回头补上午的观察。

---

## 4. 下午主任务：按主链顺序回源码，只解释上午现象

下午不是“阅读全文”，而是“定点解释”。

请按下面顺序读，顺序不要打乱。

## 4.1 文件 1：`examples/grpo_trainer/run_alfworld.sh`

建议时间：20 分钟

目标：

1. 把 shell 脚本里的“启动入口”和“关键 override”定位出来。
2. 明白为什么它不是一个简单的 bash 包装，而是整个实验的第一层设计面。

你要带着下面问题读：

1. 为什么它先运行 `examples.data_preprocess.prepare`？
2. 为什么这里用 `main_ppo` 而不是另一个 agent trainer？
3. 哪些参数是和 agentic RL 强相关的，哪些只是训练规模参数？

你应该得出的结论：

1. `prepare.py` 负责造出“形式正确”的起始样本。
2. `main_ppo` 仍是主入口，说明 agentic RL 是插在 verl 训练主干里的。
3. `env.rollout.n`、`env.max_steps`、`env.env_name` 才是 agent 机制的关键开关。

## 4.2 文件 2：`examples/data_preprocess/prepare.py`

建议时间：30 分钟

执行：

```bash
sed -n '1,220p' examples/data_preprocess/prepare.py
```

你重点看这几处：

1. `data_source = 'hiyouga/geometry3k'`
2. 注释里那句 `We do NOT use the data ...`
3. `instruction_following`
4. 返回的字段：`prompt`、`images`、`ability`、`extra_info`

这里你一定要理解一个非常重要的点：

这个脚本不是在准备“带标准答案的监督数据”，而是在准备“模态占位 + batch 占位 + 样本索引信息”。

你要明确看到：

1. `problem` 被取出来，但并没有作为监督答案参与训练。
2. `answer` 被注释掉了。
3. 文本模式下 prompt 甚至可以是空字符串。
4. 真正训练时有价值的信号主要来自后续环境 rollout，而不是 parquet 里的正确答案。

这一步直接回答原计划里的关键问题：

“为什么数据准备脚本明明读了 geometry3k，但实际却不用里面的数据内容？”

你应该在 `code_map.md` 里写一句自己的版本：

“这个 dataset 在 agent 训练里主要是一个合法的 prompt 容器和模态占位符，用来启动环境交互，而不是提供标准监督标签。”

## 4.3 文件 3：`verl/trainer/main_ppo.py`

建议时间：45 分钟

执行：

```bash
sed -n '1,240p' verl/trainer/main_ppo.py
```

你今天只看这些位置：

1. `run_ppo(config)`
2. `ray.init(...)`
3. `from agent_system.environments import make_envs`
4. `reward_manager_name == 'episode'`
5. `assert config.actor_rollout_ref.rollout.n == 1`
6. `TrajectoryCollector(config=...)`
7. `create_rl_dataset(...)`
8. `RayPPOTrainer(..., traj_collector=traj_collector, envs=envs, val_envs=val_envs)`

这一段是 Day 1 的第一个“认知翻转点”：

你会发现入口虽然叫 `main_ppo`，但它已经把 agent 训练最关键的几个插件装上去了：

1. 环境管理器 `make_envs`
2. 轨迹采集器 `TrajectoryCollector`
3. episode reward manager
4. 带 env/traj_collector 的 `RayPPOTrainer`

也就是说：

这个仓库不是写了一个完全独立于 verl 的新 trainer，而是在原 trainer 主干上接进了 agent 环境交互链。

### 这里你必须停下来，认真看这句断言

```python
assert config.actor_rollout_ref.rollout.n == 1
```

旁边注释几乎已经把答案写给你了：

在原始 verl 语境里，`actor_rollout_ref.rollout.n > 1` 常用于 GRPO 分组；
但在 `verl-agent` 里，作者故意让它保持为 1，然后把 group 机制迁移到 `env.rollout.n`。

这件事为什么重要：

1. 它说明 group 不再是“对同一个 prompt 直接采 n 个 response”那么简单。
2. 这里的 group 是“同一个初始样本衍生出的多条 agent-environment 交互轨迹”。
3. 这更符合 agentic RL 的真实对象：轨迹，而不是单次静态回答。

### 读完这个文件后，你必须能回答

1. 为什么 `main_ppo.py` 足以启动 agentic RL？
2. 为什么 Day 1 可以先不看 critic 代码？
3. 这个文件里最能证明“verl-agent 是插接式扩展”的 3 行代码是什么？

## 4.4 文件 4：`agent_system/environments/env_manager.py`

建议时间：35 分钟

执行：

```bash
rg -n "def make_envs|group_n" agent_system/environments/env_manager.py
sed -n '602,720p' agent_system/environments/env_manager.py
```

你只抓住 3 件事：

1. `group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1`
2. 训练环境构造时传入了 `env_num=config.data.train_batch_size`
3. 同时把 `group_n` 也传给了具体环境构造器

你现在不要去追 AlfWorld 的内部实现细节。Day 1 只要理解这一层：

`make_envs()` 接到 trainer 配置后，会把“训练 batch 大小”和“每个样本对应的 group size”一起交给环境系统。

这意味着：

1. 环境层从一开始就知道 group 概念。
2. 这个 group 不是 loss 层才出现，而是环境构造层就出现。
3. 这和你未来做 GUI 斗地主环境非常相关，因为你未来的 group 也应该从环境/轨迹层建模，而不是只在 loss 里临时拼。

### 下午此刻你应该回看上午日志

如果上午你设的是：

1. `data.train_batch_size=4`
2. `env.rollout.n=4`

那么你脑子里应该立即形成这张图：

1. trainer 觉得一轮训练拿 4 个起始样本。
2. 每个样本要扩成 4 条候选轨迹。
3. 所以训练时最终会围绕 16 条轨迹展开。

这就是你后面理解 GRPO group 的地基。

## 4.5 文件 5：`agent_system/multi_turn_rollout/rollout_loop.py`

建议时间：70 分钟

这是今天最重要的源码文件。建议分 3 小段读，不要一口气啃完。

### 第一段：先看 `multi_turn_loop`

执行：

```bash
sed -n '490,550p' agent_system/multi_turn_rollout/rollout_loop.py
```

重点盯住这句：

```python
gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)
```

这是今天必须记死的一行。

它直接解释上午你看到的现象：

为什么 4 个起始样本最后会变成 16 条训练轨迹。

此时你要明确：

1. group 不是在 reward 阶段才分出来。
2. prompt 在 rollout 之前就被复制了。
3. 每一份复制品后续都要进入独立环境交互。

### 第二段：看 `vanilla_multi_turn_loop`

执行：

```bash
sed -n '300,420p' agent_system/multi_turn_rollout/rollout_loop.py
```

请按下面顺序理解，不要跳读。

第一步：`envs.reset(...)`

这说明：

1. 轨迹开始前先拿环境初始观测。
2. 模型不是直接凭 parquet 静态答题，而是先进入一个交互环境。

第二步：构造 `uid_batch`

你会看到：

1. 每 `env.rollout.n` 条样本共享同一个 `uid`
2. 每条轨迹还有自己独立的 `traj_uid`

这是 Day 1 第二个必须记死的点：

1. `uid` 是 group id。
2. `traj_uid` 是单条轨迹 id。

以后你看 advantage 代码时，就知道谁负责“分组”，谁负责“区分组内不同轨迹”。

第三步：`actor_rollout_wg.generate_sequences(...)`

这一步的意义不是“采样一个最终答案”，而是：

模型在当前 observation 和 memory 拼成的上下文上，生成当前 step 的动作文本。

第四步：`envs.step(text_actions)`

这里真正发生了 agentic RL 的关键动作：

1. 模型输出被当作 action 文本送入环境。
2. 环境根据 action 返回：
   - `next_obs`
   - `rewards`
   - `dones`
   - `infos`

第五步：累计 episode 信息

你会看到：

1. `episode_rewards[active_masks] += rewards`
2. `episode_lengths[active_masks] += 1`
3. `is_action_valid` 会被塞进 `non_tensor_batch`

这说明 Day 1 里你可以先把环境交互粗略理解为：

每一步都有环境返回，但训练真正最终用的主要统计量，是整条轨迹累计下来的 episode reward、episode length，以及动作是否合法等附加信息。

### 第三段：理解这一层回答了什么

读完后，你应该能回答：

1. `TrajectoryCollector` 到底是在采样 token，还是采样轨迹？
2. 为什么 agentic RL 的基本训练对象不再只是单条 response，而是一整条 trajectory？
3. 为什么 `uid` 和 `traj_uid` 必须同时存在？

标准理解应该是：

1. 它表面上调用模型生成 token。
2. 但更高层的训练对象已经变成“多轮交互轨迹”。
3. token 只是轨迹中某一步 action 的载体。

## 4.6 文件 6：`agent_system/reward_manager/episode.py`

建议时间：25 分钟

执行：

```bash
sed -n '1,220p' agent_system/reward_manager/episode.py
```

你重点看这几件事：

1. `episode_rewards = data_item.non_tensor_batch['episode_rewards']`
2. `episode_lengths = data_item.non_tensor_batch['episode_lengths']`
3. `score = episode_rewards` 或 `episode_rewards / episode_lengths`
4. `reward_tensor[i, valid_response_length - 1] = score`

这会直接解释原计划中的两个关键问题：

1. 为什么 reward manager 里拿到的是 `episode_rewards`，不是环境逐步的细粒度 reward？
2. 为什么 reward 最后落在 response 的最后一个 token 上？

Day 1 你只需要建立这个简洁理解：

1. 环境层累计整条轨迹的 episode reward。
2. reward manager 再把这个 episode-level 信号写回 token tensor。
3. 默认写在“本条 response 的最后一个有效 token 位置”。

为什么作者这么做：

1. trainer 下面的优化接口最终还是按 token tensor 组织的。
2. 所以需要把“轨迹级信号”投影回“token 级张量”。
3. 这样才能无缝复用 verl 原本的 RL 优化框架。

这正是“agentic RL 插到 verl 主干上”的另一个证据。

## 4.7 文件 7：`verl/trainer/ppo/ray_trainer.py`

建议时间：45 分钟

今天只看 3 段，别贪多。

### 第一段：非法动作惩罚

执行：

```bash
sed -n '200,230p' verl/trainer/ppo/ray_trainer.py
```

你要记住：

1. `apply_invalid_action_penalty(...)` 会读 `is_action_valid`
2. 惩罚会直接减到最后那个 reward 位置上
3. 它还会上报 `episode/valid_action_ratio`

这一步为什么重要：

1. 它解释了上午日志里的 `episode/valid_action_ratio` 从哪来。
2. 它说明动作合法性约束不是只存在于环境内部，而是会进一步反馈到训练目标。
3. 这和你未来 GUI agent 的“点击是否合法”“操作序列是否合法”完全同构。

### 第二段：`compute_advantage`

执行：

```bash
sed -n '244,320p' verl/trainer/ppo/ray_trainer.py
sed -n '113,180p' verl/trainer/ppo/core_algos.py
```

你这里先不要深究所有 advantage 变体，只抓 GRPO：

1. `compute_advantage(... adv_estimator=GRPO ...)`
2. `compute_grpo_outcome_advantage(...)`
3. `index=data.non_tensor_batch["uid"]`
4. `traj_index=data.non_tensor_batch["traj_uid"]`

到这里，Day 1 最重要的主链就闭环了：

1. `uid` 在 rollout 里构造。
2. reward 已经变成 token-level tensor。
3. advantage 计算时按 `uid` 做 group 聚合。
4. `traj_uid` 用来区分组内不同轨迹。

你今天不需要背实现细节，只要吃透一句话：

GRPO 在这个仓库里分组的对象，是共享同一个起始样本的多条轨迹组，而不是普通的“同 prompt 多次静态回答”。

### 第三段：训练主循环中的顺序

执行：

```bash
sed -n '1180,1260p' verl/trainer/ppo/ray_trainer.py
```

你按顺序抄下这条链：

1. `token_level_scores = reward_tensor`
2. `apply_invalid_action_penalty`
3. `token_level_rewards`
4. `compute_advantage`
5. `update_actor`

这就是你晚上要手画那张单页图的后半段。

你应该非常明确：

今天你真正要懂的 trainer 主干，不是所有训练细节，而是这 5 步是如何把 episode signal 送进 actor 更新的。

---

## 5. 下午后半段：把日志字段映射回代码

这一段非常关键。很多人“跑过了，也看过了”，但就是说不清代码和日志的对应关系。今天你必须补上这一步。

建议你在 `code_map.md` 里做下面这个表。

| 日志字段 | 先用人话解释它在描述什么 | 代码产地 |
| --- | --- | --- |
| `episode/reward/mean` | 一批轨迹的平均整局回报 | `verl/trainer/ppo/metric_utils.py` 的 `compute_data_metrics`，数据源来自 `episode_rewards` |
| `episode/length/mean` | 一批轨迹平均跑了多少步 | `compute_data_metrics`，数据源来自 `episode_lengths` |
| `episode/valid_action_ratio` | 当前 batch 中动作合法的比例 | `ray_trainer.py` 里的 `apply_invalid_action_penalty` |
| `critic/score/mean` | token-level score 按序列求和后的平均值 | `compute_data_metrics`，底层来自 `token_level_scores` |
| `critic/advantages/mean` | response token 上的 advantage 平均值 | `compute_data_metrics`，底层来自 `compute_advantage` |
| `response_length/mean` | 平均 response 长度 | `compute_data_metrics` |
| `timing_s/gen` | 生成阶段耗时 | `compute_timing_metrics` |
| `perf/throughput` | 每 GPU 吞吐 | `compute_throughout_metrics` |

今天至少完成其中 5 个。

### 这里你要建立一个很重要的工程意识

工业场景里，真正拉开差距的不是“会不会背公式”，而是：

1. 看到某个指标怪了，你能迅速知道它是 rollout 问题、环境问题、reward 问题，还是训练更新问题。
2. 你能直接定位到产生日志的代码位置。
3. 你知道该插 print、该改哪个超参、该盯哪个对象。

这正是你 Day 1 开始建立的能力。

---

## 6. 晚上主任务：把理解固化成你自己的主链图

晚上这 2 小时不要再继续拓展阅读。请只做固化。

## 6.1 手画一张单页主链图

必须包含这些节点：

1. `run_alfworld.sh`
2. `prepare.py`
3. `main_ppo.py`
4. `make_envs()`
5. `TrajectoryCollector.multi_turn_loop()`
6. `envs.step()`
7. `EpisodeRewardManager`
8. `compute_advantage()`
9. `update_actor()`

### 每个节点边上必须写一句话

例如：

1. `prepare.py`：提供模态正确、大小正确的起始样本，而不是标准答案监督。
2. `make_envs()`：把 train batch 和 group size 一起交给环境系统。
3. `multi_turn_loop()`：复制 prompt，组织多轮 agent-environment 轨迹采样。
4. `EpisodeRewardManager`：把 episode-level reward 投影回 response 最后 token。
5. `compute_advantage()`：按 `uid` 做 GRPO 分组计算。

### 这张图的标准

不是画得好看，而是：

1. 任何一个节点删掉，你都知道主链断在哪里。
2. 任何一条箭头，你都能说清“数据形态发生了什么变化”。

## 6.2 写一句最关键的 Day 1 总结

写进 `study_guide/day1_notes/obs.md` 顶部。

推荐你用这种句式：

“`verl-agent` 的本质不是重写了一套全新的 RL trainer，而是在 `verl` 原本的 PPO/GRPO 主干上，插入环境、多轮轨迹采样、episode reward 回填和基于轨迹组的 advantage 计算，从而把优化对象从静态回答扩展成 agent trajectory。”

你也可以自己改写，但核心意思不能丢。

## 6.3 用自己的话回答那句 dataset 问题

必须单独写成 3 句短句：

1. parquet 数据不是主要监督来源。
2. 它主要负责提供 prompt 容器、模态信息、batch 大小和样本索引。
3. 真正让训练产生区分度的是 rollout 过程中环境返回的 reward 和轨迹结果。

如果这 3 句你写不顺，说明 Day 1 还没真正吃透。

---

## 7. 今日必须完成的最小实践任务

这部分严格对应原计划，但我把动作拆开。

### 任务 1：画主链图

完成标准：

1. 图上有 9 个节点。
2. 节点之间箭头方向正确。
3. 明确写出 `env.rollout.n` 在哪里生效。

### 任务 2：写一句 dataset 的本质解释

完成标准：

1. 不超过 60 字。
2. 必须同时出现“占位”或“起点”与“环境 rollout”这两个意思。

### 任务 3：记录 5 个真实日志字段及其产地

完成标准：

1. 字段名准确。
2. 能对应到具体函数。
3. 能说出这个字段异常时优先怀疑哪一层。

例如：

1. `episode/valid_action_ratio` 低：优先怀疑 action projection / 环境合法性判定 / 输出格式。
2. `episode/reward/mean` 一直不变：优先怀疑环境奖励、group 差异不足、reward 回填。
3. `timing_s/gen` 特别大：优先怀疑 rollout 生成慢、vllm 配置或 response 太长。

---

## 8. Day 1 产出验收清单

晚上收工前，你逐项打勾。

1. 我亲手跑过一次最小训练闭环，AlfWorld 或 Sokoban 至少成功一个。
2. 我知道这个脚本虽然走 `main_ppo`，但 agentic RL 关键能力是在哪里插进去的。
3. 我知道 `env.rollout.n` 才是这里的 group size。
4. 我知道 prompt 是在 `multi_turn_loop()` 里被 repeat 的。
5. 我知道 `uid` 是 group id，`traj_uid` 是轨迹 id。
6. 我知道 episode reward 是先累计，再被写到 response 最后一个 token 上。
7. 我知道 advantage 是按 `uid` 分组算的。
8. 我做完了那张单页主链图。
9. 我记录了至少 5 个日志字段与代码位置。
10. 我能不用看代码，口头讲 3 分钟主链路。

如果你有 8 项以上能打勾，Day 1 就合格。

---

## 9. Day 1 面试硬核拷问与参考答题方向

这一部分不是让你背标准答案，而是检验你是否真的吃透。

### 问题 1

为什么 `verl-agent` 明明调用的是 `verl.trainer.main_ppo`，却能做 agentic RL？

参考答题方向：

1. 因为它不是另起一套 trainer，而是在 `main_ppo` 里接入了环境系统、轨迹采集器和 episode reward manager。
2. `RayPPOTrainer` 也被传入了 `traj_collector`、`envs`、`val_envs` 等 agent 组件。
3. 所以底层优化框架沿用 verl，但采样对象和奖励来源已经变成 agent trajectory。

### 问题 2

`actor_rollout_ref.rollout.n` 和 `env.rollout.n` 在这个仓库里的职责差异是什么？

参考答题方向：

1. 这个仓库里 `actor_rollout_ref.rollout.n` 被强制保持为 1。
2. group 机制转移到了 `env.rollout.n`。
3. 前者不再承担 GRPO 分组职责，后者负责把一个起始样本扩展成多条环境交互轨迹。

### 问题 3

如果你把 `env.rollout.n` 从 8 改成 1，训练行为会发生什么根本变化？

参考答题方向：

1. 同一样本不再扩成多条候选轨迹。
2. GRPO 的组内相对比较几乎失去意义，退化成没有组内差异的情形。
3. 训练对象更接近单轨迹更新，group-based advantage 的价值会明显下降。

### 问题 4

为什么这里的 dataset 不需要包含标准答案？

参考答题方向：

1. 这里不是监督微调范式，主要学习信号不是 label。
2. dataset 只负责提供起始 prompt、模态信息和样本索引。
3. 真正有效的优化信号来自 agent rollout 后的环境奖励和轨迹结果。

### 问题 5

`EpisodeRewardManager` 为什么只在 response 最后一个 token 位置写 reward？

参考答题方向：

1. 环境回来的主要是 episode-level 标量回报。
2. verl 下层优化接口需要 token-level reward tensor。
3. 所以作者把整条轨迹的回报投影到当前 response 的最后有效 token 上，以兼容原 trainer 结构。

---

## 10. 今天绝对不要做的 8 件事

这部分很重要，因为 Day 1 最容易失控。

1. 不要一上来就通读整个 `verl/trainer/ppo/ray_trainer.py`。
2. 不要去深挖 critic、value、GAE。
3. 不要因为一个环境安装问题就把整个上午耗光。
4. 不要在第一次运行前就想着“调到最优吞吐”。
5. 不要把所有 WandB、logger、美化输出当成今天重点。
6. 不要把 attention、模型结构、vllm 内核当成今天重点。
7. 不要把源码阅读变成无目标摘抄。
8. 不要结束一天时还没有一张你自己手画的主链图。

---

## 11. 一句收尾提醒

Day 1 最有价值的收获，不是“我看了多少代码”，而是：

你第一次真正把这个仓库最核心的 agentic RL 主干，跑成了一个自己能解释、能定位、能复述的动态系统。

只要这一步站稳，Day 2 往后的环境、动作合法性、记忆拼接、GRPO 工程落地问题，都会变得顺理成章。
