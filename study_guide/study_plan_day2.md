# verl-agent Day 2 超详细学习指南

这份文档只展开 [study_guide/study_plan.md](./study_plan.md) 里的 Day 2，不重写整份 5 天计划。

今天你的任务非常明确，只做 3 件事，并且必须把它们做扎实：

1. 吃透环境批量构建与 group reset。
2. 吃透动作投影、动作合法性判定，以及非法动作惩罚链。
3. 吃透历史记忆是怎样被重新拼回 prompt 的。

今天故意不碰更高级的算法细节，不讨论 critic / value model / GAE，也不追大而全的源码扫读。你只围绕你未来最可能魔改的工程层动手：

`make_envs`
-> `build_*_envs`
-> `EnvironmentManager.reset/step`
-> `projection.py`
-> `SimpleMemory`
-> `prompt template`
-> `rollout_loop`
-> `apply_invalid_action_penalty`

---

## 0. Day 2 的作战原则

今天请一直遵守这 7 条，不然非常容易学散：

1. 动态调试优先，但动态调试的主战场只选一个环境，优先 `Sokoban`。
2. `WebShop` 和 `ALFWorld` 今天主要拿来做横向对照，帮助你看清“不同环境哪些是共性，哪些是特例”。
3. 任何结论都必须同时有两种证据：
   一次实际运行的现象，外加一段对应源码。
4. 今天不追“模型为什么会学会推理”，只追“环境为什么这样组织，动作为什么这样判合法，记忆为什么这样拼 prompt”。
5. 你未来做 GUI 斗地主 Agent，最该从今天带走的不是某个具体环境细节，而是 3 个模板：
   group reset 模板、输出协议解析模板、短期记忆拼接模板。
6. 一旦某个环境依赖阻塞你超过 20 分钟，立刻切回 `Sokoban`，不要把 Day 2 消耗在装环境上。
7. 今天所有笔记都围绕“职责边界”来写：
   这一层应该做什么，不应该做什么。

---

## 1. 今日总目标与时间安排

总时长按 8 小时设计。

### 你今天结束时必须达到的状态

你必须能不看代码，直接口头说清楚下面 5 件事：

1. `env.rollout.n` 是怎样在环境侧真正变成 group 的。
2. 为什么 `train_batch_size=4, env.rollout.n=4` 不等于 4 条轨迹，而是 16 个环境实例。
3. `projection.py`、`env`、`reward penalty` 三层在动作合法性问题上各自承担什么职责。
4. `SimpleMemory` 现在到底是“记忆系统”，还是“历史文本回放器”。
5. 如果你要接一个 GUI 斗地主环境，最少需要改哪些文件，以及每个文件承担什么职责。

### 时间切块

1. `09:00 - 09:30`：开工准备，搭好 Day 2 记录区，只看最必要的文件。
2. `09:30 - 11:00`：先把 `make_envs -> build_sokoban_envs -> reset` 这条 group reset 链吃透。
3. `11:00 - 12:00`：跑一个极小 Sokoban 实验，验证 group reset 和 worker 数量的直觉。
4. `14:00 - 15:30`：拆 `projection.py -> env_manager.step -> rollout_loop -> invalid penalty`。
5. `15:30 - 17:00`：拆 `SimpleMemory -> build_text_obs -> prompt template`，看历史是怎样拼回 prompt 的。
6. `20:00 - 21:00`：做 Day 2 的微型魔改任务 1，把输出协议临时改得更像你未来项目。
7. `21:00 - 22:00`：整理“接入新环境最小模板”，回答硬核拷问。

如果你作息不同，可以整体平移时间，不影响结构。

---

## 2. 开工前 30 分钟：只做最必要的准备

### Step 1：进入正确环境

执行：

```bash
conda activate verl-agent-bw
cd /home/zhangwj/science/verl-agent
```

今天所有命令默认都在这个环境和仓库根目录下执行。

### Step 2：创建 Day 2 工作记录区

执行：

```bash
mkdir -p study_guide/day2_notes
```

建议今天只用下面 4 个文件记录：

1. `study_guide/day2_notes/group_reset.md`
2. `study_guide/day2_notes/projection_and_validity.md`
3. `study_guide/day2_notes/memory_and_prompt.md`
4. `study_guide/day2_notes/new_env_template.md`

它们各自只记录一种东西：

1. `group_reset.md`：只记 group、seed、worker、batch 之间的关系。
2. `projection_and_validity.md`：只记动作解析、合法性、奖励惩罚链。
3. `memory_and_prompt.md`：只记历史是如何被存起来、取出来、拼进 prompt 的。
4. `new_env_template.md`：晚上整理成你未来可复用的“接新环境模板”。

### Step 3：先读原计划里的 Day 2，不超过 10 分钟

只看原文件 [study_plan.md](./study_plan.md) 的 Day 2 这 8 块：

1. `当日目标`
2. `时间分配`
3. `今天重点文件`
4. `今天必须抓住的实现细节`
5. `今天必须做的调试实验`
6. `今日微型魔改任务 1`
7. `Day 2 产出标准`
8. `Day 2 面试硬核拷问`

要求：

1. 不展开联想。
2. 不做源码阅读。
3. 只在纸上写 3 句话：
   - 今天动态调试主战场是谁。
   - 今天最关键的 3 个概念是什么。
   - 今天下班时我必须能独立回答什么。

### Step 4：确认今天只读这几组文件

执行：

```bash
rg --files study_guide
rg --files agent_system/environments agent_system/memory | sort
```

然后确认今天只围绕这 4 组内容推进：

1. `agent_system/environments/env_manager.py`
2. `agent_system/environments/env_package/{sokoban,webshop,alfworld}/{envs.py,projection.py}`
3. `agent_system/memory/memory.py`
4. `agent_system/environments/prompts/{sokoban,webshop,alfworld}.py`

今天不要把注意力扩散到别的目录。

---

## 3. 先搭 Day 2 的三条心智骨架

正式读代码前，你先把今天要攻下来的 3 条链在脑中立起来。

### 链 1：环境批量构建与 group reset

```text
run_sokoban.sh
-> main_ppo
-> make_envs(config)
-> build_sokoban_envs(seed, env_num, group_n, ...)
-> SokobanMultiProcessEnv.reset()
-> 随机采样 env_num 个 seed
-> np.repeat(seed, group_n)
-> env_num * group_n 个 worker 拿到 reset seed
```

这条链回答的问题是：

“为什么同一个原始样本会在环境侧被复制成一组 sibling trajectories？”

### 链 2：动作合法性与非法动作惩罚

```text
模型输出原始文本
-> projection.py 提取 <action>
-> projection.py 给出 valids
-> EnvironmentManager.step 写入 info['is_action_valid']
-> rollout_loop 挂到 batch.non_tensor_batch
-> ray_trainer.apply_invalid_action_penalty()
-> episode/valid_action_ratio
```

这条链回答的问题是：

“非法动作不是一个地方决定的，而是一条跨模块链路共同决定的。”

### 链 3：历史记忆拼回 prompt

```text
EnvironmentManager.reset()
-> memory.reset(batch_size)
-> 每次 step 后 memory.store(...)
-> 下一个 step 前 build_text_obs()
-> memory.fetch(history_length)
-> prompt template.format(action_history=...)
```

这条链回答的问题是：

“所谓 memory，在这个仓库里本质上并不是一个复杂记忆系统，而是一个非常朴素的历史文本重放器。”

把这 3 条链先抄到 `study_guide/day2_notes/` 里。后面所有现象都往这 3 条链上挂。

---

## 4. 上午主任务：把环境批量构建与 group reset 吃透

这一段是 Day 2 的地基。你如果连环境是怎么成组构建的都没吃透，后面谈 GRPO group、谈合法动作率、谈 memory 都会飘。

## 4.1 先读 `run_sokoban.sh`，但只读 15 分钟

执行：

```bash
sed -n '1,220p' examples/grpo_trainer/run_sokoban.sh
```

你今天只盯这几个配置：

1. `algorithm.adv_estimator=grpo`
2. `actor_rollout_ref.actor.use_invalid_action_penalty=True`
3. `actor_rollout_ref.actor.invalid_action_penalty_coef=0.1`
4. `env.env_name=Sokoban`
5. `env.max_steps=15`
6. `env.rollout.n=$group_size`
7. `env.sokoban.mode='rgb_array'`
8. `env.resources_per_worker.num_cpus=$num_cpus_per_env_worker`
9. `trainer.n_gpus_per_node=2`

你现在只需要得出下面 6 个结论：

1. 这个脚本明确把 group size 放在 `env.rollout.n`，不是放在 `actor_rollout_ref.rollout.n`。
2. 这个脚本已经开启了非法动作惩罚，所以 Day 2 的调试实验有明确观测点。
3. Sokoban 默认是视觉模式，这和你未来 GUI agent 的方向接近。
4. 这个仓库把“每个环境 worker 的 CPU 配额”暴露出来了，说明作者非常在意环境并发吞吐。
5. 这个脚本天然就是双卡配置，和你的机器条件匹配。
6. 今天你做的所有极短实验，优先都从这个脚本 override，而不是自己重新拼一整条命令。

把这 6 句话写进 `group_reset.md`。

## 4.2 读 `make_envs()`，搞清楚 group 是在哪一层长出来的

执行：

```bash
rg -n "def make_envs|elif \"sokoban\"|elif \"webshop\"|elif \"alfworld\"" agent_system/environments/env_manager.py
sed -n '520,760p' agent_system/environments/env_manager.py
```

你只看 5 个点：

1. `group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1`
2. 训练环境用 `env_num=config.data.train_batch_size`
3. 训练环境用 `group_n=env.rollout.n`
4. 验证环境强制 `group_n=1`
5. 返回的不是裸环境，而是 `EnvironmentManager`

你要在脑子里把下面这个等式彻底记死：

```text
训练阶段的底层环境实例数
= data.train_batch_size * env.rollout.n
```

举例：

1. 如果 `data.train_batch_size=4`
2. 如果 `env.rollout.n=4`
3. 那训练时底层 `SokobanWorker` 数量就是 `16`
4. 但“原始问题数”依然只有 `4`
5. 这 16 条轨迹会按 4 组来组织，每组 4 条 sibling trajectories

这一步非常关键，因为它直接解释了：

1. 为什么你在 agentic RL 里会看到一个 batch 对应很多环境 actor。
2. 为什么 Ray CPU 会很快被吃掉。
3. 为什么 `env.resources_per_worker.num_cpus` 是一个真实吞吐开关。

### 这一段读完后，你必须立刻写一句话

写进 `group_reset.md`：

“在 `verl-agent` 里，GRPO 的 group 不是 trainer 生成 N 次 response 得来的，而是 `make_envs()` 在环境构建阶段，通过 `group_n=env.rollout.n` 直接把底层环境实例扩出来的。”

## 4.3 读 `sokoban/envs.py`，看 group reset 具体怎样落地

执行：

```bash
sed -n '1,260p' agent_system/environments/env_package/sokoban/envs.py
```

你只读下面这些逻辑，不要被 Ray 语法分散注意力：

1. `self.num_processes = env_num * group_n`
2. 构造 `self.workers` 的 for 循环
3. `reset()` 里先采样 `self.env_num` 个 seed
4. `seeds = np.repeat(seeds, self.group_n)`
5. 每个 worker 拿一个被 repeat 后的 seed 执行 reset

你现在要完全吃透这个具体例子：

假设：

1. `env_num = 3`
2. `group_n = 4`

那么：

1. 一共创建 `12` 个 Ray worker
2. reset 时先采样 `3` 个 seed，比如 `[101, 202, 303]`
3. 然后变成 `[101, 101, 101, 101, 202, 202, 202, 202, 303, 303, 303, 303]`
4. 第 `0-3` 个 worker 拿到同一个 seed
5. 第 `4-7` 个 worker 拿到同一个 seed
6. 第 `8-11` 个 worker 拿到同一个 seed

这就是 group reset 的本质。

### 为什么这件事对 GRPO 很关键

因为 GRPO 需要“同题多答”的 sibling trajectories。

在这个仓库里，“同题多答”不是靠一份 prompt 让模型采样 4 次那么简单，而是靠：

1. 同一个初始环境状态
2. 被复制成一组并行环境实例
3. 后续每个环境分支随着模型动作逐步分叉

这和你未来的 GUI 斗地主项目高度相关。

未来如果你有：

1. 同一张游戏截图
2. 同一个回合上下文
3. 同一个玩家对话历史

你也应该在环境侧复制出一组相同起点的轨迹，然后让模型在这些轨迹上探索不同动作或策略。

## 4.4 动手做一个最小 group reset 观测实验

这一段非常重要。你今天不能只“相信代码”，你要亲手验证 group reset 真的发生了。

执行下面这个最小脚本：

```bash
python - <<'PY'
import numpy as np
from agent_system.environments.env_package.sokoban import build_sokoban_envs

env = build_sokoban_envs(
    seed=0,
    env_num=2,
    group_n=4,
    mode="tiny_rgb_array",
    is_train=True,
    resources_per_worker={"num_cpus": 0.1},
    env_kwargs={
        "dim_room": [6, 6],
        "num_boxes": 1,
        "max_steps": 6,
        "search_depth": 30,
    },
)

obs, infos = env.reset()
print("total env instances:", len(obs))
for g in range(2):
    start = g * 4
    same = [np.array_equal(obs[start], obs[start + j]) for j in range(4)]
    print(f"group {g} identical-to-first:", same)
env.close()
PY
```

### 你应该观察什么

你大概率会看到：

1. `total env instances: 8`
2. 每个 group 内部，前 4 个或后 4 个初始 observation 是相同的

这就说明：

1. 环境确实被扩成了 `env_num * group_n`
2. 同一个 group 内部确实从同一初始状态出发

### 如果这个脚本跑不通，不要慌

你按这个顺序排查：

1. 先确认 `gym_sokoban` 已安装
2. 再确认当前 conda 环境真的是 `verl-agent-bw`
3. 再确认 Ray 没有残留僵尸进程
4. 如果这里卡住超过 20 分钟，直接跳过动态验证，改为把这段代码逐行手工解释，也能完成今天目标

今天你要学的是机制，不是和环境安装死磕。

## 4.5 再读 `SokobanEnvironmentManager`，把“裸环境”变成“agent 可用环境”

执行：

```bash
rg -n "class SokobanEnvironmentManager" agent_system/environments/env_manager.py
sed -n '220,360p' agent_system/environments/env_manager.py
```

你重点看 `reset()` 和 `step()`。

### `reset()` 里发生了什么

你必须看清下面这几件事：

1. `obs, infos = self.envs.reset()`
2. 根据 `mode` 决定这是视觉模式还是非视觉模式
3. `self.memory.reset(batch_size=len(infos))`
4. 调 `self.build_text_obs(...)`
5. 返回统一格式的 observation dict：
   `{'text': ..., 'image': ..., 'anchor': ...}`

这一步的意义是：

底层环境只负责给你“原始观测”，而 manager 负责把它整理成上层 rollout 能直接吃的统一接口。

### `step()` 里发生了什么

顺序非常关键：

1. 先 `actions, valids = self.projection_f(text_actions)`
2. 再 `next_obs, rewards, dones, infos = self.envs.step(actions)`
3. 再把 `info['is_action_valid']` 填进去
4. 再把上一步观测和动作存进 `memory`
5. 再构造下一轮 agent 要看到的 observation dict

你要特别注意第 4 步：

```text
memory.store({'text_obs': self.pre_text_obs, 'action': ...})
```

这说明被记下来的不是“当前 step 后的新观测”，而是“做出这个动作之前的观测 + 动作”。

这是非常合理的，因为 prompt 里要重放的是：

“之前你看到什么，然后你采取了什么动作。”

### 一个今天必须发现的细节

在视觉 Sokoban 模式下：

1. `self.is_multi_modal = envs.mode == 'rgb_array'`
2. `build_text_obs()` 会走 `SOKOBAN_VISUAL_TEMPLATE`
3. 这个模板没有历史占位符

这意味着：

1. `memory` 仍然在存
2. 但历史并没有真正被拼回视觉 Sokoban 的 prompt

这个发现非常重要，因为它直接告诉你：

当前仓库里的 `SimpleMemory` 机制，不是所有环境、所有模态下都真正起作用。

这也是你未来做 GUI 斗地主 Agent 时必须主动设计的地方：

视觉环境里，历史不能只“存在内存里”，它必须真的被重新组织进下一轮 prompt，或者被编码成结构化 memory channel。

把这句话写进 `memory_and_prompt.md`。

## 4.6 上午收尾：跑一个最小训练命令，把 group 概念从“静态理解”变成“运行直觉”

执行：

```bash
bash examples/grpo_trainer/run_sokoban.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.rollout.n=4 \
  env.max_steps=6 \
  ray_init.num_cpus=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
```

### 这条命令今天的目的不是“训练出效果”

它只有 3 个目的：

1. 让你看到环境 actor 真的启动起来。
2. 让你对 `batch_size * env.rollout.n` 的代价有体感。
3. 给下午的合法动作率实验准备 baseline。

### 你现在只看 5 类东西

1. Hydra 配置里 `env.rollout.n=4` 是否真的生效。
2. Hydra 配置里 `data.train_batch_size=4` 是否真的生效。
3. `episode/valid_action_ratio` 有没有出现。
4. `timing_s/gen` 或其他吞吐日志是不是开始出现。
5. 整个系统是不是明显比 Day 1 的纯文本环境更吃资源。

把你的观察写成 3 句话：

1. 今天这个 batch 到底扩成了多少个底层环境。
2. 资源瓶颈更像来自 GPU 生成，还是 Ray 环境并发。
3. 视觉模式和文本模式相比，哪种更接近你未来项目。

---

## 5. 下午上半场：把动作投影、合法性判定、非法惩罚这条链打穿

这一段是你未来做 GUI agent 的核心壁垒之一。

因为未来你最痛的不是“模型不会输出”，而是：

1. 模型输出格式经常飘。
2. 输出动作经常不合法。
3. 不同层对“合法”的定义不一致。
4. 惩罚施加位置错了以后，训练信号会非常混乱。

## 5.1 先读 3 个 `projection.py`，只比较，不沉迷

执行：

```bash
sed -n '1,220p' agent_system/environments/env_package/sokoban/projection.py
sed -n '1,220p' agent_system/environments/env_package/webshop/projection.py
sed -n '1,220p' agent_system/environments/env_package/alfworld/projection.py
```

今天不要逐字抠细节。你只做“横向比较”。

### 先看 `sokoban_projection.py`

你必须抓住这 5 个事实：

1. 它先找 `<action>...</action>`。
2. 它把 action 映射成离散动作 id：
   `up/down/left/right/still -> 1/2/3/4/0`
3. 找不到 `<action>`，就给 `0`
4. 缺少 `<think>...</think>`，即使动作提取成功，也会把 `valids[i]=0`
5. 返回的是两样东西：
   `actions` 和 `valids`

### 再看 `webshop_projection.py`

你必须抓住这 4 个事实：

1. 它也要求 `<think>` 和 `<action>`
2. 但它不把 action 映射成 id，而是直接取 `<action>` 里的字符串
3. 它会额外检查原始输出里有没有中文
4. 它并不会检查这个 action 是否真的属于当前页面的 `available_actions`

### 再看 `alfworld_projection.py`

它和 WebShop 非常像：

1. 只做标签提取
2. 只做很浅层的格式合法性检查
3. 也不对照 `admissible_commands` 做成员校验

### 这一段读完后，你必须写进笔记的核心结论

写进 `projection_and_validity.md`：

“当前仓库的 `projection.py` 首先解决的是输出协议解析问题，不是完整的语义合法性判定问题。也就是说，它更像第一道闸门，而不是唯一闸门。”

这句话非常关键。

因为它直接回答了一个工程问题：

为什么动作合法性最好先在 `projection.py` 做第一层约束，而不是完全交给环境？

答案就是：

1. 因为格式错误要尽早暴露。
2. 因为上层 trainer 需要统一拿到 `is_action_valid`。
3. 因为环境只知道“这个动作能不能执行”，但不一定知道“模型有没有遵守你的输出协议”。

## 5.2 读 `env_manager.step()`，看 `valids` 是怎样被挂进 info 的

执行：

```bash
rg -n "info\\['is_action_valid'\\]|info\\[\"is_action_valid\"\\]" agent_system/environments/env_manager.py agent_system/environments/base.py
sed -n '1,120p' agent_system/environments/base.py
sed -n '240,520p' agent_system/environments/env_manager.py
```

你要看清楚一件事：

`projection.py` 返回的 `valids` 并不会停留在 env manager 内部，而是会被写进每个 step 的 `info`：

```text
info['is_action_valid'] = to_numpy(valids[i])
```

这一步的价值是：

1. 环境层和 trainer 层解耦了
2. 只要 manager 按统一接口写 `info['is_action_valid']`
3. 上层 rollout/trainer 就不需要知道你具体是什么环境

这就是工业级代码常见的设计风格：

底层环境可以不同，但中间接口必须统一。

## 5.3 读 `rollout_loop.py` 和 `apply_invalid_action_penalty()`，把整条链闭合

执行：

```bash
sed -n '320,430p' agent_system/multi_turn_rollout/rollout_loop.py
sed -n '190,240p' verl/trainer/ppo/ray_trainer.py
sed -n '1188,1215p' verl/trainer/ppo/ray_trainer.py
```

你必须顺着这条链解释：

1. `envs.step(text_actions)` 返回 `infos`
2. rollout loop 发现 `infos[0]` 里有 `is_action_valid`
3. 它把这一列塞进 `batch.non_tensor_batch['is_action_valid']`
4. trainer 里 `apply_invalid_action_penalty()` 读取这列
5. 它在 response 最后一个有效 token 的 reward 上减去惩罚
6. 同时记录指标：
   `episode/valid_action_ratio`

### 这里最该记住的不是代码细节，而是奖励链路位置

非法动作惩罚不是直接发生在 environment manager 里，而是发生在 trainer 的 reward 处理阶段。

也就是说：

1. projection 负责“解析并标记”
2. env manager 负责“把标记挂上去”
3. rollout loop 负责“把标记带入 batch”
4. trainer 负责“把标记变成 reward 惩罚”

这 4 层职责非常清楚。

### 再补一个今天很重要的工程细节

在 Sokoban 里，非法动作很可能是“双重惩罚”：

1. 一方面，projection 把非法格式动作映射成 `0`
2. 另一方面，Sokoban 底层环境把 `0` 定义为 invalid action，并且环境本身有 `PENALTY_FOR_INVALID = -1`
3. 然后 trainer 还会再额外减一层 `invalid_action_penalty_coef`

这意味着：

你未来自己做 GUI agent 时，一定要想清楚：

1. 环境本身是否已经对非法动作给了负奖励
2. trainer 层是否还要再叠一层 penalty
3. 两层叠加会不会让非法动作信号过重

这就是工业落地里非常常见的“奖励重复计罚”问题。

把这个风险写进 `projection_and_validity.md`。

## 5.4 做 Day 2 的主调试实验：故意制造格式失配

原计划说：

1. 故意让模型输出缺少 `<think>` 或 `<action>`
2. 观察 `projection.py` 如何把它判成 invalid
3. 观察 `episode/valid_action_ratio` 和 reward 的变化

这里我建议你不要赌模型“自然犯错”，而是主动制造一个更可控的格式失配实验。

### 主实验思路

我们临时改 `Sokoban` 的 prompt，让 prompt 明确要求模型：

“不要输出 `<think>` 和 `<action>` 标签，只输出一个方向单词。”

而 `sokoban_projection.py` 仍然坚持要求 `<think>` 和 `<action>`。

这样就构造出了：

1. prompt 规范
2. projection 规范
3. 二者故意不一致

这会比“等模型自己哪天抽风漏标签”稳定得多。

### 实验前先跑一个 baseline

执行：

```bash
bash examples/grpo_trainer/run_sokoban.sh \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=4 \
  data.val_batch_size=8 \
  env.rollout.n=4 \
  env.max_steps=6 \
  ray_init.num_cpus=32 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
```

记下这 3 个基线现象：

1. `episode/valid_action_ratio`
2. `episode/reward/mean`
3. 生成输出里是否大体遵守 `<think><action>` 格式

### 然后临时改 prompt

打开：

```bash
sed -n '1,220p' agent_system/environments/prompts/sokoban.py
```

你临时把 3 个模板最后的输出要求，改成这种风格：

```text
Now output only one word from: up, down, left, right.
Do not output any XML-style tags.
Do not output <think> or <action>.
```

你只需要做实验，不需要优雅。

今天的关键不是“写出最好的 prompt”，而是要稳定制造出：

1. prompt 要求模型不带标签
2. parser 依然要求带标签

### 再跑同一条短命令

继续执行刚才那条短命令。

### 你应该重点观察什么

你大概率会看到：

1. `episode/valid_action_ratio` 明显下降
2. reward 变差
3. 即使环境还能继续 step，trainer 侧也会记录更多 invalid action

这时你要写下 3 句话：

1. projection 负责的是“协议合法性”，不是“任务完成度”
2. 只改 prompt，不改 env，就足以通过 projection 改变训练信号
3. 模型输出协议是强化学习工程里的硬约束，不是可有可无的格式美观问题

### 实验做完后必须恢复 prompt

恢复原 prompt 非常重要。

不恢复的话，你后面的所有实验基线都会被污染。

## 5.5 备选实验：改 `sokoban_projection.py` 的合法动作集合

如果你不想动 prompt，也可以按原学习计划里的可选方式做。

你今天可以临时把：

```python
"still": 0,
```

删掉，或者更激进一点，只暂时保留：

```python
"up": 1,
"down": 2,
```

这样做的意义是：

1. 你在 projection 层直接收紧动作空间
2. 即使环境本身还能接受更多动作，projection 也会先把它们判成 invalid
3. 你可以观察 `valid_action_ratio` 是否上升或下降，以及 reward 如何变化

这一步帮助你理解：

动作合法性的第一道闸门放在哪里，会直接改变训练分布。

---

## 6. 下午下半场：把 `SimpleMemory` 和 prompt 拼接机制吃透

这部分是你未来做“记牌/策略记忆”的起点。

但你今天必须非常清醒：

当前仓库的 memory 机制，只是一个非常朴素的短期历史拼接器，不是长期记忆系统。

## 6.1 先独立读 `SimpleMemory`

执行：

```bash
sed -n '1,240p' agent_system/memory/memory.py
```

你今天只看 `SimpleMemory` 的 3 个方法：

1. `reset(batch_size)`
2. `store(record)`
3. `fetch(history_length, obs_key, action_key)`

### 你必须看懂的结构

`SimpleMemory` 的底层数据结构本质上就是：

```text
self._data = [
  [step1_record, step2_record, ...],   # env 0
  [step1_record, step2_record, ...],   # env 1
  ...
]
```

也就是说：

1. 它不是全局共享记忆
2. 它是“每个环境实例一条自己的历史列表”
3. 每个 step 只 append 一个字典

### `store()` 的本质

`store()` 接受的是一个“按 batch 对齐”的 record：

```python
{
    "text_obs": [...],
    "action": [...],
}
```

然后它会把第 `env_idx` 个 observation 和第 `env_idx` 个 action 组合成一条记录，append 到该环境自己的历史列表里。

### `fetch()` 的本质

`fetch()` 干的事情非常朴素：

1. 每个 env 取最近 `history_length` 条
2. 按固定字符串模板拼成文本
3. 返回 `memory_contexts` 和 `valid_lengths`

你要看懂这不是摘要，不是检索，不是压缩。

它只是把最近几步 `(obs, action)` 原样重播成 prompt 字符串。

## 6.2 用一个 3 分钟小脚本验证 `SimpleMemory` 的行为

执行：

```bash
python - <<'PY'
from agent_system.memory import SimpleMemory

m = SimpleMemory()
m.reset(batch_size=2)

m.store({
    "text_obs": ["obs0_step1", "obs1_step1"],
    "action": ["left", "search[shoes]"],
})

m.store({
    "text_obs": ["obs0_step2", "obs1_step2"],
    "action": ["up", "click[item]"],
})

ctx, lens = m.fetch(history_length=2, obs_key="text_obs", action_key="action")
print("lengths:", lens)
print("env0 ctx:")
print(ctx[0])
print("env1 ctx:")
print(ctx[1])
PY
```

### 你应该观察什么

你会发现：

1. 每个 env 都有自己单独的历史
2. 历史只是字符串拼接
3. `history_length` 控制的是“取多少条最近记录”，不是“总共存多少条”

把这个结论写进 `memory_and_prompt.md`：

“`SimpleMemory` 解决的是短上下文回放，不解决长期知识沉淀、检索、压缩、冲突消解。”

## 6.3 读 3 个 manager 的 `build_text_obs()`，比较记忆是如何拼回 prompt 的

执行：

```bash
sed -n '1,220p' agent_system/environments/prompts/alfworld.py
sed -n '1,220p' agent_system/environments/prompts/webshop.py
sed -n '1,220p' agent_system/environments/prompts/sokoban.py
sed -n '40,220p' agent_system/environments/env_manager.py
sed -n '360,520p' agent_system/environments/env_manager.py
```

今天你只对比 3 件事：

1. 哪些环境会在 prompt 中显式插入 `action_history`
2. 插入的是“原始历史”，还是“筛选后的历史”
3. 哪些环境会额外插入 `available_actions` 或 `admissible_actions`

### `AlfWorldEnvironmentManager`

你必须看明白：

1. 它会从初始 observation 里抽任务描述 `task_description`
2. 非初始步时，会把 `memory.fetch()` 的结果塞进 `ALFWORLD_TEMPLATE`
3. 同时它每一步还会把当前 `admissible_actions` 注入 prompt

这很像一个标准的“文本动作环境”写法。

### `WebshopEnvironmentManager`

你必须看明白：

1. 它会先从原始 observation 里抽出 task
2. 它会把页面当前的 `available_actions` 显式展开成文本
3. 它也会把最近的 `(obs, action)` 历史拼到 prompt 里
4. 如果 prompt 太长，还会退回 `WEBSHOP_TEMPLATE_NO_HIS`

这一点非常重要。

因为这说明：

1. memory 不是绝对优先级
2. 一旦上下文太长，历史可以被直接砍掉

这对你未来做 GUI 斗地主非常有启发：

当视觉输入、对话输入、操作历史、记忆摘要都堆在一起时，你必须显式做 budget 管理，不然 prompt 会爆。

### `SokobanEnvironmentManager`

你必须分成两种模式看：

1. 非视觉模式：
   历史可以通过 `SOKOBAN_TEMPLATE` 拼回 prompt
2. 视觉模式：
   走 `SOKOBAN_VISUAL_TEMPLATE`，不拼历史

这一步是今天很关键的发现。

因为它说明：

当前仓库并没有把“视觉 agent 的历史拼接问题”真正解决掉。

它只是搭了个接口，但视觉模式下并没有完整闭环。

这对你未来项目非常重要。

你的 GUI 斗地主 Agent 如果要做：

1. 视觉截图输入
2. `<memory>` 结构化输出
3. 多轮策略延续

那你不能只照搬 `SOKOBAN_VISUAL_TEMPLATE`。

你必须自己补上：

1. 历史如何编码
2. memory 如何进入下一轮 prompt
3. memory 是自然语言段落、结构化 JSON、还是单独 channel

## 6.4 `history_length` 究竟影响什么

执行：

```bash
rg -n "history_length" agent_system/environments/env_manager.py agent_system/memory/memory.py verl/trainer/config/ppo_trainer.yaml
sed -n '292,308p' verl/trainer/config/ppo_trainer.yaml
```

你要明确：

1. 默认 `env.history_length=2`
2. 它控制的是每次 `fetch()` 取最近多少步历史
3. 它直接决定 prompt 长度
4. 它会影响吞吐、显存、上下文预算

### 回答今天原计划里的问题

“如果把 `history_length` 调大，最可能先炸的是哪里？”

你的答案应该是：

1. 对文本环境，最先感知到的通常是吞吐下降和上下文膨胀
2. 对长上下文模型，也可能先推高显存占用
3. 对训练稳定性，影响不是最先爆炸的，但会通过更长 prompt、更多噪声历史间接影响 reward 质量

换句话说：

最先出问题的通常不是“算法公式”，而是“上下文预算和系统吞吐”。

---

## 7. 晚上主任务：做微型魔改任务 1，并整理“接入新环境最小模板”

现在开始做原计划里的微型魔改任务 1。

今天的目标不是真的把 `<chat>` 和 `<memory>` 用起来，而是先把协议入口改成你未来项目看起来顺手的样子。

## 7.1 先明确今天这次魔改的边界

今天只做：

1. 允许模型输出：
   `<think>...</think><action>...</action><chat>...</chat><memory>...</memory>`
2. projection 仍然只真正提取 `<action>`
3. 多余字段不应该导致动作解析失败

今天明确不做：

1. 真正消费 `<chat>`
2. 真正消费 `<memory>`
3. 修改 reward manager
4. 修改 env step 逻辑
5. 做长期记忆系统

这非常符合 Day 2 的边界。

## 7.2 推荐你选 `Sokoban` 做这次魔改

原因很简单：

1. `Sokoban` 的 `projection.py` 最短，最好改
2. 它没有 WebShop / ALFWorld 那种中文字符额外过滤
3. 你的未来 GUI 项目也是视觉操作类任务，迁移感最强

### 先看当前代码的真实状态

当前 `sokoban_projection.py` 其实已经对“尾部额外内容”相对宽容：

1. 它只找 `<action>...</action>`
2. 只检查有没有 `<think>...</think>`
3. 后面即使再跟 `<chat>` `<memory>`，理论上也不一定会炸

但今天你不要满足于“碰巧能用”。

你要把它改成“职责明确、可扩展、自己看得懂”的版本。

## 7.3 推荐的魔改方向：把 tag 提取显式化

你可以在文档里先按这个思路写，再自己动手改。

推荐结构如下：

```python
import re

def extract_tag(text: str, tag: str):
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()
```

然后在 `sokoban_projection()` 里显式做：

1. `think_text = extract_tag(original_str, "think")`
2. `action_text = extract_tag(original_str, "action")`
3. `chat_text = extract_tag(original_str, "chat")`
4. `memory_text = extract_tag(original_str, "memory")`

注意：

1. `chat_text` 和 `memory_text` 今天先提取但不用
2. 你的合法性只依赖 `think_text` 和 `action_text`
3. 这样代码可读性会比现在用 `find()` 更高

### 你今天改完后应该达到的效果

下面这种输出应该被视为合法：

```text
<think>先往右走一格，靠近箱子。</think>
<action>right</action>
<chat>我准备先试探一下这个位置。</chat>
<memory>箱子现在还没有靠近目标点。</memory>
```

下面这种输出应该被视为非法：

```text
<action>right</action>
<chat>我先走一步。</chat>
```

因为缺了 `<think>`。

下面这种输出也应该被视为非法：

```text
<think>我准备往右走。</think>
<chat>先动一下。</chat>
<memory>观察箱子位置。</memory>
```

因为缺了 `<action>`。

## 7.4 今天不要求你真的提交这段代码，但你至少要写出“设计说明”

把下面 4 句话写进 `new_env_template.md`：

1. 输出协议扩展时，parser 应该显式提取 tag，而不是靠字符串碰运气。
2. `<chat>` 和 `<memory>` 是否被消费，是下游模块的事；projection 的职责首先是解析。
3. `projection.py` 的第一职责是把原始文本变成“可执行动作 + 合法性标记”。
4. 结构化输出协议一旦确定，未来 reward、memory、env 都应该围绕这个协议协同演进。

## 7.5 整理“新增一个环境时，最少要改哪些文件”

这是今晚最重要的整理任务。

你要把今天学到的东西收敛成一个最小模板。

建议你在 `new_env_template.md` 写成下面这个结构。

### 模板 1：底层环境文件 `envs.py`

最少要负责：

1. 定义单个 worker 如何 `reset()` 和 `step()`
2. 定义 vectorized wrapper 如何创建 `env_num * group_n` 个 worker
3. 在 `reset()` 中把同一初始状态按 `group_n` 复制出来
4. 返回统一长度的 `obs_list, reward_list, done_list, info_list`

### 模板 2：动作协议文件 `projection.py`

最少要负责：

1. 从模型原始文本里解析 `<action>`
2. 检查最基本的输出协议，比如 `<think>` 是否存在
3. 返回 `actions` 和 `valids`
4. 明确第一层合法性边界：
   是只做格式检查，还是顺手做一部分动作空间检查

### 模板 3：环境管理器 `EnvironmentManager`

最少要负责：

1. `reset()` 时统一 observation 结构
2. `step()` 时先走 projection，再走 env
3. 把 `is_action_valid` 挂进 `info`
4. 在 step 之间维护 memory
5. 把历史、任务、可选动作等组织成最终 prompt

### 模板 4：prompt 模板文件

最少要负责：

1. 讲清楚任务
2. 讲清楚当前观测
3. 讲清楚当前合法动作空间
4. 讲清楚模型应该输出的协议
5. 在需要时接收 memory / history

### 模板 5：`make_envs()` 注册分支

最少要负责：

1. 构造训练环境
2. 构造验证环境
3. 把 `projection_f` 绑到 manager 上
4. 保证训练时 `group_n=env.rollout.n`，验证时通常 `group_n=1`

---

## 8. 今天的最终验收标准

今天结束前，你必须完成下面这些具体产出。

### 产出 1：一页 `group reset` 口头复述

你必须能脱口而出：

1. `data.train_batch_size` 决定原始样本数
2. `env.rollout.n` 决定每个样本扩成多少个环境实例
3. `env_num * group_n` 决定底层 worker 数
4. `np.repeat(seeds, group_n)` 是 Sokoban group reset 的直接实现

### 产出 2：一页“动作合法性链路图”

你必须能画出：

```text
model output
-> projection
-> is_action_valid
-> rollout_loop
-> invalid penalty
-> valid_action_ratio
```

并能解释每一层的职责边界。

### 产出 3：一页“memory 真相”

你必须能明确写出：

1. `SimpleMemory` 现在能做什么
2. `SimpleMemory` 不能做什么
3. 视觉 Sokoban 当前并没有真正把 history 拼回 prompt
4. 这对你未来 GUI 斗地主意味着什么

### 产出 4：一页“新增环境最小模板”

你必须能列出最少修改文件：

1. `envs.py`
2. `projection.py`
3. `prompt template`
4. `EnvironmentManager`
5. `make_envs()` 注册分支

---

## 9. Day 2 面试硬核拷问与答题抓手

下面这些问题，今晚必须自己闭卷回答。你可以先答，再对照“答题抓手”查漏补缺。

### 1. 为什么动作合法性最好先在 `projection.py` 做第一层约束，而不是完全丢给环境？

答题抓手：

1. projection 解决的是输出协议合法性
2. 环境更关注执行合法性
3. projection 层早判错，trainer 才能稳定记录 `is_action_valid`
4. 这样 reward penalty 链更统一，跨环境更容易复用

### 2. 如果你把 `history_length` 调大，最可能先炸的是哪里：显存、吞吐、还是 reward 稳定性？为什么？

答题抓手：

1. 先炸的通常是 prompt 长度和吞吐
2. 长上下文会推高显存占用
3. reward 稳定性更多是间接受影响，不一定是第一时间爆炸
4. 工程上你首先要关注上下文预算，而不是先谈算法本身

### 3. 为什么 `WebshopEnvironmentManager` 会在 prompt 中显式注入 `available_actions`？

答题抓手：

1. 网页交互的动作空间随状态变化
2. 如果不显式提供，模型会在巨大自由度里乱生成
3. 这是一种“先在 prompt 中缩窄动作空间，再在 projection / env 里做校验”的组合策略
4. 但当前 `webshop_projection.py` 并没有真正检查 action 是否属于 `available_actions`

### 4. `SimpleMemory` 和真正长期记忆系统的本质差距是什么？

答题抓手：

1. `SimpleMemory` 只是最近若干步的字符串回放
2. 没有摘要、检索、优先级、压缩、遗忘机制
3. 没有结构化状态更新
4. 没有跨 episode 沉淀

### 5. 如果你做斗地主 GUI agent，`projection.py` 最合理的职责边界是什么？

答题抓手：

1. 解析结构化输出协议
2. 抽取并校验 `<action>`
3. 给出第一层格式合法性标记
4. 允许额外字段如 `<chat>` `<memory>` 存在
5. 不应该在这里承担全部环境逻辑或奖励逻辑

### 6. 如果训练时 `valid_action_ratio` 很低，你应该优先查哪 4 个地方？

答题抓手：

1. prompt 有没有把输出协议写清楚
2. projection 是否过严或有 bug
3. env manager 是否正确透传 `is_action_valid`
4. trainer 是否开启了 invalid action penalty，以及惩罚是否过重

---

## 10. 今天结束前，用 5 句话总结 Day 2

今晚收工前，请你强制自己口头说出下面这 5 句话。

1. `env.rollout.n` 在 `verl-agent` 里是环境侧 group 扩张器。
2. `projection.py` 是动作合法性的第一道闸门，但不是唯一闸门。
3. 非法动作惩罚是一条跨 `projection -> env_manager -> rollout_loop -> trainer` 的链。
4. 当前 `SimpleMemory` 本质上只是最近若干步 `(obs, action)` 的 prompt 重放器。
5. 我未来做 GUI 斗地主 Agent 时，最先要魔改的是 `env_manager`、`projection`、`prompt/memory` 三层，而不是先去碰模型底层实现。

如果你能把这 5 句话说顺，而且能举出对应代码位置和一个真实实验现象，Day 2 就算学透了。
