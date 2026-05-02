# 在本仓库中添加自定义 Agent 行动环境

本文说明如何把一个新环境接入 `verl-agent`，达到可用 PPO/GRPO/GiGPO 等 agentic RL 训练完整跑通的水平。核心思想：底层环境只负责状态转移；`EnvironmentManager` 负责把 LLM 文本动作和环境动作、环境观测和模型 prompt 对齐；`TrajectoryCollector` 负责多轮采样；`EpisodeRewardManager` 把 episode 累计奖励写回训练。

## 1. 总链路

训练入口通常是 `python -m verl.trainer.main_ppo ...`：

1. `verl/trainer/main_ppo.py` 调用 `agent_system.environments.make_envs(config)` 创建训练/验证环境。
2. `TrajectoryCollector.multi_turn_loop()` 按 batch 调用 `envs.reset()`，随后循环：构造 prompt -> actor 生成 response -> `envs.step(text_actions)` -> 记录奖励、done、info。
3. 每条轨迹的有效 step 被收集成 `DataProto`，其中包含 `episode_rewards`、`episode_lengths`、`rewards`、`active_masks`、`is_projection_valid`、`anchor_obs` 等字段。
4. `EpisodeRewardManager` 默认把 `episode_rewards` 放在每条 response 的最后一个有效 token 上；若启用 `actor_rollout_ref.actor.use_projection_invalid_penalty`，非法投影动作会额外扣分。
5. 优势估计器使用这些字段训练策略。GiGPO/HGPO 还会读取 `anchor_obs` 做分组或相似性相关计算。

因此新增环境通常需要改四处：

- `agent_system/environments/env_package/<your_env>/envs.py`
- `agent_system/environments/env_package/<your_env>/projection.py`
- `agent_system/environments/env_package/<your_env>/__init__.py`
- `agent_system/environments/env_manager.py`：新增一个 manager 类和 `make_envs()` 分支

可选：新增 prompt 模板、数据预处理脚本、训练 shell 脚本、单元/冒烟测试。

## 2. 数据契约

训练数据是 parquet，至少需要 `data_source`、`prompt`、`ability`/`extra_info` 等普通 verl 字段。对于很多交互环境，`prompt` 只是占位，用来指示文本/视觉模态和 batch 大小；真正任务可以来自环境自身，也可以来自每条样本的 `env_kwargs`。

如果你的环境需要每条样本携带题目、目标、初始状态或 ground truth，在 parquet 行里写：

```python
{
    "data_source": "my_env",
    "prompt": [{"role": "user", "content": ""}],  # 视觉模型可用 "<image>" 占位
    "ability": "agent",
    "extra_info": {"index": idx, "split": split},
    "env_kwargs": {
        "task": task,
        "ground_truth": answer,
        "seed": seed,
    },
}
```

`TrajectoryCollector` 会在 `reset` 前执行：

```python
kwargs = gen_batch.non_tensor_batch.pop("env_kwargs", None)
obs, infos = envs.reset(kwargs=kwargs)
```

所以 manager 的 `reset(kwargs)` 必须能处理两类情况：`kwargs is None`，以及 `kwargs` 是长度等于当前 rollout batch 的 `List[Dict]`/object array。训练时若 `env.rollout.n > 1`，输入 batch 会先按组重复，环境数量也必须是 `data.train_batch_size * env.rollout.n`；验证环境固定 `group_n=1`。

## 3. 底层环境契约

底层并行环境可以用 Ray actor、线程池、进程池、Gym vector env，或者简单 Python list。对上只需暴露：

```python
class MyVectorEnv:
    def reset(self, kwargs=None):
        return obs_list, info_list

    def step(self, actions):
        return next_obs_list, reward_list, done_list, info_list

    def close(self):
        ...
```

硬性要求：

- `reset` 返回的样本数必须等于当前 rollout batch；训练通常为 `data.train_batch_size * env.rollout.n`，验证为 `data.val_batch_size`。
- `step(actions)` 的 `actions` 是已经投影后的环境动作，长度同 batch。
- `reward_list` 和 `done_list` 必须是一维、可转为 numpy 的数值/布尔序列。
- 已 done 的环境仍可能在外层循环后续 step 中被传入动作；奖励累计只对 `active_masks` 生效，但为了 batch 对齐，底层最好保持返回占位结果，不要缩短 batch。
- `info_list` 必须是 `List[Dict]`。默认成功率统计会读取最后一个有效 step 的 `info["won"]`。
- 随机性要由 `config.env.seed` 控制；训练/验证建议使用不同 seed 区间，避免验证泄漏。
- 重资源环境应在 `close()` 中释放 actor、进程、socket、文件句柄。

建议构造函数形态：

```python
def build_my_envs(seed, env_num, group_n, is_train=True, env_config=None, resources_per_worker=None):
    return MyVectorEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_config=env_config,
        resources_per_worker=resources_per_worker,
    )
```

`env_num` 是原始数据 batch 大小，`group_n` 是同一任务的重复采样数。若做 GRPO/GiGPO，同一组内应共享同一个任务/初始分布，便于比较不同 response；可参考 gym_cards/sokoban 对 seed 的 repeat 逻辑。

## 4. 动作投影契约

LLM 输出是自由文本，底层环境需要结构化动作。为每个环境实现 `projection.py`：

```python
from typing import List, Tuple

def my_env_projection(text_actions: List[str]) -> Tuple[List[object], List[int]]:
    env_actions = []
    valids = []
    for text in text_actions:
        action, valid = parse_first_action(text)
        env_actions.append(action)     # 可为 str/int/dict，只要底层 env.step 接受
        valids.append(int(valid))      # 1 合法，0 非法
    return env_actions, valids
```

原则：

- 只解析第一个完整动作，避免模型输出多个动作导致一步执行多次。
- 解析失败必须返回安全动作或空动作，并把 `valid=0`；不要抛异常中断整批训练。
- 合法性和可执行性分开：格式非法用 `is_projection_valid=0`；环境执行失败可由环境给低 reward、`info` 说明原因。
- Prompt 中必须明确动作格式，例如 JSON、XML tag、函数调用式字符串或有限动作列表。

manager 的 `step` 需要把 `valids` 写入 `infos[i]["is_projection_valid"]`。训练器会用它统计 `episode/projection_valid_ratio`，并可施加非法动作惩罚。

## 5. EnvironmentManager 契约

`EnvironmentManager` 是最重要的适配层，负责：

- `reset(kwargs)`：重置底层环境，构造首轮模型观测。
- `step(text_actions)`：投影 LLM 输出，推进环境，构造下一轮模型观测。
- `build_text_obs(...)`：把任务、当前观测、可用动作、历史拼成 prompt。
- `success_evaluator(...)`：从整条轨迹的 `info` 中产出日志指标。

对 `TrajectoryCollector` 的返回契约固定为：

```python
observations = {
    "text": List[str] | None,
    "image": np.ndarray | torch.Tensor | list | None,
    "anchor": Any | None,
}
infos = List[Dict]
```

`text`/`image`/`anchor` 含义：

- `text`：每个样本给 LLM 的用户消息内容。文本环境必须提供；视觉环境也建议提供，且需要包含 `<image>` 占位符以适配 Qwen-VL 类 processor。
- `image`：每个样本一张图，通常是 RGB HWC `uint8` 或 `[0, 1]` float；`TrajectoryCollector` 会转成 PIL 并送入 processor。
- `anchor`：无历史、无提示词污染的原始观测，用于 GiGPO/HGPO 分组或相似性逻辑。文本环境可放当前原始文本，视觉环境可放原图/状态数组；不用时也建议给可 hash/可 numpy 化的对象。

最小 manager 骨架：

```python
import numpy as np
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory

class MyEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        raw_obs, infos = self.envs.reset(kwargs=kwargs)
        self.memory.reset(batch_size=len(raw_obs))
        self.tasks = [info.get("task", "") for info in infos]
        self.prev_obs = raw_obs
        obs = {
            "text": self.build_text_obs(raw_obs, infos, init=True),
            "image": None,
            "anchor": np.array(raw_obs, dtype=object),
        }
        return obs, infos

    def step(self, text_actions):
        actions, valids = self.projection_f(text_actions)
        raw_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({"text_obs": self.prev_obs, "action": actions})
        self.prev_obs = raw_obs

        for i, info in enumerate(infos):
            info["is_projection_valid"] = to_numpy(valids[i])

        obs = {
            "text": self.build_text_obs(raw_obs, infos),
            "image": None,
            "anchor": np.array(raw_obs, dtype=object),
        }
        return obs, to_numpy(rewards), to_numpy(dones), infos

    def build_text_obs(self, raw_obs, infos, init=False):
        prompts = []
        if not init and self.config.env.history_length > 0:
            histories, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
            )
        for i, cur in enumerate(raw_obs):
            if init or self.config.env.history_length <= 0:
                prompts.append(f"Task:\n{self.tasks[i]}\n\nObservation:\n{cur}\n\nAction:")
            else:
                prompts.append(
                    f"Task:\n{self.tasks[i]}\n\n"
                    f"History ({valid_lens[i]} steps):\n{histories[i]}\n\n"
                    f"Observation:\n{cur}\n\nAction:"
                )
        return prompts

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for t in reversed(range(len(total_batch_list[batch_idx]))):
            if total_batch_list[batch_idx][t]["active_masks"]:
                info = total_infos[batch_idx][t]
                success["success_rate"].append(float(info.get("won", False)))
                return
```

注意：`SimpleMemory.store` 以 batch 维度存储，同一 step 的字段长度必须等于 batch。历史过长会撑爆 `data.max_prompt_length`；高风险环境应主动截断历史或摘要。

## 6. 注册到 make_envs

在 `agent_system/environments/env_package/<your_env>/__init__.py` 暴露构造器和投影函数：

```python
from .envs import build_my_envs
from .projection import my_env_projection
```

在 `agent_system/environments/env_manager.py` 中新增 manager 类，并在 `make_envs(config)` 里加分支：

```python
elif "my_env" in config.env.env_name.lower():
    from agent_system.environments.env_package.my_env import build_my_envs, my_env_projection

    _envs = build_my_envs(
        seed=config.env.seed,
        env_num=config.data.train_batch_size,
        group_n=group_n,
        is_train=True,
        env_config=config.env,
        resources_per_worker=resources_per_worker,
    )
    _val_envs = build_my_envs(
        seed=config.env.seed + 1000,
        env_num=config.data.val_batch_size,
        group_n=1,
        is_train=False,
        env_config=config.env,
        resources_per_worker=resources_per_worker,
    )

    projection_f = partial(my_env_projection)
    envs = MyEnvironmentManager(_envs, projection_f, config)
    val_envs = MyEnvironmentManager(_val_envs, projection_f, config)
    return envs, val_envs
```

配置可以直接通过命令行覆盖；若要长期维护，再把默认值加入 `verl/trainer/config/ppo_trainer.yaml` 的 `env:` 段，例如：

```yaml
env:
  env_name: my_env
  max_steps: 20
  history_length: 2
  resources_per_worker:
    num_cpus: 0.1
    num_gpus: 0
  rollout:
    n: 4
  my_env:
    difficulty: easy
```

## 7. 奖励与日志契约

默认训练使用 `reward_model.reward_manager=episode`：

- 每步环境返回的 `reward` 会累加到 `episode_rewards`。
- episode 结束或达到 `env.max_steps` 后，`episode_rewards` 被写到每个有效 step 样本里。
- `EpisodeRewardManager` 把该 episode 总分写到 response 最后一个有效 token。

因此环境 reward 可以是稀疏终局分、稠密 shaping 分，或二者相加。若 reward 量纲很大，建议在环境里归一化到稳定范围，或者启用/改造 reward manager。

`success_evaluator` 只影响日志和部分筛选统计，不直接决定 reward。默认实现取最后一个有效 step 的 `info["won"]`。如果需要多指标，重写 `_process_batch`，向 `success` 写入等长数组，例如：

```python
success["success_rate"].append(float(info["won"]))
success["task_score"].append(float(info.get("score", 0.0)))
```

`filter_groups` 动态采样会按同组 episode reward 是否全相同来过滤；新环境使用 GRPO/GiGPO 时，应保证同一 group 内任务一致、reward 可区分，否则会被大量过滤。

## 8. Prompt 与动作格式

Prompt 需要同时满足三件事：

- 给足任务目标、当前观测、可用动作或动作语法。
- 明确“每轮只输出一个动作”，并让 projection 能稳定解析。
- 控制长度，尤其是 `history_length > 0` 时；超过 `data.max_prompt_length` 且 `data.truncation=error` 会直接报错。

推荐把模板放在 `agent_system/environments/prompts/<your_env>.py`，再由 manager import。不要把 ground truth 或评测答案放进 prompt；可以放进 `env_kwargs` 或底层环境私有状态。

## 9. 训练脚本最小检查点

最小文本环境命令形态：

```bash
python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=gae \
  data.train_files=$TRAIN_DATA \
  data.val_files=$VAL_DATA \
  data.train_batch_size=32 \
  data.val_batch_size=64 \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-1.5B-Instruct \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.actor.use_projection_invalid_penalty=True \
  actor_rollout_ref.actor.projection_invalid_penalty_coef=0.05 \
  reward_model.reward_manager=episode \
  env.env_name=my_env \
  env.max_steps=20 \
  env.history_length=2 \
  env.rollout.n=1 \
  trainer.total_epochs=1
```

GRPO/GiGPO 类 group 采样不要改 `actor_rollout_ref.rollout.n`；本仓库要求它保持 `1`，通过 `env.rollout.n` 控制环境组内重复采样。

视觉环境还需要：

- 数据 parquet 的 `prompt.content` 或 manager 输出 text 中包含 `<image>`。
- `data.image_key=images` 只用于数据加载占位；实际训练图像来自 manager 的 `obs["image"]`。
- 选择 VL 模型和足够的 `data.max_prompt_length`。

## 10. 冒烟测试顺序

先不用大模型训练，按以下顺序压测：

1. 直接实例化 `build_my_envs(..., env_num=2, group_n=2)`，检查 `reset/step` 长度、reward/done 类型、`close`。
2. 调用 `my_env_projection` 覆盖合法、非法、多动作、空输出、非字符串输出。
3. 实例化 `MyEnvironmentManager`，手写两轮 `reset -> step`，确认 observation dict、`is_projection_valid`、`info["won"]`。
4. 用很小 batch 跑 `python -m verl.trainer.main_ppo ... data.train_batch_size=2 data.val_batch_size=2 env.max_steps=2 trainer.total_epochs=1 trainer.test_freq=1`。
5. 再打开目标算法、目标 `env.rollout.n`、目标模型规模。

常见失败定位：

- `gen_batch size ... does not match obs size ...`：环境数量没有按 `train_batch_size * env.rollout.n` 创建，或 `reset` 返回数量不对。
- `KeyError: won`：默认 success evaluator 找不到 `info["won"]`；补字段或重写 `_process_batch`。
- prompt 过长：降低 `history_length`，截断历史，或增大 `data.max_prompt_length`。
- 视觉 token 报错：`text` 缺 `<image>`，图像不是 RGB/HWC，或 processor 与模型不匹配。
- 非法动作率高：动作格式太松、projection 解析过宽/过窄，先调 prompt 和 projection。
- GiGPO 相似性报错：`anchor` 不可 numpy 化/不可 hash，改成字符串、数值数组或简单 dict/list。

## 11. 接入完成标准

一个自定义环境达到可训练水平，应满足：

- `make_envs(config)` 能同时创建 train/val env，且资源释放正常。
- `reset/step` 在任意 batch 下长度稳定，done 后仍保持 batch 对齐。
- manager 输出的 `text/image/anchor` 能被 `TrajectoryCollector.preprocess_batch` 编码。
- 每个 `info` 至少包含 `won`，每个 step 后包含 `is_projection_valid`。
- reward 数值稳定，episode 累计分符合训练目标。
- 小 batch、小步数能跑完一次 train 和 validation；目标 `env.rollout.n` 下 group 任务语义正确。
