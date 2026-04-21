# 训练性能优化及相关参数设置学习指南

## 0. 适用范围与结论先行

这份指南基于当前仓库的实现，重点覆盖：

- `verl.trainer.main_ppo` / `verl.trainer.ppo.ray_trainer`
- `verl.workers.fsdp_workers`
- `verl.workers/actor/dp_actor.py`
- `verl.workers/critic/dp_critic.py`
- `verl.workers/rollout/vllm_rollout/*`
- `agent_system/multi_turn_rollout/*`
- `verl/trainer/config/ppo_trainer.yaml`

不展开 Megatron，分布式部分主要看 FSDP。

你的硬件我按如下假设写：

- 单机
- 2 张 GPU
- `NVIDIA RTX PRO 6000 Blackwell`
- 总显存 192GB，即单卡约 96GB

本地当前环境里我额外确认到：

- `torch==2.8.0+cu128`
- `vllm==0.11.0`

这意味着：

1. 本仓库在你环境里更适合走“新版本 vLLM 路径”。
2. 老文档里为 `vllm<=0.6.3` 准备的 `VLLM_ATTENTION_BACKEND=XFORMERS` 兜底方案，不应作为你的默认设置。
3. 你这套 2x96GB 配置，对 `1.5B/7B` 非常宽松，对 `14B` 大概率可做高效训练，对 `32B` 可以做但要明显偏向 GRPO/GiGPO + 动态 batch + offload；若是 PPO+critic，再叠长上下文，压力会很大。

一句话总结调参方向：

- 先把 **长度上限**、**env 分组**、**rollout TP** 定对；
- 再开 **remove padding**、**gradient checkpointing**、**dynamic batch**；
- 然后分别调 **rollout 侧吞吐** 和 **actor/ref/critic 侧吞吐**；
- 最后才用 **param/optimizer/activation offload** 去换可训练性。

---

## 1. 先建立全局性能模型

这个仓库的单步训练，不是单纯的“前向 + 反向”，而是一个长链路：

1. dataloader 取 prompt
2. 环境 reset / step，进入多轮 agent-environment rollout
3. rollout 生成回复
4. actor 重新算 `old_log_probs`
5. ref 算 `ref_log_prob`（如果启用了 KL loss 或 reward KL）
6. critic 算 `values`（只有 PPO/GAE 需要）
7. driver 侧算 reward / advantage
8. critic update
9. actor update

所以总吞吐受最慢环节决定。通常瓶颈分成 4 类：

- `gen` 慢：rollout/vLLM/环境交互慢
- `old_log_prob` / `ref` / `values` 慢：训练模型前向慢
- `update_actor` / `update_critic` 慢：反向和优化器慢
- `env` 慢：CPU、Ray worker、环境步数、history 拼接、动态采样拖慢

你调优时不要只盯 GPU 利用率，要看日志里的阶段耗时：

- `timing_s/gen`
- `timing_s/old_log_prob`
- `timing_s/ref`
- `timing_s/values`
- `timing_s/update_actor`
- `timing_s/update_critic`
- `perf/throughput`
- `perf/mfu/actor`
- `perf/mfu/critic`
- `perf/max_memory_allocated_gb`
- `perf/max_memory_reserved_gb`

如果不知道该调哪个参数，先看哪一段耗时最大。

---

## 2. 这个仓库里最重要的“参数语义”

很多人调不动，不是因为不会调，而是因为把“全局 batch”和“每卡 batch”混了。

### 2.1 全局语义参数

这些参数更接近“算法语义”或“整个 step 处理多少数据”：

- `data.train_batch_size`
- `data.gen_batch_size`
- `actor_rollout_ref.actor.ppo_mini_batch_size`
- `critic.ppo_mini_batch_size`
- `env.rollout.n`
- `actor_rollout_ref.rollout.n`

其中对你这个 `verl-agent` 分支，最重要的一条是：

- `actor_rollout_ref.rollout.n` 在 `agent_system` 主训练入口里被强制要求为 `1`
- 这里真正承担“同一个 prompt 采多少条轨迹/分组多少个 env”的，是 `env.rollout.n`

也就是说，在这个仓库里：

- 通常不是 `rollout.n > 1`
- 而是 `env.rollout.n > 1`

这是和上游普通 verl 脚本最容易混淆的一点。

### 2.2 每卡局部语义参数

这些参数直接决定单次 forward/backward 吃多少显存、吞吐有多高：

- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
- `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`
- `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
- `critic.ppo_micro_batch_size_per_gpu`
- `critic.forward_micro_batch_size_per_gpu`
- `reward_model.micro_batch_size_per_gpu`

如果你开启动态 batch，则会改为“每卡 token 上限”语义：

- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`
- `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`
- `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`
- `critic.ppo_max_token_len_per_gpu`
- `critic.forward_max_token_len_per_gpu`
- `reward_model.forward_max_token_len_per_gpu`

### 2.3 这几个 batch 到底是什么关系

在 PPO 语境下，最实用的理解方式如下：

- `data.train_batch_size`
  - 一个训练 step 期望收集多少个“prompt”
- `env.rollout.n`
  - 每个 prompt 复制成多少条并行轨迹组
- `actor_rollout_ref.actor.ppo_mini_batch_size`
  - 一次 PPO 更新时，拿多少条轨迹作为一个 mini-batch
- `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`
  - 这个 mini-batch 再切成多少个每卡小块去做 forward/backward

对当前 agent 分支，可先粗略理解为：

`有效轨迹数 ≈ data.train_batch_size * env.rollout.n`

注意这还会受到动态采样、filter_groups、adjust_batch 的影响，后面单独讲。

---

## 3. 先理解这份仓库里的 FSDP 实现，别拿“教科书 FSDP”硬套

### 3.1 RL 训练里的 `fsdp`，这里更接近 ZeRO-2

在 `verl/workers/fsdp_workers.py` 里，1D device mesh 对应的 sharding 策略是：

- `ShardingStrategy.SHARD_GRAD_OP`

源码里甚至直接写了注释：

- `# modified to zero-2!!!`

这意味着：

- 在 RL worker 这条路径里，`strategy=fsdp` 不要简单理解成“标准 FULL_SHARD”
- 它更接近 ZeRO-2 风格
- 这也是为什么本仓库很多 7B/14B 配置在 2 卡上会比你想象中更容易跑起来

相反，在 `verl/trainer/fsdp_sft_trainer.py` 的 SFT 路径中：

- `fsdp` 才是 `ShardingStrategy.FULL_SHARD`

所以：

- RL FSDP 和 SFT FSDP 的显存/速度特性，不要直接互相套结论

### 3.2 `fsdp2` 是另一条实现路径

仓库同时支持：

- `fsdp`
- `fsdp2`

`fsdp2` 相关关键参数：

- `offload_policy`
- `reshard_after_forward`
- `fsdp_size`

对你的 2 卡机器，我的建议是：

- 默认优先从 `fsdp` 开始
- 先把流程和吞吐调通
- 只有在你明确要测试 `fsdp2` 特性，或者遇到 `fsdp` 路径不稳定/不够省显存时，再切换

原因很简单：

- 这个仓库的大量 RL 示例和 worker 逻辑，默认使用体验仍是 `fsdp`
- 你现在的主要目标是“把训练性能调到合适”，不是做 FSDP1/FSDP2 基准论文

### 3.3 `fsdp_size` 在 2 卡机器上一般不要乱动

代码允许用 `fsdp_size` 构建 2D mesh。

但在你的单机双卡场景中：

- 默认 `fsdp_size=-1`
- 也就是直接把 2 卡作为一个 FSDP world

这是最自然、最稳的设置。

在 2 卡场景里改 `fsdp_size` 的收益很有限，复杂度却会上升。除非你非常清楚自己在做什么，否则保持：

```yaml
actor_rollout_ref.actor.fsdp_config.fsdp_size: -1
critic.model.fsdp_config.fsdp_size: -1
reward_model.model.fsdp_config.fsdp_size: -1
```

---

## 4. 仓库里所有和性能最相关的参数，按模块拆开讲

## 4.1 数据与长度相关

### `data.max_prompt_length`

作用：

- 控制 prompt token 上限
- 影响训练显存、rollout KV cache、recompute logprob、critic 前向

调参原则：

- 不是越大越好
- 应该尽量贴近任务真实分布的 P95/P99

如果这个值明显偏大，会出现：

- 大量 padding
- remove padding 开启前吞吐骤降
- rollout `max_model_len` 也跟着变大
- vLLM KV cache 压力变大

如果这个值太小，会出现：

- `data.truncation='error'` 时直接报错
- `prompt_length/clip_ratio` 很高
- 环境历史被截断，策略质量下降

经验建议：

- AlfWorld：通常 `2048` 起步
- WebShop：常见 `4096`
- Sokoban / 短任务：`1024` 往往够用

### `data.max_response_length`

作用：

- rollout 最大生成长度
- 直接决定 vLLM 生成上限和后续训练 token 数

调参原则：

- 过大时，吞吐和显存都被拖垮
- 过小时，`response_length/clip_ratio` 高，说明经常撞上上限

建议：

- 先用任务真实需要的最小值
- 大多数 agent 任务先从 `256` 或 `512` 开始
- 只有任务确实需要长 reasoning / 长 action chain 再继续加

### `data.truncation`

常见值：

- `error`
- `left`
- `right`
- `middle`

性能层面建议：

- 开发阶段优先 `error`
  - 这样你能看到到底是长度不够还是别的问题
- 生产训练若长输入分布波动大，可改 `left`
  - 但要结合任务语义确认是否合理

### `data.filter_overlong_prompts`

作用：

- 提前过滤过长样本

何时建议开：

- 数据集较小或中等
- 你不希望训练时因为少量坏样本反复中断

何时可能拖慢：

- 大规模数据集
- worker 数不足

配套参数：

- `data.filter_overlong_prompts_workers`

---

## 4.2 agent / 环境侧参数

这个分支和上游通用 RLHF 最大的不同，是多轮环境交互本身就是性能大头。

### `env.rollout.n`

作用：

- 一个 prompt 复制成多少个并行环境轨迹
- 对 GRPO / GiGPO / HGPO / DAPO 类方法尤其关键

影响：

- 线性增加 rollout 次数
- 线性增加环境 step 开销
- 线性增加后续 logprob / update 的样本数

经验上：

- `n=1` 最省资源
- `n=4/8` 是常见折中
- `n=16` 在双卡上就已经非常吃紧了，通常只适合更轻模型或强动态采样场景

### `env.max_steps`

作用：

- 单条 episode 的最大交互步数

影响：

- 直接决定平均 response token 总量
- 也会决定一个 step 内总 rollout 时长上限

调参建议：

- 如果任务多数在 8~15 步完成，就不要设 50
- 太大的 `max_steps` 会制造尾部慢样本

### `env.history_length`

作用：

- prompt 里保留多少步历史

影响：

- 历史越长，prompt 越长
- 长度压力会传导到 rollout、old_log_prob、critic、reward model

建议：

- 除非算法必须，先用 `2`
- 从 `2 -> 4` 是很常见的性能拐点

### `env.resources_per_worker.num_cpus`

作用：

- Ray 环境 worker 的 CPU 配额

如果你出现这些现象：

- GPU 明显空闲
- `timing_s/gen` 很长
- 但显存、算力都不饱和

优先怀疑：

- 环境侧 CPU 不够
- 不是 GPU 不够

对你的双卡单机，一般建议：

- 从 `0.1` 起步
- 如果 CPU 够多、环境很重，可以尝试 `0.2 ~ 0.5`

### `algorithm.filter_groups.enable`

作用：

- 动态采样 / group filtering
- 过滤掉“组内所有结果都一样”的低信息组

收益：

- 更好的训练有效样本密度

代价：

- 可能需要重复 rollout 多次，直到攒够有效组
- 会让 wall-clock 变长

相关参数：

- `data.gen_batch_size`
- `data.train_batch_size`
- `algorithm.filter_groups.max_num_gen_batches`

理解方式：

- `gen_batch_size` 决定每轮先生成多少 prompt
- `train_batch_size` 决定最终希望保留多少 prompt
- 开了 filter 之后，系统会“边生成边过滤，直到凑够为止”

双卡建议：

- 如果你主要目标是先把吞吐打高，先关掉 filter
- 如果你主要目标是提升样本效率，再打开它

---

## 4.3 rollout / vLLM 参数

对这个仓库，rollout 常常是第一大瓶颈。

### `actor_rollout_ref.rollout.tensor_model_parallel_size`

在你 2 卡机器上，只需要考虑两个值：

- `1`
- `2`

它的本质是 rollout 推理用几卡做 TP。

#### 取 `1`

含义：

- 每卡一个独立 rollout replica
- 2 卡机器上相当于 2 个 DP 副本

优点：

- 并发高
- 吞吐常常更好

缺点：

- 每卡都要留更完整的模型权重和更多 KV cache 空间
- 对长上下文 / 大模型更容易 OOM

#### 取 `2`

含义：

- 两张卡一起做一个 TP=2 的 rollout engine

优点：

- 更稳
- 显存压力更小
- 更适合 14B 以上、长上下文、或者 `gpu_memory_utilization` 想拉高时

缺点：

- 并发副本减少
- 某些小模型下吞吐反而更差

双卡建议：

- `1.5B / 7B`，短中上下文：先试 `tensor_model_parallel_size=1`
- `7B` 长上下文、`14B+`：优先试 `2`

### `actor_rollout_ref.rollout.gpu_memory_utilization`

这是 rollout 调优最关键参数之一。

对 vLLM，它决定：

- 为模型静态内存和 KV cache 预留多少 GPU 内存

你的环境是 vLLM 0.11，建议按下面思路调：

- 保守起步：`0.5 ~ 0.6`
- 吞吐优先：`0.65 ~ 0.8`
- 极限压榨：`0.8+`

但注意：

- 这个 rollout 和 actor/ref/critic 在同一轮训练里要共享整机显存预算
- 你把它调太高，通常 first hit 的不是 rollout，而是后面的 actor/ref/critic OOM

实践建议：

- 先 `0.6`
- 观察 `timing_s/gen`、OOM 情况、step 稳定性
- 再一点点拉到 `0.7`、`0.75`

### `actor_rollout_ref.rollout.max_num_batched_tokens`

作用：

- rollout 解码时一次能打包多少 token

影响：

- 太小：并发不够，吞吐低
- 太大：KV cache / 调度压力上升，甚至 OOM

仓库默认是：

```yaml
actor_rollout_ref.rollout.max_num_batched_tokens: 8192
```

经验建议：

- 中短上下文可从 `8192` 或更大开始
- 长上下文时，最好和 `max_prompt_length + max_response_length` 联动思考

特别注意：

- 如果 `enable_chunked_prefill=True`
- 那么代码要求 `max_num_batched_tokens >= max_model_len`

否则直接报错。

### `actor_rollout_ref.rollout.max_model_len`

默认不设时：

- 使用 `max_prompt_length + max_response_length`

这通常够用。

只在你需要：

- 给多轮工具调用预留额外长度
- 或显式限制 vLLM 长度上限

时再手工改。

### `actor_rollout_ref.rollout.enable_chunked_prefill`

作用：

- 在长 prompt/高并发下提高 prefill 吞吐

适用场景：

- prompt 长
- `max_num_batched_tokens` 足够大

不适用场景：

- 小模型、短上下文
- 或者 `max_num_batched_tokens` 被设得很小

双卡建议：

- `1.5B/7B`、`prompt<=4k`：先从 `False` 开始，稳定优先
- 真正长上下文或大 batch rollout：再试 `True`

### `actor_rollout_ref.rollout.enforce_eager`
### `actor_rollout_ref.rollout.free_cache_engine`

这两个参数一起决定：

- 是否使用 CUDA graph
- 是否让 vLLM 每轮主动 free cache

在你当前 `vllm==0.11.0` 环境里，推荐默认组合是：

```yaml
actor_rollout_ref.rollout.enforce_eager: False
actor_rollout_ref.rollout.free_cache_engine: False
```

理由：

- 新版本 vLLM 更适合走 sleep mode + cudagraph 路径
- 这也是仓库 `README_vllm0.8` 的主推方向

如果你遇到 rollout 不稳定、奇怪 OOM、或者怀疑 cudagraph 问题：

退回保守组合：

```yaml
actor_rollout_ref.rollout.enforce_eager: True
actor_rollout_ref.rollout.free_cache_engine: True
```

代价就是吞吐下降。

### `actor_rollout_ref.rollout.disable_log_stats`

默认是：

```yaml
True
```

如果你在认真调 rollout，建议临时改成：

```yaml
False
```

这样你能看到更多 vLLM 侧统计信息，便于判断：

- KV cache 是否没吃满
- batch 是否太小
- 是否该加 `gpu_memory_utilization`
- 是否该加 `max_num_batched_tokens`

### `actor_rollout_ref.rollout.engine_kwargs.vllm.swap_space`

作用：

- 给 vLLM 留 CPU swap 空间

适用场景：

- 长上下文
- 多响应
- KV 压力大

它不是默认第一选择，但在双卡长上下文时可以作为缓冲项。

---

## 4.4 actor / ref / critic / reward model 参数

### `actor_rollout_ref.model.enable_gradient_checkpointing`
### `critic.model.enable_gradient_checkpointing`

这是训练侧最该优先打开的参数。

收益：

- 显著降激活显存
- 支持更大的 micro batch
- 支持更长上下文

代价：

- 额外重算，单步更慢

但在 RL 训练里，这个 tradeoff 通常是值得的。原因是：

- 你省下来的显存，经常能换来更大的 micro batch
- 更大的 micro batch 往往能把吞吐再挣回来

对双卡 96GB：

- `1.5B/7B` 基本建议一直开
- `14B/32B` 更是默认必开

### `actor_rollout_ref.model.use_remove_padding`
### `critic.model.use_remove_padding`
### `reward_model.model.use_remove_padding`

这是本仓库非常重要的优化项。

作用：

- 去掉 padding token 的无效计算
- 对长度分布离散、max length 设得较大的任务很有效

收益最明显的场景：

- `max_prompt_length` 明显大于实际平均长度
- 多轮 agent 任务，步间长度波动大
- `WebShop` / `AlfWorld` 这种 prompt 长度不均匀任务

建议：

- 文本模型默认开
- 尤其是 PPO/GRPO 训练和长 prompt 任务

### `actor_rollout_ref.actor.use_torch_compile`

作用：

- 对 actor 某些 fused kernel / entropy 计算路径启用编译优化

建议：

- 默认保持 `True`
- 如果你碰到 Triton 编译错误、JIT 不稳定，再设 `False`

这属于“有问题再关”的参数，不建议上来就关。

### `actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu`

作用：

- actor update 时每卡一次 forward/backward 处理多少条序列

调参原则：

- 能大则大
- 直到接近显存边界为止

对吞吐的影响非常直接：

- 太小：GPU 吃不满，梯度累积次数多，慢
- 太大：OOM

对 2x96GB 的经验起点：

- `1.5B`：`16 ~ 32`
- `7B`：`4 ~ 16`
- `14B`：`2 ~ 8`
- `32B`：静态 batch 往往已经不优雅，建议改动态 batch

### `actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`
### `actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`
### `critic.forward_micro_batch_size_per_gpu`

这几个都是“只前向、不反向”的参数。

强烈建议：

- 它们可以比 actor/critic update 的 micro batch 更大

常见经验：

- 设成训练 micro batch 的 `1.5x ~ 2x`

原因：

- 这些阶段不做 backward
- 激活压力小得多

比如：

- actor 用 `8`
- ref / rollout logprob 可以试 `16`

### `actor_rollout_ref.actor.ppo_mini_batch_size`

作用：

- PPO 一个 mini-batch 的全局大小

它更偏“算法/优化稳定性”参数，但也影响性能，因为它决定：

- 一次 update 中要拆多少个 mini-batch
- 每个 mini-batch 再拆多少个 micro-batch

如果 mini batch 太小：

- kernel launch 变碎
- step overhead 增加

如果太大：

- 单次 update 太重
- 需要更多 micro-batch 累积

双卡建议：

- `1.5B/7B`：`64/128/256` 都常见
- 先保证它能被你选定的每卡 micro batch 整除

### `critic.ppo_micro_batch_size_per_gpu`

critic 最后一层输出维度比 actor 小很多，因此常常可以比 actor 再大一点。

但在 agent 任务里，critic 依然会被长上下文拖慢，所以也不要盲目翻倍。

### `reward_model.enable`

如果 reward 是规则函数，就别轻易上模型 RM。

因为一旦启用模型 RM：

- 多一个 FSDP 模型
- 多一段前向
- 多一套显存压力

性能上非常贵。

---

## 4.5 动态 batch 相关参数

这是这个仓库里最值得学会的性能特性之一。

### 什么是动态 batch

不是固定“每卡 8 条序列”，而是固定：

- 每卡最多处理多少 token

这样在变长序列场景里，batch 会自动按 token 数重排和切分，通常能明显提高吞吐并降低 OOM。

对应开关：

```yaml
actor_rollout_ref.actor.use_dynamic_bsz: True
actor_rollout_ref.ref.log_prob_use_dynamic_bsz: True
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz: True
critic.use_dynamic_bsz: True
reward_model.use_dynamic_bsz: True
```

### 最关键的不是 micro batch，而是 token 上限

需要调的是：

- `actor_rollout_ref.actor.ppo_max_token_len_per_gpu`
- `actor_rollout_ref.ref.log_prob_max_token_len_per_gpu`
- `actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu`
- `critic.ppo_max_token_len_per_gpu`
- `critic.forward_max_token_len_per_gpu`
- `reward_model.forward_max_token_len_per_gpu`

### 仓库内部的硬约束

动态切分实现里有个关键断言：

- `max_token_len >= max_seq_len`

也就是：

- 每卡 token 上限至少不能小于单条样本的总长度

否则直接报错。

### 最实用的经验公式

先定义：

```text
seq_total = max_prompt_length + max_response_length
```

建议起点：

- actor：`2 * seq_total`
- ref / rollout logprob：`2 * seq_total` 到 `3 * seq_total`
- critic：`2 * actor`

如果你的长度波动极大，或者 env 分组大：

- actor 直接从 `3 * seq_total` 起步会更顺手

### 什么时候优先开启动态 batch

以下任意一个成立，就建议优先开：

- 长度分布非常不均匀
- `max_prompt_length` 很大但平均 prompt 没那么大
- `env.rollout.n` 较大
- 经常因为少数长样本把静态 micro batch 卡住
- 你想在 14B/32B 上挤显存

### 动态 batch 的隐性收益

它不只是省显存，还会：

- 减少短序列 padding 浪费
- 让每个 micro-batch 的总 token 更接近
- 让 GPU 利用率更平滑

对 agent 任务尤其有用，因为轨迹长度天然差异很大。

---

## 4.6 Ulysses Sequence Parallel（SP）

参数：

- `actor_rollout_ref.actor.ulysses_sequence_parallel_size`
- `actor_rollout_ref.ref.ulysses_sequence_parallel_size`
- `critic.ulysses_sequence_parallel_size`
- `reward_model.ulysses_sequence_parallel_size`

### 先说最重要的约束

只要在 FSDP 路径里用了 SP：

- 必须同时开启 `use_remove_padding=True`

这是 trainer 里显式检查的。

### 在你的双卡机器上，SP 只有两个现实选项

- `1`
- `2`

#### `sp=1`

默认，适合：

- 大多数 1.5B/7B 中短上下文任务
- 吞吐优先

#### `sp=2`

相当于两张卡一起切分同一条序列，适合：

- 极长上下文
- 单条样本太长，静态/动态 batch 都顶不住
- 14B/32B 长序列训练

但代价也很明确：

- DP 维度下降
- 并行粒度变差
- 对短上下文通常不划算

### 双卡上的推荐判断标准

如果满足以下任一条件，可考虑 `sp=2`：

- `max_prompt_length + max_response_length >= 8k`
- 虽然开了 checkpoint + dynamic batch 仍经常 OOM
- 任务收益高度依赖超长上下文

否则建议维持 `sp=1`。

---

## 4.7 offload 相关参数

这是“保命”手段，不是第一优先级的提速手段。

### `actor_rollout_ref.actor.fsdp_config.param_offload`
### `actor_rollout_ref.actor.fsdp_config.optimizer_offload`
### `critic.model.fsdp_config.param_offload`
### `critic.model.fsdp_config.optimizer_offload`
### `actor_rollout_ref.ref.fsdp_config.param_offload`

作用：

- 把参数或优化器状态在不同阶段搬到 CPU

收益：

- 降显存

代价：

- PCIe / CPU 内存传输开销
- step 抖动变大
- 吞吐显著下降

### 推荐使用顺序

1. 先不开 actor/critic offload
2. 只给 ref 开 `param_offload=True`
3. 仍不够时，再考虑 actor `param_offload=True`
4. 最后才考虑 optimizer offload

为什么 ref 优先 offload：

- ref 只做前向
- 对吞吐伤害相对更小

还有一个细节要知道：

- 在 `fsdp_workers.py` 的 FSDP1 路径里，`ref` 构建时默认就走 `CPUOffload(offload_params=True)`
- 这和配置项里的 `ref.fsdp_config.param_offload=True` 不是一回事
- 前者是 FSDP 包装层面的 CPU offload
- 后者是仓库在阶段切换时，额外手工把整模型搬回 CPU

所以你看到 ref 比 actor 更“省显存”，不一定全是因为你手工开了 `param_offload`，还可能是它本来就更激进地用了 CPU offload 策略

对你的 2x96GB，经验建议：

- `1.5B/7B`：actor/critic 一般都不需要 offload
- `14B`：ref 可开 param offload；actor/critic 视上下文长度决定
- `32B`：如果坚持 2 卡做，通常需要至少 ref offload，甚至 actor offload

### `enable_activation_offload`

作用：

- 对 saved tensors 做 CPU offload

特点：

- 比参数 offload 更细粒度
- 能和 gradient checkpointing 叠加

建议：

- 只有在 checkpoint + dynamic batch + remove padding 都不够时再开
- 它主要是“扩大可训练边界”，不是“加速”

---

## 4.8 内核、编译与额外优化项

### `flash_attention_2`

这条优化在仓库主要模型加载路径里已经默认启用了：

- actor
- critic
- SFT
- reward model

也就是说：

- 只要你的环境兼容，一般不需要你额外手动开
- 你真正需要关注的是版本兼容，而不是再去找一个 `enable_flash_attn=True`

### `actor_rollout_ref.model.use_fused_kernels`

作用：

- 让 actor 某些 logprob / entropy 相关路径走 fused kernels

潜在收益：

- actor 前向更快

潜在风险：

- Triton/JIT 编译问题
- 某些环境下首次编译很慢

建议：

- 先用默认 `False`
- 先把整体流程与 batch 策略调顺
- 确认真正瓶颈在 actor 前向后，再把它当“第二阶段优化项”

### `actor_rollout_ref.model.use_liger`

仓库在 RL worker 和 SFT trainer 里都接了 Liger Kernel。

适合场景：

- 你环境里已经安装 `liger-kernel`
- 主要做训练侧吞吐优化

建议：

- 文本模型可以尝试
- 先在小规模实验上验证稳定性
- 对 SFT 的价值通常比 RL 更直接

### `actor_rollout_ref.actor.use_torch_compile`

再次强调它的使用姿势：

- 默认保留 `True`
- 只在你确认编译路径出错时才关

因为在这份仓库里，它不只是“一个小优化开关”，而是部分性能路径的默认前提。

---

## 4.9 系统级参数：CPU、Ray、dataloader

### `data.dataloader_num_workers`

`RayPPOTrainer` 创建 dataloader 时默认会取：

- `data.dataloader_num_workers`
- 未设置时默认 `8`

什么时候应该调它：

- GPU 空闲、但 step 启动慢
- parquet 解码或 tokenizer 前处理偏重
- 机器 CPU 很多

什么时候别盲目调大：

- 机器 CPU 本来就紧张
- 环境 worker 已经吃掉大量 CPU

双卡 agent 训练里，CPU 往往同时被以下几部分争用：

- dataloader
- Ray driver
- env workers
- vLLM / PyTorch 辅助线程

所以这不是“越大越好”的参数。

### `ray_init.num_cpus`

如果你在受限环境里跑，例如：

- 容器 CPU quota 有限制
- 调度系统只给了一部分核

建议显式设置，避免 Ray 误判“可以用所有 CPU”，进而导致抢占过度。

### `trainer.n_gpus_per_node`
### `trainer.nnodes`

这两个除了资源声明，还会被 trainer 用来校验：

- 全局 batch 整除关系

对你当前机器，应明确固定：

```yaml
trainer.n_gpus_per_node: 2
trainer.nnodes: 1
```

不要偷懒用默认值 `8`，否则很多 batch 检查和性能判断都会偏掉。

---

## 5. 这个仓库里几个经常被忽略、但真的影响吞吐的实现细节

## 5.1 `trainer.balance_batch=True` 基本应该保留

trainer 会在 driver 侧按序列总长度重排 batch，让每个 DP rank 分到更均衡的 token 数。

作用：

- 降低 rank 之间的 straggler
- 让 update 阶段更平衡

这是非常值得保留的默认项。

如果关掉，常见表现是：

- 某一张卡明显更慢
- step 时间波动变大

## 5.2 `adjust_batch()` 会为了整除关系复制或删除样本

`agent_system/multi_turn_rollout/utils.py` 里，最终 batch 会被强制对齐到：

- rollout logprob divisor
- ref logprob divisor
- actor divisor

三者的最小公倍数。

这意味着如果你的参数组合很别扭，就会：

- 复制样本
- 或删除样本

带来两类问题：

- 额外的数据搬运和处理
- 有效样本利用率变差

最实用的建议是：

- 尽量让 `data.train_batch_size * env.rollout.n`
- 本身就能被 `2 * ppo_micro_batch_size_per_gpu`
- `2 * log_prob_micro_batch_size_per_gpu`
- 以及相关 mini-batch 约束较好整除

你是双卡，因此经验上优先选：

- `env.rollout.n` 为 `2/4/8`
- micro batch 为 `2/4/8/16`
- `train_batch_size` 为这些值的公共倍数

## 5.3 `data.gen_batch_size` 和 `data.train_batch_size` 不一定该相等

如果不开动态采样：

- 通常相等就行

如果开了 `filter_groups.enable=True`：

- `gen_batch_size` 往往应该大于 `train_batch_size`

因为系统要先多采一轮，再过滤掉低信息组。

常见思路：

- `gen_batch_size = 2x ~ 3x train_batch_size`

但这会增加 rollout 瞬时压力，所以双卡上不要一上来拉太大。

## 5.4 PPO 是否需要 critic，对性能影响巨大

如果你用：

- `algorithm.adv_estimator=gae`

那就是 PPO，需要 critic。

如果你用：

- `grpo`
- `gigpo`
- `rloo`
- `reinforce_plus_plus`

这类 critic-free 方法，训练链路会直接少掉：

- `values`
- `update_critic`
- 一整套 critic 显存

对 2 卡机器，这是能否上更大模型的重要分界线。

简单说：

- 双卡上若你想追求大模型或长上下文，优先考虑 critic-free 算法
- 若你要 PPO，请接受 critic 的额外成本

## 5.5 KL 控制也会增加前向成本

只要打开其中之一：

- `algorithm.use_kl_in_reward=True`
- `actor_rollout_ref.actor.use_kl_loss=True`

系统就需要 reference policy。

这意味着：

- 多一个 ref 前向
- 多一份显存与时延

因此：

- 如果你正在做极致性能调试，先关 KL，测清楚无 ref 的吞吐上限
- 再根据训练稳定性需要把 KL 加回来

---

## 6. 面向双卡 192GB 的推荐调参顺序

下面这套顺序是我认为最稳、也最符合这个仓库实现逻辑的。

## 阶段 A：先做“高概率正确”的基础配置

建议默认先设：

```yaml
actor_rollout_ref.model.enable_gradient_checkpointing: True
critic.model.enable_gradient_checkpointing: True
actor_rollout_ref.model.use_remove_padding: True
critic.model.use_remove_padding: True
trainer.balance_batch: True
actor_rollout_ref.rollout.enforce_eager: False
actor_rollout_ref.rollout.free_cache_engine: False
```

如果是文本任务，再加：

```yaml
actor_rollout_ref.actor.use_dynamic_bsz: True
actor_rollout_ref.ref.log_prob_use_dynamic_bsz: True
actor_rollout_ref.rollout.log_prob_use_dynamic_bsz: True
critic.use_dynamic_bsz: True
```

## 阶段 B：先把 rollout 调顺

顺序如下：

1. 先固定训练侧 batch，不动 actor/critic
2. 只调 rollout：
   - `tensor_model_parallel_size`
   - `gpu_memory_utilization`
   - `max_num_batched_tokens`
   - `enable_chunked_prefill`
3. 找到 `gen` 最快且不 OOM 的组合

对 2 卡：

- 7B 以下先试 `TP=1`
- 大模型或长上下文先试 `TP=2`

## 阶段 C：再调 ref / old_log_prob / critic 前向

优先调：

- `rollout.log_prob_micro_batch_size_per_gpu`
- `ref.log_prob_micro_batch_size_per_gpu`
- `critic.forward_micro_batch_size_per_gpu`

原则：

- 它们通常可以比训练 micro batch 更大

## 阶段 D：最后调 actor / critic update

如果 update 慢，就按这个顺序动：

1. 增大 `ppo_micro_batch_size_per_gpu`
2. 如果静态 batch 不好调，切动态 batch
3. 增大 `ppo_max_token_len_per_gpu`
4. 不够再上 offload

## 阶段 E：只有在长上下文或大模型边界上，再考虑 SP/offload

顺序建议：

1. `use_remove_padding=True`
2. `use_dynamic_bsz=True`
3. `gradient_checkpointing=True`
4. `sp=2`
5. `ref param_offload=True`
6. `actor/critic param_offload=True`
7. `optimizer_offload=True`
8. `activation_offload=True`

---

## 7. 双卡 192GB 的分档推荐

下面是工程上更有用的结论，不是仓库官方硬性边界。我会把“推断”与“代码事实”分开。

## 7.1 1.5B / 7B：最佳甜区

这类模型在你机器上通常可以做到：

- PPO 可跑
- critic 可开
- remove padding + checkpointing + dynamic batch 都能上
- rollout 也有相当大的余量

推荐策略：

- 训练优先：`TP=1`
- 中长上下文：若 rollout OOM，再切 `TP=2`
- actor/critic offload 通常不用
- ref 可以不开 offload，也可以开 `param_offload=True` 稍微省点显存

## 7.2 14B：可作为双卡高性价比上限

14B 在 2x96GB 上通常依然是可操作区间，但建议你：

- 优先用 critic-free 算法
- PPO 时减少长度上限或减小 batch
- rollout 常从 `TP=2` 开始
- ref 建议开 `param_offload=True`

## 7.3 32B：能做，但要接受明显工程约束

如果要在 2x96GB 上做 32B：

- 更推荐 GRPO / GiGPO / DAPO 这类 critic-free 路径
- 动态 batch 基本必开
- checkpointing 必开
- ref offload 基本必开
- rollout 大概率 `TP=2`
- 如果上下文很长，`sp=2` 也可能需要

如果坚持 PPO + critic：

- 很可能需要明显降低 `max_prompt_length` / `max_response_length`
- 并配合 actor/critic offload
- 吞吐不会好看

## 7.4 70B 及以上：不建议把双卡当主力训练平台

不是说绝对不能试，而是：

- 即便勉强拼起来，可调空间也太小
- 训练稳定性、吞吐、开发效率都会很差

双卡 96GB 更适合：

- 1.5B / 7B 做高吞吐算法实验
- 14B 做中高强度实验
- 32B 做偏研究型、小 batch、critic-free 尝试

---

## 8. 我最推荐的几套起步模板

下面给的是“起步模板”，不是最终最优值。

## 8.1 模板一：双卡 7B，PPO/GAE，稳定高效版

适合：

- AlfWorld / WebShop
- `max_prompt_length` 在 `2k~4k`
- `max_response_length=512`

```yaml
trainer:
  nnodes: 1
  n_gpus_per_node: 2
  balance_batch: True

data:
  train_batch_size: 64
  max_prompt_length: 2048
  max_response_length: 512
  filter_overlong_prompts: True

actor_rollout_ref:
  model:
    use_remove_padding: True
    enable_gradient_checkpointing: True
  actor:
    strategy: fsdp
    ppo_mini_batch_size: 64
    ppo_micro_batch_size_per_gpu: 8
    use_dynamic_bsz: False
    fsdp_config:
      param_offload: False
      optimizer_offload: False
  ref:
    log_prob_micro_batch_size_per_gpu: 16
    fsdp_config:
      param_offload: True
  rollout:
    name: vllm
    tensor_model_parallel_size: 1
    gpu_memory_utilization: 0.65
    log_prob_micro_batch_size_per_gpu: 16
    enforce_eager: False
    free_cache_engine: False
    enable_chunked_prefill: False
    max_num_batched_tokens: 8192

critic:
  model:
    use_remove_padding: True
    enable_gradient_checkpointing: True
    fsdp_config:
      param_offload: False
      optimizer_offload: False
  ppo_micro_batch_size_per_gpu: 8
  forward_micro_batch_size_per_gpu: 16
```

## 8.2 模板二：双卡 7B，长上下文优先版

适合：

- `max_prompt_length=4096`
- 或历史较长
- 或 episode 较长

```yaml
actor_rollout_ref:
  model:
    use_remove_padding: True
    enable_gradient_checkpointing: True
  actor:
    use_dynamic_bsz: True
    ppo_max_token_len_per_gpu: 12288
    ppo_mini_batch_size: 64
  ref:
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: 16384
    fsdp_config:
      param_offload: True
  rollout:
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.7
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: 16384
    enforce_eager: False
    free_cache_engine: False
    max_num_batched_tokens: 12288

critic:
  use_dynamic_bsz: True
  ppo_max_token_len_per_gpu: 24576
  forward_max_token_len_per_gpu: 24576
```

## 8.3 模板三：双卡 14B/32B，critic-free，显存优先版

适合：

- GRPO / GiGPO / DAPO
- 更大模型

```yaml
actor_rollout_ref:
  model:
    use_remove_padding: True
    enable_gradient_checkpointing: True
  actor:
    strategy: fsdp
    use_dynamic_bsz: True
    ppo_mini_batch_size: 32
    ppo_max_token_len_per_gpu: 8192
    fsdp_config:
      param_offload: True
      optimizer_offload: False
    ulysses_sequence_parallel_size: 1
  ref:
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: 12288
    fsdp_config:
      param_offload: True
    ulysses_sequence_parallel_size: 1
  rollout:
    name: vllm
    tensor_model_parallel_size: 2
    gpu_memory_utilization: 0.7
    log_prob_use_dynamic_bsz: True
    log_prob_max_token_len_per_gpu: 12288
    enforce_eager: False
    free_cache_engine: False

critic:
  # critic-free 算法不使用
```

如果这套还顶不住，再尝试：

```yaml
actor_rollout_ref.actor.ulysses_sequence_parallel_size: 2
actor_rollout_ref.ref.ulysses_sequence_parallel_size: 2
```

同时保持：

```yaml
actor_rollout_ref.model.use_remove_padding: True
```

---

## 9. 看到什么现象，就该调什么

## 9.1 `timing_s/gen` 明显最大

优先检查：

- `env` 是否在拖慢
- `rollout TP` 是否不合适
- `gpu_memory_utilization` 是否过低
- `max_num_batched_tokens` 是否太小

调参顺序：

1. 确认 CPU 够用：`env.resources_per_worker.num_cpus`
2. 7B 以下先试 `rollout.tensor_model_parallel_size=1`
3. 增大 `gpu_memory_utilization`
4. 增大 `max_num_batched_tokens`
5. 必要时开 `enable_chunked_prefill`

## 9.2 `timing_s/old_log_prob` 或 `timing_s/ref` 很大

优先检查：

- 是否 still 用静态 micro batch
- `log_prob_micro_batch_size_per_gpu` 是否太小
- ref 是否没开 remove padding

调参顺序：

1. 开 `use_remove_padding`
2. 增大 `log_prob_micro_batch_size_per_gpu`
3. 改动态 batch
4. ref 开 `param_offload=True` 仅在 OOM 时使用

## 9.3 `timing_s/update_actor` 或 `timing_s/update_critic` 很大

通常说明：

- 每卡 micro batch 太小
- 梯度累积轮数太多
- 或长度太长

调参顺序：

1. 增大 `ppo_micro_batch_size_per_gpu`
2. 开 `enable_gradient_checkpointing`
3. 开 `use_remove_padding`
4. 改动态 batch
5. 只在必要时上 offload

## 9.4 OOM 出现在 rollout

优先尝试：

1. `rollout.tensor_model_parallel_size=2`
2. 降 `rollout.gpu_memory_utilization`
3. 降 `rollout.max_num_batched_tokens`
4. 缩短 `max_prompt_length` / `max_response_length`
5. 必要时 `enforce_eager=True`

## 9.5 OOM 出现在 actor/critic update

优先尝试：

1. 开 checkpointing
2. 开 remove padding
3. 降 `ppo_micro_batch_size_per_gpu`
4. 改 `use_dynamic_bsz=True`
5. 降 `ppo_max_token_len_per_gpu`
6. 开 `param_offload`
7. 开 `optimizer_offload`

## 9.6 `prompt_length/clip_ratio` 很高

说明：

- prompt 经常撞上 `max_prompt_length`

这不是性能优化参数本身，但会反过来影响性能和效果。

你应该：

- 重新评估真实长度分布
- 适当加大 `max_prompt_length`
- 或减小 `history_length`

## 9.7 `response_length/clip_ratio` 很高

说明：

- 响应经常撞上 `max_response_length`

你应该：

- 判断是不是任务真的需要更长 response
- 如果需要，再增加 `max_response_length`
- 否则可能只是策略在发散

---

## 10. 新版本 vLLM / Torch 环境下的特别建议

基于你当前本地环境：

- `torch 2.8.0+cu128`
- `vllm 0.11.0`

我的建议是：

### 10.1 默认不要再设 `VLLM_ATTENTION_BACKEND=XFORMERS`

这个变量主要是给老版本 vLLM bug 兜底用的。

你当前版本默认不应该依赖它。

只有当你真的遇到 rollout 侧底层兼容问题，再把它当试验项，而不是默认项。

### 10.2 优先走新版本推荐的 rollout 组合

```yaml
actor_rollout_ref.rollout.enforce_eager: False
actor_rollout_ref.rollout.free_cache_engine: False
```

### 10.3 如果开启 fused kernels / torch compile 出现 Triton 问题

第一反应不是全盘回退，而是：

1. 先保留 `use_torch_compile=True`
2. 只在错误复现后，把 `actor_rollout_ref.actor.use_torch_compile=False`

---

## 11. 训练前必做检查清单

建议每次正式训练前，按这个顺序看一遍：

1. 长度配置是否合理
   - `max_prompt_length`
   - `max_response_length`
   - `history_length`
   - `max_steps`

2. 算法是否真的需要 critic
   - PPO 需要
   - GRPO/GiGPO/RLOO 不需要

3. rollout TP 是否合理
   - 小模型先 `1`
   - 大模型/长上下文先 `2`

4. 是否打开以下三件套
   - `enable_gradient_checkpointing=True`
   - `use_remove_padding=True`
   - `balance_batch=True`

5. 是否应改用动态 batch
   - 长度分布大时，优先开

6. 是否不小心把 offload 开太多
   - 能不用就不用

7. batch 整除是否合理
   - `train_batch_size * env.rollout.n`
   - 尽量和 micro batch 体系匹配

8. rollout 调试时是否关掉了日志
   - 调优阶段可设 `disable_log_stats=False`

---

## 12. 如果让我给你的机器一个“默认起步方案”

如果你现在就要在这台双卡机器上开始调，我会这样起步：

### 场景 A：1.5B / 7B，常规 agent 训练

- `strategy=fsdp`
- `rollout.tensor_model_parallel_size=1`
- `use_remove_padding=True`
- `enable_gradient_checkpointing=True`
- `use_dynamic_bsz=True`
- `actor.ppo_max_token_len_per_gpu = 2~3 * (max_prompt_length + max_response_length)`
- `ref/rollout logprob` token 上限略大于 actor
- `critic` token 上限为 actor 的 2 倍
- `ref.fsdp_config.param_offload=True`
- actor/critic offload 先关

### 场景 B：7B 长上下文 / 14B

- `rollout.tensor_model_parallel_size=2`
- `gpu_memory_utilization=0.65~0.75`
- `use_dynamic_bsz=True`
- `use_remove_padding=True`
- `gradient_checkpointing=True`
- ref 开 `param_offload=True`

### 场景 C：32B

- 优先不用 PPO
- 优先 critic-free
- rollout 直接 `TP=2`
- 动态 batch 必开
- 参考模型 offload 必开
- 必要时 actor 也开 param offload
- 如果长度特别长，再考虑 `SP=2`

---

## 13. 重点源码定位，便于你后续继续深挖

如果你之后要自己沿源码继续读，建议按这个顺序看：

1. `verl/trainer/config/ppo_trainer.yaml`
   - 全量参数入口

2. `verl/trainer/main_ppo.py`
   - 训练入口、资源池、角色初始化

3. `verl/trainer/ppo/ray_trainer.py`
   - 单步训练流程、validate、batch 平衡、指标记录

4. `verl/workers/fsdp_workers.py`
   - actor/ref/critic/reward 的 FSDP 构建与 offload 行为

5. `verl/workers/actor/dp_actor.py`
   - actor logprob / PPO update 细节

6. `verl/workers/critic/dp_critic.py`
   - critic values/update 细节

7. `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
   - rollout/vLLM 参数真正怎么落地

8. `verl/workers/sharding_manager/fsdp_vllm.py`
   - actor <-> vLLM 权重同步和 sleep/wake 行为

9. `agent_system/multi_turn_rollout/rollout_loop.py`
   - agent-environment loop、动态采样、group filtering

10. `agent_system/multi_turn_rollout/utils.py`
    - `adjust_batch()` 的 batch 对齐逻辑

---

## 14. 附录：如果你后续也想看 SFT 路径，该怎么类比

虽然你这次重点是 RL/FSDP，但这个仓库还有一条独立的 SFT 路径：

- 入口：`verl.trainer.fsdp_sft_trainer`
- 配置：`verl/trainer/config/sft_trainer.yaml`

它和 RL 路径的几个关键区别是：

1. SFT 的 `fsdp` 真的是 `FULL_SHARD`
2. SFT 默认没有 rollout / ref / critic 这几段
3. SFT 的 batch 更简单：
   - `data.train_batch_size`
   - `data.micro_batch_size_per_gpu`
4. SFT 同样支持：
   - `use_remove_padding`
   - `ulysses_sequence_parallel_size`
   - `use_liger`
   - `gradient_checkpointing`

如果你以后要在这台双卡机器上跑 SFT，最常见的高收益组合仍然是：

- `use_remove_padding=True`
- `enable_gradient_checkpointing=True`
- 长上下文时 `ulysses_sequence_parallel_size=2`
- 想进一步挤吞吐时尝试 `use_liger=True`

所以可以把这份 RL 指南里的大部分“长度 / remove padding / dynamic token 视角 / SP 视角”迁移过去，但要记住：

- SFT 没有 rollout/vLLM 瓶颈
- 它的核心瓶颈更单纯地集中在训练前向/反向本身

---

## 15. 最后的经验判断

对这个仓库来说，“训练性能优化”本质上不是单点调参，而是做下面这四件事：

1. 控制长度
2. 提高有效 token 密度
3. 让 rollout 和训练两侧都吃满 GPU
4. 只在必要时用 offload 换可训练性

如果你只能记住三条：

1. `use_remove_padding + gradient_checkpointing + dynamic_bsz` 是第一优先级。
2. 双卡上 rollout 的 `TP=1` 还是 `TP=2`，往往比你调一堆小参数更影响吞吐。
3. 想在双卡上做更大模型时，优先换成 critic-free 算法，而不是一上来堆 offload。
