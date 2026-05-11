# Qwen3.5 适配方案

本文面向本仓库的 agentic RL 训练路径，目标是支持通过类似
`actor_rollout_ref.model.path=Qwen/Qwen3.5-4B` 的参数直接启动 FSDP + vLLM/SGLang 训练。Megatron 路径不作为首要目标。

## 0. 结论摘要

Qwen3.5 不能按现有 Qwen3 文本模型或 Qwen3-VL 分支简单兼容。

官方 `Qwen/Qwen3.5-4B` 是 `Image-Text-to-Text` 模型，`config.json` 中的关键字段是：

- `model_type: "qwen3_5"`
- `architectures: ["Qwen3_5ForConditionalGeneration"]`
- `text_config.model_type: "qwen3_5_text"`
- `vision_config.model_type: "qwen3_5"`
- `image_token_id/video_token_id/vision_start_token_id/vision_end_token_id`
- `text_config.layer_types` 同时包含 `linear_attention` 与 `full_attention`
- `text_config.rope_parameters` 使用 `mrope_interleaved` 和 `mrope_section`

因此首版适配建议分两阶段：

1. **先打通默认训练**：FSDP actor/ref、vLLM/SGLang rollout、processor/dataset、checkpoint、FLOPs 识别。默认关闭 `use_remove_padding`、`use_fused_kernels`、`ulysses_sequence_parallel_size>1`。
2. **再做性能特性**：为 Qwen3.5 的 Gated DeltaNet/linear attention 验证 remove-padding、Ulysses、fused PPO kernel 和 text-only rollout 优化。

## 1. 环境依赖层面

### 1.1 本节依据和当前真实环境

本仓库实际使用的主环境不是按根目录 `requirements.txt` 或 `setup.py` 直接安装出来的，而是按
`study_guide/setup_env.md` 针对 Blackwell 机器调整过的路线配置的。因此 Qwen3.5 环境不能简单写成
“在 `verl-agent-bw` 中 `pip install -U transformers vllm`”。

`study_guide/setup_env.md` 的主线结论是：

- 主训练环境使用 `Python 3.12`。
- Blackwell / `sm_120` 机器优先使用 `torch 2.8.0 + cu128`，不要回退到仓库旧文档里的 `torch 2.6 + cu124`。
- vLLM 主线先用 `vllm 0.11.0`，不要默认使用旧脚本中的 `vllm 0.8.x`。
- `flash-attn` 对训练路径基本是强依赖，但要单独按 Blackwell 编译/安装。
- WebShop、Retriever、AppWorld 等环境继续拆分，不要塞进主训练环境。
- 默认不要继承旧示例脚本里的 `export VLLM_ATTENTION_BACKEND=XFORMERS`。

我在 `2026-05-11` 检查到当前 `verl-agent-bw` 的真实状态如下：

- `Python 3.12.13`
- `torch 2.8.0+cu128`，`torchvision 0.23.0+cu128`，`torchaudio 2.8.0+cu128`
- GPU compute capability 检测为 `(12, 0)`
- `transformers 4.57.3`
- `vllm 0.11.0`
- `flash_attn 2.8.3`
- `qwen-vl-utils 0.0.14`，并安装了 `decord`
- `ray 2.50.0`
- `tensordict 0.10.0`
- `peft 0.18.1`
- `xformers 0.0.32.post1`
- 已有 agent 环境依赖：`gym-cards`、`gym-sokoban`、`alfworld`、`textworld`、`jericho`
- 未安装 `sglang`

```bash
conda run -n verl-agent-bw python -c '
import importlib.util
print("qwen3_5", bool(importlib.util.find_spec("transformers.models.qwen3_5")))
print("qwen3_vl", bool(importlib.util.find_spec("transformers.models.qwen3_vl")))
'
```

结果是：

```text
qwen3_5 False
qwen3_vl True
```

所以当前主环境已经适合 Blackwell 上的 Qwen3-VL 训练基线，但不满足 Qwen3.5。Qwen3.5 环境应该新建，不应原地升级
`verl-agent-bw`。

### 1.2 现有安装约束的问题

根目录里的依赖文件仍然会影响新环境：

- `requirements.txt` 固定 `transformers==4.51.1`，不能用于 Qwen3.5。
- `setup.py` 的基础依赖写了 `transformers<=4.57.3`，会把 Qwen3.5 所需 Transformers 版本压回旧版本。
- `setup.py` 的 vLLM extra 写了 `vllm>=0.8.5,<=0.11.0`，但官方 vLLM Qwen3.5 recipe 使用的是最新 vLLM 安装方式，并且对 Blackwell Docker 明确建议 `cu130-nightly` 镜像；`0.11.0` 不能作为 Qwen3.5 支持保证。
- `setup.py` 的 SGLang extra 固定 `sglang[srt,openai]==0.5.5`，而当前 `verl-agent-bw` 实际未安装 SGLang；Qwen3.5 的 tool calling 需要确认是否有 `qwen3_coder` parser。

因此新环境有两条可选路径：

1. **正式适配路径**：先修改 `setup.py` / extras，把 Qwen3.5 依赖约束写对，再正常 `pip install -e ".[qwen35,...]"`。
2. **过渡验证路径**：在新环境里先手工装依赖，安装本仓库时用 `pip install -e . --no-deps`，避免旧约束把 Transformers/vLLM 降级。

正式适配后建议把 `setup.py` 拆出新的 extras，而不是全局升级所有用户：

```python
QWEN35_REQUIRES = [
    # 不建议只靠版本号判断；CI 应同时做 capability check。
    "transformers>=<first_release_with_qwen3_5>",
    "qwen-vl-utils[decord]>=0.0.14",
    "torchcodec",
    # Qwen3.5 Gated DeltaNet 的 fast path 依赖；缺失时 Transformers 可回退 torch 实现，但训练会明显慢。
    "flash-linear-attention",
    "causal-conv1d",
    "flash-attn>=2.8.3",
]

VLLM_QWEN35_REQUIRES = [
    # 先以官方 recipe 的最新 vLLM / nightly 能力为准，稳定后再收敛成明确版本下界。
    "vllm>=<first_release_with_qwen3_5>",
]

SGLANG_QWEN35_REQUIRES = [
    "sglang[srt,openai]>=<first_release_with_qwen3_5>",
]
```

### 1.3 推荐的新环境方案

建议新环境命名为 `verl-agent-qwen35-bw`，不要覆盖 `verl-agent-bw`：

```bash
conda create -n verl-agent-qwen35-bw python=3.12 -y
conda activate verl-agent-qwen35-bw

python -m pip install -U pip setuptools wheel packaging ninja uv
```

基础 CUDA/PyTorch 层先沿用 `study_guide/setup_env.md` 已验证的 Blackwell 组合：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装本仓库基础依赖时，不要用 `requirements.txt`。如果 `setup.py` 还没有改掉 Qwen3.5 约束，过渡期建议显式安装基础依赖，然后以 `--no-deps` 安装本仓库：

```bash
pip install \
  accelerate codetiming datasets dill hydra-core numpy pandas peft \
  "pyarrow>=19.0.0" pybind11 pylatexenc \
  "ray[default]>=2.41.0,<=2.50.0" \
  torchdata "tensordict>=0.8.0,<=0.10.0,!=0.9.0" \
  wandb "packaging>=20.0" "qwen-vl-utils[decord]>=0.0.14" \
  pillow torchcodec

cd /home/zhangwj/science/verl-agent
pip install -e . --no-deps
```

然后安装 Qwen3.5 所需 Transformers。这里不要把 `transformers>=某个猜测版本` 当作唯一标准；以 capability check 为准。若 PyPI 最新版已经包含
`qwen3_5`，用 PyPI；否则用 Hugging Face Transformers main：

```bash
# 首选：已发布 PyPI 版本，具体版本以 capability check 通过为准。
pip install -U transformers

# 如果 PyPI 版仍没有 qwen3_5，再用 main。
pip install -U "transformers @ git+https://github.com/huggingface/transformers.git"
```

能力检查必须通过：

```bash
python - <<'PY'
import importlib.util
import transformers

print("transformers:", transformers.__version__)
assert importlib.util.find_spec("transformers.models.qwen3_5"), "missing qwen3_5"
assert importlib.util.find_spec("transformers.models.qwen3_5_moe"), "missing qwen3_5_moe"

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=False)
print("model_type:", cfg.model_type)
assert cfg.model_type == "qwen3_5"
print("AutoModelForImageTextToText:", AutoModelForImageTextToText)
print("AutoProcessor:", AutoProcessor)
PY
```

`flash-attn` 继续按 `study_guide/setup_env.md` 的 Blackwell 做法单独安装：

```bash
MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation
```

如果日志没有正确包含 `sm_120`，再指定架构重装：

```bash
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 \
  pip install flash-attn==2.8.3 --no-build-isolation --force-reinstall
```

Qwen3.5 的 Gated DeltaNet fast path 在 Transformers 源码中会探测 `causal-conv1d` 和 `flash-linear-attention`。它们缺失时有 torch fallback，但训练速度会受到影响。建议把它们作为 Qwen3.5 环境的可选加速依赖安装，并在失败时先不阻塞功能验证：

```bash
pip install -U causal-conv1d --no-build-isolation
pip install -U flash-linear-attention --no-build-isolation
```

### 1.4 vLLM / SGLang 选择

首版训练优先走 vLLM，因为仓库已有 FSDP-vLLM 权重同步路径。当前 `verl-agent-bw` 的 `vllm 0.11.0` 是 Qwen3-VL 时代的可用基线，但不能推导出 Qwen3.5 可用。官方 vLLM recipe 对 Qwen3.5/Qwen3.6 的 NVIDIA pip 安装写法是使用最新 vLLM：

```bash
uv pip install --python "$CONDA_PREFIX/bin/python" -U vllm --torch-backend=auto
```

对本仓库的新环境，建议这样处理：

```bash
# 在已激活的 verl-agent-qwen35-bw 中执行。
uv pip install --python "$CONDA_PREFIX/bin/python" -U vllm --torch-backend=auto
```

`--python "$CONDA_PREFIX/bin/python"` 是为了让 `uv` 明确写入当前 conda 环境。安装后必须重新检查 torch/vLLM/flash-attn 组合，因为 vLLM 可能根据自己的 wheel 约束调整 torch：

```bash
python - <<'PY'
import torch, vllm
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("vllm:", vllm.__version__)
print("cuda available:", torch.cuda.is_available())
print("cc:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
PY
```

如果这一步把 torch 从 `2.8.0+cu128` 升/降到了另一个 CUDA 组合，需要重新安装或重编译 `flash-attn`，并重新跑环境 smoke test。如果 vLLM 安装过程又把 `transformers` 改回了不含 `qwen3_5` 的版本，重新安装 Qwen3.5 可用的 Transformers，并以 capability check 为准。Blackwell 上如果遇到 vLLM wheel 与 CUDA 组合问题，优先参考官方 recipe 的 Blackwell `cu130-nightly` 路线，而不是回退到仓库旧脚本的 `vllm 0.8.x`。

SGLang 不作为首版必需项。若要验证 SGLang，建议在同一个新环境稳定后再安装最新版，或者单独开 `verl-agent-qwen35-sglang-bw`，不要使用 `setup.py` 当前固定的 `sglang==0.5.5`：

```bash
pip install -U "sglang[srt,openai]"
```

安装后检查：

```bash
python - <<'PY'
import sglang
print("sglang:", getattr(sglang, "__version__", "unknown"))
PY
```

并在 rollout smoke test 中确认 `qwen3_coder` tool-call parser 或等价 parser 可用。官方 vLLM recipe 明确把 Qwen3.5 tool calling 配置为
`--enable-auto-tool-choice --tool-call-parser qwen3_coder`，SGLang 侧也应按同等能力验收。

### 1.5 agent 环境依赖迁移

新环境不能只装模型栈，还要复制当前 `verl-agent-bw` 已具备的 agentic RL 环境能力：

- `gym-cards`：当前是从仓库子目录以 editable 形式安装，路径对应 `agent_system/environments/env_package/gym_cards/gym-cards`。
- `gym-sokoban==0.0.6`
- `alfworld==0.4.2`
- `textworld==1.7.0`
- `jericho==3.3.1`
- `stable-baselines3==2.6.0`

建议在新环境里补齐：

```bash
pip install gym==0.26.2 gymnasium==0.29.1 gym-sokoban==0.0.6
pip install alfworld==0.4.2 textworld==1.7.0 jericho==3.3.1 stable-baselines3==2.6.0
pip install -e agent_system/environments/env_package/gym_cards/gym-cards
```

WebShop、Search Retriever、AppWorld 继续按 `study_guide/setup_env.md` 拆成独立环境；不要为了 Qwen3.5 把它们合进 `verl-agent-qwen35-bw`。

### 1.6 环境验收脚本

新环境建好后，先跑纯环境验收，不要直接跑训练：

```bash
conda activate verl-agent-qwen35-bw

python - <<'PY'
import importlib.util
import torch

checks = [
    "verl",
    "agent_system",
    "transformers.models.qwen3_5",
    "transformers.models.qwen3_5_moe",
    "flash_attn",
    "qwen_vl_utils",
    "vllm",
    "gym",
    "gym_sokoban",
    "alfworld",
    "textworld",
    "jericho",
]

for name in checks:
    print(name, bool(importlib.util.find_spec(name)))

print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("cc:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None)
PY
```

再跑 Qwen3.5 processor/model 类验收：

```bash
python - <<'PY'
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

model_id = "Qwen/Qwen3.5-4B"
cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=False)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=False)

print("model_type:", cfg.model_type)
print("architectures:", cfg.architectures)
print("processor:", processor.__class__.__name__)
print("image_token_id:", getattr(cfg, "image_token_id", None))
print("video_token_id:", getattr(cfg, "video_token_id", None))
print("AutoModel class ok:", AutoModelForImageTextToText)

assert cfg.model_type == "qwen3_5"
assert "Qwen3_5ForConditionalGeneration" in cfg.architectures
PY
```

如果只想避免下载 4B 权重，这一步不要调用 `from_pretrained` 加载模型权重；等代码适配完成后再用 tiny/random 权重或受控 smoke test 验证 forward。

### 1.7 运行配置建议

首版训练脚本应显式设置：

```bash
+data.apply_chat_template_kwargs.enable_thinking=False \
actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
actor_rollout_ref.model.trust_remote_code=False \
actor_rollout_ref.model.use_remove_padding=False \
actor_rollout_ref.model.use_fused_kernels=False \
actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
actor_rollout_ref.rollout.max_model_len=8192 \
actor_rollout_ref.rollout.enable_chunked_prefill=False
```

说明：

- Qwen3.5 默认会输出 `<think>...</think>`；agentic 环境通常已有动作格式约束，建议先用 `enable_thinking=False` 避免过长思维链破坏 action parser。
- 官方推荐 262K 上下文，但 RL smoke test 不应一开始使用超长上下文；先用 8K/16K 验证端到端。
- `use_remove_padding=True` 对 Qwen3.5 风险较高，因为 Qwen3.5 有 linear attention/Gated DeltaNet；当前仓库 remove-padding 会把多个样本打包成单条序列并传 `attention_mask=None`，需要额外验证不会跨样本泄漏线性注意力状态。
- 旧示例脚本中的 `export VLLM_ATTENTION_BACKEND=XFORMERS` 首版不要启用；`study_guide/setup_env.md` 已建议在 `vllm>=0.8` 后默认去掉它。

## 2. 仓库代码层面

### 2.1 模型类型和 AutoModel 加载

修改位置：

- `verl/workers/fsdp_workers.py`
- `verl/utils/checkpoint/fsdp_checkpoint_manager.py`
- 可选新增：`verl/models/transformers/model_types.py`

新增模型族常量：

```python
QWEN35_MODEL_TYPES = {"qwen3_5", "qwen3_5_moe"}
QWEN35_TEXT_MODEL_TYPES = {"qwen3_5_text", "qwen3_5_moe_text"}
QWEN35_ARCHS = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
}
```

在 `ActorRolloutRefWorker._build_model_optimizer()` 中，当前逻辑只在
`type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys()` 时使用 `AutoModelForVision2Seq`，否则退回 `AutoModelForCausalLM`。Qwen3.5 官方示例使用 `AutoModelForImageTextToText`，所以应改成：

1. 优先尝试 `AutoModelForImageTextToText`。
2. 其次保持 `AutoModelForVision2Seq`。
3. 最后才是 `AutoModelForCausalLM`。

示意：

```python
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoModelForVision2Seq

if type(actor_model_config) in AutoModelForImageTextToText._model_mapping.keys():
    actor_module_class = AutoModelForImageTextToText
elif type(actor_model_config) in AutoModelForVision2Seq._model_mapping.keys():
    actor_module_class = AutoModelForVision2Seq
else:
    actor_module_class = AutoModelForCausalLM
```

checkpoint 保存同理。`FSDPCheckpointManager` 目前看到 `ForConditionalGeneration` 就用 `AutoModelForVision2Seq`，Qwen3.5 应改为优先 `AutoModelForImageTextToText`，否则保存 `hf_model` 会失败或保存成不匹配的模型类。

### 2.2 Qwen3.5 multimodal RoPE / processor 适配

修改位置：

- 新增 `verl/models/transformers/qwen3_5.py`
- `verl/utils/dataset/rl_dataset.py`
- `agent_system/multi_turn_rollout/rollout_loop.py`

当前 `RLHFDataset` 的 Qwen-VL 分支依赖类名：

```python
if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
    if "Qwen3VLProcessor" in self.processor.__class__.__name__:
        from verl.models.transformers.qwen3_vl import get_rope_index
    else:
        from verl.models.transformers.qwen2_vl import get_rope_index
```

这个判断不会覆盖 Qwen3.5。应改为基于 processor/model input 能力判断：

- `model_inputs` 中存在 `image_grid_thw` 或 `video_grid_thw`。
- `model_inputs` 中存在 `mm_token_type_ids` 时走 Qwen3.5 的 mrope helper。
- processor/config 暴露 `image_token_id`、`video_token_id`、`vision_start_token_id` 时不要再依赖具体类名。

Qwen3.5 的官方 `get_rope_index` 与 Qwen2/2.5-VL、Qwen3-VL 不同：

- 它需要 `mm_token_type_ids` 来区分 text/image/video。
- 它返回 `(position_ids, mrope_position_deltas)`，其中 `position_ids` 是 `(3, batch, seq)`。
- Qwen3.5 text model 接收 `(4, batch, seq)` 时会把第 0 行作为 text position ids，其余 3 行作为 multimodal RoPE position ids。

因此新增 `verl/models/transformers/qwen3_5.py`，按官方逻辑实现一个训练前可调用的 helper：

```python
def get_rope_index(
    processor,
    input_ids,
    mm_token_type_ids,
    image_grid_thw=None,
    video_grid_thw=None,
    attention_mask=None,
):
    ...
    return vision_position_ids, mrope_position_deltas
```

然后在 dataset 中构造：

```python
vision_position_ids, _ = qwen3_5_get_rope_index(...)
text_position_ids = ...
position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)
```

注意不要丢弃 `mm_token_type_ids`。当前 `RLHFDataset` 会把 processor 产出的剩余字段放进 `row_dict["multi_modal_inputs"]`，这点应保留；Qwen3.5 官方模型在有图像/视频时会要求 `mm_token_type_ids`，否则无法正确计算 M-RoPE。

`agent_system/multi_turn_rollout/rollout_loop.py` 的视觉路径目前手写 `<|vision_start|><|image_pad|><|vision_end|>` 替换，并直接调用 `processor.image_processor`。这对 Qwen2/3-VL 可工作，但对 Qwen3.5 不够稳。建议抽出公共函数：

```python
build_multimodal_model_inputs(processor, tokenizer, messages, images=None, videos=None)
```

在 Qwen3.5 分支中直接使用：

```python
raw_prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **kwargs)
model_inputs = processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
```

这样能自然得到 `mm_token_type_ids`，避免手工 placeholder 与 processor 规则不一致。

### 2.3 monkey patch 和训练 forward

修改位置：

- `verl/models/transformers/monkey_patch.py`
- 新增 `verl/models/transformers/qwen3_5.py`

首版策略：

- 非 fused kernel：尽量使用 Transformers 官方 `Qwen3_5ForConditionalGeneration.forward`，不要强行复用 `qwen3_vl.py`。
- fused PPO kernel：先不支持，或新增 Qwen3.5 专用 `forward_with_torch_backend/forward_with_triton_backend` 后再打开。
- remove-padding / Ulysses：首版遇到 `model.config.model_type in {"qwen3_5", "qwen3_5_moe"}` 且 `use_remove_padding=True` 或 `ulysses_sp_size>1` 时抛出清晰错误，提示先关闭。

原因：

- Qwen3.5 包含 `linear_attention` 层，官方实现会用 `attention_mask` 管理 padding/状态；当前仓库 remove-padding 路径把 `attention_mask=None` 传入 packed 输入，可能让 linear attention 状态跨样本延续。
- Qwen3.5 视觉 forward 返回结构与 Qwen3-VL 不完全相同；`qwen3_vl.py` 的 `_get_input_embeds()` 假设 deepstack visual embeds，不应直接复用。

建议在 `apply_monkey_patch()` 开头加入保护：

```python
if model.config.model_type in QWEN35_MODEL_TYPES:
    if use_remove_padding or ulysses_sp_size > 1:
        raise RuntimeError(
            "Qwen3.5 contains linear attention; remove-padding/Ulysses is not validated yet. "
            "Set actor_rollout_ref.model.use_remove_padding=False and ulysses_sequence_parallel_size=1."
        )
```

后续性能阶段再补：

- 按 `mm_token_type_ids` 和 text position id 切分 packed segments。
- 为 linear attention 构造不跨样本的状态 reset/mask。
- 用 padded-vs-packed logits 对齐测试证明无泄漏。

### 2.4 vLLM rollout 适配

修改位置：

- `setup.py`
- `requirements*.txt`
- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`
- `verl/trainer/config/ppo_trainer.yaml`

当前 SPMD vLLM 已支持把 `actor_rollout_ref.rollout.engine_kwargs.vllm` 透传给 `LLM(...)`，这是 Qwen3.5 适配的主要入口。需要做的是：

1. 升级 vLLM 到支持 Qwen3.5 的版本。
2. 在配置示例中加入 Qwen3.5 常用 engine kwargs，但保持默认 `null`，避免影响旧模型。
3. 首版不要默认启用 `language_model_only`，因为 actor 是完整 multimodal HF 模型，rollout 端若只加载 text-only vLLM 模型，FSDP 权重同步的名字映射需要单独验证。

建议示例：

```yaml
actor_rollout_ref:
  rollout:
    max_model_len: 8192
    engine_kwargs:
      vllm:
        reasoning_parser: qwen3
        enable_auto_tool_choice: null
        tool_call_parser: null
        language_model_only: null
        mm_processor_kwargs: null
```

对于纯文本 agent 训练，`language_model_only=true` 是有价值的显存优化，但应作为第二阶段测试项：必须验证 `FSDPVLLMShardingManager` 中 `model.load_weights(...)` 能正确把 HF full multimodal actor 权重同步到 vLLM text-only 子模型。

### 2.5 SGLang rollout 适配

修改位置：

- `setup.py`
- `requirements_sglang.txt`
- `verl/workers/rollout/sglang_rollout/sglang_rollout.py`
- `tests/workers/rollout/test_sglang_async_rollout_*`

当前 `ppo_trainer.yaml` 暴露了 `actor_rollout_ref.rollout.engine_kwargs.sglang`，但 `SGLangRollout._init_inference_engine()` 没有像 vLLM 那样透传这些 kwargs。Qwen3.5 至少需要支持：

- `reasoning_parser=qwen3`
- `tool_call_parser=qwen3_coder`
- `default_chat_template_kwargs={"enable_thinking": false}` 或同等参数
- `context_length`

建议仿照 vLLM rollout：

```python
engine_kwargs = {}
if "engine_kwargs" in self.config and "sglang" in self.config.engine_kwargs:
    engine_kwargs = OmegaConf.to_container(deepcopy(self.config.engine_kwargs.sglang))
engine_kwargs = {key: val for key, val in engine_kwargs.items() if val is not None}

self._engine = Engine(..., **engine_kwargs)
```

同时更新 tool-call parser 测试。现有测试断言 Qwen2.5 tokenizer 解析为 `qwen25`；Qwen3.5 官方 tool use 推荐 `qwen3_coder`，所以需要新增 Qwen3.5 tokenizer 的 parser 检测测试，并确保依赖的 SGLang 版本包含这个 parser。

### 2.6 FLOPs / metrics

修改位置：

- `verl/utils/flops_counter.py`
- `tests/utils/gpu_tests/test_flops_counter.py`

当前 `VALID_CONFIG_TYPE` 没有 `qwen3_5` 和 `qwen3_5_moe`，训练会打印不支持并让 MFU 为 0。首版至少应加入：

```python
"qwen3_5": self._estimate_qwen3_5_flops,
"qwen3_5_moe": self._estimate_qwen3_5_moe_flops,
```

若不想立即精确计算 linear attention/Gated DeltaNet，可先近似映射到现有 qwen2/qwen2_moe 估计函数，保证指标不为 0，并在注释中说明不包含 Gated DeltaNet 卷积/线性注意力的精确项。更严谨的实现应读取：

- `config.text_config.layer_types`
- `linear_num_key_heads`
- `linear_num_value_heads`
- `linear_key_head_dim`
- `linear_value_head_dim`
- `linear_conv_kernel_dim`
- MoE 的 routed/shared expert 参数

### 2.7 配置和示例脚本

新增示例脚本：

- `examples/grpo_trainer/run_doudizhu_qwen3_5.sh`
- `examples/grpo_trainer/run_sokoban_qwen3_5.sh`

基础启动片段：

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
  actor_rollout_ref.model.trust_remote_code=False \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_fused_kernels=False \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
```

如果走 SGLang 多轮工具环境：

```bash
actor_rollout_ref.rollout.name=sglang \
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.engine_kwargs.sglang.reasoning_parser=qwen3 \
actor_rollout_ref.rollout.engine_kwargs.sglang.tool_call_parser=qwen3_coder \
+data.apply_chat_template_kwargs.enable_thinking=False
```

## 3. 测试验证层面

### 3.1 环境能力测试

新增 `tests/sanity/test_qwen35_env.py`：

```python
import importlib.util

def test_qwen35_transformers_available():
    assert importlib.util.find_spec("transformers.models.qwen3_5") is not None
    assert importlib.util.find_spec("transformers.models.qwen3_5_moe") is not None
```

再加一个不下载权重的 config 测试：

```python
from transformers import AutoConfig

def test_qwen35_hf_config():
    cfg = AutoConfig.from_pretrained("Qwen/Qwen3.5-4B")
    assert cfg.model_type == "qwen3_5"
    assert cfg.architectures[0] == "Qwen3_5ForConditionalGeneration"
    assert cfg.text_config.model_type == "qwen3_5_text"
```

这个测试需要网络或本地 HF cache，可标记为 `pytest.mark.remote_model`，CI 中使用预下载 cache。

### 3.2 模型加载测试

新增 `tests/models/test_qwen3_5_model.py`，在 `transformers.models.qwen3_5` 可用时运行：

1. 用极小 `Qwen3_5Config` 构造 tiny model，不下载 4B 权重。
2. 验证 `AutoModelForImageTextToText` 是本仓库选择的模型类。
3. 文本输入 forward 返回 logits。
4. `use_remove_padding=True` 时，首版应抛出明确错误。

重点不是追求性能，而是防止 `AutoModelForCausalLM` 误加载和 checkpoint 保存类错误。

### 3.3 M-RoPE helper 测试

新增对 `verl.models.transformers.qwen3_5.get_rope_index` 的纯 CPU 单测：

- 手工构造一条含 text + image placeholder + text 的 `input_ids`。
- 构造匹配的 `mm_token_type_ids`，其中 text=0、image=1、video=2。
- 传入 `image_grid_thw=torch.tensor([[1, 4, 4]])`。
- 断言返回 `(3, seq_len)`，再与 text row 拼成 `(4, seq_len)`。
- 验证 padding 位置不被写入有效递增位置。

这个测试可以不依赖真实图片，也不需要下载模型权重。

### 3.4 Dataset / rollout 数据测试

扩展 `tests/utils/gpu_tests/dataset/test_rl_dataset.py` 或新增 CPU 可跑的 fake processor 测试：

- Qwen3.5 multimodal 样本应包含 `multi_modal_inputs["mm_token_type_ids"]`。
- position ids 应为 3D batch 形态，即单样本存储 `(4, seq_len)`，batch 后为 `(bs, 4, seq_len)`。
- 文本-only Qwen3.5 样本可以使用普通 2D position ids；如果 processor 返回 `mm_token_type_ids`，也不能导致 forward 参数丢失。

对 `agent_system/multi_turn_rollout/rollout_loop.py` 增加 visual observation 单步测试：

- 输入一张环境截图。
- 断言不再手工错配 placeholder。
- 断言 rollout DataProto 包含 `multi_modal_data`、`multi_modal_inputs`、`mm_token_type_ids`、`image_grid_thw`。

### 3.5 vLLM/SGLang smoke test

vLLM：

```bash
MODEL_ID=Qwen/Qwen3.5-4B \
NGPUS_PER_NODE=1 \
pytest -s tests/workers/rollout/test_vllm_spmd.py -k qwen35
```

测试内容：

- vLLM 能加载模型。
- `prompt_token_ids` 路径能生成。
- `FSDPVLLMShardingManager` 能同步一次权重。
- 先不打开 `language_model_only`；单独测试它与权重同步是否兼容。

SGLang：

```bash
pytest -s tests/workers/rollout/test_sglang_async_rollout_search_tools.py -k qwen35
```

测试内容：

- Qwen3.5 tokenizer 能被识别为 `qwen3_coder` tool parser，或者配置显式指定 parser。
- `engine_kwargs.sglang` 确实传给 `Engine(...)`。
- `enable_thinking=False` 后不会把 `<think>` 内容加入历史 tool message。

### 3.6 端到端训练 smoke test

先跑主训练环境内已有的文本/牌类 agent，避免同时引入视觉路径变量。这里不建议首选 WebShop，因为
`study_guide/setup_env.md` 已明确 WebShop 应拆到独立 Python 3.10 环境：

```bash
conda activate verl-agent-qwen35-bw
bash examples/grpo_trainer/run_doudizhu_qwen3_5.sh \
  actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
  trainer.total_epochs=1 \
  trainer.test_freq=1 \
  trainer.save_freq=-1 \
  data.train_batch_size=8 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.use_fused_kernels=False
```

通过标准：

- actor/ref/rollout 初始化成功。
- 至少完成 1 次 rollout、log_prob、update_actor。
- `old_log_probs/log_prob/advantages` 无 NaN/Inf。
- `response_mask` 与 action parser 正常。
- checkpoint 可保存普通 sharded checkpoint。

再跑视觉 agent smoke：

```bash
bash examples/grpo_trainer/run_sokoban_qwen3_5.sh \
  actor_rollout_ref.model.path=Qwen/Qwen3.5-4B \
  trainer.total_epochs=1 \
  data.train_batch_size=2 \
  actor_rollout_ref.rollout.max_model_len=4096
```

通过标准：

- processor 产出 `mm_token_type_ids`。
- image token 数与 image feature 数匹配。
- Qwen3.5 forward 不报 `mm_token_type_ids is missing`。

### 3.7 回归测试

必须保留已有模型行为：

```bash
pytest -s -x tests/sanity
pytest -s -x tests/utils/cpu_tests
pytest -s tests/models/test_transformer.py
pytest -s tests/utils/gpu_tests/test_flops_counter.py
```

GPU 条件允许时再跑：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 pytest -s tests/workers/rollout/test_vllm_spmd.py
pytest -s tests/workers/rollout/test_sglang_async_rollout_search_tools.py
```

## 4. 风险和优先级

建议优先级：

1. 依赖升级和 capability test。
2. `AutoModelForImageTextToText` 加载与 checkpoint 保存。
3. Qwen3.5 processor/M-RoPE/dataset/multi-turn 视觉输入。
4. vLLM 默认非 text-only rollout。
5. SGLang `engine_kwargs` 透传和 `qwen3_coder` parser。
6. FLOPs 识别。
7. remove-padding/Ulysses/fused-kernel 性能适配。

最大风险点：

- Qwen3.5 linear attention 在 remove-padding packed 输入下可能跨样本泄漏状态；初版应禁止。
- vLLM `language_model_only` 与 FSDP full multimodal actor 的权重同步名称可能不一致；先不要默认开启。
- MoE Qwen3.5-A* 模型需要 vLLM expert parallel 或足够新的 SGLang；本仓库不走 Megatron 时，先用 `Qwen/Qwen3.5-4B` 这类较小 dense checkpoint 验证链路。
- Qwen3.5 默认 thinking mode 会改变 action 格式和历史内容；agentic RL 初期建议统一关闭。

## 5. 参考来源

- 本仓库 Blackwell 环境配置说明：`study_guide/setup_env.md`。
- 本地 `verl-agent-bw` 环境检查：`conda list -n verl-agent-bw` 与 `conda run -n verl-agent-bw python -m pip freeze`。
- Qwen 官方 GitHub：`https://github.com/QwenLM/Qwen3.5`（当前重定向到 Qwen3.6 仓库，但 README 明确包含 Qwen3.5 发布时间、模型列表、架构和 agent/RL 说明）。
- Hugging Face 模型卡：`https://huggingface.co/Qwen/Qwen3.5-4B`。
- Hugging Face `config.json`：`https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/config.json`。
- Hugging Face Transformers 文档：`https://huggingface.co/docs/transformers/main/model_doc/qwen3_5` 与 `https://huggingface.co/docs/transformers/main/model_doc/qwen3_5_moe`。
- vLLM Qwen3.5/Qwen3.6 recipe：`https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html`。
- ModelScope ms-swift Qwen3.5 最佳实践：`https://swift.readthedocs.io/zh-cn/v4.1/BestPractices/Qwen3_5-Best-Practice.html`。
