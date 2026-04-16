# verl-agent 在 Blackwell 机器上的环境配置指南

本文是专门结合这份代码库和你这台机器写的。

截至 **2026-04-02**，我读到你的本机 GPU 信息是：

- `NVIDIA RTX PRO 6000 Blackwell Workstation Edition`
- 共 `2` 张卡
- 驱动版本 `575.57.08`
- 单卡显存约 `97887 MiB`（约 96GB）

这很关键，因为这类 **Blackwell Workstation/Consumer** GPU 在 NVIDIA 官方 CUDA Compute Capability 页面里属于 **compute capability 12.0（sm_120）**。这和仓库里很多仍然基于 `CUDA 12.4 / torch 2.6 / vllm 0.8.x / flash-attn 2.7.4` 的旧安装说明，不是同一个时代的组合。

---

## 先说结论

如果你的目标是 **先把 `verl-agent` 在这台机器上稳稳跑起来**，我建议你这样做：

1. 主训练环境用 **Python 3.12**。
2. 主推理后端先用 **vLLM 0.11.0**，不要一上来折腾 SGLang。
3. PyTorch 先用 **CUDA 12.8** 对应的官方 wheel。
4. **不要直接照抄仓库里的 `scripts/install_vllm_sglang_mcore.sh`**，它硬编码了很多对 Blackwell 不友好的旧版本。
5. **不要默认保留示例脚本里的 `export VLLM_ATTENTION_BACKEND=XFORMERS`**。对这台机器，先不设这个环境变量更稳。
6. `flash-attn` 对这份仓库来说，**基本属于训练必需依赖**，但它也是最容易在 Blackwell 上卡住的一步，所以要单独处理。

如果你只想要一句最短建议，那就是：

> 先建一个 `Python 3.12 + torch 2.8.0/cu128 + vllm 0.11.0 + 可编辑安装仓库` 的主环境；`flash-attn` 单独安装；WebShop/Search/AppWorld 各自再开独立环境，不要混在一个环境里。

---

## 为什么不能直接照搬仓库 README

这份仓库内部现在其实有几套不完全一致的安装建议：

- 根目录 `README.md` 主安装写的是：
  - `python==3.12`
  - `vllm==0.11.0`
  - `flash-attn==2.7.4.post1`
- 但 `agent_system/environments/README.md` 和部分旧文档里，仍然有：
  - `torch==2.6.0`
  - `cu124`
  - `vllm==0.8.2 / 0.8.5`
- `scripts/install_vllm_sglang_mcore.sh` 甚至还会：
  - 直接装 `torch==2.6.0`
  - 直接装 `vllm==0.8.5.post1`
  - 下载一个 **`cp310 + cu12 + torch2.6`** 的 `flash-attn` 预编译 wheel

这些东西在老的 Ada/Hopper 机器上还能凑合，但对你的 **RTX PRO 6000 Blackwell Workstation Edition（CC 12.0）** 来说，已经明显偏旧。

更具体地说：

- NVIDIA 官方在 **CUDA 12.8 Release Notes** 里明确写了，CUDA 12.8 新增了对 **`SM_120`** 的编译支持。
- NVIDIA 官方 Compute Capability 页面里明确列出：`RTX PRO 6000 Blackwell Workstation Edition` 属于 **CC 12.0**。
- vLLM 官方安装文档（`v0.11.0`）明确说它的预编译 CUDA wheel 默认是 **CUDA 12.8**，并提醒 **Blackwell（文档里点名 B200/GB200）至少需要 CUDA 12.8**。

这里有一个我基于官方资料做出的推断：

> vLLM 文档点名的是数据中心 Blackwell（B200/GB200），而你的卡是工作站 Blackwell（RTX PRO 6000 WE，CC 12.0）。再结合 CUDA 12.8 对 `SM_120` 的官方支持，可以合理推断：**对你的卡，凡是需要源码编译 CUDA 扩展的步骤，也至少应该看齐 CUDA 12.8，而不是 12.4。**

这也是本文后面所有建议的出发点。

---

## 这台机器最推荐的主线方案

### 方案定位

这套方案适合：

- 跑仓库主线的 `vllm` 训练/rollout
- 先跑通 `ALFWorld / Sokoban / Gym Cards / Qwen3-VL` 这类主环境
- 暂时不优先碰最难装的 `Megatron + TransformerEngine + Apex`

### 为什么我推荐这条主线

因为它同时满足三件事：

1. 跟当前仓库主 README 的方向一致：`Python 3.12 + vLLM 0.11.0`
2. 跟 Blackwell 的官方支持边界一致：至少看齐 `CUDA 12.8`
3. 避开仓库里那些明显写给旧 CUDA/旧 wheel 的脚本

---

## 第 0 步：先确认系统层是否达标

本文默认你在 **Linux x86_64** 上配置，因为：

- vLLM 官方文档当前明确支持 Linux
- 这份仓库本身也主要按 Linux 生态写的

你当前驱动是 `575.57.08`，已经高于 NVIDIA 在 CUDA 12.8 release notes 里给出的 `CUDA 12.8 GA -> Linux driver >= 570.26` 的门槛，所以：

- **跑 PyTorch / vLLM 的预编译 wheel：驱动已经够了**
- **是否需要本地 CUDA Toolkit**，取决于你要不要源码编译 `flash-attn`

建议你先检查：

```bash
nvidia-smi
nvcc --version
python3 --version
```

如果：

- `nvidia-smi` 正常
- `nvcc` 不存在，或者版本低于 `12.8`

那么请记住一句话：

> **没有本地 CUDA Toolkit 12.8+，你大概率仍然能跑 PyTorch/vLLM wheel；但你在编译 `flash-attn` 这类 CUDA 扩展时，很容易翻车。**

所以：

- 如果你只想先验证 PyTorch/vLLM 能用，可以暂时不装 toolkit。
- 如果你准备认真跑训练，我建议你把本地 CUDA Toolkit 补到 **12.8 或 12.9**。

---

## 第 1 步：创建主环境

我建议名字就叫：

```bash
conda create -n verl-agent-bw python=3.12 -y
conda activate verl-agent-bw
```

然后先把构建工具补齐：

```bash
python -m pip install -U pip setuptools wheel packaging ninja
```

这里的 `ninja` 很重要。`flash-attn` 官方文档明确建议装它；没有 `ninja` 时，编译会慢很多。

---

## 第 2 步：先装 PyTorch（推荐 `cu128`）

这一步建议直接使用 PyTorch 官方 wheel。

我推荐的第一选择是：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
```

为什么不是仓库里常见的 `torch 2.6 + cu124`：

- 那套更偏老平台
- 你的卡是 `sm_120`
- `cu128` 跟 Blackwell 的官方支持边界更对齐

如果你非常保守，想稍微往回收一点，也可以退到：

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
```

但我的主推荐仍然是 `2.8.0 + cu128`。

---

## 第 3 步：安装 vLLM

这份仓库的主 README 现在已经切到 `vllm==0.11.0`，而 `setup.py` 的 `vllm` extra 也允许到 `<=0.11.0`，所以这里直接对齐仓库当前上界即可：

```bash
pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128
```

这里建议先不要装 `.[vllm]`，原因很简单：

- 你现在最在乎的是 **版本可控**
- 直接写死 `0.11.0` 更容易排障

---

## 第 4 步：安装仓库本体

在仓库根目录执行：

```bash
pip install -e .
```

这一步会把仓库核心依赖装上，包括：

- `accelerate`
- `datasets`
- `hydra-core`
- `peft`
- `ray[default]`
- `tensordict`
- `transformers`
- `wandb`
- `qwen-vl-utils[decord]`

注意：

- 这里 **不会** 自动把 `flash-attn` 和 `vllm` 都装成你想要的版本，所以我们前面才手动先装。
- 也正因为如此，这个顺序是可控的。

---

## 第 5 步：单独安装 `flash-attn`

这一段最重要，也最容易出问题。

### 先说结论

对这份仓库来说，`flash-attn` 不是“锦上添花”的小优化，而是训练路径上的 **强依赖**。仓库里很多文件直接 `import flash_attn`，尤其是：

- `verl/workers/actor/dp_actor.py`
- `verl/workers/critic/dp_critic.py`
- `verl/workers/fsdp_workers.py`
- 多个 `megatron` 层和 `transformers` 适配层

所以如果你准备真正开始训练，**基本要把它装好**。

### 为什么不建议照仓库脚本装旧 wheel

仓库脚本里抓的是：

- `flash-attn==2.7.4.post1`
- `cu12 + torch2.6 + cp310` 的预编译 wheel

这对你现在这套 `Python 3.12 + torch 2.8/cu128 + Blackwell` 组合，并不合适。

### 现在更推荐的装法

截至 **2026-04-02**，FlashAttention 在 PyPI 上最新版本是 **`2.8.3`**。官方说明里写的是：

- `PyTorch 2.2 and above`
- `CUDA 12.0 and above`
- 但“明确列出的已支持 NVIDIA GPU”仍主要是 **Ampere / Ada / Hopper**

这意味着：

- 从“编译条件”上看，它已经足够新
- 但从“文档明确列出的 GPU 名单”上看，Blackwell 还不是写得最清楚的那一档

因此我的建议是：

```bash
MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation
```

如果你的机器内存足够大，也可以把 `MAX_JOBS` 调到 `16`。

### 如果编译失败，按这个顺序排查

#### 情况 A：报 `nvcc` 不存在，或者 CUDA 版本太老

说明你本地没有可用于编译扩展的 toolkit，或者 toolkit `< 12.8`。

处理：

1. 安装 CUDA Toolkit `12.8` 或 `12.9`
2. 重新打开 shell
3. 确认：

```bash
nvcc --version
```

#### 情况 B：日志里没有看到 `sm_120`

这通常说明构建流程没有正确识别 Blackwell 架构。

可以这样重试：

```bash
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --force-reinstall
```

这是经验性建议，不是 FlashAttention 文档里的显式要求，但对新架构机器经常有帮助。

#### 情况 C：出现类似 `unsupported gpu architecture` / `Unknown CUDA arch`

这通常说明你在用的不是足够新的 CUDA 编译工具链。

对你的卡来说，看到这类错误时，第一反应应该是：

> **先检查本地 CUDA Toolkit 是否真的到了 12.8+。**

---

## 第 6 步：做一个最小验收

先测 PyTorch：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
print("cc:", torch.cuda.get_device_capability(0))
print("bf16:", torch.cuda.is_bf16_supported())
PY
```

你这台机器正常情况下，`cc` 应该看到类似：

```text
(12, 0)
```

再测 vLLM 和仓库导入：

```bash
python - <<'PY'
import vllm
import verl
print("vllm:", vllm.__version__)
print("verl import ok")
PY
```

最后测一下 `flash-attn`：

```bash
python - <<'PY'
import flash_attn
print("flash_attn import ok")
PY
```

如果这三关都过了，你的主环境就已经很像样了。

---

## 第 7 步：跑这份仓库时要改掉的一个旧习惯

很多示例脚本里都有这一行：

```bash
export VLLM_ATTENTION_BACKEND=XFORMERS
```

但仓库自己的 `docs/README_vllm0.8.md` 又明确说过：

> 升到 `vllm >= 0.8` 后，要把这个环境变量去掉。

所以对你现在的 `vllm 0.11.0`，我的建议是：

### 默认做法

先 **不要** 设它：

```bash
unset VLLM_ATTENTION_BACKEND
```

并把示例脚本里的这一行先注释掉。

### 什么时候再试 `XFORMERS`

只有当你在 rollout 或 attention backend 上遇到非常具体的问题时，再把它作为 fallback 去试。

也就是说：

- **先不用**
- **有症状再加**

而不是一上来就继承旧脚本。

---

## 第 8 步：建议你从 FSDP + vLLM 起步，不要先碰 Megatron

这不是因为 Megatron 不好，而是因为在 **新架构 + 本机 Python 环境** 里，Megatron 通常会把环境复杂度一下抬很高：

- 可能要编译 `TransformerEngine`
- 可能要额外关注 `cuDNN`
- 可能还会遇到 `Apex`

对“先在 Blackwell 机器上把仓库跑通”这个目标来说，优先级不高。

所以建议顺序是：

1. 先让主环境跑通 `FSDP + vLLM`
2. 先跑一个文本环境的最小实验
3. 再考虑要不要上 `Megatron / TransformerEngine`

这会省你很多时间。

---

## 推荐的主环境安装清单

如果你想直接复制一套我最推荐的命令，可以用下面这一版。

```bash
conda create -n verl-agent-bw python=3.12 -y
conda activate verl-agent-bw

python -m pip install -U pip setuptools wheel packaging ninja

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install vllm==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu128

cd /home/zhangwj/science/verl-agent
pip install -e .

# 如果 nvcc 不够新，先补 CUDA Toolkit 12.8/12.9 再执行
MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation

# 可选，加速类依赖
pip install liger-kernel
```

如果 `flash-attn` 编译失败，再试：

```bash
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=8 pip install flash-attn==2.8.3 --no-build-isolation --force-reinstall
```

---

## 这份仓库里不同环境，建议怎么拆

仓库 README 已经提醒你：很多环境最好分开装。对你的机器，这条建议仍然成立，而且我建议你更坚决一点地执行。

### 1. 主训练环境：`verl-agent-bw`

用途：

- 主仓库
- `ALFWorld`
- `Sokoban`
- `Gym Cards`
- 视觉模型训练（如 Qwen3-VL）

版本建议：

- `Python 3.12`
- `torch 2.8.0/cu128`
- `vllm 0.11.0`

### 2. WebShop 环境：`verl-agent-webshop`

仓库里已经明确写了 WebShop 需要 `Python <= 3.10`，所以请单独开：

```bash
conda create -n verl-agent-webshop python=3.10 -y
conda activate verl-agent-webshop
```

然后只在这个环境里安装 WebShop 本体。

**不要** 为了 WebShop 把主环境降到 Python 3.10。

### 3. Search Retriever 环境：`retriever`

仓库 README 给的方向仍然合理：

- 单独开一个环境
- 单独装 `faiss-gpu`

这一块不建议强行并回主环境。

### 4. AppWorld 环境：`appworld`

AppWorld 也建议单独开 server 环境。仓库里本来就是这么写的，保持这个拆分就好。

---

## 第一次真正运行前，我建议你顺手改两处

### 改动 1：把示例脚本里的 `XFORMERS` 先注释掉

例如：

```bash
# export VLLM_ATTENTION_BACKEND=XFORMERS
```

### 改动 2：第一次 smoke test 不要贪大

比如先跑：

- 小模型
- 文本环境
- 少量 batch

因为你的机器虽然是双 `RTX PRO 6000 Blackwell`、总显存很大，但第一次验证环境时，重点不是“压满显卡”，而是“确认整条软件栈没问题”。

---

## 常见坑速查

### 坑 1：`pip install flash-attn ...` 失败

最常见原因：

- 没有 `nvcc`
- 本地 toolkit 太老
- 架构没带上 `sm_120`

优先检查：

```bash
nvcc --version
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_capability(0))
PY
```

### 坑 2：跑示例脚本时 attention backend 出错

先做这件事：

```bash
unset VLLM_ATTENTION_BACKEND
```

并注释脚本里的 `export VLLM_ATTENTION_BACKEND=XFORMERS`。

### 坑 3：你明明是 Blackwell，却还在装 `cu124`

这不是绝对不行，但它明显不是最顺手的起点。

对于你的 `CC 12.0` 机器，我建议起点就是：

- `PyTorch cu128`
- `vLLM 0.11.0`
- `flash-attn 2.8.3`

### 坑 4：把所有环境都塞进一个 conda env

这份仓库已经明示很多环境会互相打架。你这台机器再强，也不应该靠“一个超级环境”去解决依赖冲突。

正确思路是：

- 主训练环境一个
- WebShop 一个
- Retriever 一个
- AppWorld 一个

---

## 我对你这台机器的最终建议

如果目标是“最省时间地把 `verl-agent` 用起来”，我的建议很明确：

- **主环境走 `Python 3.12 + torch 2.8.0/cu128 + vllm 0.11.0`**
- **`flash-attn` 用新版本单独装，不要吃仓库里旧 wheel**
- **第一次先走 FSDP + vLLM**
- **先别默认继承脚本里的 `XFORMERS`**
- **WebShop / Search / AppWorld 都独立环境**

你这台机器是双 `RTX PRO 6000 Blackwell Workstation Edition`，显存非常宽裕；仓库里很多示例默认就是 `2` 卡，你的硬件形态其实很合适。真正需要你小心的，不是算力不够，而是 **软件版本别落在 Blackwell 支持断层里**。

---

## 参考依据

### 代码库内依据

- 根目录安装说明：[`README.md`](../README.md)
- 环境拆分说明：[`agent_system/environments/README.md`](../agent_system/environments/README.md)
- 依赖边界：[`setup.py`](../setup.py)
- 旧安装脚本：[`scripts/install_vllm_sglang_mcore.sh`](../scripts/install_vllm_sglang_mcore.sh)
- vLLM 0.8 说明：[`docs/README_vllm0.8.md`](../docs/README_vllm0.8.md)
- 示例脚本中的 `VLLM_ATTENTION_BACKEND=XFORMERS`：`examples/gigpo_trainer/*.sh`

### 官方资料（截至 2026-04-02）

- NVIDIA CUDA GPU Compute Capability  
  https://developer.nvidia.com/cuda-gpus

- NVIDIA CUDA 12.8 Release Notes  
  https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html

- NVIDIA Blackwell Compatibility Guide  
  https://docs.nvidia.com/cuda/inline-ptx-assembly/blackwell-compatibility-guide/index.html

- PyTorch Get Started  
  https://pytorch.org/get-started/locally/

- PyTorch Previous Versions  
  https://pytorch.org/get-started/previous-versions

- vLLM `v0.11.0` GPU Installation  
  https://docs.vllm.ai/en/v0.11.0/getting_started/installation/gpu.html

- FlashAttention PyPI  
  https://pypi.org/project/flash-attn/

---

## 一句话收尾

对你的这台 **RTX PRO 6000 Blackwell Workstation Edition** 机器，最值得坚持的原则只有一条：

> **运行期尽量用官方新 wheel，编译期至少看齐 CUDA 12.8，仓库里的旧 pin 只参考，不照搬。**
