#!/usr/bin/env bash
set -euo pipefail

# Create/update the verl-agent Qwen3.5 conda env and optionally run the
# Sokoban one-step smoke script.
#
# Usage:
#   bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh
#   SKIP_INSTALL=1 bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh
#   RUN_SMOKE=0 bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh
#   RECREATE_ENV=1 bash scripts/setup_verl_agent_bw_exp_qwen3_5_sokoban_smoke.sh

ENV_NAME="${ENV_NAME:-verl-agent-bw-exp}"
SOURCE_ENV_NAME="${SOURCE_ENV_NAME:-verl-exp}"
RECREATE_ENV="${RECREATE_ENV:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"
RUN_MINIMAL_CHECK="${RUN_MINIMAL_CHECK:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"

CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"
MAX_JOBS="${MAX_JOBS:-32}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PYTORCH_CUDA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL:-https://download.pytorch.org/whl/cu129}"

MODEL_PATH="${MODEL_PATH:-/home/zhangwj/verl/models/Qwen3.5-2B}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

env_exists() {
  conda env list | awk '{print $1}' | grep -qx "$1"
}

if [[ "${RECREATE_ENV}" == "1" ]] && env_exists "${ENV_NAME}"; then
  conda env remove -y -n "${ENV_NAME}"
fi

if ! env_exists "${ENV_NAME}"; then
  if ! env_exists "${SOURCE_ENV_NAME}"; then
    echo "Source conda env '${SOURCE_ENV_NAME}' does not exist." >&2
    exit 1
  fi
  conda create -y -n "${ENV_NAME}" --clone "${SOURCE_ENV_NAME}"
fi

conda activate "${ENV_NAME}"

export CUDA_HOME
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export MAX_JOBS
export PIP_INDEX_URL
export PIP_EXTRA_INDEX_URL="${PYTORCH_CUDA_INDEX_URL}"

if [[ "${SKIP_INSTALL}" != "1" ]]; then
  python -m pip install --upgrade pip wheel
  python -m pip install --retries 20 --timeout 120 setuptools==80.10.2 packaging==25.0
  python -m pip install --retries 20 --timeout 120 ninja cmake pybind11 nvidia-mathdx
  python -m pip install --retries 20 --timeout 120 gym==0.26.2 gym-sokoban==0.0.6 imageio requests tqdm
  python -m pip install --no-deps -e .

  if ! python - <<'PY'
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

raise SystemExit(0 if is_flash_linear_attention_available() and is_causal_conv1d_available() else 1)
PY
  then
    python -m pip install --retries 20 --timeout 120 --no-build-isolation --no-cache-dir \
      causal-conv1d==1.6.2.post1
    python -m pip uninstall -y flash-linear-attention fla-core || true
    python -m pip install --retries 20 --timeout 120 --no-cache-dir --no-deps \
      git+https://github.com/fla-org/flash-linear-attention.git
  fi
fi

python - <<'PY'
import importlib.metadata as md
import os
import torch
import transformers
import vllm
from transformers import AutoConfig
from transformers.utils.import_utils import is_causal_conv1d_available, is_flash_linear_attention_available

model_path = os.environ.get("MODEL_PATH", "/home/zhangwj/verl/models/Qwen3.5-2B")
config = AutoConfig.from_pretrained(model_path, attn_implementation="sdpa")
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
for pkg in ("flash-attn", "flash-linear-attention", "causal-conv1d", "gym", "gym-sokoban"):
    try:
        print(pkg, md.version(pkg))
    except md.PackageNotFoundError:
        print(pkg, "not installed")
print("model_type", config.model_type, "attn", getattr(config, "_attn_implementation", None))
print("flash_linear_attention_available", is_flash_linear_attention_available())
print("causal_conv1d_available", is_causal_conv1d_available())
assert config.model_type == "qwen3_5"
assert is_flash_linear_attention_available()
assert is_causal_conv1d_available()
PY

if [[ "${RUN_MINIMAL_CHECK}" == "1" ]]; then
  MODEL_PATH="${MODEL_PATH}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python - <<'PY'
import os
import torch
from transformers import AutoConfig, AutoModelForImageTextToText
from verl.models.transformers.monkey_patch import apply_monkey_patch

model_path = os.environ["MODEL_PATH"]
config = AutoConfig.from_pretrained(model_path, attn_implementation="sdpa")
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    config=config,
).cuda().eval()
apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1)
vocab = model.config.text_config.vocab_size
input_ids = torch.randint(0, vocab, (1, 16), device="cuda")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = model(input_ids=input_ids)
torch.cuda.synchronize()
print("minimal Qwen3.5 sdpa full-attn + FLA linear-attn check ok", tuple(out.logits.shape))
PY
fi

if [[ "${RUN_SMOKE}" == "1" ]]; then
  ray stop --force || true
  mkdir -p logs
  start_time="$(date +%Y%m%d_%H%M%S)"
  MODEL_PATH="${MODEL_PATH}" bash study_guide/smoke_run/run_sokoban_qwen3_5_smoke.sh "$@" \
    2>&1 | tee "logs/qwen3_5-sokoban-smoke-${start_time}.log"

  latest_log="$(ls -t logs/qwen3_5-sokoban-smoke-*.log | head -n 1)"
  if grep -q "fast path is not available.*Falling back to torch implementation" "${latest_log}"; then
    echo "Qwen3.5 linear-attention fast path was not used; see ${latest_log}" >&2
    exit 1
  fi
  grep -q "Training Progress: .*1/1" "${latest_log}"
  echo "Smoke run completed: ${latest_log}"
fi
