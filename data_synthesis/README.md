# Doudizhu Grounding SFT Synthesis

This directory contains standalone data synthesis utilities. They do not modify
the training or environment code paths.

## Generate data

Run from the repository root in the environment that has the Dou Dizhu
dependencies installed:

```bash
conda activate verl-agent-bw
python data_synthesis/doudizhu_grounding_sft.py \
  --output-dir data_synthesis/doudizhu_grounding_sft \
  --train-samples 30000 \
  --val-samples 5000 \
  --test-samples 5000 \
  --language zh \
  --jitter 0.25
```

The script writes:

- `train.parquet`
- `val.parquet`
- `test.parquet`
- `metadata.json`

Each sample is verified by the environment scorer before being accepted. The
assistant response is intentionally restricted to the current
`doudizhu_grounding_projection` format:

```xml
<tool_call>left_click([x1,y1],[x2,y2])</tool_call>
```

The sampler enforces category quotas from proportions. Once a category quota is
full, matching legal actions are used only to advance the underlying game and
are not written as SFT samples. Default proportions are:

- `pass`: 0.12
- `solo`: 0.24
- `pair`: 0.19
- `trio`: 0.13
- `chain`: 0.15
- `bomb_rocket`: 0.06
- `other`: 0.11
