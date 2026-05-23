# Doudizhu SFT Synthesis

This directory contains standalone data synthesis utilities. They do not modify
the training or environment code paths.

## Generate QA data

The QA generator implements the first two phases from
`study_guide/doudizhu_QA_data_synthesis_plan.md`: hand reading, counting,
basic/compound card type recognition, legal action understanding, candidate
legality, and candidate card type QA.

Run from the repository root in the environment that has the Dou Dizhu
dependencies installed:

```bash
conda activate verl-agent-bw
python data_synthesis/doudizhu_qa_sft.py \
  --output-dir data_synthesis/doudizhu_qa_sft \
  --train-samples 15000 \
  --val-samples 2000 \
  --test-samples 2000 \
  --language zh
```

The script writes `train.parquet`, `val.parquet`, `test.parquet`, and
`metadata.json`. Each row stores:

- `data_source=doudizhu_qa_sft`
- `prompt` and `question` with one image placeholder
- `images=[{"bytes": PNG_BYTES}]`
- `answer` using `<answer>` only for direct reading/counting tasks, or
  `<plan>` plus `<answer>` for rule tasks
- `extra_info.gold`, `extra_info.plan_aux`, and verifier metadata

Optional HTML review:

```bash
python data_synthesis/visualize_doudizhu_qa_sft.py \
  --input data_synthesis/doudizhu_qa_sft/train.parquet \
  --output data_synthesis/doudizhu_qa_sft/review.html \
  --num-samples 40
```

## Generate grounding data

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
