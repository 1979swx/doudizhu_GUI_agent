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

The QA sampler also steers within-task label quotas for the most skew-prone
tasks: Task F true/false, H2 any plane attachment true/false, Task I can-pass
true/false, Task K legal-action-count buckets, and Task M legal/illegal reason
buckets. While a task still has an unmet label bucket, already-full buckets for
that task are skipped and the sampler keeps scanning later game states. The
result is recorded in `metadata.json` as `label_targets`, `label_counts`, and
`label_quota_remaining`. Use `--disable-label-quotas` only when you need to
reproduce the old task-only sampler behavior.

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

Optional HTML review:

```bash
python data_synthesis/visualize_doudizhu_grounding_sft.py \
  --input data_synthesis/doudizhu_grounding_sft/train.parquet \
  --output data_synthesis/doudizhu_grounding_sft/review.html \
  --num-samples 40
```

The review page draws the normalized gold clicks on top of each screenshot and
shows the prompt, model answer, verifier result, and full `extra_info`.

## Generate end-to-end rollout data

`doudizhu_end_to_end_sft.py` collects full main-environment episodes with a
teacher VLM, writes raw rollout JSONL first, then filters winning valid steps
into SFT parquet.

Small local dry run without API calls:

```bash
conda activate verl-agent-bw-exp
python data_synthesis/doudizhu_end_to_end_sft.py \
  --model-backend mock \
  --output-dir data_synthesis/doudizhu_end_to_end_sft_smoke \
  --train-samples 20 \
  --val-samples 0 \
  --test-samples 0 \
  --max-raw-episodes 20
```

Kimi/Moonshot collection:

```bash
export MOONSHOT_API_KEY=...
conda activate verl-agent-bw-exp
python data_synthesis/doudizhu_end_to_end_sft.py \
  --output-dir data_synthesis/doudizhu_end_to_end_sft \
  --model-backend api \
  --api-base-url https://api.moonshot.cn/v1 \
  --api-model kimi-k2.6 \
  --api-key-env MOONSHOT_API_KEY \
  --api-thinking disabled \
  --temperature 0.6 \
  --max-new-tokens 1536 \
  --terminal-max-hand 2 \
  --num-workers 8 \
  --request-concurrency 8 \
  --train-samples 5000 \
  --val-samples 500 \
  --test-samples 500
```

`--num-workers` controls how many independent Dou Dizhu episodes are active at
the same time. `--request-concurrency` caps concurrent API requests across all
workers, while `--request-rpm` and `--request-tpm` remain global rate limits.
For a new account or model, start with 4-8 workers and increase only after the
provider dashboard shows stable throughput.

Filtered SFT rows are accepted from normal-ended episodes where player 0 has at
most `--terminal-max-hand` cards left at the terminal state. The default is `2`;
set it to `0` to recover the strict winning/zero-hand mode.

The output directory contains raw audit data under `raw/`, filtered
`train.parquet`/`val.parquet`/`test.parquet`, split reports under `reports/`,
`reports/review.html`, and `metadata.json`. Rebuild filtered parquet from
existing raw rollout without calling the API:

```bash
python data_synthesis/doudizhu_end_to_end_sft.py \
  --filter-only \
  --output-dir data_synthesis/doudizhu_end_to_end_sft
```

Standalone HTML visualization for a filtered parquet split:

```bash
python data_synthesis/visualize_doudizhu_end_to_end.py \
  --input data_synthesis/doudizhu_end_to_end_sft/train.parquet \
  --output data_synthesis/doudizhu_end_to_end_sft/review.html \
  --num-samples 40
```
