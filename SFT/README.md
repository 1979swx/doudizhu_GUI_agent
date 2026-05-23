# Qwen3.5 VLM SFT

Standalone SFT entrypoints for synthetic datasets. These scripts do not modify
the main RL training chain.

## Doudizhu QA SFT

```bash
conda activate verl-agent-bw-exp

NUM_GPUS=2 \
MODEL_PATH=Qwen/Qwen3.5-4B \
DATA_DIR=data_synthesis/doudizhu_qa_sft \
bash SFT/run_qwen3_5_4B_doudizhu_qa_sft.sh
```

The script expects:

- `data_synthesis/doudizhu_qa_sft/train.parquet`
- `data_synthesis/doudizhu_qa_sft/val.parquet`

## Doudizhu grounding + QA mixed SFT

```bash
conda activate verl-agent-bw-exp

NUM_GPUS=2 \
MODEL_PATH=Qwen/Qwen3.5-4B \
GROUNDING_DATA_DIR=data_synthesis/doudizhu_grounding_sft \
QA_DATA_DIR=data_synthesis/doudizhu_qa_sft \
bash SFT/run_qwen3_5_4B_doudizhu_grounding_qa_mix_sft.sh
```

The mixed script passes both train parquet files and both validation parquet
files to `SFT/qwen3_5_vlm_sft_trainer.py`; that trainer concatenates them and
uses the distributed sampler shuffle during training.

## Doudizhu grounding example

```bash
conda activate verl-agent-bw-exp

NUM_GPUS=2 \
MODEL_PATH=Qwen/Qwen3.5-4B \
DATA_DIR=data_synthesis/doudizhu_grounding_sft \
FSDP_SHARDING=zero3 \
bash SFT/run_qwen3_5_4B_doudizhu_grounding_sft.sh
```

The trainer reads parquet datasets with configurable keys:

- `prompt`: list of chat messages, or a plain user string
- `images`: optional list of image dicts such as `{"bytes": PNG_BYTES}`
- `answer`: assistant response string

Text-only datasets can omit `images`. Future synthetic datasets can reuse this
format by changing `TRAIN_FILE`, `VAL_FILE`, `PROMPT_KEY`, `RESPONSE_KEY`, and
`IMAGE_KEY`.
