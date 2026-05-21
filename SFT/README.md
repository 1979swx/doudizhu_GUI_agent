# Qwen3.5 VLM SFT

Standalone SFT entrypoints for synthetic datasets. These scripts do not modify
the main RL training chain.

## Doudizhu grounding example

```bash
conda activate verl-agent-bw-exp

bash SFT/run_qwen3_5_sft.sh \
  DATA_DIR=data_synthesis/doudizhu_grounding_sft \
  MODEL_PATH=Qwen/Qwen3.5-4B \
  NUM_GPUS=2 \
  FSDP_SHARDING=zero3
```

The trainer reads parquet datasets with configurable keys:

- `prompt`: list of chat messages, or a plain user string
- `images`: optional list of image dicts such as `{"bytes": PNG_BYTES}`
- `answer`: assistant response string

Text-only datasets can omit `images`. Future synthetic datasets can reuse this
format by changing `TRAIN_FILE`, `VAL_FILE`, `PROMPT_KEY`, `RESPONSE_KEY`, and
`IMAGE_KEY`.

