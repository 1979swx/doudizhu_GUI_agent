#!/usr/bin/env bash
set -xeuo pipefail

DATA_DIR="${DATA_DIR:-data_synthesis/doudizhu_grounding_sft}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-20000}"
VAL_SAMPLES="${VAL_SAMPLES:-3000}"
TEST_SAMPLES="${TEST_SAMPLES:-3000}"
LANGUAGE="${LANGUAGE:-zh}"
JITTER="${JITTER:-0.25}"
SYNTH_LOG_EVERY="${SYNTH_LOG_EVERY:-1000}"

# The synthesis environment can be overridden if the caller already has all
# environment dependencies in the current shell.
SYNTH_CMD="${SYNTH_CMD:-conda run -n verl-agent-bw-exp python}"
SKIP_SYNTH="${SKIP_SYNTH:-False}"

if [[ "${SKIP_SYNTH}" != "True" && "${SKIP_SYNTH}" != "true" && "${SKIP_SYNTH}" != "1" ]]; then
  ${SYNTH_CMD} data_synthesis/doudizhu_grounding_sft.py \
    --output-dir "${DATA_DIR}" \
    --train-samples "${TRAIN_SAMPLES}" \
    --val-samples "${VAL_SAMPLES}" \
    --test-samples "${TEST_SAMPLES}" \
    --language "${LANGUAGE}" \
    --jitter "${JITTER}" \
    --log-every "${SYNTH_LOG_EVERY}"
fi

DATA_DIR="${DATA_DIR}" bash SFT/run_qwen3_5_4B_doudizhu_grounding_sft.sh
