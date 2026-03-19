#!/bin/bash

ROOT="/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train"
DATA_DIR="$ROOT/../data"
LOG_DIR="$ROOT/experiments/gsm_grpo/logs"
OUT_DIR="$ROOT/experiments/gsm_grpo/outputs/smoke"
TRAIN_FILE="$DATA_DIR/gsm_sample_train.jsonl"
EVAL_FILE="$DATA_DIR/gsm_sample_valid.jsonl"
TEST_FILE="$DATA_DIR/gsm_sample_test.jsonl"
SPLIT_SCRIPT="$ROOT/gsm/prepare_gsm_sample_splits.py"
SOURCE_JSON="${GSM_SOURCE_JSON:-}"
HF_DATASET="${GSM_HF_DATASET:-YMinglai/GSM-DC-Dataset-Sample}"
HF_DATA_FILE="${GSM_HF_DATA_FILE:-all_problems.json}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
export PYTHONPATH="$ROOT:$ROOT/gsm/script${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$EVAL_FILE" ]; then
  if [ -n "$SOURCE_JSON" ]; then
    if [ ! -f "$SOURCE_JSON" ]; then
      echo "[run_smoke] error=missing_source_json path=$SOURCE_JSON" >&2
      exit 1
    fi
    echo "[run_smoke] stage=prepare_gsm_sample_splits source=$SOURCE_JSON"
    python3 "$SPLIT_SCRIPT" --source_json "$SOURCE_JSON"
  else
    echo "[run_smoke] stage=prepare_gsm_sample_splits dataset=$HF_DATASET data_file=$HF_DATA_FILE"
    python3 "$SPLIT_SCRIPT" --dataset_name "$HF_DATASET" --data_file "$HF_DATA_FILE"
  fi
else
  echo "[run_smoke] stage=dataset_ready train=$TRAIN_FILE eval=$EVAL_FILE test=$TEST_FILE"
fi

exec accelerate launch \
  --num_processes 4 \
  "$ROOT/grpo_task.py" \
  --task gsm \
  --loss_type grpo \
  --model_name Qwen/Qwen3-4B \
  --train_file "$TRAIN_FILE" \
  --eval_file "$EVAL_FILE" \
  --output_dir "$OUT_DIR" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_generations 4 \
  --max_completion_length 2048 \
  --max_train_samples 8 \
  --max_eval_samples 8 \
  --max_steps 1 \
  --logging_steps 1 \
  --save_steps 1000 \
  --eval_steps 1 \
  --bf16 \
  --use_vllm \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 4 \
  >"$LOG_DIR/smoke.out" 2>"$LOG_DIR/smoke.err"
