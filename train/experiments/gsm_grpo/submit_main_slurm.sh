#!/bin/bash

ROOT="/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train"
SCRIPT="$ROOT/experiments/gsm_grpo/run_main_slurm.sh"

mkdir -p "$ROOT/experiments/gsm_grpo/logs/slurm"
cd "$ROOT"

exec sh "$SCRIPT"
