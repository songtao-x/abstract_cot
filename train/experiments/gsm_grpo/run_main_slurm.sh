#!/bin/bash

#SBATCH --job-name=gsm_grpo_main
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/experiments/gsm_grpo/logs/slurm/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/experiments/gsm_grpo/logs/slurm/%j.err

ROOT="/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train"

cd "$ROOT"
mkdir -p \
  "$ROOT/experiments/gsm_grpo/logs" \
  "$ROOT/experiments/gsm_grpo/logs/slurm" \
  "$ROOT/experiments/gsm_grpo/outputs/main"

bash "$ROOT/experiments/gsm_grpo/run_main.sh"
