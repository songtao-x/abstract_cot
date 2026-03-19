#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/gsm_grpo_pure_eval/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/gsm_grpo_pure_eval/%j.err

#SBATCH --job-name=gsm_grpo_pure_eval
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=12:00:00
#SBATCH --mail-user=songtao2@ualberta.ca

train_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
abstract_root="$train_root/.."
script="$abstract_root/eval/evaluate_gsm_run.py"

run_dir="$train_root/outputs/grpo_gsm_pure"
checkpoint_dir=""
eval_file="$abstract_root/data/gsm_sample_test.jsonl"
summary_file=""
max_samples=200
max_new_tokens=2048

echo "[gsm_grpo_pure_eval] abstract_root=$abstract_root"
echo "[gsm_grpo_pure_eval] script=$script"
echo "[gsm_grpo_pure_eval] run_dir=$run_dir"
echo "[gsm_grpo_pure_eval] eval_file=$eval_file"
echo "[gsm_grpo_pure_eval] max_samples=$max_samples"
echo "[gsm_grpo_pure_eval] max_new_tokens=$max_new_tokens"

if [ -n "$checkpoint_dir" ]; then
    echo "[gsm_grpo_pure_eval] checkpoint_dir=$checkpoint_dir"
fi

if [ -n "$summary_file" ]; then
    echo "[gsm_grpo_pure_eval] summary_file=$summary_file"
fi

export PYTHONPATH="$abstract_root${PYTHONPATH:+:$PYTHONPATH}"

set -- \
    python "$script" \
    --run_dir "$run_dir" \
    --eval_file "$eval_file" \
    --max_samples "$max_samples" \
    --max_new_tokens "$max_new_tokens"

if [ -n "$checkpoint_dir" ]; then
    set -- "$@" --checkpoint_dir "$checkpoint_dir"
fi

if [ -n "$summary_file" ]; then
    set -- "$@" --summary_file "$summary_file"
fi

exec "$@"
