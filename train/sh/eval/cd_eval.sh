#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_grpo_eval_no_think/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_grpo_eval_no_think/%j.err

#SBATCH --job-name=cd_eval
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=18:00:00
#SBATCH --mail-user=songtao2@ualberta.ca

train_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
abstract_root="$train_root/.."
script="$abstract_root/eval/evaluate_countdown_run.py"

run_dir="$train_root/outputs/grpo_abstract_no_think"
checkpoint_dir=""
eval_file="$abstract_root/data/cd4_eval.jsonl"
summary_file=""
max_samples=""
max_new_tokens=2048

echo "[cd_eval] abstract_root=$abstract_root"
echo "[cd_eval] script=$script"
echo "[cd_eval] run_dir=$run_dir"
echo "[cd_eval] eval_file=$eval_file"
echo "[cd_eval] max_new_tokens=$max_new_tokens"

if [ -n "$checkpoint_dir" ]; then
    echo "[cd_eval] checkpoint_dir=$checkpoint_dir"
fi

if [ -n "$summary_file" ]; then
    echo "[cd_eval] summary_file=$summary_file"
fi

if [ -n "$max_samples" ]; then
    echo "[cd_eval] max_samples=$max_samples"
fi

export PYTHONPATH="$abstract_root${PYTHONPATH:+:$PYTHONPATH}"

set -- \
    python "$script" \
    --run_dir "$run_dir" \
    --eval_file "$eval_file" \
    --max_new_tokens "$max_new_tokens"

if [ -n "$checkpoint_dir" ]; then
    set -- "$@" --checkpoint_dir "$checkpoint_dir"
fi

if [ -n "$summary_file" ]; then
    set -- "$@" --summary_file "$summary_file"
fi

if [ -n "$max_samples" ]; then
    set -- "$@" --max_samples "$max_samples"
fi

exec "$@"
