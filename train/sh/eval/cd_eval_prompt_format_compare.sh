#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_eval_prompt_format_compare/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_eval_prompt_format_compare/%j.err

#SBATCH --job-name=cd_eval_pfmt
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=04:00:00
#SBATCH --mail-user=songtao2@ualberta.ca

train_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
abstract_root="$train_root/.."
script="$abstract_root/eval/eval_prompt_format_compare.py"

checkpoint_dir="$train_root/outputs/grpo_abstract_contrastive/checkpoint-80"
eval_file="$abstract_root/data/cd4_eval.jsonl"
max_samples=100
max_new_tokens=512

mkdir -p "$train_root/log/cd_eval_prompt_format_compare"

echo "[cd_eval_pfmt] abstract_root=$abstract_root"
echo "[cd_eval_pfmt] script=$script"
echo "[cd_eval_pfmt] checkpoint_dir=$checkpoint_dir"
echo "[cd_eval_pfmt] eval_file=$eval_file"
echo "[cd_eval_pfmt] max_samples=$max_samples"
echo "[cd_eval_pfmt] max_new_tokens=$max_new_tokens"

export PYTHONPATH="$abstract_root${PYTHONPATH:+:$PYTHONPATH}"

exec python "$script" \
    --checkpoint_dir "$checkpoint_dir" \
    --eval_file "$eval_file" \
    --max_samples "$max_samples" \
    --max_new_tokens "$max_new_tokens"
