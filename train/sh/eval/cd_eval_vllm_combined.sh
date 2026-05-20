#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_eval_vllm_combined/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_eval_vllm_combined/%j.err

#SBATCH --job-name=cd_eval_vllm_combined
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=12:00:00
#SBATCH --mail-user=songtao2@ualberta.ca

run_name=grpo_abstract_combined

train_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
abstract_root="$train_root/.."
script="$abstract_root/eval/evaluate_countdown_run_vllm.py"
run_dir="$train_root/outputs/$run_name"
eval_file="$abstract_root/data/cd4_eval.jsonl"
max_samples=200
max_new_tokens=2048
gpu_memory_utilization=0.9
max_model_len=4096

mkdir -p "$train_root/log/cd_eval_vllm_combined"

echo "[cd_eval_vllm] run_name=$run_name run_dir=$run_dir"
echo "[cd_eval_vllm] eval_file=$eval_file max_samples=$max_samples max_new_tokens=$max_new_tokens"

unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="$abstract_root${PYTHONPATH:+:$PYTHONPATH}"

exec python "$script" \
    --run_dir "$run_dir" \
    --eval_file "$eval_file" \
    --max_samples "$max_samples" \
    --max_new_tokens "$max_new_tokens" \
    --gpu_memory_utilization "$gpu_memory_utilization" \
    --max_model_len "$max_model_len" \
    --tensor_parallel_size 1 \
    --skip_existing
