#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/sh/log/gsm_grpo_pure/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/sh/log/gsm_grpo_pure/%j.err

#SBATCH --job-name=gsm_grpo_pure
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=16:00:00
#SBATCH --mail-user=songtao2@ualberta.ca

project_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
script="$project_root/grpo_task.py"

model_name="Qwen/Qwen3-4B"
train_file="$project_root/../data/gsm_sample_train.jsonl"
eval_file="$project_root/../data/gsm_sample_valid.jsonl"
output_dir="$project_root/outputs/grpo_gsm_pure"
num_processes=4

per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=2e-6
num_train_epochs=1
max_steps=-1
num_generations=4
num_generations_eval=1
max_completion_length=2048
generation_batch_size=4
logging_steps=2
save_steps=10
save_total_limit=15
eval_steps=10
torch_empty_cache_steps=1

use_vllm=1
vllm_mode="colocate"
vllm_gpu_memory_utilization=0.5
vllm_tensor_parallel_size=4

use_bf16=1
use_fp16=0

wandb_project="${WANDB_PROJECT:-abstract_grpo_runs}"
wandb_entity="${WANDB_ENTITY:-}"
wandb_run_name="${WANDB_RUN_NAME:-}"

if [ -z "$model_name" ]; then
    echo "set model_name in gsm_grpo_pure.sh before running this script"
    exit 1
fi

echo "[gsm_grpo_pure] project_root=$project_root"
echo "[gsm_grpo_pure] model=$model_name train=$train_file eval=$eval_file"
echo "[gsm_grpo_pure] stage=launch_accelerate num_processes=$num_processes"
echo "[gsm_grpo_pure] wandb_project=$wandb_project"

if [ "$use_vllm" = "1" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
    echo "[gsm_grpo_pure] unset PYTORCH_CUDA_ALLOC_CONF for vLLM compatibility"
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "[gsm_grpo_pure] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
fi

set -- \
    accelerate launch \
    --num_processes "$num_processes" \
    "$script" \
    --task gsm \
    --loss_type grpo \
    --reward_variant pure \
    --model_name "$model_name" \
    --train_file "$train_file" \
    --eval_file "$eval_file" \
    --output_dir "$output_dir" \
    --per_device_train_batch_size "$per_device_train_batch_size" \
    --gradient_accumulation_steps "$gradient_accumulation_steps" \
    --learning_rate "$learning_rate" \
    --num_train_epochs "$num_train_epochs" \
    --max_steps "$max_steps" \
    --num_generations "$num_generations" \
    --num_generations_eval "$num_generations_eval" \
    --max_completion_length "$max_completion_length" \
    --generation_batch_size "$generation_batch_size" \
    --logging_steps "$logging_steps" \
    --save_steps "$save_steps" \
    --save_total_limit "$save_total_limit" \
    --eval_steps "$eval_steps" \
    --wandb_project "$wandb_project" \
    --torch_empty_cache_steps "$torch_empty_cache_steps" \
    --vllm_mode "$vllm_mode" \
    --vllm_gpu_memory_utilization "$vllm_gpu_memory_utilization" \
    --vllm_tensor_parallel_size "$vllm_tensor_parallel_size"

if [ -n "$wandb_entity" ]; then
    set -- "$@" --wandb_entity "$wandb_entity"
fi

if [ -n "$wandb_run_name" ]; then
    set -- "$@" --wandb_run_name "$wandb_run_name"
fi

if [ "$use_bf16" = "1" ]; then
    set -- "$@" --bf16
fi

if [ "$use_fp16" = "1" ]; then
    set -- "$@" --fp16
fi

if [ "$use_vllm" = "1" ]; then
    set -- "$@" --use_vllm
fi

if [ "$use_vllm_sleep_mode" = "1" ]; then
    set -- "$@" --vllm_enable_sleep_mode
fi

exec "$@"
