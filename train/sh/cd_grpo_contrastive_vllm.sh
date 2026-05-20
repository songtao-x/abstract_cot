#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_grpo_contrastive_vllm/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_grpo_contrastive_vllm/%j.err

#SBATCH --job-name=grpo_contrast_vllm
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=12:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


project_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
script="$project_root/grpo.py"

model_name="Qwen/Qwen3-4B"
train_file="$project_root/../data/cd4_train.jsonl"
eval_file="$project_root/../data/cd4_test.jsonl"
output_dir="$project_root/outputs/grpo_abstract_contrastive_vllm_n8"
num_processes=4

per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=2e-6
num_train_epochs=1
max_steps=180
num_generations=8
max_completion_length=2048
logging_steps=2
save_steps=10
save_total_limit=10
reward_beta=0.5
think_delta_weight=1.0
num_generations_eval=1
generation_batch_size=8
torch_empty_cache_steps=1
vllm_gpu_memory_utilization=0.25
use_vllm=1
vllm_mode="colocate"

# Contrastive CoT settings.
# NOTE: contrastive_max_tokens is left at 2048 per request. The big speedup
# comes from routing the no-plan rollouts through the trainer's vLLM engine
# (PlanAwareReward.bind_vllm_engine) instead of HF model.generate(), which
# under DeepSpeed ZeRO-3 all-gathers sharded params at every token-step.
contrastive_weight=0.3
contrastive_max_tokens=2048

use_bf16=1
use_fp16=0

wandb_project="${WANDB_PROJECT:-abstract_grpo_runs}"
wandb_entity="${WANDB_ENTITY:-}"
wandb_run_name="${WANDB_RUN_NAME:-}"

if [ -z "$model_name" ]; then
    echo "set model_name in run_grpo.sh before running this script"
    exit 1
fi

mkdir -p "$project_root/log/cd_grpo_contrastive_vllm"

echo "[run_grpo] project_root=$project_root"
echo "[run_grpo] model=$model_name train=$train_file eval=$eval_file"
echo "[run_grpo] stage=launch_accelerate num_processes=$num_processes"
echo "[run_grpo] wandb_project=$wandb_project"
echo "[run_grpo] mode=contrastive CoT via vLLM (weight=$contrastive_weight, max_tokens=$contrastive_max_tokens)"

if [ "$use_vllm" = "1" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
    echo "[run_grpo] unset PYTORCH_CUDA_ALLOC_CONF for vLLM compatibility"
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    echo "[run_grpo] PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"
fi

# Keep the bumped NCCL timeout — the plan-attention path still does ZeRO-3
# all-gathers and the original 600 s default has tripped the watchdog before.
export NCCL_TIMEOUT=1800
export TORCH_NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo "[run_grpo] NCCL_TIMEOUT=$NCCL_TIMEOUT s"

set -- \
    accelerate launch \
    --num_processes "$num_processes" \
    "$script" \
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
    --reward_beta "$reward_beta" \
    --think_delta_weight "$think_delta_weight" \
    --wandb_project "$wandb_project" \
    --torch_empty_cache_steps "$torch_empty_cache_steps" \
    --vllm_gpu_memory_utilization "$vllm_gpu_memory_utilization" \
    --vllm_mode "$vllm_mode" \
    --contrastive_cot \
    --contrastive_weight "$contrastive_weight" \
    --contrastive_max_tokens "$contrastive_max_tokens" \
    --no_eval

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
