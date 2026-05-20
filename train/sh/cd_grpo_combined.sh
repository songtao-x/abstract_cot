#!/bin/bash

# Combined recipe:
#   reward = base_reward_weight * base
#          + think_delta_weight * (think_with - think_without)
#          + contrastive_weight * (base - no_plan_base)
# (reward_beta=0 → answer_delta term disabled; the answer_only ablation showed
# it collapses think_chars on its own.)
#
# Base is highlighted (base_reward_weight=2.0) because the prior ablations show
# the auxiliary terms grow much larger than the [0,1] base reward and end up
# dominating the GRPO advantage. With the bump, base ∈ {0,2} and the auxiliary
# terms (typically |Δ| ≲ 1) stay subordinate.

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_grpo_combined/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_grpo_combined/%j.err

#SBATCH --job-name=grpo_combined
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=18:00:00
#SBATCH --mail-user=songtao2@ualberta.ca


project_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
script="$project_root/grpo.py"

model_name="Qwen/Qwen3-4B"
train_file="$project_root/../data/cd4_train.jsonl"
eval_file="$project_root/../data/cd4_test.jsonl"
output_dir="$project_root/outputs/grpo_abstract_combined"
num_processes=4

per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=2e-6
num_train_epochs=1
max_steps=180

# num_generations=8 doubles per-step rollouts vs the 4-gen ablations. To keep
# loss-side activation memory in check, vllm_gpu_memory_utilization is lowered
# from 0.30 → 0.25, leaving the training side ~5% more headroom.
num_generations=8
generation_batch_size=8
max_completion_length=2048

logging_steps=2
save_steps=10
save_total_limit=10

# --- reward composition ---
base_reward_weight=2.0
think_delta_weight=0.3
contrastive_weight=0.3
reward_beta=0.0   # answer_delta disabled

num_generations_eval=1
torch_empty_cache_steps=1
vllm_gpu_memory_utilization=0.25
use_vllm=1
vllm_mode="colocate"

# Contrastive CoT settings.
contrastive_max_tokens=2048

use_bf16=1
use_fp16=0

wandb_project="${WANDB_PROJECT:-abstract_grpo_runs}"
wandb_entity="${WANDB_ENTITY:-}"
wandb_run_name="${WANDB_RUN_NAME:-}"

mkdir -p "$project_root/log/cd_grpo_combined"

echo "[run_grpo] project_root=$project_root"
echo "[run_grpo] model=$model_name train=$train_file eval=$eval_file"
echo "[run_grpo] mode=combined  base_w=$base_reward_weight  think_w=$think_delta_weight  contrastive_w=$contrastive_weight"
echo "[run_grpo] num_generations=$num_generations  vllm_gpu_mem=$vllm_gpu_memory_utilization"

if [ "$use_vllm" = "1" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

export NCCL_TIMEOUT=1800
export TORCH_NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

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
    --base_reward_weight "$base_reward_weight" \
    --think_delta_weight "$think_delta_weight" \
    --reward_beta "$reward_beta" \
    --wandb_project "$wandb_project" \
    --torch_empty_cache_steps "$torch_empty_cache_steps" \
    --vllm_gpu_memory_utilization "$vllm_gpu_memory_utilization" \
    --vllm_mode "$vllm_mode" \
    --contrastive_cot \
    --contrastive_weight "$contrastive_weight" \
    --contrastive_max_tokens "$contrastive_max_tokens" \
    --no_eval

[ -n "$wandb_entity" ]    && set -- "$@" --wandb_entity "$wandb_entity"
[ -n "$wandb_run_name" ]  && set -- "$@" --wandb_run_name "$wandb_run_name"
[ "$use_bf16" = "1" ]     && set -- "$@" --bf16
[ "$use_fp16" = "1" ]     && set -- "$@" --fp16
[ "$use_vllm" = "1" ]     && set -- "$@" --use_vllm

exec "$@"
