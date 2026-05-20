#!/bin/bash

# DAPO + answer-only reward.
#   Reward: base_reward − reward_beta * (answer_with − answer_without)
#     - think_delta_weight=0, reward_beta=0.5, contrastive off  (same as cd_grpo_answer_only)
#   DAPO knobs (vs. plain GRPO):
#     - loss_type=dapo                  → token-level loss
#     - epsilon=0.2, epsilon_high=0.28  → clip-higher: asymmetric ratio bound
#     - top_entropy_quantile=0.2        → train only on the top-20% highest-entropy tokens
#     - mask_truncated_completions      → drop truncated rollouts from the loss
#     - scale_rewards=group             → standardize rewards per group
#   Group size bumped (num_generations=16) so the entropy filter has enough signal.

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_dapo_answer_only/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_dapo_answer_only/%j.err

#SBATCH --job-name=dapo_answer
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
output_dir="$project_root/outputs/dapo_abstract_answer_only_n16"
num_processes=4

# --- batch / optimization ---
per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=1e-6           # halved vs GRPO 2e-6: clip-higher widens effective updates
num_train_epochs=1
max_steps=240                # +33% vs GRPO 180 (DAPO converges slower per step but stabler)
num_generations=8           # bumped from 8 for top_entropy_quantile=0.2 signal
max_completion_length=2048
logging_steps=2
save_steps=20
save_total_limit=12

# --- reward composition (answer-only, identical to cd_grpo_answer_only) ---
think_delta_weight=0.0
reward_beta=0.5

# --- DAPO knobs ---
loss_type="dapo"
epsilon=0.2
epsilon_high=0.28
top_entropy_quantile=0.2
scale_rewards="group"

# --- generation / vLLM ---
num_generations_eval=1
generation_batch_size=16
torch_empty_cache_steps=1
vllm_gpu_memory_utilization=0.25
use_vllm=1
vllm_mode="colocate"

use_bf16=1
use_fp16=0

wandb_project="${WANDB_PROJECT:-abstract_dapo_runs}"
wandb_entity="${WANDB_ENTITY:-}"
wandb_run_name="${WANDB_RUN_NAME:-}"

mkdir -p "$project_root/log/cd_dapo_answer_only"

echo "[run_dapo] project_root=$project_root"
echo "[run_dapo] model=$model_name train=$train_file eval=$eval_file"
echo "[run_dapo] mode=answer-only  loss=$loss_type  eps=$epsilon/$epsilon_high  q=$top_entropy_quantile  G=$num_generations"

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
    --reward_beta "$reward_beta" \
    --think_delta_weight "$think_delta_weight" \
    --loss_type "$loss_type" \
    --epsilon "$epsilon" \
    --epsilon_high "$epsilon_high" \
    --top_entropy_quantile "$top_entropy_quantile" \
    --scale_rewards "$scale_rewards" \
    --mask_truncated_completions \
    --wandb_project "$wandb_project" \
    --torch_empty_cache_steps "$torch_empty_cache_steps" \
    --vllm_gpu_memory_utilization "$vllm_gpu_memory_utilization" \
    --vllm_mode "$vllm_mode" \
    --no_eval

[ -n "$wandb_entity" ]    && set -- "$@" --wandb_entity "$wandb_entity"
[ -n "$wandb_run_name" ]  && set -- "$@" --wandb_run_name "$wandb_run_name"
[ "$use_bf16" = "1" ]     && set -- "$@" --bf16
[ "$use_fp16" = "1" ]     && set -- "$@" --fp16
[ "$use_vllm" = "1" ]     && set -- "$@" --use_vllm

exec "$@"
