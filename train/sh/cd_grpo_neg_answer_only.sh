#!/bin/bash

# Reward = base_reward − reward_beta * (answer_with − answer_without).
#   think_delta_weight=0   (no think-delta term)
#   reward_beta=-0.5       (answer-delta sign flipped vs. answer_only)
#   contrastive_cot off
# Negative-sign counterpart of cd_grpo_answer_only.sh.
#
# GRPO twin of cd_dapo_neg_answer_only.sh: same data / model / reward /
# rollout settings (max_completion_length=3096, generation_batch_size=16) and
# the same torch.compile cache isolation, but plain GRPO — no DAPO knobs, and
# GRPO's own lr=2e-6 / max_steps=180 baseline (DAPO deliberately differs).

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_grpo_neg_answer_only/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/cd_grpo_neg_answer_only/%j.err

#SBATCH --job-name=grpo_neg_answer
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=7:50:00
#SBATCH --mail-user=songtao2@ualberta.ca


project_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
script="$project_root/grpo.py"

model_name="Qwen/Qwen3-4B"
train_file="$project_root/../data/cd4_train.jsonl"
eval_file="$project_root/../data/cd4_test.jsonl"
output_dir="$project_root/outputs/grpo_abstract_neg_answer_only"
num_processes=4

per_device_train_batch_size=1
gradient_accumulation_steps=8
learning_rate=2e-6
num_train_epochs=1
max_steps=180
num_generations=8
max_completion_length=3096   # matched to cd_dapo_neg_answer_only.sh
logging_steps=2
save_steps=10
save_total_limit=10

# --- reward composition ---
think_delta_weight=0.0
reward_beta=-0.5

num_generations_eval=1
generation_batch_size=16     # matched to cd_dapo_neg_answer_only.sh (16/8=2 prompts/round)
torch_empty_cache_steps=1
vllm_gpu_memory_utilization=0.25
use_vllm=1
vllm_mode="colocate"

use_bf16=1
use_fp16=0

wandb_project="${WANDB_PROJECT:-abstract_grpo_runs}"
wandb_entity="${WANDB_ENTITY:-}"
wandb_run_name="${WANDB_RUN_NAME:-}"

mkdir -p "$project_root/log/cd_grpo_neg_answer_only"

echo "[run_grpo] project_root=$project_root"
echo "[run_grpo] model=$model_name train=$train_file eval=$eval_file"
echo "[run_grpo] mode=neg-answer-only (think_delta_weight=$think_delta_weight, reward_beta=$reward_beta)"

if [ "$use_vllm" = "1" ]; then
    unset PYTORCH_CUDA_ALLOC_CONF
else
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
fi

export NCCL_TIMEOUT=1800
export TORCH_NCCL_TIMEOUT=1800
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- torch.compile cache isolation ---------------------------------------
# Same fix as cd_dapo_neg_answer_only.sh. vLLM's torch.compile path crashed
# the DAPO twin two different ways, both rooted in caches on shared/NFS:
#   1) inductor C++ codecache flock() → OSError [Errno 116] Stale file handle
#   2) vLLM graph save → standalone_compile.py:74 assert (aot_autograd: [])
# Pin all compile caches to node-local scratch (SLURM_TMPDIR is per-job local
# disk, never NFS), disable the persistent inductor fx-graph cache, and
# disable vLLM's compile-cache so it skips the failing save/load entirely
# (model is still compiled in-process — only graph serialization is skipped).
compile_cache_root="${SLURM_TMPDIR:-/tmp/$USER}/torch_compile_cache.${SLURM_JOB_ID:-$$}"
mkdir -p "$compile_cache_root/inductor" "$compile_cache_root/triton" "$compile_cache_root/vllm"
export TORCHINDUCTOR_CACHE_DIR="$compile_cache_root/inductor"
export TRITON_CACHE_DIR="$compile_cache_root/triton"
export TORCHINDUCTOR_FX_GRAPH_CACHE=0
export TORCHINDUCTOR_AUTOGRAD_CACHE=0
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT="$compile_cache_root/vllm"
echo "[run_grpo] torch compile cache → $compile_cache_root (vLLM compile-cache disabled)"

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
    --no_eval

[ -n "$wandb_entity" ]    && set -- "$@" --wandb_entity "$wandb_entity"
[ -n "$wandb_run_name" ]  && set -- "$@" --wandb_run_name "$wandb_run_name"
[ "$use_bf16" = "1" ]     && set -- "$@" --bf16
[ "$use_fp16" = "1" ]     && set -- "$@" --fp16
[ "$use_vllm" = "1" ]     && set -- "$@" --use_vllm

exec "$@"
