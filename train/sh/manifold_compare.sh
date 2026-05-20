#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/manifold_compare/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/log/manifold_compare/%j.err

#SBATCH --job-name=manifold_compare
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --cpus-per-task=10
#SBATCH --time=06:00:00
#SBATCH --mail-type=end
#SBATCH --mail-user=songtao2@ualberta.ca


project_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract
script="$project_root/analysis/manifold_compare.py"

base_model="Qwen/Qwen3-4B"
trained_model="$project_root/train/outputs/grpo_abstract_contrastive_vllm_n8"

data_file="$project_root/data/cd4_test.jsonl"
out_root="$project_root/train/outputs/manifolds"
run_tag="${RUN_TAG:-base_vs_contrastive_vllm_n8}"

n_samples="${N_SAMPLES:-256}"
max_new_tokens="${MAX_NEW_TOKENS:-1024}"
max_seq_len="${MAX_SEQ_LEN:-4096}"
pca_components="${PCA_COMPONENTS:-50}"

# Paper-style shared-geometry / isometry analysis (arXiv:2605.05115):
# M_h = input-span, M_y = plan-span <abstract>, concept axis = countdown target.
# Set GEOM_CORR=0 to skip it.
geom_corr="${GEOM_CORR:-1}"
geom_pca_dim="${GEOM_PCA_DIM:-64}"
geom_min_count="${GEOM_MIN_COUNT:-3}"
geom_dense="${GEOM_DENSE:-2000}"

# vLLM knobs (used by phase=all and phase=generate)
phase="${PHASE:-all}"
vllm_tp="${VLLM_TP:-4}"
vllm_gpu_mem="${VLLM_GPU_MEM:-0.70}"
vllm_max_len="${VLLM_MAX_LEN:-4096}"
vllm_dtype="${VLLM_DTYPE:-bfloat16}"
vllm_temperature="${VLLM_TEMPERATURE:-0.6}"
vllm_top_p="${VLLM_TOP_P:-0.95}"
vllm_seed="${VLLM_SEED:-0}"

# vLLM does not like expandable_segments; unset to match the training scripts.
unset PYTORCH_CUDA_ALLOC_CONF

echo "[manifold] project_root=$project_root"
echo "[manifold] models = $base_model | $trained_model"
echo "[manifold] data=$data_file  out_root=$out_root  run_tag=$run_tag"
echo "[manifold] n_samples=$n_samples  max_new_tokens=$max_new_tokens  phase=$phase"
echo "[manifold] vllm: tp=$vllm_tp  gpu_mem=$vllm_gpu_mem  max_len=$vllm_max_len  dtype=$vllm_dtype"
echo "[manifold] sampling: temperature=$vllm_temperature  top_p=$vllm_top_p  seed=$vllm_seed"
echo "[manifold] geom_corr=$geom_corr  pca_dim=$geom_pca_dim  min_count=$geom_min_count  dense=$geom_dense"

set -- \
    python -u "$script" \
    --models "$base_model" "$trained_model" \
    --data_file "$data_file" \
    --out_root "$out_root" \
    --run_tag "$run_tag" \
    --n_samples "$n_samples" \
    --max_new_tokens "$max_new_tokens" \
    --max_seq_len "$max_seq_len" \
    --pca_components "$pca_components" \
    --phase "$phase" \
    --vllm_tensor_parallel "$vllm_tp" \
    --vllm_gpu_memory_utilization "$vllm_gpu_mem" \
    --vllm_max_model_len "$vllm_max_len" \
    --vllm_dtype "$vllm_dtype" \
    --vllm_temperature "$vllm_temperature" \
    --vllm_top_p "$vllm_top_p" \
    --vllm_seed "$vllm_seed"

if [ "${USE_UMAP:-0}" = "1" ]; then
    set -- "$@" --umap
fi
if [ "${USE_TSNE:-0}" = "1" ]; then
    set -- "$@" --tsne
fi
# geom params are always forwarded (defaults match argparse, so harmless) so
# that PHASE=geom honours GEOM_* overrides even when GEOM_CORR=0. The
# --geom_corr toggle (which turns the analysis on during extract/all) is added
# only when requested; phase=geom runs the analysis regardless.
set -- "$@" \
    --geom_pca_dim "$geom_pca_dim" \
    --geom_min_count "$geom_min_count" \
    --geom_dense "$geom_dense"
if [ "$geom_corr" = "1" ]; then
    set -- "$@" --geom_corr
fi

exec "$@"
