#!/bin/bash

#SBATCH --output=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/sh/log/gsm_dc_baseline/%j.out
#SBATCH --error=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train/sh/log/gsm_dc_baseline/%j.err

#SBATCH --job-name=gsm_baseline
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --time=16:00:00
#SBATCH --mail-user=songtao2@ualberta.ca

project_root=/home/songtaow/projects/aip-qchen/songtaow/reward_hack/abstract/train
gsm_dc_root="$project_root/gsm/gsm_dc"

model_path="Qwen/Qwen3-4B"
dataset_path="YMinglai/GSM-DC-Dataset-Sample"
op_values="16,17,18,19,20,21,22"
nshots=5
prm_model_name=""

if [ -z "$model_path" ]; then
    echo "set model_path in gsm_dc_baseline.sh before running this script"
    exit 1
fi

echo "[gsm_dc_baseline] root=$gsm_dc_root"
echo "[gsm_dc_baseline] model=$model_path dataset=$dataset_path op_values=$op_values nshots=$nshots"

export GSM_DC_MODEL_PATH="$model_path"
export GSM_DC_DATASET_PATH="$dataset_path"
export GSM_DC_OP_VALUES="$op_values"
export GSM_DC_NSHOTS="$nshots"
if [ -n "$prm_model_name" ]; then
    export GSM_DC_PRM_MODEL_NAME="$prm_model_name"
fi

cd "$gsm_dc_root" || exit 1
exec python evaluate.py
