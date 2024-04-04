#!/bin/bash

#SBATCH -o slurm_outputs/bm_log.txt
#SBATCH -e slurm_outputs/bm_error.txt
#SBATCH -J bm_ctrans
#SBATCH -p gpu_p
#SBATCH -c 18
#SBATCH --mem=160G
#SBATCH -q gpu_normal
#SBATCH --time=8:00:00
#SBATCH --nice=1
#SBATCH --gres=gpu:1

# Environment setup
source $HOME/.bashrc

# activate your own conda environment
conda activate feature_ex

# set checkpoint to evaluate as input 
python dinov2/eval/miccai/general_fixed_split_patch_eval.py --model_path "" --model_name dinov2_vitl14 --experiment_name vitl --run_name bonemarrow


