#!/bin/bash

#SBATCH -o slurm_outputs/eval_log.txt
#SBATCH -e slurm_outputs/eval_error.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH -c 18
#SBATCH --mem=150G
#SBATCH -q gpu_normal
#SBATCH --time=5:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

# Environment setup
source $HOME/.bashrc

# activate your own conda environment
conda activate feature_ex

# set checkpoint to evaluate as input 
python dinov2/eval/miccai/general_patch_eval.py --model_name dinov2_vits14 --experiment_name acevedo


