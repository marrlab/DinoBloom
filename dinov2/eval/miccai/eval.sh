#!/bin/bash

#SBATCH -o slurm_outputs/eval_output.txt
#SBATCH -e slurm_outputs/eval_error.txt
#SBATCH -J feature_extraction
#SBATCH -p gpu_p
#SBATCH -c 20
#SBATCH --mem=150G
#SBATCH -q gpu_normal
#SBATCH --time=4:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
source $HOME/.bashrc
conda activate histo_env
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python "/lustre/groups/shared/histology_data/eval/evaluation.py"
