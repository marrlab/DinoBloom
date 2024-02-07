#!/bin/bash

#SBATCH -o eval_all.txt
#SBATCH -e eval_all_error.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH -c 18
#SBATCH --mem=240G
#SBATCH -q gpu_long
#SBATCH --time=15:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

# Environment setup
source $HOME/.bashrc

# choose the right directory (Valentin: /home/icb/valentin.koch/dinov2, Sophia: /home/haicu/sophia.wagner/projects/dinov2)    
cd /home/icb/valentin.koch/dinov2

# activate your own conda environment (Valentin: feature_ex, Sophia: vim)
conda activate feature_ex

# set checkpoint to evaluate as input 
python dinov2/eval/miccai/extract_and_eval.py --run_path /home/icb/valentin.koch/dinov2/debug/eval


