#!/bin/bash

#SBATCH -o extract_features_slide_tasks_out.txt
#SBATCH -e extract_features_slide_tasks_err.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH --time=15:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpusrv27

#SBATCH -c 20
#SBATCH --mem=160G
#SBATCH -q gpu_normal

# Environment setup
source $HOME/.bashrc

# choose the right directory
# Valentin: /home/icb/valentin.koch/dinov2 
# Sophia: /home/haicu/sophia.wagner/projects/dinov2
cd /home/haicu/sophia.wagner/projects/dinov2

# activate your own conda environment 
# Valentin: feature_ex
# Sophia: vim
source /home/haicu/sophia.wagner/miniforge3/bin/activate dinov2

# set checkpoint for feature extraction as input 
python dinov2/eval/miccai/extract_slide_features.py --model $1 --run $2 --dataset


