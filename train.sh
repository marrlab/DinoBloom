#!/bin/bash

#SBATCH -o run.txt
#SBATCH -e error.txt
#SBATCH -J dino
#SBATCH -p gpu_p
#SBATCH -c 28
#SBATCH --mem=240G
#SBATCH -q gpu_long
#SBATCH --time=96:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:2
#SBATCH -C a100_80gb
# Environment setup
source $HOME/.bashrc
conda activate dinov2

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

python /dinov2/dinov2/train/train.py --no-resume --config-file /dinov2/dinov2/configs/train/custom.yaml
