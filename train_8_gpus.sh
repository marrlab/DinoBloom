#!/bin/bash

#SBATCH -o run.txt
#SBATCH -e error.txt
#SBATCH -J dino
#SBATCH --reservation=test_supergpu05
#SBATCH --qos gpu_reservation
#SBATCH -p gpu_p
#SBATCH -c 126
#SBATCH --mem=1800G
#SBATCH --time=96:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:8
# Environment setup
source $HOME/.bashrc
conda activate dinov2

export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

torchrun --nproc_per_node=8 dinov2/dinov2/train/train.py --name "vitb_f1" --output_dir "" --config-file dinov2/dinov2/configs/train/custom.yaml
