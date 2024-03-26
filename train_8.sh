#!/bin/bash

#SBATCH -o /home/icb/valentin.koch/dinov2/slurm_outputs/swiglu_run.txt
#SBATCH -e /home/icb/valentin.koch/dinov2/slurm_outputs/swiglu_error.txt
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
conda activate /home/icb/valentin.koch/anaconda3/envs/dino/envs/dinov2
cd /home/icb/valentin.koch/dinov2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

torchrun --nproc_per_node=8 /home/icb/valentin.koch/dinov2/dinov2/train/train.py --name "vitb_f1" --output_dir "" --config-file /home/icb/valentin.koch/dinov2/dinov2/configs/train/custom_8.yaml
