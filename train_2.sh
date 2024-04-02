#!/bin/bash

#SBATCH -o /home/icb/valentin.koch/dinov2/slurm_outputs/swiglu_run.txt
#SBATCH -e /home/icb/valentin.koch/dinov2/slurm_outputs/swiglu_error.txt
#SBATCH -J dino
#SBATCH -p gpu_p
#SBATCH -q gpu_long
#SBATCH -c 28
#SBATCH --mem=240G
#SBATCH --time=48:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:2
#SBATCH -C a100_80gb
# Environment setup
source $HOME/.bashrc
conda activate /home/icb/valentin.koch/anaconda3/envs/dino/envs/dinov2
cd /home/icb/valentin.koch/dinov2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

torchrun --nproc_per_node=2 /home/icb/valentin.koch/dinov2/dinov2/train/train.py --no-resume --name "vits_beluga" --output_dir "/lustre/groups/shared/users/peng_marr/DinoBloomv2" --config-file /home/icb/valentin.koch/dinov2/dinov2/configs/train/custom_2.yaml
