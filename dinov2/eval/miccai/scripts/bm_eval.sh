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

# choose the right directory (Valentin: /home/icb/valentin.koch/dinov2, Sophia: /home/haicu/sophia.wagner/projects/dinov2)    
cd /home/icb/valentin.koch/dinov2

# activate your own conda environment (Valentin: feature_ex, Sophia: vim)
conda activate feature_ex

# set checkpoint to evaluate as input 
python dinov2/eval/miccai/general_patch_eval.py --model_path "/home/icb/valentin.koch/dinov2/vits_8nodes_hema_no_local_new_aug/eval" --dataset_path /lustre/groups/shared/histology_data/BM_cytomorphology_data --model_name dinov2_vits14 --experiment_name real_acevedo_vits14_no_local_crops_new_aug --num_workers 16


