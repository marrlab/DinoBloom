#!/bin/bash

#SBATCH -o slurm_outputs/raabin_log.txt
#SBATCH -e slurm_outputs/raabin_error.txt
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
python dinov2/eval/miccai/general_fixed_split_patch_eval.py --model_path "/home/icb/valentin.koch/dinov2/vits_ablation_baseline-local_crop/eval"  --evaluate_untrained_baseline --model_name dinov2_vits14 --experiment_name vits14_split --run_name cerv_cancer --image_path_train /home/icb/valentin.koch/dinov2/dinov2/eval/miccai/splits/cerv_cancer_train.csv --image_path_test /home/icb/valentin.koch/dinov2/dinov2/eval/miccai/splits/cerv_cancer_test.csv


