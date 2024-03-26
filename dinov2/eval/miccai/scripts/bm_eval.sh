#!/bin/bash

#SBATCH -o slurm_outputs/bm_log.txt
#SBATCH -e slurm_outputs/bm_error.txt
#SBATCH -J bm_ctrans
#SBATCH -p gpu_p
#SBATCH -c 18
#SBATCH --mem=160G
#SBATCH -q gpu_normal
#SBATCH --time=8:00:00
#SBATCH --nice=1
#SBATCH --gres=gpu:1

# Environment setup
source $HOME/.bashrc

# choose the right directory (Valentin: /home/icb/valentin.koch/dinov2, Sophia: /home/haicu/sophia.wagner/projects/dinov2)    
cd /home/icb/valentin.koch/dinov2

# activate your own conda environment (Valentin: feature_ex, Sophia: vim)
conda activate feature_ex

# set checkpoint to evaluate as input 
python dinov2/eval/miccai/general_fixed_split_patch_eval.py --model_path "/home/icb/valentin.koch/dinov2/vitl+cerv/eval/training_4999/" --model_name dinov2_vitl14 --experiment_name vitl_cerv --run_name bonemarrow --image_path_train /home/icb/valentin.koch/dinov2/dinov2/eval/miccai/splits/bm_train.csv --image_path_test /home/icb/valentin.koch/dinov2/dinov2/eval/miccai/splits/bm_test.csv


