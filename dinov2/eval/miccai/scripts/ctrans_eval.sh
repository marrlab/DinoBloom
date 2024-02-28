#!/bin/bash

#SBATCH -o slurm_outputs/ctrans_vits.txt
#SBATCH -e slurm_outputs/ctrans_error.txt
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
python dinov2/eval/miccai/general_patch_eval.py --model_path "/lustre/groups/shared/users/peng_marr/HistoDINO/logs/baseline_models/ctranspath/ctranspath.pth" --dataset_path /lustre/groups/labs/marr/qscd01/datasets/armingruber/_Domains/Acevedo_cropped/ --model_name ctranspath --experiment_name real_acevedo_ctranspath --num_workers 16 


