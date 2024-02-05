#!/bin/bash

#SBATCH -o /lustre/groups/shared/users/peng_marr/HistoDINO/logs/slurm/feature_run2.txt
#SBATCH -e /lustre/groups/shared/users/peng_marr/HistoDINO/logs/slurm/feature_error2.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH -c 18
#SBATCH --mem=240G
#SBATCH -q gpu_long
#SBATCH --time=5:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

# Environment setup
source $HOME/.bashrc

# choose the right directory (Valentin: /home/icb/valentin.koch/dinov2, Sophia: /home/haicu/sophia.wagner/projects/dinov2)    
cd /home/haicu/sophia.wagner/projects/dinov2

# activate your own conda environment (Valentin: feature_ex, Sophia: vim)
conda activate vim

# set checkpoint to evaluate as input 
python dinov2/eval/miccai/extract_patch_features.py --model_name vim_finetuned --checkpoint $1

# activate your own conda environment (Valentin: histo_env, Sophia: eval)
conda activate eval

python dinov2/eval/miccai/evaluation.py --model_name vim_finetuned --checkpoint $1


