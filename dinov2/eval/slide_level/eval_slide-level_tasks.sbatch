#!/bin/bash

#SBATCH -o extract_features_slide_tasks_out.txt
#SBATCH -e extract_features_slide_tasks_err.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH --time=15:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

#SBATCH -c 20
#SBATCH --mem=160G
#SBATCH -q gpu_long

# Environment setup
source $HOME/.bashrc

# choose the right directory
# Valentin: /home/icb/valentin.koch/dinov2 
# Sophia: /home/haicu/sophia.wagner/projects/dinov2
cd /home/haicu/sophia.wagner/projects/dinov2

# activate your own conda environment 
# Valentin: feature_ex
# Sophia: vim
conda activate vim

# set checkpoint to evaluate as input 
python dinov2/eval/slide_level/eval_slide_task.py --feature_dir /lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data --result_dir /lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data/results

