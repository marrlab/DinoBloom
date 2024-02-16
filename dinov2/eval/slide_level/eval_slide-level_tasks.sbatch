#!/bin/bash

#SBATCH -o /lustre/groups/shared/users/peng_marr/HistoDINO/logs/slurm/eval_slide_tasks_out.txt
#SBATCH -e /lustre/groups/shared/users/peng_marr/HistoDINO/logs/slurm/eval_slide_tasks_err.txt
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
conda activate dinov2

# set checkpoint to evaluate as input 
# python dinov2/eval/slide_level/eval_slide_task.py --base_dir /lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data           --feature_extractor vim_balanced_data
python dinov2/eval/slide_level/eval_slide_task.py --base_dir /lustre/groups/shared/users/peng_marr/HistoDINO/logs/owkin                       --feature_extractor owkin
# python dinov2/eval/slide_level/eval_slide_task.py --base_dir /lustre/groups/shared/users/peng_marr/HistoDINO/logs/DINOv2_finetuned            --feature_extractor dinov2_finetuned
# python dinov2/eval/slide_level/eval_slide_task.py --base_dir /lustre/groups/shared/users/peng_marr/HistoDINO/logs/ctranspath                  --feature_extractor ctranspath
# python dinov2/eval/slide_level/eval_slide_task.py --base_dir /lustre/groups/shared/users/peng_marr/HistoDINO/logs/dinov2_vits14_downloaded    --feature_extractor dinov2_vits14_downloaded

