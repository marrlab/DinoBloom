#!/bin/bash

#SBATCH -o slurm_outputs/eval_all_ctran.txt
#SBATCH -e slurm_outputs/eval_all_error_ctran.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH -c 18
#SBATCH --mem=150G
#SBATCH -q gpu_long
#SBATCH --time=5:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

# Environment setup
source $HOME/.bashrc

# choose the right directory 
cd /home/haicu/sophia.wagner/projects/dinov2  
# cd /home/icb/valentin.koch/dinov2

# activate your own conda environment (Valentin: feature_ex, Sophia: vim)
conda activate dinov2

# set checkpoint to evaluate as input 
python dinov2/eval/miccai/extract_and_eval_patches.py --run_path "/lustre/groups/shared/users/peng_marr/HistoDINO/logs/baseline_models/retccl" --model_name resnet50  --image_path_train /home/icb/valentin.koch/dinov2/dinov2/eval/miccai/matek_train.csv --image_path_test "/home/icb/valentin.koch/dinov2/dinov2/eval/miccai/matek_test.csv" --experiment_name matek  --num_workers 16 


