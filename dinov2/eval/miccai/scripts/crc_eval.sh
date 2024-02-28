#!/bin/bash

#SBATCH -o slurm_outputs/matek_vits.txt
#SBATCH -e slurm_outputs/matek_vits__error.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH -c 18
#SBATCH --mem=150G
#SBATCH -q gpu_normal
#SBATCH --time=15:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

# Environment setup
source $HOME/.bashrc

# choose the right directory (Valentin: /home/icb/valentin.koch/dinov2, Sophia: /home/haicu/sophia.wagner/projects/dinov2)    
cd /home/icb/valentin.koch/dinov2

# activate your own conda environment (Valentin: feature_ex, Sophia: vim)
conda activate feature_ex

# set checkpoint to evaluate as input 
python dinov2/eval/miccai/crc_eval.py --run_path "/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits_NCT-CRC_224px_orig_aug_correct_length1708925578/eval" --model_name dinov2_vits14  --image_path_train /home/icb/valentin.koch/dinov2/dinov2/eval/miccai/nct_crc_train.csv --image_path_test "/home/icb/valentin.koch/dinov2/dinov2/eval/miccai/nct_crc_test.csv" --experiment_name nct_crc  --num_workers 16 


