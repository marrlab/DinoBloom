#!/bin/bash

#SBATCH -o /lustre/groups/shared/histology_data/eval/slurm_outputs/feature_run2.txt
#SBATCH -e /lustre/groups/shared/histology_data/eval/slurm_outputs/feature_error2.txt
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
conda activate feature_ex
cd /home/icb/valentin.koch/dinov2/
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
python "/home/icb/valentin.koch/dinov2/dinov2/eval/miccai/extract_patch_features.py"
conda activate histo_env
python "/home/icb/valentin.koch/dinov2/dinov2/eval/miccai/evaluation.py" --path_folder "/lustre/groups/shared/histology_data/features_NCT-CRC-100k-nonorm/benedikt_baseline_vits"
