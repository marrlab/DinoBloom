#!/bin/bash

#SBATCH -o slurm_outputs/slide.txt
#SBATCH -e slurm_outputs/slide_error.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH -c 18
#SBATCH --mem=240G
#SBATCH -q gpu_long
#SBATCH --time=24:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1

# Environment setup
source $HOME/.bashrc

# Activate your conda environment  (Valentin: /home/icb/valentin.koch/dinov2, Sophia: /home/haicu/sophia.wagner/projects/dinov2)    
conda activate feature_ex

# choose the right directory (Valentin: /home/icb/valentin.koch/dinov2, Sophia: /home/haicu/sophia.wagner/projects/dinov2)    

cd /home/icb/valentin.koch/dinov2

# Directory where checkpoints are stored
CHECKPOINTS_DIR="/home/icb/valentin.koch/dinov2/debug/eval"

# Find all teacher_checkpoint.pth files in CHECKPOINTS_DIR
CHECKPOINTS=$(find $CHECKPOINTS_DIR -name "*teacher_checkpoint.pth" | tr '\n' ' ')

# Array of slide paths
SLIDE_PATHS=(
    "/lustre/groups/shared/histology_data/NewTCGA/TCGA-BRCA/"
    "/lustre/groups/shared/histology_data/NewTCGA/TCGA-NSLCL/"
    "/lustre/groups/shared/histology_data/NewTCGA/TCGA-CRC/"
)

# Loop over the slide paths
for SLIDE_PATH in "${SLIDE_PATHS[@]}"; do
    # Call the Python script with the current slide path and all checkpoints
    python "./dinov2/eval/miccai/extract_slide_features.py" \
        --slide_path "$SLIDE_PATH" \
        --file_extension .svs \
        --resolution_in_mpp 0.5 \
        --checkpoints $CHECKPOINTS
done

