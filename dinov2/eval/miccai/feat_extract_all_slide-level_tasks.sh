#!/bin/bash

# --- vim checkpoints
# MODEL="vim_finetuned"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data/eval/training_99999/teacher_checkpoint.pth"

# --- vit checkpoints (Benedikt)
# MODEL="dinov2_finetuned"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/DINOv2_finetuned/dinov2_vits_TCGA_training_29999_teacher_checkpoint.pth"

# --- ctranspath featuer extraction
# MODEL="ctranspath"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/ctranspath/eval/pretrained/ctranspath.pth"

# --- pretrained dinov2 networks
# MODEL="dinov2_vits14_downloaded"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/dinov2_vits14_downloaded"

# --- owkin model
MODEL="owkin"
CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/owkin"

DATASETS=("TCGA-CRC" "TCGA-BRCA" "TCGA-NSCLC")
# DATASETS=("TCGA-NSCLC")

for dataset in "${DATASETS[@]}"; do
    sbatch feat_extract_slide-level_tasks.sbatch $MODEL $CHECKPOINT $dataset
done
