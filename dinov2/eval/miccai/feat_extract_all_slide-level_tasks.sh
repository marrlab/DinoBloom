#!/bin/bash

# --- vim checkpoints
# MODEL="vim_finetuned"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data/eval/training_99999/teacher_checkpoint.pth"

# --- vit checkpoints (Benedikt)
# MODEL="dinov2_finetuned"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/DINOv2_finetuned"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/DINOv2_finetuned/dinov2_vits_TCGA_training_29999_teacher_checkpoint.pth"

# --- ctranspath 
# MODEL="ctranspath"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/ctranspath"
# CHECKPOINT="/ltre/groups/shared/users/peng_marr/HistoDINO/logs/ctranspath/eval/pretrained/ctranspath.pth"

# --- pretrained dinov2 networks
# MODEL="dinov2_vits14_downloaded"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/dinov2_vits14_downloaded"
# CHECKPOINT="dinov2_meta_vits14/downloaded.pth"

# --- owkin model
# MODEL="owkin"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/owkin"
# CHECKPOINT="owkin/phikon.pth"

# --- resnet model
# MODEL="resnet50"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/resnet50"
# CHECKPOINT="resnet50/checkpoint.pth"

# --- vit checkpoints
MODEL="dinov2_finetuned"
BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits"
CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits/eval/training_59999/teacher_checkpoint.pth"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits/eval/training_14999/teacher_checkpoint.pth"


# TRAINING_CONFIG="--model AttentionMIL --model_config={} --lr_scheduler_config={max_lr: 1.0e-04}"
# TRAINING_CONFIG="--model Transformer --model_config={dropout: 0.2} --lr_scheduler_config={max_lr: 1.0e-05}"


DATASETS=("CPTAC-CRC" "TCGA-CRC" "TCGA-BRCA" "TCGA-NSCLC")
# DATASETS=("TCGA-CRC" "TCGA-BRCA" "TCGA-NSCLC")
# DATASETS=("TCGA-BRCA")

for dataset in "${DATASETS[@]}"; do
    sbatch feat_extract_slide-level_tasks.sbatch $MODEL $BASE_DIR $CHECKPOINT $dataset
done
