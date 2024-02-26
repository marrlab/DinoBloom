#!/bin/bash

# --- vim checkpoints
# MODEL="vim_finetuned"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data/eval/training_99999/teacher_checkpoint.pth"

# --- vit checkpoints (Benedikt)
# MODEL="dinov2_vits14"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/DINOv2_finetuned"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/DINOv2_NCT_100K_finetuned"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/DINOv2_finetuned/dinov2_vits_NCT_10k_training_1999_teacher_checkpoint.pth"

# --- ctranspath 
# MODEL="ctranspath"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/ctranspath"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/ctranspath/eval/pretrained/ctranspath.pth"

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
MODEL="dinov2_vits14"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits/eval/training_59999/teacher_checkpoint.pth"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits/eval/training_14999/teacher_checkpoint.pth"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits_v2"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits_v2/training_99999/teacher_checkpoint.pth"
# MODEL="dinov2_vitl14"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vitl"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vitl/training_99999/teacher_checkpoint.pth"
# MODEL="dinov2_vitb14"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vitb"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vitb/training_99999/teacher_checkpoint.pth"
# MODEL="dinov2_vits14"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/merged_models"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/merged_models/dinov2_finetuned_vits_TCGA_NCT_uniform_soup.pth"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits14_brca"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/vits14_brca/training_29999/teacher_checkpoint.pth"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/vits14_brca/training_74999/teacher_checkpoint.pth"
# BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vits14_nsclc"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/vits14_nsclc/training_29999/teacher_checkpoint.pth"
# CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/vits14_nsclc/training_74999/teacher_checkpoint.pth"
BASE_DIR="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/merged_models_brca-nsclc"
CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/merged_models/dinov2_finetuned_vits14_TCGA-BRCA-NSCLC_uniform_soup.pth"

# TRAINING_CONFIG="--model AttentionMIL --model_config={} --lr_scheduler_config={max_lr: 1.0e-04}"
# TRAINING_CONFIG="--model Transformer --model_config={dropout: 0.2} --lr_scheduler_config={max_lr: 1.0e-05}"


DATASETS=("CPTAC-CRC" "TCGA-CRC" "TCGA-BRCA" "TCGA-NSCLC")
# DATASETS=("TCGA-CRC" "TCGA-BRCA" "TCGA-NSCLC")
# DATASETS=("TCGA-BRCA")

for dataset in "${DATASETS[@]}"; do
    sbatch feat_extract_slide-level_tasks.sbatch $MODEL $BASE_DIR $CHECKPOINT $dataset
done
