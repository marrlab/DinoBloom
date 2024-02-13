#!/bin/bash

CHECKPOINT="/lustre/groups/shared/users/peng_marr/HistoDINO/logs/vim_balanced_data/eval/training_99999/teacher_checkpoint.pth"
DATASETS=("TCGA-NSLCL")
# DATASETS=("TCGA-CRC" "TCGA-BRCA" "TCGA-NSLCL")

for dataset in "${DATASETS[@]}"; do
    sbatch feat_extract_slide-level_tasks.sbatch $CHECKPOINT $dataset
done
