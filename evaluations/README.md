# Evaluations

use the following folder to store features

```
/lustre/groups/shared/users/peng_marr/HistoDINO/features /<dataset-name>/<model_name + run_name + checkpoint>
```
and evaluations
```
/lustre/groups/shared/users/peng_marr/HistoDINO/features /<dataset-name>/<model_name + run_name + checkpoint>
```

1. Extract features of trained model

```bash
conda activate vim

python evaluations/extract_patch_features.py \
    --model_name vim_finetuned \
    --save_dir /lustre/groups/shared/users/peng_marr/HistoDINO/features \ 
    --dataset NCT-CRC-100k-nonorm \ 
    --checkpoint /home/haicu/sophia.wagner/projects/Vim/vim/vim_tiny_73p1.pth 
```

2. Run evaluation pipeline
change conda env to fitting environment
```
conda activate eval

python evaluations/evaluation.py \
    --path-folder /lustre/groups/shared/users/peng_marr/HistoDINO/features/NCT-CRC-100k-nonorm/vim_tiny_224_pretrained
    --save-dir /lustre/groups/shared/users/peng_marr/HistoDINO/eval \
    --dataset NCT-CRC-100k-nonorm
```

3. Run HistoBistro evaluation pipeline
```
conda activate eval

cd ~/projects/HistoBistro
python train.py
```