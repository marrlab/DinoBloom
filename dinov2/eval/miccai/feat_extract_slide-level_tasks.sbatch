#!/bin/bash

#SBATCH -o /lustre/groups/shared/users/peng_marr/HistoDINO/logs/slurm/extract_features_slide_out.txt
#SBATCH -e /lustre/groups/shared/users/peng_marr/HistoDINO/logs/slurm/extract_features_slide_err.txt
#SBATCH -J feature_ex
#SBATCH -p gpu_p
#SBATCH --time=2-00:00:00
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpusrv27
#SBATCH --constraint=a100_80gb

#SBATCH -c 20
#SBATCH --mem=160G
#SBATCH -q gpu_normal

# Environment setup
source $HOME/.bashrc

# choose the right directory
# Valentin: /home/icb/valentin.koch/dinov2 
# Sophia: /home/haicu/sophia.wagner/projects/dinov2
cd /home/haicu/sophia.wagner/projects/dinov2

# activate your own conda environment 
# Valentin: feature_ex
# Sophia: dinov2
# Sophia: vim_new
# source /home/haicu/sophia.wagner/miniforge3/bin/activate dinov2
source /home/haicu/sophia.wagner/miniforge3/bin/activate dinov2

# set checkpoint for feature extraction as input 
python dinov2/eval/miccai/extract_slide_features.py --model $1 --base_dir $2 --checkpoints $3 --dataset $4

# choose env for slide-level evaluations
source /home/haicu/sophia.wagner/miniforge3/bin/activate dinov2

# evaluate the extracted features
# choose feature folder according to following naming
# {args.dataset}_{Path(c).parent.name}_{args.patch_size}px_{args.model[i]}_{args.resolution_in_mpp}mpp_{args.downscaling_factor}xdown_normal
parent_name=$(basename "$(dirname "$3")")
patch_size=224
resolution_in_mpp=1
feature_dir=$4_${parent_name}_${patch_size}px_$1_${resolution_in_mpp}mpp
echo $feature_dir
echo $2/features/$feature_dir

python dinov2/eval/slide_level/eval_slide_task.py --base_dir $2 --feature_dir $2/features/$feature_dir --feature_extractor $1 


