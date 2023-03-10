#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=8:00:00
#SBATCH --job-name=dino_lin_labeled_s
#SBATCH --output=dino_lin_labeled_s_%A_%a.out
#SBATCH --array=0-1

module purge
module load cuda/11.3.1

MODELS=(vitb14 vitb14 vitb14 vitb14 vitb14 vitb14 vitb14 resnext50 resnext50 resnext50 resnext50 vitb14 vitb14 vitb14 vitb14 vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(ego4d kinetics random imagenet_100 imagenet_10 imagenet_3 imagenet_1 say s a y say s a y say s a y say s a y say s a y)
ARCHS=(vit_base vit_base vit_base vit_base vit_base vit_base vit_base resnext50_32x4d resnext50_32x4d resnext50_32x4d resnext50_32x4d vit_base vit_base vit_base vit_base vit_large vit_large vit_large vit_large vit_base vit_base vit_base vit_base vit_small vit_small vit_small vit_small)
PATCHES=(14 14 14 14 14 14 14 16 16 16 16 14 14 14 14 16 16 16 16 16 16 16 16 16 16 16 16)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
PATCH=${PATCHES[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH
echo $PATCH

# labeled_s
python -u /scratch/eo41/dino/eval_linear.py \
	--arch ${ARCH} \
	--patch_size ${PATCH} \
	--pretrained_weights "/scratch/eo41/dino/models_${MODEL}/${SUBJECT}_5fps_${MODEL}_checkpoint.pth" \
	--save_prefix ${SUBJECT}_${MODEL} \
	--checkpoint_key "teacher" \
	--batch_size 1024 \
	--epochs 300 \
	--num_workers 8 \
	--lr 0.0005 \
	--output_dir "/scratch/eo41/dino/evals/labeled_s" \
	--train_data_path "/vast/eo41/data/labeled_s" \
	--val_data_path "" \
	--num_labels 26 \
	--split \
	--subsample
	
echo "Done"
