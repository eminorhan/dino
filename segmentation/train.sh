#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=123GB
#SBATCH --time=08:00:00
#SBATCH --job-name=train_coco
#SBATCH --output=train_coco_%A_%a.out
#SBATCH --array=0-24

module purge
module load cuda/11.6.2

MODELS=(vitb14 vitb14 vitb14 vitb14 vitb14 resnext50 resnext50 resnext50 resnext50 vitb14 vitb14 vitb14 vitb14 vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(random imagenet_100 imagenet_10 imagenet_3 imagenet_1 say s a y say s a y say s a y say s a y say s a y)
ARCHS=(vit_base vit_base vit_base vit_base vit_base resnext50_32x4d resnext50_32x4d resnext50_32x4d resnext50_32x4d vit_base vit_base vit_base vit_base vit_large vit_large vit_large vit_large vit_base vit_base vit_base vit_base vit_small vit_small vit_small vit_small)
PATCHES=(14 14 14 14 14 16 16 16 16 14 14 14 14 16 16 16 16 16 16 16 16 16 16 16 16)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
PATCH=${PATCHES[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH
echo $PATCH

python -u train.py \
	--dataset coco \
	--data_path '/vast/eo41/data/coco' \
	--arch ${ARCH} \
	--patch_size ${PATCH} \
	--pretrained_weights "/scratch/eo41/dino/models_${MODEL}/${SUBJECT}_5fps_${MODEL}_checkpoint.pth" \
	--save_prefix ${SUBJECT}_${MODEL} \
	--checkpoint_key "teacher" \
	--output_dir "/scratch/eo41/dino/segmentation/evals/coco" \
	--epochs 10 \
	--batch_size 32 \
	--lr 0.0005 \
	--workers 16

echo "Done"