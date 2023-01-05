#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=12:00:00
#SBATCH --job-name=dino_lin_imagenet_sayavakepicutego4d
#SBATCH --output=dino_lin_imagenet_sayavakepicutego4d_%A_%a.out
#SBATCH --array=1

module purge
module load cuda/11.3.1

MODELS=(resnext50 vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(sayavakepicutego4d sayavakepicutego4d sayavakepicutego4d_10_1 sayavakepicutego4d_10_2 sayavakepicutego4d_10_3)
ARCHS=(resnext50_32x4d vit_large vit_large vit_large vit_large vit_base vit_base vit_base vit_base vit_small vit_small vit_small vit_small)
PATCHES=(16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
PATCH=${PATCHES[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH
echo $PATCH

# imagenet
python -u /scratch/eo41/dino/eval_linear.py \
	--arch ${ARCH} \
	--patch_size ${PATCH} \
	--pretrained_weights "/vast/eo41/sayavakepicutego4d_models/dino_vitl16/${SUBJECT}_${MODEL}_checkpoint.pth" \
	--save_prefix ${SUBJECT}_${MODEL} \
	--checkpoint_key "teacher" \
	--batch_size 1024 \
	--epochs 50 \
	--num_workers 16 \
	--lr 0.0005 \
	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--num_labels 1000
	
echo "Done"
