#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=8:00:00
#SBATCH --job-name=dino_finetune_imagenet_sayavakepicutego4d
#SBATCH --output=dino_finetune_imagenet_sayavakepicutego4d_%A_%a.out
#SBATCH --array=0-12

SUBJECTS=(
	"sayavakepicutego4d" 
	"sayavakepicutego4d_0.1_1" 
	"sayavakepicutego4d_0.1_2" 
	"sayavakepicutego4d_0.1_3" 
	"sayavakepicutego4d_0.01_1" 
	"sayavakepicutego4d_0.01_2" 
	"sayavakepicutego4d_0.01_3" 
	"sayavakepicutego4d_0.001_1" 
	"sayavakepicutego4d_0.001_2" 
	"sayavakepicutego4d_0.001_3" 
	"sayavakepicutego4d_0.0001_1" 
	"sayavakepicutego4d_0.0001_2" 
	"sayavakepicutego4d_0.0001_3"
	)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
echo $SUBJECT

# vit-b/14
python -u /scratch/eo41/dino/eval_finetune.py \
	--arch "vit_base" \
	--patch_size 14 \
	--pretrained_weights "/vast/eo41/sayavakepicutego4d_models/dino_vitb14/${SUBJECT}_vitb14_checkpoint.pth" \
	--save_prefix dino_${SUBJECT}_vitb14 \
	--checkpoint_key "teacher" \
	--batch_size 256 \
	--epochs 25 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--frac_retained 0.010147 \
	--num_labels 1000
	
# vit-s/14
python -u /scratch/eo41/dino/eval_finetune.py \
	--arch "vit_small" \
	--patch_size 14 \
	--pretrained_weights "/vast/eo41/sayavakepicutego4d_models/dino_vits14/${SUBJECT}_vits14_checkpoint.pth" \
	--save_prefix dino_${SUBJECT}_vits14 \
	--checkpoint_key "teacher" \
	--batch_size 256 \
	--epochs 25 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--frac_retained 0.010147 \
	--num_labels 1000

# resnext50
python -u /scratch/eo41/dino/eval_finetune.py \
	--arch "resnext50_32x4d" \
	--patch_size 14 \
	--pretrained_weights "/vast/eo41/sayavakepicutego4d_models/dino_resnext50/${SUBJECT}_resnext50_checkpoint.pth" \
	--save_prefix dino_${SUBJECT}_resnext50 \
	--checkpoint_key "teacher" \
	--batch_size 256 \
	--epochs 25 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir "/vast/eo41/sayavakepicutego4d_evals" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--frac_retained 0.010147 \
	--num_labels 1000

echo "Done"