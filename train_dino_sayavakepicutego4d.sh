#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --job-name=dino_train_sayavakepicutego4d
#SBATCH --output=dino_train_sayavakepicutego4d_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

module purge
module load cuda/11.3.1

# vit
#srun python -u /scratch/eo41/dino/train_dino.py \
#	--use_fp16 false \
#	--arch "vit_large" \
#	--patch_size 16 \
#	--batch_size_per_gpu 52 \
#	--num_workers 8 \
#	--freeze_last_layer 0 \
#	--lr 0.0001 \
#	--min_lr 0.0001 \
#	--global_crops_scale 0.2 1 \
#	--local_crops_scale 0.05 0.2 \
#	--optimizer adamw \
#	--weight_decay 0.0 \
#	--weight_decay_end 0.0 \
#	--clip_grad 1.0 \
#	--saveckp_freq 10000 \
#	--print_freq 10000 \
#	--output_dir "/vast/eo41/sayavakepicutego4d_models" \
#	--data_path "/vast/eo41/sayavakepicutego4d/sayavakepicutego4d_{000000..000017}.tar" \
#	--save_prefix "sayavakepicutego4d_vitl16"

# resnext
srun python -u /scratch/eo41/dino/train_dino.py \
	--use_fp16 false \
	--arch "resnext50_32x4d" \
	--batch_size_per_gpu 128 \
	--num_workers 8 \
	--freeze_last_layer 0 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--global_crops_scale 0.2 1 \
	--local_crops_scale 0.05 0.2 \
	--optimizer adamw \
	--weight_decay 0.0 \
	--weight_decay_end 0.0 \
	--clip_grad 1.0 \
	--saveckp_freq 10000 \
	--print_freq 10000 \
	--output_dir "/vast/eo41/sayavakepicutego4d_models" \
	--data_path "/vast/eo41/sayavakepicutego4d/sayavakepicutego4d_{000000..000017}.tar" \
	--save_prefix "sayavakepicutego4d_resnext50"

echo "Done"
