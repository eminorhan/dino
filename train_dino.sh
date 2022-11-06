#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --job-name=dino_train
#SBATCH --output=dino_train_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=4

module purge
module load cuda/11.3.1

### ########################################## IMAGENET ########################################## ###

#srun python -u /scratch/eo41/dino/train_dino.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--patch_size 14 \
#	--batch_size_per_gpu 116 \
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
#	--output_dir "/scratch/eo41/dino/models_vitb14" \
#	--data_path "/scratch/eo41/data/imagenet/imagenet_train_{000000..000001}.tar" \
#	--save_prefix "imagenet_100_vitb14"

#srun python -u /scratch/eo41/dino/train_dino.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--patch_size 14 \
#	--batch_size_per_gpu 115 \
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
#	--output_dir "/scratch/eo41/dino/models_vitb14" \
#	--data_path "/scratch/eo41/data/imagenet_fractional/imagenet_train_0.1_1_000000.tar" \
#	--save_prefix "imagenet_10_vitb14"

#srun python -u /scratch/eo41/dino/train_dino.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--patch_size 14 \
#	--batch_size_per_gpu 115 \
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
#	--output_dir "/scratch/eo41/dino/models_vitb14" \
#	--data_path "/scratch/eo41/data/imagenet_fractional/imagenet_train_0.03_1_000000.tar" \
#	--save_prefix "imagenet_3_vitb14"

#srun python -u /scratch/eo41/dino/train_dino.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--patch_size 14 \
#	--batch_size_per_gpu 116 \
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
#	--output_dir "/scratch/eo41/dino/models_vitb14" \
#	--data_path "/scratch/eo41/data/imagenet_fractional/imagenet_train_0.01_1_000000.tar" \
#	--save_prefix "imagenet_1_vitb14"

### ########################################## SAYCAM ########################################## ###

srun python -u /scratch/eo41/dino/train_dino.py \
	--use_fp16 false \
	--arch "vit_base" \
	--patch_size 14 \
	--batch_size_per_gpu 116 \
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
	--output_dir "/scratch/eo41/dino/models_vitb14" \
	--data_path "/scratch/eo41/data/saycam/SAY_5fps_300s_{000000..000009}.tar" \
	--save_prefix "say_5fps_vitb14"

#srun python -u /scratch/eo41/dino/train_dino.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--patch_size 14 \
#	--batch_size_per_gpu 100 \
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
#	--output_dir "/scratch/eo41/dino/models_vitb14" \
#	--data_path "/scratch/eo41/data/saycam/S_5fps_300s_{000000..000003}.tar" \
#	--save_prefix "s_5fps_vitb14"
		
#srun python -u /scratch/eo41/dino/train_dino.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--patch_size 14 \
#	--batch_size_per_gpu 100 \
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
#	--output_dir "/scratch/eo41/dino/models_vitb14" \
#	--data_path "/scratch/eo41/data/saycam/A_5fps_300s_{000000..000002}.tar" \
#	--save_prefix "a_5fps_vitb14"

#srun python -u /scratch/eo41/dino/train_dino.py \
#	--use_fp16 false \
#	--arch "vit_base" \
#	--patch_size 14 \
#	--batch_size_per_gpu 100 \
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
#	--output_dir "/scratch/eo41/dino/models_vitb14" \
#	--data_path "/scratch/eo41/data/saycam/Y_5fps_300s_{000000..000002}.tar" \
#	--save_prefix "y_5fps_vitb14"
				
echo "Done"