#!/bin/bash

#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=492GB
#SBATCH --time=48:00:00
#SBATCH --job-name=dino_train_nowds
#SBATCH --output=dino_train_nowds_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

### ########################################## KINETICS-200h ########################################## ###

srun python -u /scratch/eo41/dino/train_dino_nowds.py \
	--use_fp16 false \
	--arch "vit_base" \
	--patch_size 14 \
	--batch_size 256 \
	--num_workers 16 \
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
	--data_path "/vast/eo41/data/kinetics-200h" \
	--save_prefix "kinetics200h_vitb14"

				
echo "Done"