#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=8:00:00
#SBATCH --job-name=dino_finetune_imagenet_vimlps
#SBATCH --output=dino_finetune_imagenet_vimlps_%A_%a.out
#SBATCH --array=0

# vimlp_large
python -u /scratch/eo41/dino/eval_finetune.py \
	--arch "vimlp_large" \
	--pretrained_weights "/scratch/eo41/dino/models_vimlps_liaon2b/dino_vimlp_liaon2b_large_checkpoint.pth" \
	--save_prefix "dino_vimlp_large" \
	--checkpoint_key "teacher" \
	--batch_size 2048 \
	--epochs 100 \
	--num_workers 16 \
	--lr 0.0001 \
	--output_dir "/scratch/eo41/dino/models_vimlps_liaon2b_finetuned" \
	--train_data_path "/scratch/work/public/imagenet/train" \
	--val_data_path "/scratch/eo41/imagenet/val" \
	--frac_retained 1.0 \
	--num_labels 1000
	
echo "Done"