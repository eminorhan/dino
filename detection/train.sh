#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:59:00
#SBATCH --job-name=train_coco
#SBATCH --output=train_coco_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

python -u train.py \
	--dataset coco \
	--data-path '/vast/eo41/data/coco' \
	--model maskrcnn_resnet50_fpn \
	--epochs 1 \
	--batch-size 32 \
	--lr-steps 16 22 \
	--aspect-ratio-group-factor 3

echo "Done"