#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=240GB
#SBATCH --time=48:00:00
#SBATCH --job-name=dino_eval_linear
#SBATCH --output=dino_eval_linear_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.3.1

MODELS=(vitb14 vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(say say s a y say s a y say s a y)
ARCHS=(vit_base vit_large vit_large vit_large vit_large vit_base vit_base vit_base vit_base vit_small vit_small vit_small vit_small)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH

# ecoset
python -u /scratch/eo41/dino/eval_linear.py \
	--arch $ARCH \
	--patch_size 14 \
	--pretrained_weights '' \
	--save_prefix imagenet_${MODEL} \
	--checkpoint_key "teacher" \
	--batch_size 1024 \
	--epochs 500 \
	--num_workers 16 \
	--lr 0.0005 \
	--output_dir "/scratch/eo41/dino/evals/ecoset" \
	--train_data_path "/vast/eo41/data/ecoset/train" \
	--val_data_path "/vast/eo41/data/ecoset/val" \
	--num_labels 565
	
echo "Done"
