#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=00:05:00
#SBATCH --job-name=visualize_resnext
#SBATCH --output=visualize_resnext_%A_%a.out
#SBATCH --array=0-4

srun python -u /scratch/eo41/dino/visualize_resnext.py \
	--data_path '/scratch/eo41/dino/labeled_s_examples/basket' \
	--class_idx 1 \
	--n_out 26 \
	--batch_size 8 \
	--seed $SLURM_ARRAY_TASK_ID \
	--pretrained_backbone '/scratch/eo41/dino/models_resnext50/y_5fps_resnext50_checkpoint.pth' \
	--pretrained_fc '/scratch/eo41/dino/evals/labeled_s/y_resnext50_checkpoint.pth.tar'

# srun python -u /scratch/eo41/dino/visualize_resnext.py \
# 	--data_path '/scratch/eo41/dino/ecoset_examples/woman' \
# 	--class_idx 3 \
# 	--n_out 565 \
# 	--batch_size 8 \
# 	--seed $SLURM_ARRAY_TASK_ID \
# 	--pretrained_backbone '/scratch/eo41/dino/models_resnext50/say_5fps_resnext50_checkpoint.pth' \
# 	--pretrained_fc '/scratch/eo41/dino/evals/ecoset/say_resnext50_checkpoint.pth.tar'

echo "Done"