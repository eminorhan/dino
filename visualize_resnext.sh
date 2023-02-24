#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=00:05:00
#SBATCH --job-name=visualize_resnext
#SBATCH --output=visualize_resnext_%A_%a.out
#SBATCH --array=0-4

module purge
module load cuda/11.6.2

srun python -u /scratch/eo41/dino/visualize_resnext.py \
	--data_path '/scratch/eo41/dino/labeled_s_examples/table' \
	--class_idx 23 \
	--n_out 26 \
	--batch_size 12 \
	--seed $SLURM_ARRAY_TASK_ID \
	--pretrained_backbone '/scratch/eo41/dino/models_resnext50/s_5fps_resnext50_checkpoint.pth' \
	--pretrained_fc '/scratch/eo41/dino/evals/labeled_s/s_resnext50_checkpoint.pth.tar'

echo "Done"
