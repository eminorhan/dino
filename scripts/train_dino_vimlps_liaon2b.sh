#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=240GB
#SBATCH --time=2:00:00
#SBATCH --job-name=train_dino_vimlps_liaon2b
#SBATCH --output=train_dino_vimlps_liaon2b_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

DATAS=(
	"{00000..99999}" 
	)

SAVES=(
	"dino_vimlp_liaon2b" 
	)

DATA=${DATAS[$SLURM_ARRAY_TASK_ID]}
SAVE=${SAVES[$SLURM_ARRAY_TASK_ID]}

echo $DATA
echo $SAVE

# vimlp_small
srun python -u /scratch/eo41/dino/train_dino.py \
	--use_fp16 false \
	--arch "vimlp_small" \
	--batch_size_per_gpu 256 \
	--num_workers 8 \
	--freeze_last_layer 0 \
	--lr 0.0001 \
	--min_lr 0.0001 \
	--global_crops_scale 0.4 1 \
	--local_crops_scale 0.05 0.4 \
	--optimizer adamw \
	--weight_decay 0.0 \
	--weight_decay_end 0.0 \
	--clip_grad 1.0 \
	--saveckp_freq 1000 \
	--print_freq 1000 \
	--output_dir "/scratch/eo41/dino/models_vimlps_liaon2b" \
	--data_path "/scratch/work/public/ml-datasets/laion2B-en-data/${DATA}.tar" \
	--save_prefix "${SAVE}_small"

echo "Done"