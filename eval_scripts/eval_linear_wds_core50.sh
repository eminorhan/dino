#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --job-name=dino_eval_linear_wds
#SBATCH --output=dino_eval_linear_wds_%A_%a.out
#SBATCH --array=0-11

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

# for reasons, this should only be run on a single gpu with num_workers=1 for now. I'm sorry.

MODELS=(vitl16 vitl16 vitl16 vitl16 vitb16 vitb16 vitb16 vitb16 vits16 vits16 vits16 vits16)
SUBJECTS=(say s a y say s a y say s a y)
ARCHS=(vit_large vit_large vit_large vit_large vit_base vit_base vit_base vit_base vit_small vit_small vit_small vit_small)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

echo $MODEL
echo $SUBJECT
echo $ARCH

# core50
srun python -u /scratch/eo41/dino/eval_linear_wds.py \
	--arch $ARCH \
	--pretrained_weights /scratch/eo41/dino/models_${MODEL}/${SUBJECT}_5fps_${MODEL}_checkpoint.pth \
	--save_prefix ${SUBJECT}_5fps_${MODEL} \
	--checkpoint_key "teacher" \
	--batch_size_per_gpu 1024 \
	--epochs 500 \
	--num_workers 1 \
	--lr 0.0005 \
	--output_dir "/scratch/eo41/dino/evals/core50" \
	--train_data_path "/scratch/eo41/data/core50/core50_train_000000.tar" \
	--val_data_path "/scratch/eo41/data/core50/core50_val_000000.tar" \
	--n_train 90000 \
	--n_val 75000 \
	--num_labels 50
	
echo "Done"
