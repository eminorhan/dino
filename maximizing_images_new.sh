#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:50:00
#SBATCH --job-name=maximizing_images
#SBATCH --output=maximizing_images_%A_%a.out
#SBATCH --array=4

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

# --pretrained_weights /scratch/eo41/dino/models_${MODEL}/${SUBJECT}_5fps_${MODEL}_checkpoint.pth \
# konkle
srun python -u /scratch/eo41/dino/maximizing_images_new.py \
	--arch $ARCH \
	--patch_size 16 \
	--pretrained_weights '' \
	--save_prefix ${SUBJECT}_5fps_${MODEL} \
	--checkpoint_key "teacher" \
	--batch_size_per_gpu 1024 \
	--num_workers 1 \
	--output_dir "/scratch/eo41/dino/maximizing_images/konkle_objects" \
	--val_data_path "/scratch/eo41/data/konkle_objects/konkle_objects_000000.tar" \
	--n_val 4040 \
	--num_labels 240
	
echo "Done"
