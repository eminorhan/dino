#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=01:15:00
#SBATCH --job-name=video_generation
#SBATCH --output=video_generation_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

srun python -u /scratch/eo41/dino/video_generation.py \
	--arch "vit_base" \
	--patch_size 14 \
	--input_path "output/frames" \
	--output_path "output/" \
	--fps 25 \
	--resize 1400 \
	--pretrained_weights "/scratch/eo41/dino/models_vitb14/imagenet_100_5fps_vitb14_checkpoint.pth"
				
echo "Done"
