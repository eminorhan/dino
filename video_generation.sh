#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=01:00:00
#SBATCH --job-name=video_generation
#SBATCH --output=video_generation_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

srun python -u /scratch/eo41/dino/video_generation.py \
	--pretrained_weights "/scratch/eo41/dino/models_vitb14/say_5fps_vitb14_checkpoint.pth" \
	--arch "vit_base" \
	--patch_size 14 \
	--fps 25 \
	--resize 1400 \
	--input_path "video_atts/output/ade20k_say" \
	--output_path "video_atts/output/" \
	--head_idx $SLURM_ARRAY_TASK_ID \
	--save_prefix "ade20k_say" \
	--video_only
				
echo "Done"
