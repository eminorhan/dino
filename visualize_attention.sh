#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:02:00
#SBATCH --job-name=visualize_attention
#SBATCH --output=visualize_attention_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

srun python -u /scratch/eo41/dino/visualize_attention.py \
	--arch "vit_base" \
	--patch_size 14 \
	--image_size 1400 1400 \
	--threshold 0.7 \
	--output_dir "/scratch/eo41/dino/visualizations" \
	--image_path "/scratch/eo41/dino/visualizations/imgs/saycam_5.jpeg" \
	--save_name "atts_saycam_5" \
	--pretrained_weights "/scratch/eo41/dino/models_vitb16/say_5fps_vitb16_checkpoint.pth"
				
echo "Done"
