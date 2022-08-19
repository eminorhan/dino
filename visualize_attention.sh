#!/bin/bash

##SBATCH --account=cds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200GB
#SBATCH --time=00:20:00
#SBATCH --job-name=visualize_attention
#SBATCH --output=visualize_attention_%A_%a.out
#SBATCH --array=0

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$(shuf -i 10000-65500 -n 1)
export WORLD_SIZE=1

module purge
module load cuda/11.3.1

srun python -u /scratch/eo41/dino/visualize_attention.py \
	--arch "vit_large" \
	--patch_size 16 \
	--output_dir "/scratch/eo41/dino/visualizations" \
	--image_path "/scratch/eo41/dino/visualizations/imgs/ILSVRC2012_val_00013137.JPEG" \
	--pretrained_weights "/scratch/eo41/dino/models_vitl/say_5fps_vitl16_checkpoint.pth"
				
echo "Done"
