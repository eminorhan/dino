#!/bin/bash

#SBATCH --gres=gpu:rtx8000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=00:16:00
#SBATCH --job-name=visualize_segmentations
#SBATCH --output=visualize_segmentations_%A_%a.out
#SBATCH --array=0

module purge
module load cuda/11.6.2

python -u visualize.py

echo "Done"