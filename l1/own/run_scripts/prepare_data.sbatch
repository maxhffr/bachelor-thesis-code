#!/bin/bash
#SBATCH --job-name=l1_prepare_data
#SBATCH --output=artifacts/logs/prepare/%x_%j.out
#SBATCH --error=artifacts/logs/prepare/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate l1

./own/replicate/prepare_data.sh
