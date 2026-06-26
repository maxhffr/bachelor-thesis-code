#!/bin/bash
#SBATCH --job-name=l1_max
#SBATCH --output=artifacts/logs/paper/%x_%j.out
#SBATCH --error=artifacts/logs/paper/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate l1

./scripts/replicate/eval_inference_max.sh l3lab/L1-Qwen-1.5B-Max

./scripts/replicate/eval_inference_max_others.sh l3lab/L1-Qwen-1.5B-Max
