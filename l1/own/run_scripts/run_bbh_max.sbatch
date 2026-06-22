#!/bin/bash
#SBATCH --job-name=l1_bbh_max
#SBATCH --output=artifacts/logs/bbh/%x_%j.out
#SBATCH --error=artifacts/logs/bbh/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate l1_testverl

./own/replicate/eval_bbh_max.sh l3lab/L1-Qwen-1.5B-Max
