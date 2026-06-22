#!/bin/bash
#SBATCH --job-name=seal_vector_generation
#SBATCH --output=artifacts/logs/paper/%x_%j.out
#SBATCH --error=artifacts/logs/paper/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

python vector_generation.py \
  --data_dir $WORK/SEAL/artifacts/output/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000 \
  --prefixs correct_0_500 incorrect_0_500 \
  --layers 20 \
  --save_prefix 500_500
