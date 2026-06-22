#!/bin/bash
#SBATCH --job-name=seal_hiddenstates
#SBATCH --output=artifacts/logs/paper/%x_%j.out
#SBATCH --error=artifacts/logs/paper/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

python hidden_analysis.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data_path "./data/MATH/train.jsonl" \
  --data_dir "./artifacts/output/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000" \
  --type correct \
  --start 0 \
  --sample 500

python hidden_analysis.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data_path "./data/MATH/train.jsonl" \
  --data_dir "./artifacts/output/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000" \
  --type incorrect \
  --start 0 \
  --sample 500
