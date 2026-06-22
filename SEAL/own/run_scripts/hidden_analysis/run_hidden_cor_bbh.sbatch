#!/bin/bash
#SBATCH --job-name=seal_hidden_cor_bbh
#SBATCH --output=artifacts/logs/bbh/vector/%x_%j.out
#SBATCH --error=artifacts/logs/bbh/vector/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

python hidden_analysis_bbh.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data_dir $WORK/SEAL/artifacts/output/own/BBH_vllm \
  --type correct \
  --start 0 \
  --sample 500

python hidden_analysis_bbh.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data_dir $WORK/SEAL/artifacts/output/own/BBH_vllm \
  --type incorrect \
  --start 0 \
  --sample 500
