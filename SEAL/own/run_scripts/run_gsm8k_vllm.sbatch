#!/bin/bash
#SBATCH --job-name=seal_gsm8k_vllm
#SBATCH --output=artifacts/logs/gsm8k/%x_%j.out
#SBATCH --error=artifacts/logs/gsm8k/%x_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

python eval_gsm8k_vllm.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --save_dir "./artifacts/output/own/GSM8K_vllm" \
  --max_tokens 10000 \
  --use_chat_format \
  --dataset GSM8K \
  --remove_bos
