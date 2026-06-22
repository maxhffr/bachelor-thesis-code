#!/bin/bash
#SBATCH --job-name=seal_bbh_vllm
#SBATCH --output=artifacts/logs/bbh/%x_%j.out
#SBATCH --error=artifacts/logs/bbh/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

python eval_BBH_vllm.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --save_dir "./artifacts/output/own/BBH_vllm" \
  --max_tokens 10000 \
  --use_chat_format \
  --dataset BBH \
  --bbh_data_dir data/bbh \
  --remove_bos
