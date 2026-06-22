#!/bin/bash
#SBATCH --job-name=seal_humaneval_vllm
#SBATCH --output=artifacts/logs/humaneval/%x_%j.out
#SBATCH --error=artifacts/logs/humaneval/%x_%j.err
#SBATCH --time=18:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-32gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

python eval_humaneval_vllm.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --save_dir $WORK/SEAL/artifacts/output/own/HUMANEVAL_vllm/ \
  --benchmark humaneval \
  --data_path data/humaneval/humaneval_test.jsonl \
  --max_tokens 10000 \
  --use_chat_format \
  --remove_bos
