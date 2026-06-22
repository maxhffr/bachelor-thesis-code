#!/bin/bash
#SBATCH --job-name=seal_mgsm_vllm
#SBATCH --output=artifacts/logs/mgsm/%x_%j.out
#SBATCH --error=artifacts/logs/mgsm/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

LANGS=("de" "fr" "es" "ru" "zh" "ja")

for LANG in "${LANGS[@]}"; do
    python eval_gsm8k_vllm.py \
      --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --save_dir "./artifacts/output/own/MGSM_vllm//$LANG" \
      --max_tokens 10000 \
      --use_chat_format \
      --dataset MGSM \
      --lang "$LANG" \
      --remove_bos
done
