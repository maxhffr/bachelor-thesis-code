#!/bin/bash
#SBATCH --job-name=seal_mgsm_ov_steering_lc
#SBATCH --output=artifacts/logs/mgsm/lc/ov/%x_%j.out
#SBATCH --error=artifacts/logs/mgsm/lc/ov/%x_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

LANGS=("de" "fr" "es" "ru" "zh" "ja")

for LANG in "${LANGS[@]}"; do
    python eval_mgsm_lc_steering.py \
      --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --save_dir "/work/maxhoffe/SEAL/artifacts/output/own/MGSM_steering/lc/ov/$LANG" \
      --max_tokens 10000 \
      --use_chat_format \
      --batch_size 16 \
      --dataset MGSM \
      --lang "$LANG" \
      --remove_bos \
      --steering \
      --steering_vector $WORK/SEAL/artifacts/output/own/GSM8K_vllm/vector_500_500/layer_20_transition_reflection_steervec.pt \
      --steering_layer 20 \
      --steering_coef -1.0 \
      --start 0
done
