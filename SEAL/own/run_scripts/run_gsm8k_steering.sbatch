#!/bin/bash
#SBATCH --job-name=seal_gsm8k_steering
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

python eval_gsm8k_steering.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --save_dir "./artifacts/output/own/GSM8K_steering" \
  --max_tokens 10000 \
  --use_chat_format \
  --batch_size 16 \
  --dataset GSM8K \
  --remove_bos \
  --steering \
  --steering_vector "./artifacts/output/MATH_train/DeepSeek-R1-Distill-Qwen-1.5B/baseline_10000/vector_500_500/layer_20_transition_reflection_steervec.pt" \
  --steering_layer 20 \
  --steering_coef -1.0 \
  --start 0
