#!/bin/bash
#SBATCH --job-name=seal_bbh_ov_steering
#SBATCH --output=artifacts/logs/bbh/ov/%x_%j.out
#SBATCH --error=artifacts/logs/bbh/ov/%x_%j.err
#SBATCH --time=90:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

python eval_BBH_steering.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --save_dir "./artifacts/output/own/BBH_steering/ov" \
  --max_tokens 10000 \
  --use_chat_format \
  --batch_size 16 \
  --dataset BBH \
  --bbh_data_dir data/bbh \
  --max_examples_per_task 100 \
  --remove_bos \
  --steering \
  --steering_vector "./artifacts/output/own/BBH_vllm/vector_500_500/layer_20_transition_reflection_steervec.pt"  \
  --steering_layer 20 \
  --steering_coef -1.0 \
  --start 0
