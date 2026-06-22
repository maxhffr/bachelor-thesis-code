#!/bin/bash
#SBATCH --job-name=seal_hidden_humaneval
#SBATCH --output=artifacts/logs/humaneval/vector/%x_%j.out
#SBATCH --error=artifacts/logs/humaneval/vector/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

python hidden_analysis_humaneval.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data_dir $WORK/SEAL/artifacts/output/own/HUMANEVAL_vllm \
  --dataset humaneval \
  --humaneval_results_path $WORK/SEAL/artifacts/output/own/HUMANEVAL_vllm/humaneval_samples.jsonl_results.jsonl \
  --humaneval_reasoning_path $WORK/SEAL/artifacts/output/own/HUMANEVAL_vllm/humaneval_raw_outputs.jsonl \
  --type correct

python hidden_analysis_humaneval.py \
  --model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --data_dir $WORK/SEAL/artifacts/output/own/HUMANEVAL_vllm \
  --dataset humaneval \
  --humaneval_results_path $WORK/SEAL/artifacts/output/own/HUMANEVAL_vllm/humaneval_samples.jsonl_results.jsonl \
  --humaneval_reasoning_path $WORK/SEAL/artifacts/output/own/HUMANEVAL_vllm/humaneval_raw_outputs.jsonl \
  --type incorrect
