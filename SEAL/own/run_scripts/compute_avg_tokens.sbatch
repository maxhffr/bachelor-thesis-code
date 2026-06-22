#!/bin/bash
#SBATCH --job-name=seal_humaneval_steering_tokens
#SBATCH --output=artifacts/logs/token_length/%x_%j.out
#SBATCH --error=artifacts/logs/token_length/%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

LANGS=("de" "fr" "es" "ru" "zh" "ja")

#for LANG in "${LANGS[@]}"; do
    python compute_avg_tokens.py \
      "./artifacts/output/own/HUMANEVAL_steering/alpha+0.2/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-0.2_remove_bos/humaneval_raw_outputs.jsonl" \
      --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --output_path "./artifacts/output/own/HUMANEVAL_steering/alpha+0.2/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-0.2_remove_bos/token_metrics.json" \
      --humaneval_results_path "./artifacts/output/own/HUMANEVAL_steering/alpha+0.2/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-0.2_remove_bos/humaneval_samples.jsonl_results.jsonl"
#done

