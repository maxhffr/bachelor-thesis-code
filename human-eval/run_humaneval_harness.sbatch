#!/bin/bash
#SBATCH --job-name=humaneval_harness_seal
#SBATCH --output=artifacts/logs/%x_%j.out
#SBATCH --error=artifacts/logs/%x_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate seal

TOKENS=("512" "1024" "2048" "3600")

#for TOKEN in "${TOKENS[@]}"; do
    evaluate_functional_correctness \
      "/work/maxhoffe/SEAL/artifacts/output/own/HUMANEVAL_steering/alpha+0.2/baseline_10000_vector_500_500_layer_20_transition_reflection_steervec/coef_-0.2_remove_bos/humaneval_samples.jsonl"
#done
