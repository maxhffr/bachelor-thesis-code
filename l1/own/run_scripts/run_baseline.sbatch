#!/bin/bash
#SBATCH --job-name=l1_baseline
#SBATCH --output=artifacts/logs/baseline/%x_%j.out
#SBATCH --error=artifacts/logs/baseline/%x_%j.err
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-vram-48gb
#SBATCH --mail-type=BEGIN,END,FAIL

source "$HOME/miniconda3/etc/profile.d/conda.sh"

conda activate l1_testverl

./own/replicate/eval_baseline.sh agentica-org/DeepScaleR-1.5B-Preview

#./scripts/eval/eval_model_bbh.sh \
 # --model agentica-org/DeepScaleR-1.5B-Preview \
 # --datasets bbh

#./scripts/eval/eval_model.sh \
 # --model agentica-org/DeepScaleR-1.5B-Preview \
 # --datasets gpqa lsat mmlu_1000 gsm8k_1000 mgsm_de mgsm_fr mgsm_es mgsm_ru mgsm_zh mgsm_ja humaneval mgsm_de_lc mgsm_fr_lc mgsm_es_lc mgsm_ru_lc mgsm_zh_lc mgsm_ja_lc gsm8k mmlu

#./scripts/eval/eval_model_math.sh \
 # --model agentica-org/DeepScaleR-1.5B-Preview \
 # --datasets aime math amc minerva olympiad_bench
