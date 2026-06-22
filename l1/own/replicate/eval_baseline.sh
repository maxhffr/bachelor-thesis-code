MODEL_PATH=$1

./scripts/eval/eval_model_bbh.sh \
  --model $MODEL_PATH \
  --datasets bbh

./scripts/eval/eval_model.sh \
  --model $MODEL_PATH \
  --datasets gpqa lsat mmlu_1000 gsm8k_1000 mgsm_de mgsm_fr mgsm_es mgsm_ru mgsm_zh mgsm_ja humaneval mgsm_de_lc mgsm_fr_lc mgsm_es_lc mgsm_ru_lc mgsm_zh_lc mgsm_ja_lc gsm8k mmlu

./scripts/eval/eval_model_math.sh \
  --model $MODEL_PATH \
  --datasets aime math amc minerva olympiad_bench
