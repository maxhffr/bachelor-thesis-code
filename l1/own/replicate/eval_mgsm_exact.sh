MODEL_PATH=$1

./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 512 --datasets mgsm_de mgsm_de_lc mgsm_fr mgsm_fr_lc mgsm_es mgsm_es_lc mgsm_ru mgsm_ru_lc mgsm_zh mgsm_zh_lc mgsm_ja mgsm_ja_lc
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 1024 --datasets mgsm_de mgsm_de_lc mgsm_fr mgsm_fr_lc mgsm_es mgsm_es_lc mgsm_ru mgsm_ru_lc mgsm_zh mgsm_zh_lc mgsm_ja mgsm_ja_lc
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 2048 --datasets mgsm_de mgsm_de_lc mgsm_fr mgsm_fr_lc mgsm_es mgsm_es_lc mgsm_ru mgsm_ru_lc mgsm_zh mgsm_zh_lc mgsm_ja mgsm_ja_lc
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 3600 --datasets mgsm_de mgsm_de_lc mgsm_fr mgsm_fr_lc mgsm_es mgsm_es_lc mgsm_ru mgsm_ru_lc mgsm_zh mgsm_zh_lc mgsm_ja mgsm_ja_lc
