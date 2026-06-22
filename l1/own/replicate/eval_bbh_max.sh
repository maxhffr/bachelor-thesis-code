MODEL_PATH=$1

./scripts/eval/eval_model_token_bbh_max.sh --model $MODEL_PATH  --num-tokens -512 --datasets bbh
./scripts/eval/eval_model_token_bbh_max.sh --model $MODEL_PATH  --num-tokens -1024 --datasets bbh
./scripts/eval/eval_model_token_bbh_max.sh --model $MODEL_PATH  --num-tokens -2048 --datasets bbh
./scripts/eval/eval_model_token_bbh_max.sh --model $MODEL_PATH  --num-tokens -3600 --datasets bbh
