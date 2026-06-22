MODEL_PATH=$1

./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 512 --datasets mmlu
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 1024 --datasets mmlu
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 2048 --datasets mmlu
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 3600 --datasets mmlu
