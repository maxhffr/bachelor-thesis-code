MODEL_PATH=$1

./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 512 --datasets gsm8k
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 1024 --datasets gsm8k
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 2048 --datasets gsm8k
./scripts/eval/eval_model_token_own.sh --model $MODEL_PATH  --num-tokens 3600 --datasets gsm8k
