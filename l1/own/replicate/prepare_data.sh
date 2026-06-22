for tokens in 512 1024 2048 3600; do
  mkdir -p "artifacts/data_${tokens}"
  mkdir -p "artifacts/data9_-${tokens}"
done

mkdir -p "artifacts/data"
mkdir -p "artifacts/data9_-1"
mkdir -p "artifacts/data_-1"

python ./scripts/data/generate_gsm8k.py
python ./scripts/data/generate_mmlu_full.py
python ./scripts/data/generate_mgsm.py
python ./scripts/data/generate_bbh.py
python ./scripts/data/generate_humaneval.py
python ./scripts/data/generate_mgsm_lc.py
