python scripts/data/deepscaler_dataset.py --num_tokens 512 --local_dir "./artifacts/deepscaler"
python scripts/data/deepscaler_dataset.py --num_tokens 1024 --local_dir "./artifacts/deepscaler"
python scripts/data/deepscaler_dataset.py --num_tokens 2048 --local_dir "./artifacts/deepscaler"
python scripts/data/deepscaler_dataset.py --num_tokens 3600 --local_dir "./artifacts/deepscaler"

python scripts/data/deepscaler_dataset.py --num_tokens -512 --local_dir "./artifacts/deepscaler"
python scripts/data/deepscaler_dataset.py --num_tokens -1024 --local_dir "./artifacts/deepscaler"
python scripts/data/deepscaler_dataset.py --num_tokens -2048 --local_dir "./artifacts/deepscaler"
python scripts/data/deepscaler_dataset.py --num_tokens -3600 --local_dir "./artifacts/deepscaler"

python scripts/data/generate_aime.py
python scripts/data/generate_gpqa.py
python scripts/data/generate_lsat.py
python scripts/data/generate_mmlu.py
