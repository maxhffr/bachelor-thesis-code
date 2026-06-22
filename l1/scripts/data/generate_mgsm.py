import os
import pandas as pd
import numpy as np
from datasets import load_dataset

LANGS = ["de", "fr", "es", "ru", "zh", "ja"]

out_dir = os.path.expandvars("./artifacts")
os.makedirs(out_dir, exist_ok=True)

for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    for lang in LANGS:
        ds = load_dataset("alibashir/mgsm-gold", lang, split="test")
        all_data = []

        for i in range(len(ds)):
            row = ds[i]

            question = row["question"]

            ground_truth = row["answer"].strip().replace(",", "")

            if num_tokens < -1:
                question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + f" Think for maximum {abs(num_tokens)} tokens."
            else:
                question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + f" Think for {num_tokens} tokens."

            all_data.append({
                    "data_source": "mgsm",
                    "prompt": [{
                        "role": "user",
                        "content": question
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": ground_truth,
                        "num_tokens": num_tokens
                    },
                    "extra_info": {
                        'split': 'test',
                        'index': i,
                        'language': lang
                    }
                })
        if num_tokens != -1:
            if num_tokens < -1:
                pd.DataFrame(all_data).to_parquet(f'{out_dir}/data9_{num_tokens}/mgsm_{lang}.parquet')
            else:
                pd.DataFrame(all_data).to_parquet(f'{out_dir}/data_{num_tokens}/mgsm_{lang}.parquet')
        else:
            pd.DataFrame(all_data).to_parquet(f'{out_dir}/data/mgsm_{lang}.parquet')
