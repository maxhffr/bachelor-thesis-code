import os
import pandas as pd
import numpy as np
from datasets import load_dataset

ds_gsm8k = load_dataset("openai/gsm8k", "main")

out_dir = os.path.expandvars("./artifacts")
os.makedirs(out_dir, exist_ok=True)

for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []
    for i in range(len(ds_gsm8k['test'])):
        row = ds_gsm8k['test'][i]

        question = row["question"]

        correct_choice = row["answer"].split("####")[-1].strip().replace(",", "")

        if num_tokens < -1:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + f" Think for maximum {abs(num_tokens)} tokens."
        else:
            question = f"{question}"+"\n\nLet's think step by step and output the final answer within \\boxed{}." + f" Think for {num_tokens} tokens."

        all_data.append({
                    "data_source": "gsm8k",
                    "prompt": [{
                        "role": "user",
                        "content": question
                    }],
                    "ability": "math",
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": correct_choice,
                        "num_tokens": num_tokens
                    },
                    "extra_info": {
                        'split': 'test',
                        'index': i
                    }
                })
    # Suffle all_data randomly
    np.random.seed(42)
    indices = np.arange(len(all_data))
    np.random.shuffle(indices)
    if num_tokens != -1:
        if num_tokens < -1:
            pd.DataFrame(all_data).to_parquet(f'{out_dir}/data9_{num_tokens}/gsm8k.parquet')
        else:
            pd.DataFrame(all_data).to_parquet(f'{out_dir}/data_{num_tokens}/gsm8k.parquet')
    else:
        pd.DataFrame(all_data).to_parquet(f'{out_dir}/data/gsm8k.parquet')
