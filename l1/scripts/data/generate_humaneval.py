import os
import pandas as pd
from datasets import load_dataset

ds_humaneval = load_dataset("openai/openai_humaneval", split="test")

out_dir = os.path.expandvars("./artifacts")
os.makedirs(out_dir, exist_ok=True)

for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []

    for i in range(len(ds_humaneval)):
        row = ds_humaneval[i]

        question = row["prompt"]

        if num_tokens < -1:
            question = (
                f"{question}"
                + "\n\nLet's think step by step and complete the Python function."
                + "\nOutput only valid Python code."
                + f" Think for maximum {abs(num_tokens)} tokens."
            )
        elif num_tokens == -1:
            question = (
                f"{question}"
                + "\n\nLet's think step by step and complete the Python function."
                + "\nOutput only valid Python code."
            )
        else:
            question = (
                f"{question}"
                + "\n\nLet's think step by step and complete the Python function."
                + "\nOutput only valid Python code."
                + f" Think for {num_tokens} tokens."
            )

        all_data.append({
            "data_source": "humaneval",
            "prompt": [{
                "role": "user",
                "content": question
            }],
            "ability": "code",
            "reward_model": {
                "style": "rule",
                "ground_truth": row["entry_point"],
                "num_tokens": num_tokens
            },
            "extra_info": {
                "split": "test",
                "index": i,
                "task_id": row["task_id"],
                "entry_point": row["entry_point"],
                "test": row["test"],
                "canonical_solution": row["canonical_solution"]
            }
        })

    if num_tokens != -1:
        if num_tokens < -1:
            pd.DataFrame(all_data).to_parquet(f'{out_dir}/data9_{num_tokens}/humaneval.parquet')
        else:
            pd.DataFrame(all_data).to_parquet(f'{out_dir}/data_{num_tokens}/humaneval.parquet')
    else:
        pd.DataFrame(all_data).to_parquet(f'{out_dir}/data/humaneval.parquet')
