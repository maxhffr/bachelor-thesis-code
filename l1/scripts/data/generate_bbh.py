import os
import pandas as pd
from datasets import load_dataset

TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

out_dir = os.path.expandvars("./artifacts")
os.makedirs(out_dir, exist_ok=True)

for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    all_data = []

    for task in TASKS:
        ds_bbh = load_dataset("lukaemon/bbh", task, split="test")

        for i in range(len(ds_bbh)):
            row = ds_bbh[i]
            question = row["input"]
            ground_truth = str(row["target"]).strip()

            if num_tokens < -1:
                question = (
                    f"{question}"
                    + "\n\nLet's think step by step and output the final answer within \\boxed{}."
                    + f" Think for maximum {abs(num_tokens)} tokens."
                )
            elif num_tokens == -1:
                question = (
                    f"{question}"
                    + "\n\nLet's think step by step and output the final answer within \\boxed{}."
                )
            else:
                question = (
                    f"{question}"
                    + "\n\nLet's think step by step and output the final answer within \\boxed{}."
                    + f" Think for {num_tokens} tokens."
                )

            all_data.append({
                "data_source": "bbh",
                "prompt": [{
                    "role": "user",
                    "content": question
                }],
                "ability": "reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth,
                    "num_tokens": num_tokens
                },
                "extra_info": {
                    "split": "test",
                    "index": i,
                    "task": task
                }
            })

    if num_tokens != -1:
        if num_tokens < -1:
            pd.DataFrame(all_data).to_parquet(f'{out_dir}/data9_{num_tokens}/bbh.parquet')
        else:
            pd.DataFrame(all_data).to_parquet(f'{out_dir}/data_{num_tokens}/bbh.parquet')
    else:
        pd.DataFrame(all_data).to_parquet(f'{out_dir}/data/bbh.parquet')
