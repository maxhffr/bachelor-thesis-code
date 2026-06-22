import os
import pandas as pd
import numpy as np
from datasets import load_dataset

LANGS = ["de", "fr", "es", "ru", "zh", "ja"]

COT_PROMPT_BY_LANG = {
    "de": "Lass uns Schritt für Schritt denken und die finale Antwort in \\boxed{} ausgeben.",
    "fr": "Réfléchissons étape par étape et donnons la réponse finale dans \\boxed{}.",
    "es": "Pensemos paso a paso y demos la respuesta final en \\boxed{}.",
    "ru": "Давайте подумаем шаг за шагом и выведем окончательный ответ в \\boxed{}.",
    "zh": "让我们一步一步地思考，并将最终答案输出在 \\boxed{} 中。",
    "ja": "段階的に考え、最終的な答えを \\boxed{} の中に出力しましょう。",
}

THINK_FOR_EXACT_BY_LANG = {
    "de": "Denke für {tokens} Tokens.",
    "fr": "Réfléchis pendant {tokens} tokens.",
    "es": "Piensa durante {tokens} tokens.",
    "ru": "Думай в течение {tokens} токенов.",
    "zh": "思考 {tokens} 个 token。",
    "ja": "{tokens} トークン分考えてください。",
}

THINK_FOR_MAX_BY_LANG = {
    "de": "Denke für maximal {tokens} Tokens.",
    "fr": "Réfléchis pendant au maximum {tokens} tokens.",
    "es": "Piensa durante un máximo de {tokens} tokens.",
    "ru": "Думай не более {tokens} токенов.",
    "zh": "最多思考 {tokens} 个 token。",
    "ja": "最大 {tokens} トークン分考えてください。",
}

out_dir = os.path.expandvars("./artifacts")
os.makedirs(out_dir, exist_ok=True)

for num_tokens in [512, 1024, 2048, 3600, -512, -1024, -2048, -3600, -1]:
    for lang in LANGS:
        ds = load_dataset("alibashir/mgsm-gold", lang, split="test")
        all_data = []

        cot_prompt = COT_PROMPT_BY_LANG[lang]
        think_for_exact_template = THINK_FOR_EXACT_BY_LANG[lang]
        think_for_max_template = THINK_FOR_MAX_BY_LANG[lang]

        for i in range(len(ds)):
            row = ds[i]

            question = row["question"]

            ground_truth = row["answer"].strip().replace(",", "")

            if num_tokens < -1:
                question = (
                    f"{question}"
                    + f"\n\n{cot_prompt}"
                    + " "
                    + think_for_max_template.format(tokens=abs(num_tokens))
                )
            else:
                question = (
                    f"{question}"
                    + f"\n\n{cot_prompt}"
                    + (
                        " " + think_for_exact_template.format(tokens=num_tokens)
                        if num_tokens != -1
                        else ""
                    )
                )

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
                pd.DataFrame(all_data).to_parquet(f'{out_dir}/data9_{num_tokens}/mgsm_{lang}_lc.parquet')
            else:
                pd.DataFrame(all_data).to_parquet(f'{out_dir}/data_{num_tokens}/mgsm_{lang}_lc.parquet')
        else:
            pd.DataFrame(all_data).to_parquet(f'{out_dir}/data/mgsm_{lang}_lc.parquet')
