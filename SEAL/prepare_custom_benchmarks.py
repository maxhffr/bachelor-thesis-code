from datasets import load_dataset
from pathlib import Path
import json
import re

OUT_DIR = Path("data")

MGSM_LANGS = ["de", "fr", "es", "ru", "zh", "ja"]

BBH_TASKS = [
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

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize_number_string(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    s = s.replace("$", "")
    return s

def extract_gsm8k_final_answer(answer: str) -> str:
    # Standard GSM8K format: "... #### 72"
    if "####" in answer:
        return normalize_number_string(answer.split("####")[-1])
    return normalize_number_string(answer)

def prepare_gsm8k():
    ds = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for i, ex in enumerate(ds):
        rows.append({
            "id": f"gsm8k_{i}",
            "dataset": "gsm8k",
            "question": ex["question"],
            "gt": extract_gsm8k_final_answer(ex["answer"]),
        })
    benchmark_dir = OUT_DIR / "gsm8k"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / "gsm8k" / "gsm8k_test.jsonl"
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")

def prepare_mmlu():
    ds = load_dataset("cais/mmlu", "all", split="test")
    rows = []
    idx_to_letter = ["A", "B", "C", "D"]
    for i, ex in enumerate(ds):
        labeled_choices = [
            f"A: {ex['choices'][0]}",
            f"B: {ex['choices'][1]}",
            f"C: {ex['choices'][2]}",
            f"D: {ex['choices'][3]}",
        ]

        row = {
            "id": f"mmlu_{i}",
            "dataset": "mmlu",
            "question": ex["question"],
            "choices": labeled_choices,
            "gt": idx_to_letter[ex["answer"]],
        }

        # Falls das Dataset ein subject / category Feld mitliefert, mitnehmen
        for key in ("subject", "subcategory", "category"):
            if key in ex:
                row[key] = ex[key]

        rows.append(row)

    benchmark_dir = OUT_DIR / "mmlu"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / "mmlu" / "mmlu_test.jsonl"
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")

def prepare_mgsm():
    for lang in MGSM_LANGS:
        ds = load_dataset("alibashir/mgsm-gold", lang, split="test")
        rows = []
        for i, ex in enumerate(ds):
            rows.append({
                "id": f"mgsm_{lang}_{i}",
                "dataset": "mgsm",
                "language": lang,
                "question": ex["question"],
                "gt": normalize_number_string(str(ex["answer"])),
            })

        benchmark_dir = OUT_DIR / "mgsm"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        out_path = OUT_DIR / "mgsm" / f"mgsm_{lang}_test.jsonl"
        write_jsonl(out_path, rows)
        print(f"Wrote {len(rows)} rows -> {out_path}")

def prepare_humaneval():
    ds = load_dataset("openai/openai_humaneval", split="test")
    rows = []
    for ex in ds:
        rows.append({
            "task_id": ex["task_id"],
            "dataset": "humaneval",
            "prompt": ex["prompt"],
            "canonical_solution": ex["canonical_solution"],
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        })

    benchmark_dir = OUT_DIR / "humaneval"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    out_path = OUT_DIR / "humaneval" / "humaneval_test.jsonl"
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} rows -> {out_path}")

def prepare_bbh():
    for task in BBH_TASKS:
        ds = load_dataset("lukaemon/bbh", task, split="test")
        rows = []
        for i, ex in enumerate(ds):
            rows.append({
                "id": f"bbh_{task}_{i}",
                "dataset": "bbh",
                "task": task,
                "input": ex["input"],
                "gt": ex["target"],
            })

        benchmark_dir = OUT_DIR / "bbh"
        benchmark_dir.mkdir(parents=True, exist_ok=True)

        out_path = OUT_DIR / "bbh" / f"bbh_{task}_test.jsonl"
        write_jsonl(out_path, rows)
        print(f"Wrote {len(rows)} rows -> {out_path}")

if __name__ == "__main__":
    prepare_gsm8k()
    prepare_mmlu()
    prepare_mgsm()
    prepare_humaneval()
    prepare_bbh()
