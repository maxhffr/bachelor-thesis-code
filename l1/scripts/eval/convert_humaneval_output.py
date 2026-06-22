import os
import re
import json
import pandas as pd


def extract_code(response: str) -> str:
    if not response:
        return ""

    text = response.strip()

    # Remove <think>...</think>
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

    # Prefer fenced python code blocks
    fenced = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced[0].strip()

    fenced_generic = re.findall(r"```\s*(.*?)```", text, flags=re.DOTALL)
    if fenced_generic:
        return fenced_generic[0].strip()

    # If the model starts directly with code, keep it
    lines = text.splitlines()

    # Try to find the first likely code line
    start_idx = 0
    code_starts = (
        "def ",
        "from ",
        "import ",
        "class ",
        "@",
        "if ",
        "for ",
        "while ",
        "return ",
    )

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(code_starts):
            start_idx = i
            break

    candidate = "\n".join(lines[start_idx:]).strip()

    # Remove trailing explanation after code if it clearly starts
    stop_markers = [
        "\nTo solve this problem",
        "\n### ",
        "\nExplanation",
        "\nApproach",
    ]
    for marker in stop_markers:
        pos = candidate.find(marker)
        if pos != -1:
            candidate = candidate[:pos].strip()

    return candidate


def parquet_to_humaneval_jsonl(input_path: str, output_path: str) -> None:
    df = pd.read_parquet(input_path)

    rows_out = []

    for _, row in df.iterrows():
        task_id = row["extra_info"]["task_id"]
        responses = row["responses"]

        for resp in responses:
            completion = extract_code(resp)
            rows_out.append({
                "task_id": task_id,
                "completion": completion
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in rows_out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(rows_out)} completions to {output_path}")


if __name__ == "__main__":
    input_path = os.path.expandvars("$WORK/l1/artifacts/output/output_own_512/humaneval.parquet")
    output_path = os.path.expandvars("$WORK/l1/artifacts/output/output_own_512/humaneval_samples.jsonl")
    parquet_to_humaneval_jsonl(input_path, output_path)
