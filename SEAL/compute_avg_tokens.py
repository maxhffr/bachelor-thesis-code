import argparse
import json
import os
import re
from collections import defaultdict
from statistics import mean, median

from transformers import AutoTokenizer


def extract_reasoning_part(text):
    """
    Approximation:
    reasoning = alles vor der letzten boxed-Antwort.
    Falls keine boxed-Antwort existiert, nehmen wir die komplette Generation.
    """
    if text is None:
        return ""

    text = str(text)

    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        matches = list(re.finditer(r"boxed\s*\{", text))

    if not matches:
        return text

    return text[:matches[-1].start()]


def count_tokens(tokenizer, text):
    if text is None:
        text = ""
    return len(tokenizer.encode(str(text), add_special_tokens=False))


def get_group_name(row):
    """
    Für BBH gibt es tasktypes, z.B. date_understanding.
    Für MMLU/GSM/MGSM/MATH gibt es oft keine echte task-Spalte.
    Dann gruppieren wir alles unter 'all'.
    """
    task = row.get("task", None)

    if task is not None and str(task).strip() and str(task).strip().lower() not in {"mmlu", "gsm", "gsm8k", "mgsm", "math", "math500"}:
        return str(task).strip()

    return "all"


def get_eval_value(row, generation_index=0):
    """
    Versucht, aus bereits evaluierten Dateien die Korrektheit zu lesen.
    Unterstützt u.a.:
    - BBH/MATH: mv_eval, all_eval
    - LiveCodeBench: graded_list, pass@1
    - allgemeine Felder: correct, is_correct, eval
    """

    # LiveCodeBench: pro Generation eine Bewertung
    if "graded_list" in row:
        graded_list = row["graded_list"]

        if isinstance(graded_list, list) and len(graded_list) > 0:
            generation_index = max(0, min(generation_index, len(graded_list) - 1))
            return bool(graded_list[generation_index])

        if isinstance(graded_list, bool):
            return bool(graded_list)

    # LiveCodeBench: aggregierter pass@1
    if "pass@1" in row:
        return bool(row["pass@1"])

    # BBH/MATH/GSM etc.
    for key in ["mv_eval", "correct", "is_correct", "eval", "passed"]:
        if key in row:
            return bool(row[key])

    if "all_eval" in row:
        all_eval = row["all_eval"]

        if isinstance(all_eval, list) and len(all_eval) > 0:
            generation_index = max(0, min(generation_index, len(all_eval) - 1))
            return bool(all_eval[generation_index])

        if isinstance(all_eval, bool):
            return bool(all_eval)

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--use_mv_index", action="store_true")
    parser.add_argument(
        "--humaneval_results_path",
        type=str,
        default=None,
        help="Optional path to humaneval_samples.jsonl_results.jsonl. Used only for reading passed true/false while token counts are computed from input_path.",
    )
    args = parser.parse_args()

    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def load_json_or_jsonl(path):
        with open(path, "r") as f:
            content = f.read().strip()

        if not content:
            return []

        # Fall 1: normale JSON-Datei, z.B. Liste von Dicts
        if content[0] in ["[", "{"]:
            try:
                data = json.loads(content)

                if isinstance(data, list):
                    return data

                if isinstance(data, dict):
                    # Falls die Beispiele in einem Feld liegen
                    for key in ["data", "results", "examples", "predictions"]:
                        if key in data and isinstance(data[key], list):
                            return data[key]

                    # Sonst einzelnes Dict als eine Zeile behandeln
                    return [data]

            except json.JSONDecodeError:
                pass

    # Fall 2: JSONL-Datei
        rows = []
        with open(path, "r") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Could not parse line {line_no}: {line[:200]}")
                    raise e

        return rows


    rows = load_json_or_jsonl(args.input_path)

    humaneval_passed_by_task_id = {}

    if args.humaneval_results_path is not None:
        humaneval_result_rows = load_json_or_jsonl(args.humaneval_results_path)

        for r in humaneval_result_rows:
            task_id = r.get("task_id")
            if task_id is not None and "passed" in r:
                humaneval_passed_by_task_id[task_id] = bool(r["passed"])

        print(
            f"Loaded HumanEval results: "
            f"{sum(1 for v in humaneval_passed_by_task_id.values() if v)}/"
            f"{len(humaneval_passed_by_task_id)} passed"
        )
    stats = []
    by_group = defaultdict(list)

    for i, row in enumerate(rows):
        generations = (
            row.get("model_generation")
            or row.get("model_output")
            or row.get("output_list")        # LiveCodeBench
            or row.get("raw_response")
            or row.get("generated_code")
            or row.get("completion")
            or row.get("prediction")
            or row.get("response")
            or row.get("output")
            or []
        )

        if isinstance(generations, str):
            generations = [generations]

        if not generations:
            generation = ""
            generation_index = 0
        else:
            if args.use_mv_index and "mv_index" in row:
                generation_index = int(row["mv_index"])
                generation_index = max(0, min(generation_index, len(generations) - 1))
            else:
                generation_index = 0

            generation = generations[generation_index]

        completion_tokens = count_tokens(tokenizer, generation)
        reasoning_text = extract_reasoning_part(generation)
        reasoning_tokens_approx = count_tokens(tokenizer, reasoning_text)

        group = get_group_name(row)
        eval_value = get_eval_value(row, generation_index=generation_index)

        task_id = row.get("task_id")
        if task_id is not None and task_id in humaneval_passed_by_task_id:
            eval_value = humaneval_passed_by_task_id[task_id]

        item = {
            "idx": i,
            "id": row.get("id") or row.get("task_id"),
            "group": group,
            "generation_index": generation_index,
            "completion_tokens": completion_tokens,
            "reasoning_tokens_approx": reasoning_tokens_approx,
            "has_boxed": "boxed" in str(generation),
            "correct": eval_value,
        }

        stats.append(item)
        by_group[group].append(item)

    completion_values = [x["completion_tokens"] for x in stats]
    reasoning_values = [x["reasoning_tokens_approx"] for x in stats]

    eval_items = [x for x in stats if x["correct"] is not None]
    correct = sum(1 for x in eval_items if x["correct"])
    total_evaluated = len(eval_items)
    acc = correct / total_evaluated if total_evaluated > 0 else None

    summary = {
        "num_examples": len(stats),
        "avg_completion_tokens": mean(completion_values) if completion_values else 0,
        "median_completion_tokens": median(completion_values) if completion_values else 0,
        "avg_reasoning_tokens_approx": mean(reasoning_values) if reasoning_values else 0,
        "median_reasoning_tokens_approx": median(reasoning_values) if reasoning_values else 0,
        "boxed_rate": sum(x["has_boxed"] for x in stats) / len(stats) if stats else 0,
        "acc": acc,
        "correct": correct if total_evaluated > 0 else None,
        "total_evaluated": total_evaluated if total_evaluated > 0 else None,
        "per_task": {},
    }

    for group, items in sorted(by_group.items()):
        c_vals = [x["completion_tokens"] for x in items]
        r_vals = [x["reasoning_tokens_approx"] for x in items]

        group_eval_items = [x for x in items if x["correct"] is not None]
        group_correct = sum(1 for x in group_eval_items if x["correct"])
        group_total_evaluated = len(group_eval_items)
        group_acc = group_correct / group_total_evaluated if group_total_evaluated > 0 else None

        summary["per_task"][group] = {
            "num_examples": len(items),
            "avg_completion_tokens": mean(c_vals) if c_vals else 0,
            "median_completion_tokens": median(c_vals) if c_vals else 0,
            "avg_reasoning_tokens_approx": mean(r_vals) if r_vals else 0,
            "median_reasoning_tokens_approx": median(r_vals) if r_vals else 0,
            "boxed_rate": sum(x["has_boxed"] for x in items) / len(items) if items else 0,
            "acc": group_acc,
            "correct": group_correct if group_total_evaluated > 0 else None,
            "total_evaluated": group_total_evaluated if group_total_evaluated > 0 else None,
        }

    print("Overall:")
    print(f"  examples:                      {summary['num_examples']}")
    print(f"  avg completion tokens:          {summary['avg_completion_tokens']:.2f}")
    print(f"  median completion tokens:       {summary['median_completion_tokens']:.2f}")
    print(f"  avg reasoning tokens approx:    {summary['avg_reasoning_tokens_approx']:.2f}")
    print(f"  median reasoning tokens approx: {summary['median_reasoning_tokens_approx']:.2f}")
    print(f"  boxed rate:                     {summary['boxed_rate']:.2%}")

    if acc is not None:
        print(f"  accuracy:                       {acc:.4f} ({correct}/{total_evaluated})")
    else:
        print("  accuracy:                       not available in input file")

    # Nur sinnvoll groß ausgeben, wenn es mehr als eine Gruppe gibt.
    if len(summary["per_task"]) > 1:
        print()
        print("Per task:")
        for group, vals in summary["per_task"].items():
            acc_str = "n/a"
            if vals["acc"] is not None:
                acc_str = f"{vals['acc']:.4f} ({vals['correct']}/{vals['total_evaluated']})"

            print(
                f"{group:45s} "
                f"acc={acc_str:18s} "
                f"avg_tokens={vals['avg_completion_tokens']:.2f} "
                f"reasoning_avg={vals['avg_reasoning_tokens_approx']:.2f} "
                f"boxed={vals['boxed_rate']:.2%} "
                f"n={vals['num_examples']}"
            )

    if args.output_path is None:
        out_dir = os.path.dirname(args.input_path)
        args.output_path = os.path.join(out_dir, "token_metrics.json")

    with open(args.output_path, "w") as f:
        json.dump(summary, f, indent=2)

    detail_path = args.output_path.replace(".json", "_details.jsonl")
    with open(detail_path, "w") as f:
        for item in stats:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print()
    print(f"Saved summary to: {args.output_path}")
    print(f"Saved details to: {detail_path}")


if __name__ == "__main__":
    main()
