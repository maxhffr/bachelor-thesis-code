import argparse
import json
import os
import re
import string
from collections import Counter, defaultdict
from tqdm import tqdm


YES_NO_TASKS = {
    "web_of_lies",
    "navigate",
    "sports_understanding",
    "causal_judgement",
}

DYCK_TASKS = {
    "dyck_languages",
}

VALID_INVALID_TASKS = {
    "formal_fallacies",
}

TRUE_FALSE_TASKS = {
    "boolean_expressions",
}

WORD_SORTING_TASKS = {
    "word_sorting",
}

NUMBER_TASKS = {
    "multistep_arithmetic_two",
    "object_counting",
}

MULTIPLE_CHOICE_TASKS = {
    "date_understanding",
    "disambiguation_qa",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "movie_recommendation",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "temporal_sequences",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
}


def strip_thinking(text):
    if text is None:
        return ""

    text = str(text)

    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)

    return text.strip()


def extract_boxed(text):
    """
    extracts final \\boxed{...} answer.
    """
    if text is None:
        return None

    text = str(text)

    matches = list(re.finditer(r"\\boxed\s*\{", text))
    if not matches:
        matches = list(re.finditer(r"boxed\s*\{", text))

    if not matches:
        return None

    start = matches[-1].end() - 1

    depth = 0
    out = []

    for c in text[start:]:
        if c == "{":
            depth += 1
            if depth > 1:
                out.append(c)
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(c)
        else:
            if depth >= 1:
                out.append(c)

    return None


def get_final_text(text):
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed

    text = strip_thinking(text)
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]

    if lines:
        return lines[-1]

    return str(text).strip()


def normalize_basic(text):
    text = get_final_text(text)
    text = str(text).strip().lower()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_choice(text):
    """
    normalizes multiple choice answers
    """
    text = get_final_text(text)
    text = str(text).strip()

    matches = re.findall(r"\(([A-G])\)", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    matches = re.findall(
        r"(?:answer|option|choice|final answer)\s*(?:is|:)?\s*\(?([A-G])\)?",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()

    matches = re.findall(r"\b([A-G])\b", text, flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    return normalize_basic(text).upper()


def normalize_yes_no(text):
    text = normalize_basic(text)
    matches = re.findall(r"\b(yes|no)\b", text)
    if matches:
        return matches[-1]
    return text


def normalize_true_false(text):
    text = normalize_basic(text)
    matches = re.findall(r"\b(true|false)\b", text)
    if matches:
        return matches[-1]
    return text


def normalize_valid_invalid(text):
    text = normalize_basic(text)
    matches = re.findall(r"\b(valid|invalid)\b", text)
    if matches:
        return matches[-1]
    return text


def normalize_number(text):
    text = get_final_text(text)
    text = str(text).replace(",", "")

    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text)

    if matches:
        num = matches[-1]
        if num.endswith(".0"):
            num = num[:-2]
        return num

    return normalize_basic(text)


def normalize_dyck(text):
    text = get_final_text(text)

    chars = re.findall(r"[\}\]\)\>]", str(text))

    return " ".join(chars)


def extract_word_sorting_final_text(text):
    """
    extracts final texts from word_sorting
    """
    if text is None:
        return ""

    raw = str(text)

    boxed = extract_boxed(raw)
    if boxed is not None:
        return boxed

    visible = strip_thinking(raw)

    marker_pattern = re.compile(
        r"(?:\*\*)?\s*(?:final\s+answer|answer)\s*(?:\*\*)?\s*[:：]\s*(.*)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    marker_matches = list(marker_pattern.finditer(visible))

    if marker_matches:
        candidate = marker_matches[-1].group(1).strip()

        numbered = extract_word_sorting_numbered_list(candidate)
        if numbered:
            return numbered

        lines = [clean_word_sorting_line(line) for line in candidate.splitlines()]
        lines = [line for line in lines if line]

        if lines:
            return lines[0]

        return candidate.strip()

    numbered = extract_word_sorting_numbered_list(visible)
    if numbered:
        return numbered

    comma_line = extract_word_sorting_comma_line(visible)
    if comma_line:
        return comma_line

    lines = [clean_word_sorting_line(line) for line in visible.splitlines()]
    lines = [line for line in lines if line]

    if lines:
        return lines[-1]

    return visible.strip()


def clean_word_sorting_line(line):
    line = str(line).strip()

    line = re.sub(r"^\*+\s*", "", line)
    line = re.sub(r"\s*\*+$", "", line)

    line = re.sub(r"[.;:]+$", "", line)

    return line.strip()


def extract_word_sorting_numbered_list(text):
    """
    extracts final word lists into the correct answer format
    """
    items = []

    for line in str(text).splitlines():
        line = line.strip()

        m = re.match(r"^\s*(?:\d+[\.\)]|[-*])\s+(.+?)\s*$", line)
        if not m:
            continue

        item = clean_word_sorting_line(m.group(1))
        item = re.sub(r"[,\.;:]+$", "", item).strip()

        if item:
            items.append(item)

    if len(items) >= 2:
        return ", ".join(items)

    return None


def extract_word_sorting_comma_line(text):
    """
    extracts final words seprated by , correctly
    """
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]

    for line in reversed(lines):
        cleaned = clean_word_sorting_line(line)

        if "," in cleaned and len([x for x in cleaned.split(",") if x.strip()]) >= 2:
            return cleaned

    return None


def normalize_word_sorting(text):
    text = extract_word_sorting_final_text(text)
    text = str(text).strip().lower()

    text = re.sub(
        r"^(the words sorted alphabetically are|the sorted words are|sorted words are|"
        r"the answer is|answer is|final answer is|answer)\s*:?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\b\d+[\.\)]\s*", " ", text)

    keep_apostrophe = "'"
    punctuation_without_apostrophe = string.punctuation.replace(keep_apostrophe, "")
    text = text.translate(str.maketrans(punctuation_without_apostrophe, " " * len(punctuation_without_apostrophe)))

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_bbh_answer(text, task):
    task = str(task).strip()

    if task in YES_NO_TASKS:
        return normalize_yes_no(text)

    if task in TRUE_FALSE_TASKS:
        return normalize_true_false(text)

    if task in VALID_INVALID_TASKS:
        return normalize_valid_invalid(text)

    if task in NUMBER_TASKS:
        return normalize_number(text)

    if task in DYCK_TASKS:
        return normalize_dyck(text)

    if task in WORD_SORTING_TASKS:
        return normalize_word_sorting(text)

    if task in MULTIPLE_CHOICE_TASKS:
        return normalize_choice(text)

    return normalize_basic(text)


def main(res_path, save=False, k=None, output_dir=None):
    with open(res_path, "r") as f:
        data = [json.loads(line) for line in f if line.strip()]

    for example in tqdm(data):
        if "model_generation" not in example:
            example["model_generation"] = example["model_output"]

        if k is not None:
            example["model_generation"] = example["model_generation"][:k]

        task = example.get("task", "")

        gt_raw = example["answer"]
        gt_norm = normalize_bbh_answer(gt_raw, task)

        all_extracted = [
            extract_word_sorting_final_text(pred) if task in WORD_SORTING_TASKS else get_final_text(pred)
            for pred in example["model_generation"]
        ]

        all_pred = [
            normalize_bbh_answer(pred, task)
            for pred in example["model_generation"]
        ]

        all_eval = [
            pred == gt_norm
            for pred in all_pred
        ]

        effective_pred = [
            pred
            for pred, raw in zip(all_pred, example["model_generation"])
            if "boxed" in str(raw)
        ]

        if len(effective_pred) == 0:
            effective_pred = all_pred

        counter = Counter(effective_pred)
        mv_pred = counter.most_common(1)[0][0]
        mv_index = all_pred.index(mv_pred)
        mv_eval = all_eval[mv_index]

        example["gt_norm"] = gt_norm
        example["all_pred"] = all_pred
        example["all_eval"] = all_eval
        example["all_extracted"] = all_extracted
        example["mv_pred"] = mv_pred
        example["mv_eval"] = mv_eval
        example["mv_index"] = mv_index

    acc = sum(example["mv_eval"] for example in data) / len(data) if data else 0.0
    print(f"Accuracy: {acc:.3f}")

    by_task = defaultdict(lambda: {"correct": 0, "total": 0})

    for example in data:
        task = example.get("task", "unknown")
        by_task[task]["correct"] += int(example["mv_eval"])
        by_task[task]["total"] += 1

    print()
    print("Per-task accuracy:")
    for task in sorted(by_task):
        c = by_task[task]["correct"]
        t = by_task[task]["total"]
        task_acc = c / t if t else 0.0
        print(f"{task:45s} {task_acc:.3f} ({c}/{t})")

    if save:
        if output_dir is None:
            output_dir = os.path.dirname(res_path)

        os.makedirs(output_dir, exist_ok=True)

        out_file = os.path.join(output_dir, "bbh_eval.jsonl")
        with open(out_file, "w") as f:
            for example in data:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")

        metric_file = os.path.join(output_dir, "metrics.json")
        with open(metric_file, "w") as f:
            json.dump(
                {
                    "acc": acc,
                    "per_task": {
                        task: {
                            "acc": vals["correct"] / vals["total"] if vals["total"] else 0.0,
                            "correct": vals["correct"],
                            "total": vals["total"],
                        }
                        for task, vals in sorted(by_task.items())
                    },
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("res_path", type=str)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    main(args.res_path, save=args.save, k=args.k, output_dir=args.output_dir)
