import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_existing(row, keys, default=None):
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return default


def normalize_text(text):
    return (text or "").strip().replace("\r\n", "\n")


def generate_eval_data_math(data_dir, data_path=None):
    """Original SEAL/MATH-style loader.

    Expects rows with prompt, model_generation and all_eval. Returns two lists:
    correct and incorrect, each containing {prompt, response, gt, task}.
    """
    correct, incorrect = [], []

    eval_path = data_path or os.path.join(data_dir, "math_eval.jsonl")
    eval_data = read_jsonl(eval_path)

    for e in eval_data:
        local_correct, local_incorrect = [], []

        prompt = e.get("prompt")
        generations = e.get("model_generation")
        evals = e.get("all_eval")

        gt = (
            e.get("answer")
            or e.get("gt")
            or e.get("ground_truth")
            or e.get("target")
        )

        task = e.get("task", e.get("dataset", ""))

        if prompt is None:
            raise KeyError(f"Missing prompt in eval row. Keys: {list(e.keys())}")
        if generations is None:
            raise KeyError(f"Missing model_generation in eval row. Keys: {list(e.keys())}")
        if evals is None:
            raise KeyError(f"Missing all_eval in eval row. Keys: {list(e.keys())}")

        for o, c in zip(generations, evals):
            item = {
                "prompt": prompt,
                "response": o,
                "gt": gt,
                "task": task,
            }
            if c:
                local_correct.append(item)
            else:
                local_incorrect.append(item)

        correct.extend(local_correct)
        incorrect.extend(local_incorrect)

    return correct, incorrect


def build_humaneval_result_lookup(results_rows):
    """Build lookup structures from humaneval_samples.jsonl_results.jsonl.

    The result file usually contains one row per generated completion, e.g.:
    {"task_id": "HumanEval/0", "completion": "...", "passed": true}

    We keep both:
    1. exact lookup by (task_id, completion), useful when the completion can be
       extracted from the reasoning row.
    2. ordered lookup by task_id, useful when both files were written in the same
       generation order.
    """
    by_exact = {}
    by_task_order = defaultdict(list)

    for row in results_rows:
        task_id = row.get("task_id")
        completion = normalize_text(row.get("completion", ""))
        passed = bool(row.get("passed", row.get("result") == "passed"))

        if task_id is None:
            raise KeyError(f"HumanEval result row misses task_id. Keys: {list(row.keys())}")

        by_exact[(task_id, completion)] = passed
        by_task_order[task_id].append(passed)

    return by_exact, by_task_order


def maybe_extract_completion_from_response(full_response):
    """Best-effort extraction of the final code completion from a full reasoning response.

    This is only a fallback for exact matching. For DeepSeek-style outputs the full
    response may look like <think>...</think>\n    def ... or directly contain the body.
    The hidden-state generation still uses the complete full_response.
    """
    text = full_response or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return normalize_text(text)


def generate_eval_data_humaneval(results_path, reasoning_path, match_by="order"):
    """Create correct/incorrect lists for HumanEval hidden-state extraction.

    results_path:
        JSONL file from the HumanEval evaluator, containing task_id, completion,
        and passed/result.

    reasoning_path:
        JSONL file containing the original prompt and the full model output with
        reasoning trace. The loader accepts several common key names:
        prompt: prompt/question/input
        response: response/model_generation/output/generation/completion/text
        task_id: task_id/task

    match_by:
        "order"  -> match per task_id by generation order. This is usually safest
                    if the result file was produced from the same samples file.
        "exact"  -> try exact match with the evaluated completion first, fallback
                    to order if no exact match is found.
    """
    results_rows = read_jsonl(results_path)
    reasoning_rows = read_jsonl(reasoning_path)

    result_by_exact, result_by_task_order = build_humaneval_result_lookup(results_rows)
    used_index_per_task = defaultdict(int)

    correct, incorrect = [], []
    unmatched = []

    for row_idx, row in enumerate(reasoning_rows):
        task_id = first_existing(row, ["task_id", "task"])
        prompt = first_existing(row, ["prompt", "question", "input"], "")
        response = first_existing(
            row,
            ["response", "model_generation", "output", "generation", "completion", "text"],
        )

        # Some scripts store model_generation as a list, even for one sample.
        # In that case, use each generation as a separate candidate.
        if isinstance(response, list):
            responses = response
        else:
            responses = [response]

        if task_id is None:
            raise KeyError(f"Reasoning row {row_idx} misses task_id/task. Keys: {list(row.keys())}")

        for resp in responses:
            if resp is None:
                raise KeyError(f"Reasoning row {row_idx} misses response/generation text. Keys: {list(row.keys())}")

            passed = None
            if match_by == "exact":
                completion_candidate = maybe_extract_completion_from_response(resp)
                passed = result_by_exact.get((task_id, completion_candidate))

            if passed is None:
                pos = used_index_per_task[task_id]
                task_results = result_by_task_order.get(task_id, [])
                if pos < len(task_results):
                    passed = task_results[pos]
                    used_index_per_task[task_id] += 1
                else:
                    unmatched.append({"row_idx": row_idx, "task_id": task_id, "pos": pos})
                    continue

            item = {
                "prompt": prompt,
                "response": resp,  # IMPORTANT: full reasoning trace, not only final function
                "gt": None,
                "task": task_id,
            }

            if passed:
                correct.append(item)
            else:
                incorrect.append(item)

    if unmatched:
        preview = unmatched[:5]
        raise ValueError(
            f"Could not match {len(unmatched)} reasoning generations to HumanEval results. "
            f"First unmatched entries: {preview}. "
            "Check that reasoning_path and results_path come from the same run/order, "
            "or try --humaneval_match_by exact."
        )

    return correct, incorrect


def generate_index(text, tokenizer, split_id, think_only=True):
    check_words = [
        "verify", "make sure", "hold on", "think again", "'s correct",
        "'s incorrect", "Let me check", "seems right",
    ]
    check_prefix = ["Wait"]
    switch_words = [
        "think differently", "another way", "another approach", "another method",
        "another solution", "another strategy", "another technique",
    ]
    switch_prefix = ["Alternatively"]

    tokens = tokenizer.encode(text)
    if think_only:
        think_begin_ids = tokenizer.encode("<think>", add_special_tokens=False)
        think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if not think_begin_ids or think_begin_ids[0] not in tokens:
            return [], [], []

        think_begin_id = think_begin_ids[0]
        think_end_id = think_end_ids[0] if think_end_ids else None

        start = tokens.index(think_begin_id) + 1
        if think_end_id is None or think_end_id not in tokens[start:]:
            end = len(tokens)
        else:
            end = tokens.index(think_end_id, start)
        think_tokens = tokens[start:end]
    else:
        think_tokens = tokens
        start = 0

    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    step_index = []
    check_index = []
    switch_index = []

    for i in range(len(index) - 1):
        # This keeps the original behavior: step hidden state is taken at the
        # split token / beginning of each step.
        step_index.append(index[i] + start)
        step = think_tokens[index[i] + 1:index[i + 1]]
        step = tokenizer.decode(step).strip(" ").strip("\n")
        step_lower = step.lower()

        if any(step_lower.startswith(p.lower()) for p in check_prefix) or any(
            w.lower() in step_lower for w in check_words
        ):
            check_index.append(i)
        elif any(step_lower.startswith(p.lower()) for p in switch_prefix) or any(
            w.lower() in step_lower for w in switch_words
        ):
            switch_index.append(i)

    return step_index, check_index, switch_index


def generate(model_path, data, save_dir):
    think_only = "deepseek" in model_path.lower()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    split_id = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]

    prompts = [d["prompt"] + d["response"] for d in data]

    layer_num = model.config.num_hidden_layers + 1
    hidden_dict = [{} for _ in range(layer_num)]

    for k, p in tqdm(enumerate(prompts), total=len(prompts)):
        tokenized_batch = tokenizer([p], return_tensors="pt", padding=True)
        tokenized_batch = {key: value.to(model.device) for key, value in tokenized_batch.items()}

        with torch.no_grad():
            output = model(**tokenized_batch, output_hidden_states=True)
            hidden_states = [h.detach().cpu() for h in output.hidden_states]

        step_index, check_index, switch_index = generate_index(
            p, tokenizer, split_id, think_only=think_only
        )
        step_index = torch.LongTensor(step_index)
        check_index = torch.LongTensor(check_index)
        switch_index = torch.LongTensor(switch_index)

        for i in range(len(hidden_states)):
            h = hidden_states[i][0]
            step_h = h[step_index] if len(step_index) > 0 else torch.empty(0, h.shape[-1])
            hidden_dict[i][k] = {
                "step": step_h,
                "check_index": check_index,
                "switch_index": switch_index,
            }

        del hidden_states

    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, f"{save_dir}/hidden.pt")
    with open(f"{save_dir}/prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "humaneval"])
    parser.add_argument("--humaneval_results_path", type=str, default=None)
    parser.add_argument("--humaneval_reasoning_path", type=str, default=None)
    parser.add_argument("--humaneval_match_by", type=str, default="order", choices=["order", "exact"])
    parser.add_argument("--type", type=str, default="correct", choices=["correct", "incorrect"])
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    if args.dataset == "math":
        correct, incorrect = generate_eval_data_math(args.data_dir, args.data_path)
    else:
        if not args.humaneval_results_path or not args.humaneval_reasoning_path:
            raise ValueError(
                "For --dataset humaneval you must provide "
                "--humaneval_results_path and --humaneval_reasoning_path."
            )
        correct, incorrect = generate_eval_data_humaneval(
            results_path=args.humaneval_results_path,
            reasoning_path=args.humaneval_reasoning_path,
            match_by=args.humaneval_match_by,
        )

    print(f"Loaded {len(correct)} correct and {len(incorrect)} incorrect samples.")

    data = correct if args.type == "correct" else incorrect
    save_dir = f"{args.data_dir}/hidden_{args.dataset}_{args.type}"

    if args.start != -1:
        data = data[args.start:]
        if args.sample != -1:
            data = data[:args.sample]
            save_dir = f"{save_dir}_{args.start}_{args.start + args.sample}"
        else:
            save_dir = f"{save_dir}_{args.start}_-1"
    elif args.sample != -1:
        data = data[:args.sample]
        save_dir = f"{save_dir}_0_{args.sample}"

    print(f"Generating hidden states for {len(data)} {args.type} samples.")
    print(save_dir)
    generate(args.model_path, data, save_dir)


if __name__ == "__main__":
    main()
import argparse
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def first_existing(row, keys, default=None):
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return default


def normalize_text(text):
    return (text or "").strip().replace("\r\n", "\n")


def generate_eval_data_math(data_dir, data_path=None):
    """Original SEAL/MATH-style loader.

    Expects rows with prompt, model_generation and all_eval. Returns two lists:
    correct and incorrect, each containing {prompt, response, gt, task}.
    """
    correct, incorrect = [], []

    eval_path = data_path or os.path.join(data_dir, "math_eval.jsonl")
    eval_data = read_jsonl(eval_path)

    for e in eval_data:
        local_correct, local_incorrect = [], []

        prompt = e.get("prompt")
        generations = e.get("model_generation")
        evals = e.get("all_eval")

        gt = (
            e.get("answer")
            or e.get("gt")
            or e.get("ground_truth")
            or e.get("target")
        )

        task = e.get("task", e.get("dataset", ""))

        if prompt is None:
            raise KeyError(f"Missing prompt in eval row. Keys: {list(e.keys())}")
        if generations is None:
            raise KeyError(f"Missing model_generation in eval row. Keys: {list(e.keys())}")
        if evals is None:
            raise KeyError(f"Missing all_eval in eval row. Keys: {list(e.keys())}")

        for o, c in zip(generations, evals):
            item = {
                "prompt": prompt,
                "response": o,
                "gt": gt,
                "task": task,
            }
            if c:
                local_correct.append(item)
            else:
                local_incorrect.append(item)

        correct.extend(local_correct)
        incorrect.extend(local_incorrect)

    return correct, incorrect


def build_humaneval_result_lookup(results_rows):
    """Build lookup structures from humaneval_samples.jsonl_results.jsonl.

    The result file usually contains one row per generated completion, e.g.:
    {"task_id": "HumanEval/0", "completion": "...", "passed": true}

    We keep both:
    1. exact lookup by (task_id, completion), useful when the completion can be
       extracted from the reasoning row.
    2. ordered lookup by task_id, useful when both files were written in the same
       generation order.
    """
    by_exact = {}
    by_task_order = defaultdict(list)

    for row in results_rows:
        task_id = row.get("task_id")
        completion = normalize_text(row.get("completion", ""))
        passed = bool(row.get("passed", row.get("result") == "passed"))

        if task_id is None:
            raise KeyError(f"HumanEval result row misses task_id. Keys: {list(row.keys())}")

        by_exact[(task_id, completion)] = passed
        by_task_order[task_id].append(passed)

    return by_exact, by_task_order


def maybe_extract_completion_from_response(full_response):
    """Best-effort extraction of the final code completion from a full reasoning response.

    This is only a fallback for exact matching. For DeepSeek-style outputs the full
    response may look like <think>...</think>\n    def ... or directly contain the body.
    The hidden-state generation still uses the complete full_response.
    """
    text = full_response or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return normalize_text(text)


def generate_eval_data_humaneval(results_path, reasoning_path, match_by="order"):
    """Create correct/incorrect lists for HumanEval hidden-state extraction.

    results_path:
        JSONL file from the HumanEval evaluator, containing task_id, completion,
        and passed/result.

    reasoning_path:
        JSONL file containing the original prompt and the full model output with
        reasoning trace. The loader accepts several common key names:
        prompt: prompt/question/input
        response: response/model_generation/output/generation/completion/text
        task_id: task_id/task

    match_by:
        "order"  -> match per task_id by generation order. This is usually safest
                    if the result file was produced from the same samples file.
        "exact"  -> try exact match with the evaluated completion first, fallback
                    to order if no exact match is found.
    """
    results_rows = read_jsonl(results_path)
    reasoning_rows = read_jsonl(reasoning_path)

    result_by_exact, result_by_task_order = build_humaneval_result_lookup(results_rows)
    used_index_per_task = defaultdict(int)

    correct, incorrect = [], []
    unmatched = []

    for row_idx, row in enumerate(reasoning_rows):
        task_id = first_existing(row, ["task_id", "task"])
        prompt = first_existing(row, ["prompt", "question", "input"], "")
        response = first_existing(
            row,
            ["response", "model_generation", "output", "generation", "completion", "text"],
        )

        # Some scripts store model_generation as a list, even for one sample.
        # In that case, use each generation as a separate candidate.
        if isinstance(response, list):
            responses = response
        else:
            responses = [response]

        if task_id is None:
            raise KeyError(f"Reasoning row {row_idx} misses task_id/task. Keys: {list(row.keys())}")

        for resp in responses:
            if resp is None:
                raise KeyError(f"Reasoning row {row_idx} misses response/generation text. Keys: {list(row.keys())}")

            passed = None
            if match_by == "exact":
                completion_candidate = maybe_extract_completion_from_response(resp)
                passed = result_by_exact.get((task_id, completion_candidate))

            if passed is None:
                pos = used_index_per_task[task_id]
                task_results = result_by_task_order.get(task_id, [])
                if pos < len(task_results):
                    passed = task_results[pos]
                    used_index_per_task[task_id] += 1
                else:
                    unmatched.append({"row_idx": row_idx, "task_id": task_id, "pos": pos})
                    continue

            item = {
                "prompt": prompt,
                "response": resp,  # IMPORTANT: full reasoning trace, not only final function
                "gt": None,
                "task": task_id,
            }

            if passed:
                correct.append(item)
            else:
                incorrect.append(item)

    if unmatched:
        preview = unmatched[:5]
        raise ValueError(
            f"Could not match {len(unmatched)} reasoning generations to HumanEval results. "
            f"First unmatched entries: {preview}. "
            "Check that reasoning_path and results_path come from the same run/order, "
            "or try --humaneval_match_by exact."
        )

    return correct, incorrect


def generate_index(text, tokenizer, split_id, think_only=True):
    check_words = [
        "verify", "make sure", "hold on", "think again", "'s correct",
        "'s incorrect", "Let me check", "seems right",
    ]
    check_prefix = ["Wait"]
    switch_words = [
        "think differently", "another way", "another approach", "another method",
        "another solution", "another strategy", "another technique",
    ]
    switch_prefix = ["Alternatively"]

    tokens = tokenizer.encode(text)
    if think_only:
        think_begin_ids = tokenizer.encode("<think>", add_special_tokens=False)
        think_end_ids = tokenizer.encode("</think>", add_special_tokens=False)
        if not think_begin_ids or think_begin_ids[0] not in tokens:
            return [], [], []

        think_begin_id = think_begin_ids[0]
        think_end_id = think_end_ids[0] if think_end_ids else None

        start = tokens.index(think_begin_id) + 1
        if think_end_id is None or think_end_id not in tokens[start:]:
            end = len(tokens)
        else:
            end = tokens.index(think_end_id, start)
        think_tokens = tokens[start:end]
    else:
        think_tokens = tokens
        start = 0

    index = [i for i, t in enumerate(think_tokens) if t in split_id] + [len(think_tokens)]
    step_index = []
    check_index = []
    switch_index = []

    for i in range(len(index) - 1):
        # This keeps the original behavior: step hidden state is taken at the
        # split token / beginning of each step.
        step_index.append(index[i] + start)
        step = think_tokens[index[i] + 1:index[i + 1]]
        step = tokenizer.decode(step).strip(" ").strip("\n")
        step_lower = step.lower()

        if any(step_lower.startswith(p.lower()) for p in check_prefix) or any(
            w.lower() in step_lower for w in check_words
        ):
            check_index.append(i)
        elif any(step_lower.startswith(p.lower()) for p in switch_prefix) or any(
            w.lower() in step_lower for w in switch_words
        ):
            switch_index.append(i)

    return step_index, check_index, switch_index


def generate(model_path, data, save_dir):
    think_only = "deepseek" in model_path.lower()
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    vocab = tokenizer.get_vocab()
    split_id = [vocab[token] for token in vocab.keys() if "ĊĊ" in token]

    prompts = [d["prompt"] + d["response"] for d in data]

    layer_num = model.config.num_hidden_layers + 1
    hidden_dict = [{} for _ in range(layer_num)]

    for k, p in tqdm(enumerate(prompts), total=len(prompts)):
        tokenized_batch = tokenizer([p], return_tensors="pt", padding=True)
        tokenized_batch = {key: value.to(model.device) for key, value in tokenized_batch.items()}

        with torch.no_grad():
            output = model(**tokenized_batch, output_hidden_states=True)
            hidden_states = [h.detach().cpu() for h in output.hidden_states]

        step_index, check_index, switch_index = generate_index(
            p, tokenizer, split_id, think_only=think_only
        )
        step_index = torch.LongTensor(step_index)
        check_index = torch.LongTensor(check_index)
        switch_index = torch.LongTensor(switch_index)

        for i in range(len(hidden_states)):
            h = hidden_states[i][0]
            step_h = h[step_index] if len(step_index) > 0 else torch.empty(0, h.shape[-1])
            hidden_dict[i][k] = {
                "step": step_h,
                "check_index": check_index,
                "switch_index": switch_index,
            }

        del hidden_states

    os.makedirs(save_dir, exist_ok=True)
    torch.save(hidden_dict, f"{save_dir}/hidden.pt")
    with open(f"{save_dir}/prompts.json", "w", encoding="utf-8") as f:
        json.dump(prompts, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="math", choices=["math", "humaneval"])
    parser.add_argument("--humaneval_results_path", type=str, default=None)
    parser.add_argument("--humaneval_reasoning_path", type=str, default=None)
    parser.add_argument("--humaneval_match_by", type=str, default="order", choices=["order", "exact"])
    parser.add_argument("--type", type=str, default="correct", choices=["correct", "incorrect"])
    parser.add_argument("--start", type=int, default=-1)
    parser.add_argument("--sample", type=int, default=-1)
    args = parser.parse_args()

    if args.dataset == "math":
        correct, incorrect = generate_eval_data_math(args.data_dir, args.data_path)
    else:
        if not args.humaneval_results_path or not args.humaneval_reasoning_path:
            raise ValueError(
                "For --dataset humaneval you must provide "
                "--humaneval_results_path and --humaneval_reasoning_path."
            )
        correct, incorrect = generate_eval_data_humaneval(
            results_path=args.humaneval_results_path,
            reasoning_path=args.humaneval_reasoning_path,
            match_by=args.humaneval_match_by,
        )

    print(f"Loaded {len(correct)} correct and {len(incorrect)} incorrect samples.")

    data = correct if args.type == "correct" else incorrect
    save_dir = f"{args.data_dir}/hidden_{args.dataset}_{args.type}"

    if args.start != -1:
        data = data[args.start:]
        if args.sample != -1:
            data = data[:args.sample]
            save_dir = f"{save_dir}_{args.start}_{args.start + args.sample}"
        else:
            save_dir = f"{save_dir}_{args.start}_-1"
    elif args.sample != -1:
        data = data[:args.sample]
        save_dir = f"{save_dir}_0_{args.sample}"

    print(f"Generating hidden states for {len(data)} {args.type} samples.")
    print(save_dir)
    generate(args.model_path, data, save_dir)


if __name__ == "__main__":
    main()
