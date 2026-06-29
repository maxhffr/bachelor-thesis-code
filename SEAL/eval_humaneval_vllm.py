import argparse
import os
import re
import json
import random
import torch
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from collections import Counter
from datasets import load_dataset
from functools import partial


import sys
import os
import gc
from code_evaluation import codegen_metrics, load_code_generation_dataset, get_deepseekcode_question_template_answer, extract_code, extract_instance_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"



def logit_adjustment(token_ids, logits, adjust_ids, values, max_len=-1):
    if max_len <= 0 or len(token_ids) <= max_len:
        logits[adjust_ids.to(logits.device)] += values
    return logits

#Helper
def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def build_humaneval_prompt(example):
    return (
        example["prompt"]
        + "\n\n"
        + "Let's think step by step and complete the Python function.\n"
        + "Output only valid Python code."
    )

def write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def strip_code_fences(text):
    text = text.strip()

    # if model outputs ```python ... ```
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL)
    if m:
        text = m.group(1).strip()

    return text


def clean_humaneval_completion(text, prompt=None, entry_point=None):
    text = text.strip()

    # Normalize model outputs for the HumanEval Harness
    code_blocks = re.findall(
        r"```(?:python)?\s*(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if code_blocks:
        text = code_blocks[-1].strip()

    final_markers = [
        "Final code:",
        "Final Code:",
        "final code:",
        "Final answer:",
        "Final Answer:",
        "Answer:",
        "Solution:",
    ]
    for marker in final_markers:
        if marker in text:
            text = text.split(marker, 1)[1].strip()

    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()

    if entry_point:
        pattern = rf"def\s+{re.escape(entry_point)}\s*\(.*?\)\s*(?:->\s*.*?)?:\s*\n"
        m = re.search(pattern, text, flags=re.DOTALL)
        if m:
            text = text[m.end():]

    lines = text.splitlines()
    code_start = None

    code_line_pattern = re.compile(
        r"^\s*(return\b|for\b|while\b|if\b|elif\b|else:|try:|except\b|with\b|raise\b|import\b|from\b|"
        r"[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^:]+$)"
    )

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if code_line_pattern.match(line):
            code_start = i
            break

    if code_start is not None:
        text = "\n".join(lines[code_start:])

    stop_markers = [
        "\n\n# Explanation",
        "\n\nExplanation:",
        "\n\nThe function",
        "\n\nThis function",
        "\n\nIn this code",
        "\n\nif __name__",
        "\n\nMETADATA",
        "\n\ndef check(",
        "\n\nassert ",
        "\nclass ",
        "\ndef ",
        "\n\nWait,",
        "\n\nHmm,",
        "\n\nSo,",
        "\n\nLet me",
        "\n\nTesting",
        "\n\nAlternatively,",
        "\n\nExample ",
        "\n\nAnother test",
        "\n\nWhat about",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0]

    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()

        if not stripped:
            cleaned_lines.append(line)
            continue

        reasoning_starts = (
            "Wait,",
            "Hmm,",
            "So,",
            "Example ",
            "Another test",
            "What about",
            "Check ",
            "The ",
            "This ",
            "Let ",
        )

        if stripped.startswith(reasoning_starts):
            break

        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^=]+:", stripped):
            break

        cleaned_lines.append(line)

    lines = "\n".join(cleaned_lines).strip("\n").splitlines()

    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()

    if not lines:
        return ""

    first_nonempty = next((line for line in lines if line.strip()), "")
    if first_nonempty and not first_nonempty.startswith((" ", "\t")):
        lines = ["    " + line if line.strip() else line for line in lines]

    return "\n".join(lines).rstrip() + "\n"


def main(args):
    random.seed(42)

    print("Loading data...")

    if args.benchmark == "humaneval":
        if args.data_path is None:
            args.data_path = "data/humaneval/humaneval_test.jsonl"

        benchmark = load_jsonl(args.data_path)

    else:
        if args.release == "v5-v1":
            benchmark_v5 = load_code_generation_dataset(release_version="release_v5")
            benchmark_v1 = load_code_generation_dataset(release_version="release_v1")
            benchmark = [d for d in benchmark_v5 if d not in benchmark_v1]
            assert len(benchmark)==480
        else:
            benchmark = load_code_generation_dataset(release_version=args.release)
    
    if args.max_examples and len(benchmark) > args.max_examples:
        benchmark = benchmark[:args.max_examples]

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)

     # set padding side to left for batch generation
    tokenizer.padding_side = "left"

    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts = []

    for i, example in enumerate(benchmark):
        if args.benchmark == "humaneval":
            prompt = build_humaneval_prompt(example)
        else:
            prompt = get_deepseekcode_question_template_answer(example)

        if args.use_chat_format:
            messages = [{"role": "user", "content": prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if args.remove_bos and tokenizer.bos_token is not None and prompt.startswith(tokenizer.bos_token):
                prompt = prompt[len(tokenizer.bos_token):]

        prompts.append(prompt)
    with open(os.path.join(args.save_dir, "example_prompt.txt"), 'w') as fout:
        fout.write(prompts[0])

    model = LLM(model=args.model_name_or_path, tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path, swap_space=16, gpu_memory_utilization=0.93, enable_lora=args.peft is not None, tensor_parallel_size=torch.cuda.device_count(), max_lora_rank=128, max_model_len=args.max_tokens+2000)

    if not args.logit_adjustment:

        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens)
    else:
        vocab = tokenizer.get_vocab()
        logit_adjustment_tokens = torch.LongTensor([vocab[token] for token in vocab.keys() if any([x in token for x in args.logit_adjustment_tokens])]).to("cuda")
        logit_adjustment_process = partial(logit_adjustment, adjust_ids=logit_adjustment_tokens, values=args.logit_adjustment_value, max_len=args.logit_adjustment_max_len)
        sampling_params = SamplingParams(n=1,
                                        temperature=0,
                                        max_tokens=args.max_tokens,
                                        logits_processors=[logit_adjustment_process]
                                        )
    
    if args.peft is not None:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params, lora_request=LoRARequest("lora_path", 1, lora_path=args.peft))
    else:
        outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

    results = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            attempts.append(ith_output.text)
        results.append(attempts)

    if args.benchmark == "humaneval":
        harness_rows = []
        raw_rows = []

        for example, outputs_list in zip(benchmark, results):
            raw_response = outputs_list[0]

            completion = clean_humaneval_completion(
                raw_response,
                prompt=example.get("prompt"),
                entry_point=example.get("entry_point"),
            )

            harness_rows.append({
                "task_id": example["task_id"],
                "completion": completion,
            })

            raw_rows.append({
                "task_id": example["task_id"],
                "entry_point": example.get("entry_point"),
                "prompt": example.get("prompt"),
                "raw_response": raw_response,
                "completion": completion,
            })

        write_jsonl(
            os.path.join(args.save_dir, "humaneval_samples.jsonl"),
            harness_rows,
        )

        write_jsonl(
            os.path.join(args.save_dir, "humaneval_raw_outputs.jsonl"),
            raw_rows,
        )

        print("Saved HumanEval harness file:")
        print(os.path.join(args.save_dir, "humaneval_samples.jsonl"))
        return    

    combined_results = [
        (
            outputs_list,
            [extract_code(output) for output in outputs_list],
        )
        for outputs_list in results
    ]

    save_results = [
        instance.insert_output(outputs_list, extracted_list)
        for instance, (outputs_list, extracted_list) in zip(
            benchmark, combined_results
        )
    ]

    with open(os.path.join(args.save_dir, "predictions.jsonl"), "w") as f:
        json.dump(save_results, f, indent=4)


    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    generations = [extracted for _, extracted in combined_results]

    metrics = codegen_metrics(
        eval_samples,
        generations,
        num_process_evaluate=12,
        timeout=10,
    )

    print(metrics[0]["pass@1"])

    graded = extract_instance_results(metrics[1])
    metadatas = metrics[2]
    save_eval_results = [
        instance.insert_output_evaluation(
            outputs_list, extracted_list, graded_list, metadata=meta
        )
        for instance, (outputs_list, extracted_list), graded_list, meta in zip(
            benchmark, combined_results, graded, metadatas
        )
    ]

    with open(os.path.join(args.save_dir, "metrics.jsonl"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(args.save_dir, "code_eval.jsonl"), "w") as f:
        json.dump(save_eval_results, f, indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/gsm"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--use_chat_format",
        action="store_true",
        help="If given, we will use the chat format for the prompts."
    )
    parser.add_argument(
        "--release",
        type=str,
        default="release_v1",
    )
    parser.add_argument(
        "--remove_bos",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--logit_adjustment",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--logit_adjustment_tokens",
        type=str,
        nargs="*",
        default=[]
    )
    parser.add_argument(
        "--logit_adjustment_value",
        type=float,
        default=0.0
    )
    parser.add_argument(
        "--logit_adjustment_max_len",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="livecodebench",
        choices=["livecodebench", "humaneval"],
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    if args.logit_adjustment:
        name = "_".join(args.logit_adjustment_tokens)+f"_value_{args.logit_adjustment_value}"
        if args.logit_adjustment_max_len>0:
            name += f"_first{args.logit_adjustment_max_len}"
        
        args.save_dir = os.path.join(args.save_dir, "logit-adjustment", name)



    main(args)

        
