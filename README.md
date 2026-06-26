# Bachelor Thesis Code

This repository contains the code used for the experiments in my bachelor thesis:

**Evaluating the Impact of Length Calibration on the Efficiency of Reasoning Models**

The repository is based on modified versions of the L1 and SEAL repositories used for the experiments.

The original repositories are:

* L1: https://github.com/cmu-l3/l1
* SEAL: https://github.com/VITA-Group/SEAL

## Repository Structure

```text
.
├── l1/          # Modified L1 repository
├── SEAL/        # Modified SEAL repository
├── human-eval/  # HumanEval evaluation code
├── envs/        # Environment files
└── README.md
```

## Setup

### Installation

```bash
git clone https://github.com/maxhffr/bachelor-thesis-code
cd bachelor-thesis-code
```

The environment files used for the experiments are stored in the `envs/` folder.

The `.yml` files create the basic Conda environments. The required Python packages are installed afterwards using the corresponding `requirements.txt` files.

### L1 Environment

```bash
conda env create -f envs/l1_environment.yml
conda activate l1
pip install -r envs/l1_requirements.txt
pip install --no-deps --no-cache-dir "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
./l1/scripts/setup/patch_verl_fsdp.sh
```

The patch is required, because without it, verl may try to build an optimizer during generation although no optimizer configuration is used.

### SEAL Environment

```bash
conda env create -f envs/seal_environment.yml
conda activate seal
pip install -r envs/seal_requirements.txt
```

## Usage

The general workflow for using this repository is:

1. Set up the corresponding environment.
2. Prepare the required benchmark.
3. Run the evaluation scripts for the selected benchmark.
4. Compute the final metrics.

The folders `l1/` and `SEAL/` contain the modified code and the scripts used for the experiments in the thesis. For the basic usage of the original frameworks, please refer to the original READMEs of L1 and SEAL.

## L1 Experiments

To run the L1 experiments, switch to the L1 folder:

```bash
cd l1
```

The L1 experiments follow the structure of the original L1 repository. First, the required benchmark data has to be prepared. Afterwards, the evaluation scripts can be executed for the selected model, benchmark, and token length.

The general structure is:

```bash
# prepare benchmark data
./own/replicate/prepare_data.sh
or
sbatch own/run_scripts/prepare_data.sbatch

# baseline
./own/replicate/eval_baseline.sh agentica-org/DeepScaleR-1.5B-Preview

# L1 model
./own/replicate/eval_<benchmark>_exact.sh l3lab/L1-Qwen-1.5B-Exact
./own/replicate/eval_<benchmark>_max.sh l3lab/L1-Qwen-1.5B-Max
```

The evaluation and data preparation files for the extended benchmark tests are included in:
```bash
l1/own/replicate/
```

The run scripts used for the thesis experiments are included in the repository. 
```bash
l1/own/run_scripts/
```

These scripts can be used but some scripts may contain machine-specific paths, model paths, output paths, or SLURM settings and should be adapted before running them on another system.

For more details on the original L1 workflow, see:

https://github.com/cmu-l3/l1

## SEAL Experiments

To run the SEAL experiments, switch to the SEAL folder:

```bash
cd SEAL
```

The SEAL experiments follow the structure of the original SEAL repository. First, the required benchmark data has to be prepared. Then, either baseline evaluations or steering evaluations can be executed.

The general structure is:

```bash
# prepare benchmark data
python prepare_custom_benchmarks.py

# baseline 
python eval_<benchmark>_vllm.py ...
or
sbatch own/run_scripts/run_<benchmark>_vllm.sbatch

# SEAL steering
python eval_<benchmark>_steering.py ...
or
sbatch own/run_scripts/run_<benchmark>_steering.sbatch
sbatch own/run_scripts/run_<benchmark>_ov_steering.sbatch
```

The exact script depends on the benchmark. The run scripts used for the thesis experiments are included in:

```text
SEAL/own/run_scripts/
```

Some scripts may contain machine-specific paths, model paths, output paths, or SLURM settings and should be adapted before running them on another system.

### MGSM

For the benchmark MGSM, test with translated prompts are included in 
```bash
# baseline
sbatch own/run_scripts/run_mgsm_vllm_lc.sbatch

# SEAL steering
sbatch own/run_scripts/run_mgsm_steering_lc.sbatch
sbatch own/run_scripts/run_mgsm_ov_steering_lc.sbatch
```
### Token Usage

To capture the token usage of a benchmark run, `compute_avg_tokens.py` has to be used. An example run could look like this:

```bash
python compute_avg_tokens.py \
  "<input_path>/math_eval.jsonl" \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --output_path "<output_path>/token_metrics.json"
```
For HumanEval the results and output file have to be used:

```bash
python compute_avg_tokens.py \
  "<input_path>/humaneval_raw_outputs.jsonl" \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
   --output_path "<output_path>/token_metrics.json" \
   --humaneval_results_path "<input_path>/humaneval_samples.jsonl_results.jsonl"
```


For more details on the original SEAL workflow, see:

https://github.com/VITA-Group/SEAL

## HumanEval

The HumanEval benchmark is evaluated using the HumanEval-Harness `human-eval/`.

The output files have to be prepared to be used by the harness.

### L1

For L1, the .parquet output file has to be converted into a .jsonl file. For this the file `create_json_for_harness.py` is used.

```bash
python create_json_for_harness.py \
  --input <input_path>/humaneval.parquet \
  --output <output_path>/humaneval_samples_pass1.jsonl \
  --first-only
```

### SEAL

For SEAL, the .jsonl output file `humaneval_samples.jsonl` can be used.

### Using the Harness

```bash
evaluate_functional_correctness \
  "<input_path>/humaneval_samples.jsonl"
```

For more details on the HumanEval Harness, see

https://github.com/openai/human-eval

## Benchmarks

The experiments use several benchmarks, including:

* GSM8K
* MGSM
* MMLU
* BBH
* HumanEval

Not every benchmark is used in exactly the same way for L1 and SEAL. Some benchmarks require additional preprocessing or separate evaluation scripts.

## Notes

This repository is intended as a research artifact for the bachelor thesis and not as a fully packaged software library.

Large model files, generated outputs, and benchmark result files are not included in this repository. Some paths and settings may need to be adapted depending on the local system.
