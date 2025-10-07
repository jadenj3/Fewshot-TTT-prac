# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Test-Time Training (TTT) research project for few-shot learning on language models. It uses vLLM for inference and Torchtune for LoRA fine-tuning to improve performance on BIG-Bench-Hard reasoning tasks.

The core idea: given a task, use k-shot examples to fine-tune a LoRA adapter at test time, then evaluate on the remaining examples. This contrasts with baseline zero-shot/few-shot prompting.

## Commands

### Environment Setup
```bash
# Activate virtual environment (Python 3.12 required)
source tttenv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Initialize git submodules (required for BIG-Bench-Hard and Torchtune)
git submodule update --init --recursive
```

### Running Experiments
```bash
# Baseline evaluation (zero-shot/few-shot)
python src/methods/baseline.py --model_dir <MODEL_PATH> --output_file results.json --k 0

# TTT with LoRA fine-tuning (per-task adapters)
python src/methods/ttt.py --model_dir <MODEL_PATH> --output_file results.json --k 10 --num_training_steps 40

# Shared TTT (single adapter across all tasks)
python src/methods/shared_ttt.py --model_dir <MODEL_PATH> --output_file results.json --k 10 --num_training_steps 40

# End-to-end fine-tuning (no in-context learning)
python src/methods/e2e.py --model_dir <MODEL_PATH> --output_file results.json

# Quick smoke test (first task only)
python src/methods/baseline.py --model_dir <MODEL_PATH> --output_file logs/current/baseline.json --task_start 0 --task_end 1
```

### SLURM Job Submission
```bash
# Submit baseline experiments
sbatch scripts/baseline.sh

# Submit TTT parameter sweeps (runs array jobs)
sbatch scripts/ttt_sweep.sh

# Submit shared TTT experiments
sbatch scripts/shared_ttt_sweep.sh

# Submit E2E experiments
sbatch scripts/e2e.sh
```

## Architecture

### Core Modules
- **src/methods/**: Method implementations
  - `baseline.py`: Zero/few-shot evaluation with optional majority voting
  - `ttt.py`: Per-task TTT using LoRA adaptation (creates one adapter per task)
  - `shared_ttt.py`: Shared TTT (trains single adapter on all tasks combined)
  - `e2e.py`: End-to-end fine-tuning without in-context learning
  - `shared_e2e.py`: Shared E2E variant
  - `random_guessing.py`: Random baseline for comparison

- **src/tasks.py**: Defines 27 BIG-Bench-Hard reasoning tasks with metadata:
  - `generation_length`: Max tokens to generate
  - `task_prompt`: Task description
  - `answer_format`: Expected format (e.g., "(A)", "Yes/No", integer)
  - `choices`: Number of multiple choice options (or None for free-form)

- **src/utils.py**: Shared utilities
  - `inference_vllm()`: Main inference function with vLLM, handles prompt formatting
  - `post_process_answer()`: Cleans model outputs to match expected format
  - `compute_accuracy()`: Case-insensitive exact match accuracy

### Key Design Patterns

1. **vLLM Integration**: All inference goes through `inference_vllm()` for efficient batch processing
   - Sets temperature=0.0 for deterministic outputs
   - Uses LoRARequest parameter for adapter inference
   - Stops at "\nQ:" to prevent generating additional examples

2. **LoRA Fine-tuning Workflow** (in ttt.py):
   - Create training dataset: k-shot examples formatted as "Q: ... A: ..." pairs
   - Generate YAML config using `create_torchtune_config()`
   - Run Torchtune recipe: `tune run lora_finetune_single_device --config <yaml>`
   - Copy checkpoint safetensors to adapter directory
   - Load adapter with vLLM using LoRARequest

3. **Dataset Types** (for TTT training):
   - `text_completion_dataset`: All inputs and outputs (Q: ... A: ... pairs)
   - `masked_text_completion_dataset`: Only last output (for single-example focus)
   - `masked_inputs_text_completion_dataset`: All outputs only

4. **Task Abstraction**: All methods iterate over `TASKS.items()` and load data from `external/BIG-Bench-Hard/bbh/{task}.json`
   - Each JSON has "examples" list with "input" and "target" fields
   - Methods split examples into training (k-shot) and evaluation sets

5. **Majority Voting**: Optional mode that runs inference with multiple random permutations of few-shot examples and picks most common answer

6. **JSON Logging**: Results saved with structured metadata (experiment name, hyperparameters, per-task accuracies)

### External Dependencies
- **external/BIG-Bench-Hard/**: Evaluation benchmark (git submodule)
- **external/torchtune/**: Custom fork for LoRA fine-tuning (git submodule)

## Development Guidelines

### Code Style
- 4-space indentation
- snake_case for functions and variables
- UpperCamelCase for classes
- Type hints required for public functions
- Import order: standard library → third-party → local

### Testing
No formal test suite exists. Validate changes by:
1. Running smoke tests with `--task_start 0 --task_end 1`
2. Checking deterministic output with fixed seeds
3. Comparing baseline results against expected BIG-Bench performance

### Common Modifications
- **Adding new tasks**: Update `TASKS` dictionary in `src/tasks.py` with generation_length, task_prompt, answer_format, choices
- **Changing hyperparameters**: Modify SLURM scripts (scripts/*.sh) or pass command-line arguments
  - Key TTT params: `--lr`, `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--num_training_steps`, `--batch_size`, `--epochs`
  - Key baseline params: `--k` (num shots), `--majority_vote`, `--vote_permutations`
- **Adjusting LoRA config**: Edit `create_torchtune_config()` in ttt.py or shared_ttt.py
- **Modifying inference**: Update `inference_vllm()` in `src/utils.py`

### Important Implementation Details
- TTT methods shuffle examples by default before creating training data (controlled by `--shuffle` flag)
- `--leave_one_out` mode excludes the last example from the inference prompt (for ablation studies)
- Model checkpoints follow Llama naming: `model-0000{1-4}-of-00004.safetensors`
- Adapter paths must be formatted as: `{adapter_dir}:{adapter_name}` for vLLM LoRARequest
- All methods support `--task_start` and `--task_end` for running subsets of the 27 tasks
