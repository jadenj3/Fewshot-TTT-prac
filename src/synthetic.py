import os
import sys
import json
import time
import random
import shutil
import argparse
import gc
from collections import Counter

import torch
from vllm import LLM
from vllm.lora.request import LoRARequest

# Local imports
from src.tasks import TASKS
from src.utils import (
    compute_accuracy,
    inference_vllm
)
CHECKPOINT_FILES = [
    'model-00001-of-00004.safetensors',
    'model-00002-of-00004.safetensors',
    'model-00003-of-00004.safetensors',
    'model-00004-of-00004.safetensors'
]

def create_ttt_dataset(
    prefix: str,
    correct_examples: list,
    num_training_steps: int,
    dataset_filename: str,
    shuffle_examples: bool
):
    """
    Create a multi-sample JSON dataset for Torchtune finetuning, with no chain-of-thought.

    - If shuffle_examples=True, each item will be a random permutation of correct_examples.
    - If shuffle_examples=False, each item uses the same order of correct_examples.

    We prepend `prefix` to each sample. Then, for each (Q, A) pair, we append:
       Q: ...
       A: ...
    """
    data_samples = []
    for _ in range(num_training_steps):
        if shuffle_examples:
            perm = correct_examples[:]
            random.shuffle(perm)
            ex_list = perm
        else:
            ex_list = correct_examples

        text = prefix
        for (q, a) in ex_list:
            # Each example has question/answer only (no chain-of-thought)
            text += f"Q: {q}\nA: {a}\n\n"
        data_samples.append({"text": text})

    with open(dataset_filename, 'w') as f:
        json.dump(data_samples, f)

def build_inference_prompt(
    correct_examples: list,
    leave_one_out: bool = False,
    shuffle_examples: bool = False
) -> str:
    """
    Build a single few-shot prompt prefix from the list of correct examples (Q, A).

    If leave_one_out=True, we exclude the *last* example in 'correct_examples'.
    If shuffle_examples=True, we shuffle them first.
    """
    if not correct_examples:
        return ""

    examples_to_use = correct_examples[:]
    if shuffle_examples:
        random.shuffle(examples_to_use)

    if leave_one_out and len(examples_to_use) > 1:
        examples_to_use = examples_to_use[:-1]

    prefix = ""
    for (q, a) in examples_to_use:
        prefix += f"Q: {q}\nA: {a}\n\n"
    return prefix

def majority_vote(answers):
    """
    Given a list of answers (strings), pick the one that appears most frequently.
    If there's a tie, pick whichever candidate is lexicographically smallest.
    """
    counter = Counter(answers)
    max_count = max(counter.values())
    winners = [ans for ans, cnt in counter.items() if cnt == max_count]
    winners.sort()
    return winners[0]


def create_torchtune_config(
    model_dir: str,
    dataset_type: str,
    task: str,
    dataset_filename: str,
    output_dir: str,
    batch_size: int,
    epochs: int,
    lr: float,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float
) -> str:
    """
    Create a YAML config for Torchtune finetuning (LoRA).
    We'll do multiple samples => steps_per_epoch = (#samples / batch_size).
    """
    import yaml

    tokenizer_path = os.path.join(model_dir, "original", "tokenizer.model")
    checkpoint_dir = model_dir

    # Count samples in dataset
    try:
        with open(dataset_filename, 'r') as f:
            data = json.load(f)
        n_samples = len(data)
    except Exception:
        n_samples = 0

    steps_per_epoch = max(1, n_samples // batch_size)
    total_steps = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)

    config = {
        'tokenizer': {
            '_component_': 'torchtune.models.llama3.llama3_tokenizer',
            'path': tokenizer_path,
            'max_seq_len': None
        },
        'dataset': {
            '_component_': f'torchtune.datasets.{dataset_type}',
            'source': 'json',
            'data_files': dataset_filename,
            'column': 'text',
            'split': 'train'
        },
        'model': {
            '_component_': 'torchtune.models.llama3_1.lora_llama3_1_8b',
            'lora_attn_modules': ['q_proj', 'v_proj', 'output_proj'],
            'apply_lora_to_mlp': True,
            'apply_lora_to_output': False,
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout
        },
        'checkpointer': {
            '_component_': 'torchtune.training.FullModelHFCheckpointer',
            'checkpoint_dir': checkpoint_dir,
            'checkpoint_files': CHECKPOINT_FILES,
            'output_dir': output_dir,
            'recipe_checkpoint': None,
            'model_type': 'LLAMA3'
        },
        'save_adapter_weights_only': True,
        'resume_from_checkpoint': False,
        'metric_logger': {
            '_component_': 'torchtune.training.metric_logging.DiskLogger',
            'log_dir': output_dir
        },
        'log_every_n_steps': 1,
        'seed': 42,
        'shuffle': True,
        'batch_size': batch_size,
        'epochs': epochs,
        'max_steps_per_epoch': None,
        'optimizer': {
            '_component_': 'torch.optim.AdamW',
            'lr': lr,
            'eps': 1e-8
        },
        'lr_scheduler': {
            '_component_': 'torchtune.training.lr_schedulers.get_cosine_schedule_with_warmup',
            'num_warmup_steps': warmup_steps,
            'num_training_steps': total_steps
        },
        'loss': {
            '_component_': 'torch.nn.CrossEntropyLoss'
        },
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'dtype': 'bf16',
        'output_dir': output_dir,
        'enable_activation_checkpointing': True,
        'enable_activation_offloading': False,
        'gradient_accumulation_steps': 1,
        'optimizer_in_bwd': True,
        'compile': False
    }

    config_filename = os.path.join(output_dir, f"{task}_config.yaml")
    with open(config_filename, 'w') as f:
        yaml.dump(config, f)

    return config_filename


def finetune_with_torchtune(config_filename: str):
    """
    Run the torchtune finetuning command fixed.
    """
    cmd = f"tune run lora_finetune_single_device --config {config_filename}"
    print("Running command:", cmd)
    ret = os.system(cmd)
    if ret != 0:
        print(f"[TTT] Torchtune command failed with config: {config_filename}")

def generate_data(data_dict, num_training_steps):

    prefix = (
        f"{data_dict['task_prompt']} "
        f"{data_dict['answer_format']}\n\n"
    )
    generation_length = data_dict["generation_length"]
    answer_format = data_dict["answer_format"]
    task_prompt = data_dict["task_prompt"]
    eval_questions = data_dict["eval_questions"]
    eval_targets = data_dict["eval_targets"]
    output_dir = data_dict["output_dir"]
    task_name = data_dict["task_name"]

    correct_examples = data_dict["correct_examples"]

    dataset_filename = os.path.join(data_dict["output_dir"], f"{task_name}_ttt_dataset.json")

    create_ttt_dataset( # create the actual data to be used in training
        prefix=prefix,
        correct_examples=correct_examples,
        num_training_steps=num_training_steps,
        dataset_filename=dataset_filename,
        shuffle_examples=True
    )

    config_filename = create_torchtune_config(
        model_dir=args.model_dir,
        dataset_type=args.dataset_type,
        task=task_name,
        dataset_filename=dataset_filename,
        output_dir=data_dict["output_dir"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    finetune_with_torchtune(config_filename) #fine tune on dataset created above

    prefix = build_inference_prompt(
        correct_examples,
        leave_one_out=True,
        shuffle_examples=True
    )
    eval_outputs = inference_vllm(
        llm=llm_eval,
        prompts=eval_questions,
        max_new_tokens=generation_length,
        task_prompt=task_prompt,
        answer_format=answer_format,
        few_shot_prompt_prefix=prefix,
        lora_request=lora_request
    )
    preds = eval_outputs

    acc = compute_accuracy(preds, eval_targets) # compute accuracy
    print(f"[synthetic data] Compute accuracy: {acc}")