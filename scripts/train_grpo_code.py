#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from code_grpo_rewards import code_reward_func


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/home/wangmengxin/assignment_A/sft/outputs/qwen15_code_full_sft")
    parser.add_argument("--train_file", default="dpo/data/code_grpo_train.json")
    parser.add_argument("--output_dir", default="dpo/outputs/qwen15_code_lora_grpo_v4")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=128)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true", default=True)
    return parser.parse_args()


def load_policy_model(model_name_or_path: str):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # 兼容当前 TRL GRPOTrainer：
    # 某些模型类没有 warnings_issued 属性，但 GRPOTrainer 初始化时会访问它。
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    return model


def main() -> None:
    args = parse_args()

    train_path = Path(args.train_file)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing GRPO train file: {train_path}")

    dataset = load_dataset("json", data_files=str(train_path), split="train")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = load_policy_model(args.model_name_or_path)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_generations=args.num_generations,
        temperature=0.8,
        beta=0.04,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=code_reward_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()