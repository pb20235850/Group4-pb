#!/usr/bin/env python3
"""Prepare Python-code DPO data for LLaMA-Factory."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "py-dpo-v0.1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dpo" / "data"
DEFAULT_PROMPT_TEMPLATE = (
    "Complete the following Python coding task. Return a correct, readable "
    "Python solution and include brief reasoning when helpful.\n\nTask:\n{prompt}"
)


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd

        return pd.read_parquet(path).to_dict("records")
    except Exception:
        try:
            import pyarrow.parquet as pq

            return pq.read_table(path).to_pylist()
        except Exception:
            try:
                from datasets import load_dataset

                dataset = load_dataset("parquet", data_files=str(path), split="train")
                return [dict(row) for row in dataset]
            except Exception as datasets_error:
                raise RuntimeError(
                    "Failed to read parquet. Install pandas, pyarrow, or datasets with parquet support "
                    "in the active environment."
                ) from datasets_error


def find_parquet_files(source_dir: Path) -> list[Path]:
    candidates = sorted(source_dir.glob("*.parquet"))
    data_dir = source_dir / "data"
    if data_dir.exists():
        candidates.extend(sorted(data_dir.glob("*.parquet")))
    if not candidates:
        raise FileNotFoundError(f"No parquet files found under {source_dir}")
    return candidates


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def build_instruction(prompt: str, prompt_template: str) -> str:
    return prompt_template.format(prompt=prompt).strip()


def convert_train_row(row: dict[str, Any], prompt_template: str) -> dict[str, str] | None:
    prompt = first_text(row, ("prompt", "instruction", "question"))
    chosen = first_text(row, ("chosen", "accepted", "preferred", "output"))
    rejected = first_text(row, ("rejected", "reject", "dispreferred"))
    input_text = first_text(row, ("input", "query"))

    if not prompt or not chosen or not rejected:
        return None

    return {
        "instruction": build_instruction(prompt, prompt_template),
        "input": input_text,
        "chosen": chosen,
        "rejected": rejected,
    }


def convert_test_row(row: dict[str, Any], prompt_template: str) -> dict[str, str] | None:
    prompt = first_text(row, ("prompt", "instruction", "question"))
    output = first_text(row, ("chosen", "accepted", "preferred", "output"))
    input_text = first_text(row, ("input", "query"))

    if not prompt or not output:
        return None

    return {
        "instruction": build_instruction(prompt, prompt_template),
        "input": input_text,
        "output": output,
    }


def deduplicate(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: set[tuple[str, ...]] = set()
    unique_rows = []
    for row in rows:
        key = tuple(row.get(field, "") for field in ("instruction", "input", "chosen", "rejected", "output"))
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 means use all remaining train rows.")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    args = parser.parse_args()

    parquet_files = find_parquet_files(args.source_dir)
    raw_rows: list[dict[str, Any]] = []
    for parquet_file in parquet_files:
        raw_rows.extend(read_parquet(parquet_file))

    rng = random.Random(args.seed)
    shuffled_rows = raw_rows[:]
    rng.shuffle(shuffled_rows)

    raw_test_rows = shuffled_rows[: args.test_size]
    raw_train_rows = shuffled_rows[args.test_size :]
    if args.max_train_samples > 0:
        raw_train_rows = raw_train_rows[: args.max_train_samples]

    train_rows = deduplicate(
        [row for row in (convert_train_row(raw, args.prompt_template) for raw in raw_train_rows) if row is not None]
    )
    test_rows = deduplicate(
        [row for row in (convert_test_row(raw, args.prompt_template) for raw in raw_test_rows) if row is not None]
    )

    if not train_rows:
        raise RuntimeError("No valid DPO train rows were converted.")
    if not test_rows:
        raise RuntimeError("No valid code test rows were converted.")

    save_json(args.output_dir / "code_dpo_train.json", train_rows)
    save_json(args.output_dir / "code_dpo_test.json", test_rows)
    save_json(
        args.output_dir / "dataset_info.json",
        {
            "code_dpo_train": {
                "file_name": "code_dpo_train.json",
                "ranking": True,
                "columns": {
                    "prompt": "instruction",
                    "query": "input",
                    "chosen": "chosen",
                    "rejected": "rejected",
                },
            },
            "code_dpo_test": {
                "file_name": "code_dpo_test.json",
            },
        },
    )

    print(f"Read {len(raw_rows)} raw rows from {len(parquet_files)} parquet file(s).")
    print(f"Wrote train: {len(train_rows)} -> {args.output_dir / 'code_dpo_train.json'}")
    print(f"Wrote test : {len(test_rows)} -> {args.output_dir / 'code_dpo_test.json'}")
    print(f"Wrote registry -> {args.output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()
