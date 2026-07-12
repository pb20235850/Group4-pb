#!/usr/bin/env python3
"""Prepare MBPP GRPO data for Python code reward training.

Output:
  dpo/data/code_grpo_train.json
  dpo/data/code_grpo_test.json
  dpo/data/grpo_prepare_stats.json
  dpo/data/dataset_info.json
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = PROJECT_ROOT.parent

DEFAULT_SOURCE_DIR = WORK_ROOT / "mbpp" / "sanitized"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "dpo" / "data"

DEFAULT_PROMPT_TEMPLATE = (
    "You are an expert Python programmer.\n"
    "Write only Python code. Do not explain.\n"
    "The function name and signature must match the tests.\n\n"
    "Task:\n{prompt}\n\n"
    "Your code must pass these tests:\n{tests}\n\n"
    "Return only the Python function implementation."
)

FUNC_RE = re.compile(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")


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
                    "Failed to read parquet. Install pandas, pyarrow, or datasets with parquet support."
                ) from datasets_error


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\r\n", "\n").strip()


def listify(value: Any) -> list[str]:
    if value is None:
        return []

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, tuple):
        value = list(value)

    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]

    text = clean_text(value)
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [clean_text(item) for item in parsed if clean_text(item)]
        except Exception:
            pass

    return [text]


def first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(row.get(key))
        if value:
            return value
    return ""


def get_tests(row: dict[str, Any]) -> list[str]:
    tests = listify(row.get("test_list"))
    if not tests:
        tests = listify(row.get("tests"))
    if not tests:
        tests = listify(row.get("challenge_test_list"))
    return tests


def extract_entry_point(tests: list[str]) -> str:
    for test in tests:
        match = FUNC_RE.search(test)
        if match:
            return match.group(1)
    return ""


def build_prompt(prompt: str, tests: list[str], prompt_template: str) -> str:
    return prompt_template.format(
        prompt=prompt,
        tests="\n".join(tests),
    ).strip()


def convert_row(row: dict[str, Any], prompt_template: str) -> dict[str, Any] | None:
    prompt = first_text(row, ("prompt", "text", "instruction", "question"))
    code = first_text(row, ("code", "reference_code", "output"))
    tests = get_tests(row)

    task_id_raw = row.get("task_id", row.get("id", -1))
    try:
        task_id = int(task_id_raw)
    except Exception:
        task_id = -1

    if not prompt or not tests:
        return None

    entry_point = extract_entry_point(tests)

    return {
        "task_id": task_id,
        "prompt": build_prompt(prompt, tests, prompt_template),
        "raw_prompt": prompt,
        "tests": tests,
        "reference_code": code,
        "entry_point": entry_point,
    }


def deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    unique_rows = []

    for row in rows:
        key = (
            str(row.get("task_id", "")),
            row.get("raw_prompt", ""),
        )
        if key in seen:
            continue

        seen.add(key)
        unique_rows.append(row)

    return unique_rows


def find_split_file(source_dir: Path, split: str) -> Path:
    exact = source_dir / f"{split}-00000-of-00001.parquet"
    if exact.exists():
        return exact

    candidates = sorted(source_dir.glob(f"{split}-*.parquet"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No parquet file found for split={split} under {source_dir}")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def update_dataset_info(output_dir: Path) -> None:
    dataset_info_path = output_dir / "dataset_info.json"

    if dataset_info_path.exists():
        try:
            with dataset_info_path.open("r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        except Exception:
            dataset_info = {}
    else:
        dataset_info = {}

    dataset_info["code_grpo_train"] = {
        "file_name": "code_grpo_train.json",
        "columns": {
            "prompt": "prompt"
        },
    }

    dataset_info["code_grpo_test"] = {
        "file_name": "code_grpo_test.json",
        "columns": {
            "prompt": "prompt"
        },
    }

    save_json(dataset_info_path, dataset_info)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--test_split", type=str, default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0, help="0 means use all train rows.")
    parser.add_argument("--max_test_samples", type=int, default=0, help="0 means use all test rows.")
    parser.add_argument("--prompt_template", type=str, default=DEFAULT_PROMPT_TEMPLATE)
    args = parser.parse_args()

    train_file = find_split_file(args.source_dir, args.train_split)
    test_file = find_split_file(args.source_dir, args.test_split)

    raw_train_rows = read_parquet(train_file)
    raw_test_rows = read_parquet(test_file)

    train_rows = [
        row
        for row in (convert_row(raw, args.prompt_template) for raw in raw_train_rows)
        if row is not None
    ]

    test_rows = [
        row
        for row in (convert_row(raw, args.prompt_template) for raw in raw_test_rows)
        if row is not None
    ]

    train_rows = deduplicate(train_rows)
    test_rows = deduplicate(test_rows)

    rng = random.Random(args.seed)
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)

    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    if args.max_test_samples > 0:
        test_rows = test_rows[: args.max_test_samples]

    if not train_rows:
        raise RuntimeError(
            "No valid GRPO train rows were converted. "
            "Please check parquet columns: prompt/text, test_list/tests, task_id."
        )

    if not test_rows:
        raise RuntimeError(
            "No valid GRPO test rows were converted. "
            "Please check parquet columns: prompt/text, test_list/tests, task_id."
        )

    output_train_path = args.output_dir / "code_grpo_train.json"
    output_test_path = args.output_dir / "code_grpo_test.json"

    save_json(output_train_path, train_rows)
    save_json(output_test_path, test_rows)
    update_dataset_info(args.output_dir)

    stats = {
        "source_dir": str(args.source_dir),
        "train_file": str(train_file),
        "test_file": str(test_file),
        "raw_train_rows": len(raw_train_rows),
        "raw_test_rows": len(raw_test_rows),
        "train_samples": len(train_rows),
        "test_samples": len(test_rows),
        "train_path": str(output_train_path),
        "test_path": str(output_test_path),
        "dataset_info": str(args.output_dir / "dataset_info.json"),
        "example": train_rows[0] if train_rows else None,
    }

    save_json(args.output_dir / "grpo_prepare_stats.json", stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"Wrote train: {len(train_rows)} -> {output_train_path}")
    print(f"Wrote test : {len(test_rows)} -> {output_test_path}")
    print(f"Wrote registry -> {args.output_dir / 'dataset_info.json'}")


if __name__ == "__main__":
    main()