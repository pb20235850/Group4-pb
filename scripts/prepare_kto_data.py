#!/usr/bin/env python3
"""
Convert DPO chosen/rejected data into KTO binary preference data.

Input:
  dpo/data/code_dpo_train.json
  dpo/data/code_effect_test.json

Output:
  dpo/data/code_kto_train.json
  dpo/data/code_kto_test.json
  update dpo/data/dataset_info.json

KTO format:
  {
    "instruction": "...",
    "input": "",
    "output": "...",
    "kto_tag": true
  }

For every DPO sample:
  chosen   -> kto_tag: true
  rejected -> kto_tag: false
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def read_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list JSON file: {path}")

    return data


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def get_prompt(item: dict[str, Any]) -> str:
    """
    Compatible with common DPO formats:
    - prompt
    - instruction + input
    """

    prompt = normalize_text(item.get("prompt"))
    if prompt:
        return prompt

    instruction = normalize_text(item.get("instruction"))
    input_text = normalize_text(item.get("input"))

    if instruction and input_text:
        return instruction + "\n" + input_text

    if instruction:
        return instruction

    return ""


def get_chosen(item: dict[str, Any]) -> str:
    return normalize_text(item.get("chosen"))


def get_rejected(item: dict[str, Any]) -> str:
    return normalize_text(item.get("rejected"))


def dpo_to_kto_rows(
    rows: list[dict[str, Any]],
    shuffle: bool,
    seed: int,
    max_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kto_rows: list[dict[str, Any]] = []

    bad_missing_prompt = 0
    bad_missing_chosen = 0
    bad_missing_rejected = 0

    for item in rows:
        prompt = get_prompt(item)
        chosen = get_chosen(item)
        rejected = get_rejected(item)

        if not prompt:
            bad_missing_prompt += 1
            continue

        if not chosen:
            bad_missing_chosen += 1
            continue

        if not rejected:
            bad_missing_rejected += 1
            continue

        kto_rows.append(
            {
                "instruction": prompt,
                "input": "",
                "output": chosen,
                "kto_tag": True,
            }
        )

        kto_rows.append(
            {
                "instruction": prompt,
                "input": "",
                "output": rejected,
                "kto_tag": False,
            }
        )

    if shuffle:
        random.seed(seed)
        random.shuffle(kto_rows)

    if max_samples > 0:
        kto_rows = kto_rows[:max_samples]

    stats = {
        "source_samples": len(rows),
        "kto_samples": len(kto_rows),
        "positive_samples": sum(1 for x in kto_rows if x["kto_tag"] is True),
        "negative_samples": sum(1 for x in kto_rows if x["kto_tag"] is False),
        "bad_missing_prompt": bad_missing_prompt,
        "bad_missing_chosen": bad_missing_chosen,
        "bad_missing_rejected": bad_missing_rejected,
    }

    return kto_rows, stats


def update_dataset_info(data_dir: Path) -> None:
    dataset_info_path = data_dir / "dataset_info.json"

    if dataset_info_path.exists():
        with dataset_info_path.open("r", encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info["code_kto_train"] = {
        "file_name": "code_kto_train.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "kto_tag": "kto_tag"
        }
    }

    dataset_info["code_kto_test"] = {
        "file_name": "code_kto_test.json",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "kto_tag": "kto_tag"
        }
    }

    save_json(dataset_info_path, dataset_info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("dpo/data"))
    parser.add_argument("--train_file", default="code_dpo_train.json")
    parser.add_argument("--test_file", default="code_effect_test.json")
    parser.add_argument("--output_train_file", default="code_kto_train.json")
    parser.add_argument("--output_test_file", default="code_kto_test.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--no_shuffle", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_path = args.data_dir / args.train_file
    test_path = args.data_dir / args.test_file

    train_rows = read_json(train_path)
    test_rows = read_json(test_path)

    train_kto, train_stats = dpo_to_kto_rows(
        train_rows,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        max_samples=args.max_train_samples,
    )

    test_kto, test_stats = dpo_to_kto_rows(
        test_rows,
        shuffle=not args.no_shuffle,
        seed=args.seed,
        max_samples=args.max_test_samples,
    )

    output_train_path = args.data_dir / args.output_train_file
    output_test_path = args.data_dir / args.output_test_file

    save_json(output_train_path, train_kto)
    save_json(output_test_path, test_kto)

    update_dataset_info(args.data_dir)

    stats = {
        "train": train_stats,
        "test": test_stats,
        "output_train": str(output_train_path),
        "output_test": str(output_test_path),
        "dataset_info": str(args.data_dir / "dataset_info.json"),
    }

    save_json(args.data_dir / "kto_prepare_stats.json", stats)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()