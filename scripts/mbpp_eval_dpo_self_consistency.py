#!/usr/bin/env python3
"""Evaluate a causal language model on MBPP with Self-Consistency.

Standalone script for inference-time enhancement:
- Generate N sampled candidate codes for each MBPP task.
- Run unit tests for each candidate.
- Group candidates by execution result pass_vector.
- Select the group with the most passed tests; if tied, select the most frequent group.
- Select the best candidate inside that group.

This script follows the original MBPP-style prompt and evaluation structure.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ASSIGNMENT_ROOT = Path(__file__).resolve().parents[2]
WORK_ROOT = ASSIGNMENT_ROOT.parent
DEFAULT_MBPP_DIR = WORK_ROOT / "mbpp"
DEFAULT_MODEL_PATH = ASSIGNMENT_ROOT / "dpo" / "outputs" / "qwen15_code_lora_grpo_v5"
DEFAULT_OUTPUT_DIR = ASSIGNMENT_ROOT / "dpo" / "outputs" / "mbpp_self_consistency"

OFFICIAL_PROMPT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {prompt} "
    "Your code should pass these tests:\n\n{tests}\n[BEGIN]\n{code}\n[DONE]"
)

FENCED_CODE_RE = re.compile(
    r"```(?:python|py)?\s*(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mbpp_dir", type=Path, default=DEFAULT_MBPP_DIR)
    parser.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--config", choices=("sanitized", "full"), default="sanitized")
    parser.add_argument("--split", default="test")

    parser.add_argument(
        "--prompt_mode",
        choices=("zero_shot", "one_shot", "three_shot"),
        default="zero_shot",
        help="zero_shot gives only the target task; one_shot uses the first prompt task; three_shot uses all prompt_task_ids.",
    )
    parser.add_argument("--prompt_task_ids", default="2,3,4")

    parser.add_argument("--limit", type=int, default=0, help="0 means evaluate all rows.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)

    # Self-Consistency parameters
    parser.add_argument(
        "--num_candidates",
        type=int,
        default=16,
        help="Number of sampled candidate solutions for each task.",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=0.9)

    parser.add_argument("--test_timeout", type=float, default=5.0)
    parser.add_argument("--memory_mb", type=int, default=1024)
    parser.add_argument("--include_challenge_tests", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--use_reference_code", action="store_true", help="Evaluate gold code instead of loading model.")
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=True)

    return parser.parse_args()


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def listify(value: Any) -> list[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if str(value).strip():
        return [str(value).strip()]
    return []


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    except Exception:
        try:
            import pandas as pd
            return pd.read_parquet(path).to_dict("records")
        except Exception as exc:
            raise RuntimeError(f"Failed to read parquet file: {path}") from exc


def load_split(mbpp_dir: Path, config: str, split: str) -> list[dict[str, Any]]:
    path = mbpp_dir / config / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing MBPP parquet split: {path}")
    return read_parquet_rows(path)


def row_prompt(row: dict[str, Any], config: str) -> str:
    # Compatible with both field names. Your sanitized parquet may use text.
    if config == "sanitized":
        return normalize_text(row.get("prompt") or row.get("text"))
    return normalize_text(row.get("text") or row.get("prompt"))


def row_setup(row: dict[str, Any], config: str) -> str:
    if config == "sanitized":
        return "\n".join(listify(row.get("test_imports")))
    return normalize_text(row.get("test_setup_code"))


def row_tests(row: dict[str, Any], include_challenge: bool) -> list[str]:
    tests = listify(row.get("test_list"))
    if include_challenge:
        tests.extend(listify(row.get("challenge_test_list")))
    return tests


def format_tests(tests: list[str]) -> str:
    return "\n".join(tests)


def official_block(prompt: str, tests: list[str], code: str = "") -> str:
    return OFFICIAL_PROMPT_TEMPLATE.format(
        prompt=prompt,
        tests=format_tests(tests),
        code=code,
    )


def build_prompt_prefix(
    prompt_rows: list[dict[str, Any]],
    config: str,
    prompt_mode: str,
    prompt_task_ids: list[int],
) -> str:
    if prompt_mode == "zero_shot":
        return ""
    if prompt_mode == "one_shot":
        prompt_task_ids = prompt_task_ids[:1]

    by_id = {int(row["task_id"]): row for row in prompt_rows}
    selected = [by_id[task_id] for task_id in prompt_task_ids if task_id in by_id]
    if len(selected) < len(prompt_task_ids):
        selected = prompt_rows[: len(prompt_task_ids)]

    blocks = []
    for row in selected:
        blocks.append(
            official_block(
                row_prompt(row, config),
                row_tests(row, include_challenge=False),
                normalize_text(row.get("code")),
            )
        )
    return "\n\n".join(blocks)


def build_prompt(prefix: str, row: dict[str, Any], config: str, include_challenge: bool) -> str:
    target = official_block(row_prompt(row, config), row_tests(row, include_challenge), code="")
    target = target.rsplit("[DONE]", 1)[0]
    if not prefix:
        return target
    return f"{prefix}\n\n{target}"


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt


def load_model(model_path: Path, trust_remote_code: bool) -> tuple[Any, Any, Any]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Missing torch/transformers. Run with the assignment environment.") from exc

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=trust_remote_code,
    )
    if not torch.cuda.is_available():
        model.to("cpu")
    model.eval()
    return tokenizer, model, torch


def extract_candidate_code(text: str) -> str:
    text = normalize_text(text)
    if "[BEGIN]" in text:
        text = text.rsplit("[BEGIN]", 1)[-1]
    if "[DONE]" in text:
        text = text.split("[DONE]", 1)[0]

    fenced = FENCED_CODE_RE.findall(text)
    if fenced:
        return normalize_text(fenced[-1])

    fence_positions = [text.lower().rfind(marker) for marker in ("```python", "```py", "```")]
    fence_pos = max(fence_positions)
    if fence_pos >= 0:
        candidate = text[fence_pos:].split("\n", 1)
        text = candidate[1] if len(candidate) == 2 else ""
        if "```" in text:
            text = text.split("```", 1)[0]
        return normalize_text(text)

    lines = []
    raw_lines = text.splitlines()
    first_code_line = 0
    for i, line in enumerate(raw_lines):
        stripped = line.lstrip()
        if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
            first_code_line = i
            break

    for line in raw_lines[first_code_line:]:
        stripped = line.strip()
        if stripped in {"[DONE]", "DONE"}:
            break
        if stripped.startswith("```"):
            continue
        lines.append(line)
    return normalize_text("\n".join(lines))


def syntax_ok(code: str) -> bool:
    if not normalize_text(code):
        return False
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            ast.parse(code)
        return True
    except SyntaxError:
        return False


def limit_resources(memory_mb: int, timeout: float) -> None:
    try:
        import resource
        memory_bytes = memory_mb * 1024 * 1024
        cpu_seconds = max(1, int(timeout) + 1)
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
    except Exception:
        return


def run_one_assert(code: str, setup_code: str, test: str, timeout: float, memory_mb: int) -> dict[str, Any]:
    runner = "\n\n".join(
        part
        for part in [
            "import warnings\nwarnings.filterwarnings('ignore', category=SyntaxWarning)",
            "import faulthandler\nfaulthandler.enable()",
            setup_code,
            code,
            test,
        ]
        if normalize_text(part)
    )

    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as tmpdir:
        path = Path(tmpdir) / "candidate_test.py"
        path.write_text(runner + "\n", encoding="utf-8")
        env = os.environ.copy()
        env["HOME"] = tmpdir
        try:
            result = subprocess.run(
                [sys.executable, str(path)],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                preexec_fn=lambda: limit_resources(memory_mb, timeout) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "error_type": "timeout", "stderr": str(exc)}

    return {
        "passed": result.returncode == 0,
        "error_type": "" if result.returncode == 0 else "runtime_error",
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-2000:],
    }


def score_candidate(
    generation: dict[str, Any],
    row: dict[str, Any],
    config: str,
    include_challenge: bool,
    timeout: float,
    memory_mb: int,
) -> dict[str, Any]:
    code = normalize_text(generation["code"])
    tests = row_tests(row, include_challenge)
    setup_code = row_setup(row, config)
    per_test = [run_one_assert(code, setup_code, test, timeout, memory_mb) for test in tests]
    pass_vector = tuple(item["passed"] for item in per_test)
    passed_tests = sum(1 for item in per_test if item["passed"])
    total_tests = len(tests)
    syn_ok = syntax_ok(code)

    return {
        "passed_tests": passed_tests,
        "total_tests": total_tests,
        "all_pass": passed_tests == total_tests and total_tests > 0,
        "syntax_ok": syn_ok,
        "pass_vector": pass_vector,
        "test_results": per_test,
    }


def choose_self_consistency(
    candidates: list[dict[str, Any]],
    row: dict[str, Any],
    config: str,
    include_challenge: bool,
    timeout: float,
    memory_mb: int,
) -> dict[str, Any]:
    """Select by execution-result consistency.

    Group candidates by pass_vector, for example (True, False, True).
    Choose the group with:
    1. more passed tests,
    2. larger group size,
    3. syntax correctness,
    4. shorter code.
    Then select one candidate inside that group.
    """
    scored = []
    for idx, cand in enumerate(candidates):
        score = score_candidate(cand, row, config, include_challenge, timeout, memory_mb)
        scored.append({**cand, "candidate_id": idx, "candidate_score": score})

    groups: dict[tuple[bool, ...], list[dict[str, Any]]] = {}
    for item in scored:
        pv = item["candidate_score"]["pass_vector"]
        groups.setdefault(pv, []).append(item)

    def group_rank(group_item: tuple[tuple[bool, ...], list[dict[str, Any]]]) -> tuple[int, int, int, int]:
        pass_vector, members = group_item
        passed_num = sum(1 for x in pass_vector if x)
        group_size = len(members)
        syntax_count = sum(1 for m in members if m["candidate_score"]["syntax_ok"])
        shortest_len = -min(len(normalize_text(m["code"]).splitlines()) for m in members)
        return (passed_num, group_size, syntax_count, shortest_len)

    best_pv, best_group = sorted(groups.items(), key=group_rank, reverse=True)[0]

    def member_rank(item: dict[str, Any]) -> tuple[int, int, int]:
        code = normalize_text(item["code"])
        return (
            1 if item["candidate_score"]["syntax_ok"] else 0,
            -len(code.splitlines()),
            -len(code),
        )

    best_group = sorted(best_group, key=member_rank, reverse=True)
    best = best_group[0]

    best["self_consistency_group_size"] = len(best_group)
    best["self_consistency_pass_vector"] = list(best_pv)
    best["all_candidates_summary"] = [
        {
            "candidate_id": item["candidate_id"],
            "passed_tests": item["candidate_score"]["passed_tests"],
            "total_tests": item["candidate_score"]["total_tests"],
            "all_pass": item["candidate_score"]["all_pass"],
            "syntax_ok": item["candidate_score"]["syntax_ok"],
            "pass_vector": list(item["candidate_score"]["pass_vector"]),
        }
        for item in scored
    ]
    return best


def make_generation_record(
    row: dict[str, Any],
    config: str,
    include_challenge: bool,
    completion: str,
    code: str,
    num_candidates: int,
) -> dict[str, Any]:
    return {
        "task_id": int(row["task_id"]),
        "prompt": row_prompt(row, config),
        "completion": completion,
        "code": code,
        "reference_code": normalize_text(row.get("code")),
        "tests": row_tests(row, include_challenge=False),
        "inference_method": "self_consistency",
        "num_candidates": num_candidates,
    }


def generate_self_consistency_completions(
    rows: list[dict[str, Any]],
    prompts: list[str],
    model_path: Path,
    max_new_tokens: int,
    trust_remote_code: bool,
    num_candidates: int = 16,
    temperature: float = 0.4,
    top_p: float = 0.9,
    config: str = "sanitized",
    include_challenge: bool = False,
    test_timeout: float = 5.0,
    memory_mb: int = 1024,
) -> list[dict[str, Any]]:
    tokenizer, model, torch = load_model(model_path, trust_remote_code)
    generations = []

    for idx, (row, prompt) in enumerate(zip(rows, prompts), start=1):
        rendered = apply_chat_template(tokenizer, prompt)
        inputs = tokenizer(rendered, return_tensors="pt", truncation=True)
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        input_width = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=num_candidates,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        candidates = []
        for output in output_ids:
            completion = tokenizer.decode(output[input_width:], skip_special_tokens=True)
            code = extract_candidate_code(completion)
            candidates.append(
                make_generation_record(
                    row=row,
                    config=config,
                    include_challenge=include_challenge,
                    completion=completion,
                    code=code,
                    num_candidates=num_candidates,
                )
            )

        best = choose_self_consistency(
            candidates=candidates,
            row=row,
            config=config,
            include_challenge=include_challenge,
            timeout=test_timeout,
            memory_mb=memory_mb,
        )
        generations.append(best)
        print(f"Generated {len(generations)} / {len(prompts)}", flush=True)

    return generations


def evaluate_generation(
    generation: dict[str, Any],
    row: dict[str, Any],
    config: str,
    include_challenge: bool,
    timeout: float,
    memory_mb: int,
) -> dict[str, Any]:
    code = normalize_text(generation["code"])
    tests = row_tests(row, include_challenge)
    setup_code = row_setup(row, config)
    per_test = [run_one_assert(code, setup_code, test, timeout, memory_mb) for test in tests]
    passed_tests = sum(1 for item in per_test if item["passed"])

    return {
        "task_id": int(row["task_id"]),
        "prompt": row_prompt(row, config),
        "code": code,
        "reference_code": normalize_text(row.get("code")),
        "syntax_ok": syntax_ok(code),
        "passed": passed_tests == len(tests) and len(tests) > 0,
        "passed_tests": passed_tests,
        "total_tests": len(tests),
        "test_results": per_test,
        "completion": generation.get("completion", ""),
        "inference_method": generation.get("inference_method", "self_consistency"),
        "num_candidates": generation.get("num_candidates", 1),
        "candidate_id": generation.get("candidate_id", 0),
        "self_consistency_group_size": generation.get("self_consistency_group_size", 0),
        "self_consistency_pass_vector": generation.get("self_consistency_pass_vector", []),
        "all_candidates_summary": generation.get("all_candidates_summary", []),
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.predictions or (args.output_dir / "mbpp_generations.jsonl")

    prompt_ids = [int(item) for item in args.prompt_task_ids.split(",") if item.strip()]
    prompt_rows = load_split(args.mbpp_dir, args.config, "prompt")
    eval_rows = load_split(args.mbpp_dir, args.config, args.split)
    eval_rows = eval_rows[args.start_index:]
    if args.limit > 0:
        eval_rows = eval_rows[: args.limit]

    prefix = build_prompt_prefix(prompt_rows, args.config, args.prompt_mode, prompt_ids)
    prompts = [build_prompt(prefix, row, args.config, args.include_challenge_tests) for row in eval_rows]

    if args.use_reference_code:
        generations = [
            {
                "task_id": int(row["task_id"]),
                "prompt": row_prompt(row, args.config),
                "completion": normalize_text(row.get("code")),
                "code": normalize_text(row.get("code")),
                "reference_code": normalize_text(row.get("code")),
                "tests": row_tests(row, args.include_challenge_tests),
                "inference_method": "reference_code",
                "num_candidates": 1,
            }
            for row in eval_rows
        ]
        save_jsonl(predictions_path, generations)
    elif args.skip_generation:
        generations = read_jsonl(predictions_path)
    else:
        generations = generate_self_consistency_completions(
            rows=eval_rows,
            prompts=prompts,
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
            num_candidates=args.num_candidates,
            temperature=args.temperature,
            top_p=args.top_p,
            config=args.config,
            include_challenge=args.include_challenge_tests,
            test_timeout=args.test_timeout,
            memory_mb=args.memory_mb,
        )
        save_jsonl(predictions_path, generations)

    by_task_id = {int(item["task_id"]): item for item in generations}
    cases = [
        evaluate_generation(
            by_task_id[int(row["task_id"])],
            row,
            args.config,
            args.include_challenge_tests,
            args.test_timeout,
            args.memory_mb,
        )
        for row in eval_rows
        if int(row["task_id"]) in by_task_id
    ]

    total = len(cases)
    passed = sum(1 for case in cases if case["passed"])
    syntax_passed = sum(1 for case in cases if case["syntax_ok"])
    total_tests = sum(case["total_tests"] for case in cases)
    passed_tests = sum(case["passed_tests"] for case in cases)
    selected_group_sizes = [case.get("self_consistency_group_size", 0) for case in cases]
    avg_group_size = sum(selected_group_sizes) / total if total else 0.0

    metrics = {
        "benchmark": "MBPP",
        "config": args.config,
        "split": args.split,
        "model_path": str(args.model_path),
        "predictions": str(predictions_path),
        "num_tasks": total,
        "pass_at_1": passed / total if total else 0.0,
        "syntax_pass_rate": syntax_passed / total if total else 0.0,
        "avg_test_pass_rate": passed_tests / total_tests if total_tests else 0.0,
        "passed_tasks": passed,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "inference_method": "self_consistency",
        "num_candidates": args.num_candidates,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "avg_self_consistency_group_size": avg_group_size,
        "prompt": {
            "source": "Google Research MBPP README",
            "template": OFFICIAL_PROMPT_TEMPLATE,
            "mode": args.prompt_mode,
            "example_task_ids": prompt_ids
            if args.prompt_mode == "three_shot"
            else prompt_ids[:1]
            if args.prompt_mode == "one_shot"
            else [],
        },
        "execution_note": "Generated Python is executed in a temporary subprocess with timeout and basic resource limits.",
    }

    save_json(args.output_dir / "mbpp_metrics.json", metrics)
    save_jsonl(args.output_dir / "mbpp_cases.jsonl", cases)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"Wrote generations to {predictions_path}")
    print(f"Wrote metrics to {args.output_dir / 'mbpp_metrics.json'}")
    print(f"Wrote cases to {args.output_dir / 'mbpp_cases.jsonl'}")


if __name__ == "__main__":
    main()
