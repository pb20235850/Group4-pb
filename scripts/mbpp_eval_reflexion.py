#!/usr/bin/env python3
"""MBPP evaluation with Reflexion-style one/multi-step repair.

流程：
1. 用官方 MBPP prompt 生成多个候选代码。
2. 用单元测试选择当前最好候选。
3. 如果未全通过，把失败测试/错误信息反馈给模型，让模型生成反思和修正代码。
4. 修正后如果通过更多测试，则替换原答案。
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
DEFAULT_OUTPUT_DIR = ASSIGNMENT_ROOT / "dpo" / "outputs" / "mbpp_reflexion"

OFFICIAL_PROMPT_TEMPLATE = (
    "You are an expert Python programmer, and here is your task: {prompt} "
    "Your code should pass these tests:\n\n{tests}\n[BEGIN]{code}\n[DONE]"
)
FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mbpp_dir", type=Path, default=DEFAULT_MBPP_DIR)
    p.add_argument("--model_path", type=Path, default=DEFAULT_MODEL_PATH)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--config", choices=("sanitized", "full"), default="sanitized")
    p.add_argument("--split", default="test")
    p.add_argument("--prompt_mode", choices=("zero_shot", "one_shot", "three_shot"), default="zero_shot")
    p.add_argument("--prompt_task_ids", default="2,3,4")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--start_index", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_candidates", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--repair_attempts", type=int, default=1)
    p.add_argument("--repair_temperature", type=float, default=0.2)
    p.add_argument("--test_timeout", type=float, default=5.0)
    p.add_argument("--memory_mb", type=int, default=1024)
    p.add_argument("--include_challenge_tests", action="store_true")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--use_reference_code", action="store_true")
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    return p.parse_args()


def normalize_text(v: Any) -> str:
    return str(v or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def listify(v: Any) -> list[str]:
    if v is None:
        return []
    if hasattr(v, "tolist"):
        v = v.tolist()
    if isinstance(v, tuple):
        v = list(v)
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    return [str(v).strip()] if str(v).strip() else []


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
    # 兼容你当前 sanitized 数据：字段是 text；若有 prompt 则优先使用 prompt。
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


def official_block(prompt: str, tests: list[str], code: str = "") -> str:
    return OFFICIAL_PROMPT_TEMPLATE.format(prompt=prompt, tests="\n".join(tests), code=code)


def build_prompt_prefix(prompt_rows: list[dict[str, Any]], config: str, prompt_mode: str, prompt_task_ids: list[int]) -> str:
    if prompt_mode == "zero_shot":
        return ""
    if prompt_mode == "one_shot":
        prompt_task_ids = prompt_task_ids[:1]
    by_id = {int(r["task_id"]): r for r in prompt_rows}
    selected = [by_id[i] for i in prompt_task_ids if i in by_id]
    if len(selected) < len(prompt_task_ids):
        selected = prompt_rows[: len(prompt_task_ids)]
    return "\n\n".join(official_block(row_prompt(r, config), row_tests(r, False), normalize_text(r.get("code"))) for r in selected)


def build_prompt(prefix: str, row: dict[str, Any], config: str, include_challenge: bool) -> str:
    target = official_block(row_prompt(row, config), row_tests(row, include_challenge), code="")
    target = target.rsplit("[DONE]", 1)[0]
    return target if not prefix else f"{prefix}\n\n{target}"


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    return prompt


def load_model(model_path: Path, trust_remote_code: bool) -> tuple[Any, Any, Any]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    lines = []
    raw = text.splitlines()
    first = 0
    for i, line in enumerate(raw):
        if line.lstrip().startswith(("import ", "from ", "def ", "class ", "@")):
            first = i
            break
    for line in raw[first:]:
        s = line.strip()
        if s in {"[DONE]", "DONE"}:
            break
        if s.startswith("```"):
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
        part for part in [
            "import warnings\nwarnings.filterwarnings('ignore', category=SyntaxWarning)",
            "import faulthandler\nfaulthandler.enable()",
            setup_code,
            code,
            test,
        ] if normalize_text(part)
    )
    with tempfile.TemporaryDirectory(prefix="mbpp_eval_") as tmpdir:
        path = Path(tmpdir) / "candidate_test.py"
        path.write_text(runner + "\n", encoding="utf-8")
        env = os.environ.copy(); env["HOME"] = tmpdir
        try:
            result = subprocess.run(
                [sys.executable, str(path)], cwd=tmpdir, env=env, text=True,
                capture_output=True, timeout=timeout,
                preexec_fn=lambda: limit_resources(memory_mb, timeout) if os.name == "posix" else None,
            )
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "error_type": "timeout", "stdout": "", "stderr": str(exc)}
    return {
        "passed": result.returncode == 0,
        "error_type": "" if result.returncode == 0 else "runtime_error",
        "stdout": result.stdout[-1000:],
        "stderr": result.stderr[-2000:],
    }


def score_code(code: str, row: dict[str, Any], config: str, include_challenge: bool, timeout: float, memory_mb: int) -> dict[str, Any]:
    tests = row_tests(row, include_challenge)
    setup = row_setup(row, config)
    per_test = [run_one_assert(code, setup, t, timeout, memory_mb) for t in tests]
    passed = sum(1 for x in per_test if x["passed"])
    total = len(tests)
    return {
        "passed_tests": passed,
        "total_tests": total,
        "all_pass": passed == total and total > 0,
        "syntax_ok": syntax_ok(code),
        "test_results": per_test,
    }


def generation_record(row: dict[str, Any], config: str, include_challenge: bool, completion: str, code: str, method: str, num_candidates: int) -> dict[str, Any]:
    return {
        "task_id": int(row["task_id"]),
        "prompt": row_prompt(row, config),
        "completion": completion,
        "code": code,
        "reference_code": normalize_text(row.get("code")),
        "tests": row_tests(row, include_challenge=False),
        "inference_method": method,
        "num_candidates": num_candidates,
    }


def choose_best_by_tests(candidates: list[dict[str, Any]], row: dict[str, Any], config: str, include_challenge: bool, timeout: float, memory_mb: int) -> dict[str, Any]:
    scored = []
    for idx, cand in enumerate(candidates):
        s = score_code(cand["code"], row, config, include_challenge, timeout, memory_mb)
        scored.append({**cand, "candidate_id": idx, "candidate_score": s})
    best = scored[0]
    for item in scored[1:]:
        if item["candidate_score"]["passed_tests"] > best["candidate_score"]["passed_tests"]:
            best = item
    best["all_candidates_summary"] = [
        {"candidate_id": x["candidate_id"], "passed_tests": x["candidate_score"]["passed_tests"],
         "total_tests": x["candidate_score"]["total_tests"], "all_pass": x["candidate_score"]["all_pass"],
         "syntax_ok": x["candidate_score"]["syntax_ok"]}
        for x in scored
    ]
    return best


def summarize_failures(score: dict[str, Any], tests: list[str], max_items: int = 2) -> str:
    parts = []
    for test, result in zip(tests, score.get("test_results", [])):
        if result.get("passed"):
            continue
        err = normalize_text(result.get("stderr") or result.get("stdout") or result.get("error_type"))
        parts.append(f"Failed test: {test}\nError: {err[-800:]}")
        if len(parts) >= max_items:
            break
    if not parts and not score.get("syntax_ok"):
        parts.append("The code has a Python syntax error.")
    return "\n\n".join(parts) if parts else "Some tests failed."


def build_reflexion_prompt(row: dict[str, Any], config: str, include_challenge: bool, old_code: str, failure_info: str) -> str:
    task = row_prompt(row, config)
    tests = "\n".join(row_tests(row, include_challenge))
    return f"""You are an expert Python programmer.

Task:
{task}

The code should pass these tests:
{tests}

Previous code:
[BEGIN]
{old_code}
[DONE]

The previous code failed with the following information:
{failure_info}

First, briefly reflect on the likely bug. Then output the corrected Python code.
Return the final corrected code between [BEGIN] and [DONE].

Reflection:
"""


def generate_text(tokenizer: Any, model: Any, torch: Any, prompt: str, max_new_tokens: int, do_sample: bool, temperature: float, top_p: float) -> str:
    rendered = apply_chat_template(tokenizer, prompt)
    inputs = tokenizer(rendered, return_tensors="pt", truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_width = inputs["input_ids"].shape[1]
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    if do_sample:
        kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        kwargs.update(do_sample=False, num_beams=1)
    with torch.no_grad():
        out = model.generate(**inputs, **kwargs)
    return tokenizer.decode(out[0][input_width:], skip_special_tokens=True)


def generate_completions(rows: list[dict[str, Any]], prompts: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    tokenizer, model, torch = load_model(args.model_path, args.trust_remote_code)
    generations = []
    for idx, (row, prompt) in enumerate(zip(rows, prompts), start=1):
        candidates = []
        # greedy candidate
        completion = generate_text(tokenizer, model, torch, prompt, args.max_new_tokens, False, args.temperature, args.top_p)
        candidates.append(generation_record(row, args.config, args.include_challenge_tests, completion, extract_candidate_code(completion), "reflexion", args.num_candidates))
        # sampled candidates
        for _ in range(max(args.num_candidates - 1, 0)):
            completion = generate_text(tokenizer, model, torch, prompt, args.max_new_tokens, True, args.temperature, args.top_p)
            candidates.append(generation_record(row, args.config, args.include_challenge_tests, completion, extract_candidate_code(completion), "reflexion", args.num_candidates))
        best = choose_best_by_tests(candidates, row, args.config, args.include_challenge_tests, args.test_timeout, args.memory_mb)
        # Reflexion repair loop
        repairs = []
        for attempt in range(args.repair_attempts):
            score = score_code(best["code"], row, args.config, args.include_challenge_tests, args.test_timeout, args.memory_mb)
            if score["all_pass"]:
                break
            failure_info = summarize_failures(score, row_tests(row, args.include_challenge_tests))
            repair_prompt = build_reflexion_prompt(row, args.config, args.include_challenge_tests, best["code"], failure_info)
            repair_completion = generate_text(tokenizer, model, torch, repair_prompt, args.max_new_tokens, True, args.repair_temperature, args.top_p)
            repair_code = extract_candidate_code(repair_completion)
            repair_rec = generation_record(row, args.config, args.include_challenge_tests, repair_completion, repair_code, "reflexion", args.num_candidates)
            repair_score = score_code(repair_code, row, args.config, args.include_challenge_tests, args.test_timeout, args.memory_mb)
            repairs.append({"attempt": attempt + 1, "failure_info": failure_info, "passed_tests": repair_score["passed_tests"], "total_tests": repair_score["total_tests"], "syntax_ok": repair_score["syntax_ok"]})
            if repair_score["passed_tests"] > score["passed_tests"]:
                best = {**repair_rec, "candidate_id": f"repair_{attempt+1}", "candidate_score": repair_score}
        best["reflexion_repairs"] = repairs
        generations.append(best)
        print(f"Generated {len(generations)} / {len(prompts)}", flush=True)
    return generations


def evaluate_generation(gen: dict[str, Any], row: dict[str, Any], config: str, include_challenge: bool, timeout: float, memory_mb: int) -> dict[str, Any]:
    code = normalize_text(gen.get("code"))
    score = score_code(code, row, config, include_challenge, timeout, memory_mb)
    return {
        "task_id": int(row["task_id"]),
        "prompt": row_prompt(row, config),
        "code": code,
        "reference_code": normalize_text(row.get("code")),
        "syntax_ok": score["syntax_ok"],
        "passed": score["all_pass"],
        "passed_tests": score["passed_tests"],
        "total_tests": score["total_tests"],
        "test_results": score["test_results"],
        "completion": gen.get("completion", ""),
        "candidate_id": gen.get("candidate_id", 0),
        "all_candidates_summary": gen.get("all_candidates_summary", []),
        "reflexion_repairs": gen.get("reflexion_repairs", []),
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.predictions or (args.output_dir / "mbpp_generations.jsonl")
    prompt_ids = [int(x) for x in args.prompt_task_ids.split(",") if x.strip()]
    prompt_rows = load_split(args.mbpp_dir, args.config, "prompt")
    eval_rows = load_split(args.mbpp_dir, args.config, args.split)[args.start_index:]
    if args.limit > 0:
        eval_rows = eval_rows[:args.limit]
    prefix = build_prompt_prefix(prompt_rows, args.config, args.prompt_mode, prompt_ids)
    prompts = [build_prompt(prefix, row, args.config, args.include_challenge_tests) for row in eval_rows]
    if args.use_reference_code:
        generations = [generation_record(r, args.config, args.include_challenge_tests, normalize_text(r.get("code")), normalize_text(r.get("code")), "reference", 1) for r in eval_rows]
        save_jsonl(predictions_path, generations)
    elif args.skip_generation:
        generations = read_jsonl(predictions_path)
    else:
        generations = generate_completions(eval_rows, prompts, args)
        save_jsonl(predictions_path, generations)
    by_id = {int(g["task_id"]): g for g in generations}
    cases = [evaluate_generation(by_id[int(r["task_id"])], r, args.config, args.include_challenge_tests, args.test_timeout, args.memory_mb) for r in eval_rows if int(r["task_id"]) in by_id]
    total = len(cases)
    passed = sum(1 for c in cases if c["passed"])
    syntax_passed = sum(1 for c in cases if c["syntax_ok"])
    total_tests = sum(c["total_tests"] for c in cases)
    passed_tests = sum(c["passed_tests"] for c in cases)
    repair_used = sum(1 for c in cases if c.get("reflexion_repairs"))
    metrics = {
        "benchmark": "MBPP", "config": args.config, "split": args.split,
        "model_path": str(args.model_path), "predictions": str(predictions_path),
        "num_tasks": total, "pass_at_1": passed / total if total else 0.0,
        "syntax_pass_rate": syntax_passed / total if total else 0.0,
        "avg_test_pass_rate": passed_tests / total_tests if total_tests else 0.0,
        "passed_tasks": passed, "total_tests": total_tests, "passed_tests": passed_tests,
        "inference_method": "reflexion", "num_candidates": args.num_candidates,
        "temperature": args.temperature, "top_p": args.top_p,
        "repair_attempts": args.repair_attempts, "repair_used": repair_used,
        "prompt": {"source": "Google Research MBPP README", "template": OFFICIAL_PROMPT_TEMPLATE, "mode": args.prompt_mode,
                   "example_task_ids": prompt_ids if args.prompt_mode == "three_shot" else prompt_ids[:1] if args.prompt_mode == "one_shot" else []},
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
