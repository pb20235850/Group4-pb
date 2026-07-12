from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


FENCED_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def normalize_text(value: Any) -> str:
    return str(value or "").replace("\r\n", "\n").strip()


def extract_code(text: str) -> str:
    text = normalize_text(text)

    if "[BEGIN]" in text:
        text = text.rsplit("[BEGIN]", 1)[-1]
    if "[DONE]" in text:
        text = text.split("[DONE]", 1)[0]

    fenced = FENCED_CODE_RE.findall(text)
    if fenced:
        return normalize_text(fenced[-1])

    lines = text.splitlines()
    start = 0

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "class ", "import ", "from ")):
            start = i
            break

    code_lines = []
    for line in lines[start:]:
        if line.strip().startswith("```"):
            continue
        code_lines.append(line)

    return normalize_text("\n".join(code_lines))


def syntax_reward(code: str) -> float:
    if not code.strip():
        return 0.0
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0


def format_reward(code: str) -> float:
    if not code.strip():
        return 0.0

    score = 0.0

    if "def " in code:
        score += 0.5

    bad_phrases = [
        "Here is",
        "One possible",
        "Explanation",
        "To solve",
        "The function",
        "This code",
        "You can",
        "For example",
    ]

    if not any(p.lower() in code.lower() for p in bad_phrases):
        score += 0.3

    if len(code.splitlines()) <= 80:
        score += 0.2

    return min(score, 1.0)


def function_name_reward(code: str, entry_point: str) -> float:
    if not entry_point:
        return 0.0

    pattern = rf"def\s+{re.escape(entry_point)}\s*\("
    return 1.0 if re.search(pattern, code) else 0.0


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


def run_tests(code: str, tests: list[str], timeout: float = 3.0, memory_mb: int = 512) -> tuple[int, int]:
    if not code.strip() or not tests:
        return 0, len(tests)

    passed = 0
    total = len(tests)

    for test in tests:
        runner = "\n\n".join(
            part
            for part in [
                "import faulthandler\nfaulthandler.enable()",
                code,
                test,
            ]
            if normalize_text(part)
        )

        with tempfile.TemporaryDirectory(prefix="grpo_code_reward_") as tmpdir:
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

                if result.returncode == 0:
                    passed += 1

            except subprocess.TimeoutExpired:
                continue
            except Exception:
                continue

    return passed, total


def unit_test_reward(code: str, tests: list[str]) -> float:
    passed, total = run_tests(code, tests)
    return passed / total if total else 0.0


def code_reward_func(completions: list[str], **kwargs: Any) -> list[float]:
    """
    Reward = 0.10 * syntax correctness
           + 0.15 * function name match
           + 0.05 * output format
           + 0.70 * unit test pass rate
    """

    tests_batch = kwargs.get("tests", [[] for _ in completions])
    entry_points = kwargs.get("entry_point", ["" for _ in completions])

    rewards = []

    for completion, tests, entry_point in zip(completions, tests_batch, entry_points):
        code = extract_code(completion)

        r_syntax = syntax_reward(code)
        r_func = function_name_reward(code, entry_point)
        r_format = format_reward(code)
        r_tests = unit_test_reward(code, tests)

        total_reward = (
            0.10 * r_syntax
            + 0.15 * r_func
            + 0.05 * r_format
            + 0.70 * r_tests
        )

        rewards.append(float(total_reward))

    return rewards