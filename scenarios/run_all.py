#!/usr/bin/env python3
"""Run all LiteLLM release scenarios with colored output and summary."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from dotenv import find_dotenv, load_dotenv

SCENARIO_DIR = Path(__file__).parent
PYTHON = sys.executable

SCENARIOS: List[Tuple[str, List[str]]] = [
    ("mini_agent_live.py", [PYTHON, str(SCENARIO_DIR / "mini_agent_live.py")]),
    (
        "mini_agent_http_release.py",
        [PYTHON, str(SCENARIO_DIR / "mini_agent_http_release.py")],
    ),
    (
        "parallel_acompletions_demo.py",
        [PYTHON, str(SCENARIO_DIR / "parallel_acompletions_demo.py")],
    ),
    (
        "router_parallel_release.py",
        [PYTHON, str(SCENARIO_DIR / "router_parallel_release.py")],
    ),
    (
        "router_batch_release.py",
        [PYTHON, str(SCENARIO_DIR / "router_batch_release.py")],
    ),
    (
        "image_compression_release.py",
        [PYTHON, str(SCENARIO_DIR / "image_compression_release.py")],
    ),
    (
        "codex_agent_router.py",
        [PYTHON, str(SCENARIO_DIR / "codex_agent_router.py")],
    ),
    (
        "chutes_release.py",
        [PYTHON, str(SCENARIO_DIR / "chutes_release.py")],
    ),
    (
        "code_agent_release.py",
        [PYTHON, str(SCENARIO_DIR / "code_agent_release.py")],
    ),
]

RESET = "\033[0m"
BLUE = "\033[1;34m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
load_dotenv(find_dotenv())
def main() -> None:
    results = []
    env = os.environ.copy()
    env.setdefault("LITELLM_ENABLE_MINI_AGENT", "1")
    env.setdefault("LITELLM_ENABLE_CODEX_AGENT", "1")

    for name, cmd in SCENARIOS:
        print(f"{BLUE}▶ Running {name}{RESET}")
        proc = subprocess.run(cmd, env=env)
        success = proc.returncode == 0
        results.append((name, success))
        if success:
            print(f"{GREEN}✓ {name} succeeded{RESET}\n")
        else:
            print(f"{RED}✗ {name} failed (exit code {proc.returncode}){RESET}\n")
            if env.get("SCENARIOS_STOP_ON_FIRST_FAILURE"):
                break

    passed = [name for name, ok in results if ok]
    failed = [name for name, ok in results if not ok]

    print("\n" + "=" * 60)
    print("Scenario Summary")
    print("=" * 60)
    for name in passed:
        print(f"{GREEN}  ✓ {name}{RESET}")
    for name in failed:
        print(f"{RED}  ✗ {name}{RESET}")

    if not failed:
        print(f"\n{GREEN}All scenarios succeeded!{RESET}")
        sys.exit(0)
    else:
        print(f"\n{YELLOW}{len(failed)} scenario(s) failed. See logs above.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
