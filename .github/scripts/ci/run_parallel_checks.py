#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class CheckResult:
    command: str
    returncode: int
    seconds: float
    output: str

    @property
    def passed(self) -> bool:
        return self.returncode == 0


def _run(command: str) -> CheckResult:
    started: Final = time.monotonic()
    completed: Final = subprocess.run(
        command, shell=True, executable="/bin/bash", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    return CheckResult(command, completed.returncode, time.monotonic() - started, completed.stdout)


def _reported(result: CheckResult) -> CheckResult:
    status: Final = "PASS" if result.passed else f"FAIL (exit {result.returncode})"
    print(f"::group::{status} {result.seconds:.0f}s {result.command}", flush=True)
    print(result.output, end="" if result.output.endswith("\n") else "\n", flush=True)
    print("::endgroup::", flush=True)
    if not result.passed:
        print(f"::error title=Check failed::{result.command}", flush=True)
    return result


def main() -> int:
    parser: Final = argparse.ArgumentParser(description="Run one shell command per stdin line, in parallel.")
    parser.add_argument("--jobs", type=int, required=True)
    jobs: Final[int] = parser.parse_args().jobs
    commands: Final = tuple(line.strip() for line in sys.stdin if line.strip())
    if not commands:
        print("::error::No checks were given on stdin.", flush=True)
        return 1
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures: Final = tuple(pool.submit(_run, command) for command in commands)
        results: Final = tuple(_reported(future.result()) for future in as_completed(futures))
    failed: Final = tuple(result.command for result in results if not result.passed)
    print(f"{len(results) - len(failed)} of {len(results)} checks passed.", flush=True)
    for command in failed:
        print(f"Failed: {command}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
