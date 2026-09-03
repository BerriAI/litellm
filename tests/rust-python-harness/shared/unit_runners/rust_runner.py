from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Final


@dataclass(frozen=True, slots=True)
class RustReport:
    tests: tuple[str, ...]
    exit_code: int
    output: str


def run_rust_tests(manifest: Path, package: str, test_filter: str, *, collect_only: bool = False) -> RustReport:
    command: Final = (
        "cargo",
        "test",
        "--manifest-path",
        str(manifest),
        "--package",
        package,
        "--lib",
        test_filter,
        "--",
        *(("--list",) if collect_only else ("--format=pretty",)),
    )
    try:
        result: Final = subprocess.run(command, capture_output=True, text=True, check=False, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as error:
        return RustReport((), 1, str(error))
    tests: Final = (
        tuple(line.removesuffix(": test") for line in result.stdout.splitlines() if line.endswith(": test"))
        if collect_only
        else tuple(
            line.removeprefix("test ").removesuffix(" ... ok")
            for line in result.stdout.splitlines()
            if line.startswith("test ") and line.endswith(" ... ok")
        )
    )
    return RustReport(tests, result.returncode, result.stdout + result.stderr)


_RUST_TEST_PATTERN = re.compile(
    r"#\[(?:test|tokio::test)\][^\n]*\n(?:[^\n]*\n)*?\s*(?:async\s+)?fn\s+(\w+)\s*\("
)


def enumerate_rust_tests(repo_root: Path, relative_path: str) -> frozenset[str]:
    source = (repo_root / relative_path).read_text(encoding="utf-8")
    return frozenset(match.group(1) for match in _RUST_TEST_PATTERN.finditer(source))
