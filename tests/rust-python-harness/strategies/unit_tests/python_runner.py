from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final, TypeAlias

CommandRunner: TypeAlias = Callable[[tuple[str, ...], Path], str]

_COLLECTED_COUNT: Final = re.compile(r"^(\d+) tests? collected\b")
_FUNCTION_ID: Final = re.compile(r"^(?:\w+::)*test_\w+$")


def run_collect_command(command: tuple[str, ...], cwd: Path) -> str:
    result: Final = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ValueError(
            f"Python inventory command failed ({result.returncode}): {' '.join(command)}\n"
            f"{result.stderr}\n{result.stdout}"
        )
    return result.stdout


def _collect_command(relative_paths: tuple[str, ...]) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "pytest",
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        *relative_paths,
    )


def _parse_collected_output(
    output: str, relative_paths: tuple[str, ...]
) -> dict[str, frozenset[str]]:
    requested: Final = frozenset(relative_paths)
    collected: dict[str, set[str]] = {}
    node_count = 0
    reported_count: int | None = None
    for line in output.splitlines():
        summary_match: Final = _COLLECTED_COUNT.match(line)
        if summary_match:
            reported_count = int(summary_match.group(1))
            continue
        if "::" not in line:
            continue
        python_file, _, node = line.partition("::")
        if python_file not in requested:
            continue
        function: Final = re.sub(r"\[.*$", "", node)
        if not _FUNCTION_ID.fullmatch(function):
            raise ValueError(f"Unrecognized pytest node id: {line!r}")
        collected.setdefault(python_file, set()).add(function)
        node_count += 1
    if reported_count is not None and reported_count != node_count:
        raise ValueError(
            f"pytest collected {reported_count} tests but {node_count} node ids were parsed"
        )
    empty: Final = tuple(sorted(requested - frozenset(collected)))
    if empty:
        raise ValueError(f"pytest collected no tests in: {', '.join(empty)}")
    return {path: frozenset(tests) for path, tests in collected.items()}


def collect_python_inventory(
    repo_root: Path,
    relative_paths: tuple[str, ...],
    *,
    command_runner: CommandRunner = run_collect_command,
) -> dict[str, frozenset[str]]:
    output: Final = command_runner(_collect_command(relative_paths), repo_root)
    return _parse_collected_output(output, relative_paths)
