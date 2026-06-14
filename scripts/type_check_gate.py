#!/usr/bin/env python3
"""Per-file count gate for mypy and basedpyright.

Each tool's output is reduced to a count of errors per file and checked against
a committed budget, the max errors allowed per file. Counts ignore line and
column numbers, so they survive code moving within a file; a file fails only
when it gains more errors than its ceiling. Run with --update to re-capture the
budget from the current tree (ratchet). The tool output is read from stdin, so
the caller decides how to invoke mypy or basedpyright.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

PATTERNS: Mapping[str, re.Pattern[str]] = {
    "mypy": re.compile(r"^(?P<file>.+?):\d+: error:"),
    "basedpyright": re.compile(r"^\s*(?P<file>.+?):\d+:\d+ - error:"),
}

# Headroom per file on top of its recorded count, so an inference ripple in an
# unrelated file (basedpyright especially) does not fail the build over a couple
# of errors. Small on purpose: a file still fails once it drifts past this.
PER_FILE_SLACK = 5


class Breach(NamedTuple):
    file: str
    count: int
    cap: int


def _to_repo_relative(raw: str) -> str | None:
    path = Path(raw)
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        return absolute.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def count_errors(lines: Iterable[str], pattern: re.Pattern[str]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for line in lines:
        match = pattern.match(line)
        if match is None:
            continue
        rel = _to_repo_relative(match.group("file"))
        if rel is not None:
            counts[rel] += 1
    return dict(counts)


def evaluate(
    counts: Mapping[str, int], budget: Mapping[str, int], slack: int
) -> list[Breach]:
    return sorted(
        Breach(file, count, budget.get(file, 0) + slack)
        for file, count in counts.items()
        if count > budget.get(file, 0) + slack
    )


def budget_path(tool: str) -> Path:
    return REPO_ROOT / f"{tool}-file-budget.json"


def cmd_update(tool: str, counts: Mapping[str, int]) -> None:
    path = budget_path(tool)
    path.write_text(json.dumps(dict(sorted(counts.items())), indent=2) + "\n")
    print(
        f"Re-captured {tool} per-file budget: {len(counts)} files, {sum(counts.values())} errors"
    )


def cmd_check(tool: str, counts: Mapping[str, int]) -> None:
    budget = json.loads(budget_path(tool).read_text())
    breaches = evaluate(counts, budget, PER_FILE_SLACK)
    if not breaches:
        print(
            f"OK: every file is within its {tool} ceiling ({sum(counts.values())} errors total)"
        )
        return
    print(f"FAIL: {tool} errors exceed the per-file ceiling:")
    for breach in breaches:
        print(f"  {breach.file}: {breach.count} errors over cap {breach.cap}")
    print(
        f"Resolve the new errors, or run 'make lint-{tool}-budget-update' if the ceiling should move."
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=tuple(PATTERNS), required=True)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    counts = count_errors(sys.stdin, PATTERNS[args.tool])
    cmd_update(args.tool, counts) if args.update else cmd_check(args.tool, counts)


if __name__ == "__main__":
    main()
