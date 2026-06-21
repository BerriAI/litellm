#!/usr/bin/env python3
"""Total-count gate for the strict ruff rules in ruff-strict.toml.

Each rule has a hard ceiling (baseline + slack) in ruff-strict-budget.json. The
gate counts each rule across the whole tree and fails when a rule is both over
its ceiling and higher than the base it merges into, so a change is blamed for
the violations it adds, never for drift that already exists in the base.

The base is the merge-base of the current branch with --base; this matches CI,
which checks out the PR head sha and runs the gate against the PR's base sha.
"""

import argparse
import contextlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
STRICT_CONFIG = REPO_ROOT / "ruff-strict.toml"
BUDGET_PATH = REPO_ROOT / "ruff-strict-budget.json"
TARGET = "litellm"
DEFAULT_BASE = "origin/litellm_internal_staging"

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


class Violation(NamedTuple):
    file: str
    line: int
    code: str


class Breach(NamedTuple):
    rule: str
    total: int
    cap: int
    added: int


class GateInputs(NamedTuple):
    head: list[Violation]
    base: dict[str, int]
    changed: dict[str, set[int]]


def _run(cmd: list, cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def _ruff_json(cwd: Path, config: Path) -> list:
    raw = _run(
        ["ruff", "check", TARGET, "--config", str(config), "--output-format", "json"],
        cwd=cwd,
    )
    return json.loads(raw or "[]")


def collect_violations(root: Path, config: Path) -> list:
    out = []
    for item in _ruff_json(root, config):
        name = Path(item["filename"])
        rel = (
            (name if name.is_absolute() else root / name)
            .resolve()
            .relative_to(root)
            .as_posix()
        )
        out.append(Violation(rel, item["location"]["row"], item["code"]))
    return out


def count_by_rule(violations: list) -> dict:
    return dict(Counter(v.code for v in violations))


@contextlib.contextmanager
def _temp_worktree(ref: str) -> Iterator[Path]:
    parent = Path(tempfile.mkdtemp(prefix="ruff_wt_"))
    worktree = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref])
        yield worktree
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)


def base_counts(ref: str) -> dict:
    with _temp_worktree(ref) as worktree:
        shutil.copy(STRICT_CONFIG, worktree / "ruff-strict.toml")
        return count_by_rule(
            collect_violations(worktree, worktree / "ruff-strict.toml")
        )


def parse_changed_lines(diff_text: str) -> dict:
    changed: dict = {}
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif path and (match := _HUNK.match(line)):
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            changed.setdefault(path, set()).update(range(start, start + count))
    return changed


def evaluate(head: dict, base: dict, budget: dict) -> list:
    breaches = []
    for rule, spec in budget.items():
        cap = spec["baseline"] + spec["slack"]
        total = head.get(rule, 0)
        if total > cap and total > base.get(rule, 0):
            breaches.append(Breach(rule, total, cap, total - base.get(rule, 0)))
    return sorted(breaches)


def introduced(violations: list, changed: dict) -> list:
    return [v for v in violations if v.line in changed.get(v.file, set())]


def gather(base: str) -> GateInputs:
    base_point = _run(["git", "merge-base", base, "HEAD"]).strip() or base
    diff = _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])
    return GateInputs(
        collect_violations(REPO_ROOT, STRICT_CONFIG),
        base_counts(base_point),
        parse_changed_lines(diff),
    )


def report(breaches: list, new: list, base: str) -> None:
    print(f"FAIL: strict-rule totals exceed their ceiling (base {base}):")
    for breach in breaches:
        print(
            f"  {breach.rule}: total {breach.total} over cap {breach.cap} (this change added {breach.added})"
        )
        for violation in sorted(v for v in new if v.code == breach.rule):
            print(f"    {violation.file}:{violation.line}")
    print(
        "Reduce the new violations or remove an equal number elsewhere; the ceiling is baseline + slack in ruff-strict-budget.json."
    )
    summary = "; ".join(f"{b.rule} {b.total}/{b.cap} (+{b.added})" for b in breaches)
    print(f"BREACHED RULES: {summary}")


def cmd_check(base: str) -> None:
    budget = json.loads(BUDGET_PATH.read_text())
    inputs = gather(base)
    breaches = evaluate(count_by_rule(inputs.head), inputs.base, budget)
    if not breaches:
        print(f"OK: every strict rule is within its codebase ceiling (base {base})")
        return
    report(breaches, introduced(inputs.head, inputs.changed), base)
    raise SystemExit(1)


def cmd_update() -> None:
    budget = json.loads(BUDGET_PATH.read_text())
    head = count_by_rule(collect_violations(REPO_ROOT, STRICT_CONFIG))
    for rule in budget:
        budget[rule]["baseline"] = head.get(rule, 0)
    BUDGET_PATH.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n")
    print("Re-captured per-rule baselines from the current tree")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    cmd_update() if args.update else cmd_check(args.base)


if __name__ == "__main__":
    main()
