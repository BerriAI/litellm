#!/usr/bin/env python3
"""Total-count gate for the strict ruff rules in ruff-strict.toml.

Each rule has a hard ceiling (baseline + slack) in ruff-strict-budget.json. The
gate counts each rule across the whole tree and fails when a rule is both over
its ceiling and higher than the base it merges into, so a change is blamed for
the violations it adds, never for drift that already exists in the base.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
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


def head_violations() -> list:
    out = []
    for item in _ruff_json(REPO_ROOT, STRICT_CONFIG):
        name = Path(item["filename"])
        rel = (
            (name if name.is_absolute() else REPO_ROOT / name)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix()
        )
        out.append(Violation(rel, item["location"]["row"], item["code"]))
    return out


def count_by_rule(violations: list) -> dict:
    return dict(Counter(v.code for v in violations))


def base_counts(ref: str) -> dict:
    parent = Path(tempfile.mkdtemp(prefix="ruff_base_"))
    worktree = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref])
        shutil.copy(STRICT_CONFIG, worktree / "ruff-strict.toml")
        items = _ruff_json(worktree, worktree / "ruff-strict.toml")
        return dict(Counter(item["code"] for item in items))
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)])
        shutil.rmtree(parent, ignore_errors=True)


def evaluate(head: dict, base: dict, budget: dict) -> list:
    breaches = []
    for rule, spec in budget.items():
        cap = spec["baseline"] + spec["slack"]
        total = head.get(rule, 0)
        if total > cap and total > base.get(rule, 0):
            breaches.append(Breach(rule, total, cap, total - base.get(rule, 0)))
    return sorted(breaches)


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


def introduced(violations: list, changed: dict) -> list:
    return [v for v in violations if v.line in changed.get(v.file, set())]


def cmd_check(base: str) -> None:
    budget = json.loads(BUDGET_PATH.read_text())
    head = head_violations()
    base_point = _run(["git", "merge-base", base, "HEAD"]).strip() or base
    breaches = evaluate(count_by_rule(head), base_counts(base_point), budget)
    if not breaches:
        print(f"OK: every strict rule is within its codebase ceiling (base {base})")
        return
    new = introduced(
        head,
        parse_changed_lines(
            _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])
        ),
    )
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
    raise SystemExit(1)


def cmd_update() -> None:
    budget = json.loads(BUDGET_PATH.read_text())
    head = count_by_rule(head_violations())
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
