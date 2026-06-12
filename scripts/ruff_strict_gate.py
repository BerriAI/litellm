#!/usr/bin/env python3
"""Drift-proof gate for the strict ruff rules in ruff-strict.toml.

Runs ruff on the current tree, keeps only the violations that land on lines this
change added relative to a base ref, and fails when a rule's count of new
violations exceeds its allowance in ruff-strict-budget.json. Because the base is
measured live, anything that merged into the base does not count against the
change; only what this change introduces is gated.
"""

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
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


class GateResult(NamedTuple):
    ok: bool
    by_rule: dict
    breaches: dict


def _run(cmd: list) -> str:
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def parse_changed_lines(diff_text: str) -> dict:
    changed: dict = defaultdict(set)
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif path and (match := _HUNK.match(line)):
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            changed[path].update(range(start, start + count))
    return changed


def head_violations() -> list:
    raw = _run(
        [
            "ruff",
            "check",
            TARGET,
            "--config",
            str(STRICT_CONFIG),
            "--output-format",
            "json",
        ]
    )
    out = []
    for item in json.loads(raw or "[]"):
        name = Path(item["filename"])
        rel = (
            (name if name.is_absolute() else REPO_ROOT / name)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix()
        )
        out.append(Violation(rel, item["location"]["row"], item["code"]))
    return out


def introduced(violations: list, changed: dict) -> list:
    return [v for v in violations if v.line in changed.get(v.file, set())]


def evaluate(introduced_violations: list, allowances: dict) -> GateResult:
    by_rule: dict = defaultdict(int)
    for violation in introduced_violations:
        by_rule[violation.code] += 1
    breaches = {
        code: count
        for code, count in by_rule.items()
        if count > allowances.get(code, 0)
    }
    return GateResult(not breaches, dict(by_rule), breaches)


def cmd_check(base: str) -> None:
    allowances = json.loads(BUDGET_PATH.read_text())
    base_point = _run(["git", "merge-base", base, "HEAD"]).strip() or base
    diff = _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])
    new = introduced(head_violations(), parse_changed_lines(diff))
    result = evaluate(new, allowances)
    if result.ok:
        print(
            f"OK: this change introduces {sum(result.by_rule.values())} strict-rule violation(s), all within budget (base {base})"
        )
        return
    print(
        f"FAIL: this change introduces strict-rule violations over budget (base {base}):"
    )
    for code, count in sorted(result.breaches.items()):
        print(f"  {code}: +{count} (allowed {allowances.get(code, 0)})")
    for violation in sorted(new):
        if violation.code in result.breaches:
            print(f"    {violation.file}:{violation.line} {violation.code}")
    print(
        "Type the new code, reduce complexity/arg count, or add `# noqa: <CODE>` on a justified line."
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    cmd_check(parser.parse_args().base)


if __name__ == "__main__":
    main()
