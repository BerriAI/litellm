#!/usr/bin/env python3
"""Budget gate for the strict ruff rules defined in ruff-strict.toml.

`update` regenerates ruff-suppressions.json from the current tree.
`check` fails when the total strict-rule violation count grows past the
grandfathered baseline plus a small slack margin.
"""

import argparse
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
STRICT_CONFIG = REPO_ROOT / "ruff-strict.toml"
BASELINE_PATH = REPO_ROOT / "ruff-suppressions.json"
TARGET = "litellm"
SLACK_RATIO = 0.005


def current_counts() -> dict:
    proc = subprocess.run(
        [
            "ruff",
            "check",
            TARGET,
            "--config",
            str(STRICT_CONFIG),
            "--output-format",
            "json",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"ruff exited {proc.returncode}")
    counts: dict = defaultdict(lambda: defaultdict(int))
    for violation in json.loads(proc.stdout or "[]"):
        raw = Path(violation["filename"])
        path = (
            (raw if raw.is_absolute() else REPO_ROOT / raw)
            .resolve()
            .relative_to(REPO_ROOT)
            .as_posix()
        )
        counts[path][violation["code"]] += 1
    return {path: dict(rules) for path, rules in counts.items()}


def total(counts: dict) -> int:
    return sum(n for rules in counts.values() for n in rules.values())


class BudgetResult(NamedTuple):
    ok: bool
    current: int
    baseline: int
    slack: int
    ceiling: int
    regressions: list


def evaluate_budget(
    baseline: dict, counts: dict, slack_ratio: float = SLACK_RATIO
) -> BudgetResult:
    base_total, cur_total = total(baseline), total(counts)
    slack = math.ceil(base_total * slack_ratio)
    ceiling = base_total + slack
    regressions = []
    for path in sorted(set(counts) | set(baseline)):
        cur, base = counts.get(path, {}), baseline.get(path, {})
        for code in sorted(set(cur) | set(base)):
            grew = cur.get(code, 0) - base.get(code, 0)
            if grew > 0:
                regressions.append((path, code, base.get(code, 0), cur.get(code, 0)))
    return BudgetResult(
        cur_total <= ceiling, cur_total, base_total, slack, ceiling, regressions
    )


def write_baseline(counts: dict) -> None:
    BASELINE_PATH.write_text(json.dumps(counts, indent=2, sort_keys=True) + "\n")


def cmd_update() -> None:
    counts = current_counts()
    write_baseline(counts)
    print(
        f"Wrote {BASELINE_PATH.name}: {total(counts)} violations across {len(counts)} files"
    )


def _print_regressions(regressions: list) -> None:
    for path, code, before, after in regressions:
        print(f"  {path}: {code} {before} -> {after} (+{after - before})")


def cmd_check() -> None:
    if not BASELINE_PATH.exists():
        raise SystemExit(
            f"{BASELINE_PATH.name} missing; run: python scripts/ruff_suppressions.py update"
        )
    result = evaluate_budget(json.loads(BASELINE_PATH.read_text()), current_counts())
    if result.ok:
        print(
            f"OK: {result.current} strict-rule violations (baseline {result.baseline}, ceiling {result.ceiling}, headroom {result.ceiling - result.current})"
        )
        if result.regressions:
            print(
                f"warning: {len(result.regressions)} per-file regression(s) absorbed by slack (offset by fixes elsewhere):"
            )
            _print_regressions(result.regressions)
        return
    print(
        f"FAIL: {result.current} strict-rule violations exceeds ceiling {result.ceiling} (baseline {result.baseline} + {SLACK_RATIO:.1%} slack = {result.slack})"
    )
    _print_regressions(result.regressions)
    print(
        "Fix the new violations, add `# noqa: <CODE>` on a justified line, or re-baseline with `make lint-suppressions-update`."
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["check", "update"])
    {"check": cmd_check, "update": cmd_update}[parser.parse_args().mode]()


if __name__ == "__main__":
    main()
