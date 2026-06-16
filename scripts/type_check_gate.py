#!/usr/bin/env python3
"""Per-rule count gate for mypy and basedpyright.

Each tool's output is reduced to a count of errors per *rule* (mypy error codes
like ``arg-type``, basedpyright rules like ``reportAny``) and checked against a
committed budget of the form ``{rule: {baseline, slack}}``, the same shape as
``ruff-strict-budget.json``. A rule fails when its codebase-wide total exceeds
``baseline + slack``. Counts ignore file, line, and column, so a violation
moving anywhere in the tree is invisible; only the per-rule total moves the
needle.

Unlike ``ruff_strict_gate.py`` this does *not* re-run the tool on the merge base
to compute a delta: a second mypy/basedpyright pass is minutes and gigabytes,
whereas ruff is milliseconds. The committed budget is the baseline instead --
exactly how the previous per-file gate worked -- so keep it fresh with
``--update`` (ratchet), which re-captures every rule's count from the current
tree while preserving each rule's slack. Tool output is read from stdin, so the
caller decides how to invoke the tool (and from which cwd).

mypy is parsed from its text output (one error per line, the rule code in a
trailing ``[bracket]``). basedpyright is parsed from ``--outputjson``: its text
diagnostics routinely wrap across lines, leaving the ``(reportRule)`` on a
continuation line away from the ``- error:`` marker, so line parsing
mis-attributes ~60% of errors -- the JSON carries an unambiguous ``rule`` field.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# mypy: one error per line, e.g. `path:12: error: msg  [arg-type]`. ERROR_LINE
# recognizes the line; MYPY_CODE pulls the trailing [code]. Kept separate so an
# error emitted without a code is still counted (under UNCODED), never dropped.
MYPY_ERROR = re.compile(r"^(?P<file>.+?):\d+: error:")
MYPY_CODE = re.compile(r"\[(?P<code>[a-z][a-z0-9-]*)\]\s*$")

# Bucket for an error whose rule code we couldn't read (a mypy error with no
# code, or a basedpyright diagnostic with no `rule`). Counted so it's gated.
UNCODED = "<uncoded>"

# Ceiling for a rule that shows up at HEAD but isn't in the budget at all -- a
# brand-new error category (new construct, or a tool/version change). baseline
# is treated as 0, so the rule fails once it clears this much slack.
DEFAULT_SLACK = 10


class Breach(NamedTuple):
    code: str
    total: int
    cap: int


def _seed_slack(baseline: int) -> int:
    """Slack written for a rule first captured into a budget; busy rules get
    more headroom, mirroring the tiering in ruff-strict-budget.json. Existing
    rules keep whatever slack their JSON already declares."""
    return 10 if baseline >= 50 else 3


def _to_repo_relative(raw: str) -> str | None:
    path = Path(raw)
    absolute = path if path.is_absolute() else Path.cwd() / path
    try:
        return absolute.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def count_mypy(lines: Iterable[str]) -> dict[str, int]:
    """Count in-repo mypy errors per rule code from text output. Errors for
    files outside the repo (third-party stubs) are ignored, as before."""
    counts: Counter[str] = Counter()
    for raw in lines:
        line = raw.rstrip("\n")
        match = MYPY_ERROR.match(line)
        if match is None or _to_repo_relative(match.group("file")) is None:
            continue
        code = MYPY_CODE.search(line)
        counts[code.group("code") if code else UNCODED] += 1
    return dict(counts)


def count_basedpyright(payload: str) -> dict[str, int]:
    """Count in-repo basedpyright errors per rule from `--outputjson`. Warnings
    and information are ignored; only `severity == "error"` is gated."""
    data = json.loads(payload or "{}")
    counts: Counter[str] = Counter()
    for diag in data.get("generalDiagnostics", []):
        if diag.get("severity") != "error":
            continue
        if _to_repo_relative(diag.get("file", "")) is None:
            continue
        counts[diag.get("rule") or UNCODED] += 1
    return dict(counts)


def count_errors(stdin_text: str, tool: str) -> dict[str, int]:
    if tool == "basedpyright":
        return count_basedpyright(stdin_text)
    return count_mypy(stdin_text.splitlines())


def evaluate(counts: Mapping[str, int], budget: Mapping[str, Mapping[str, int]]) -> list[Breach]:
    breaches = []
    for code, total in counts.items():
        spec = budget.get(code)
        cap = spec["baseline"] + spec["slack"] if spec else DEFAULT_SLACK
        if total > cap:
            breaches.append(Breach(code, total, cap))
    return sorted(breaches)


def budget_path(tool: str) -> Path:
    return REPO_ROOT / f"{tool}-code-budget.json"


def cmd_update(tool: str, counts: Mapping[str, int]) -> None:
    path = budget_path(tool)
    existing = json.loads(path.read_text()) if path.exists() else {}
    budget = {
        code: {
            "baseline": count,
            "slack": existing[code]["slack"] if code in existing else _seed_slack(count),
        }
        for code, count in sorted(counts.items())
    }
    path.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n")
    print(
        f"Re-captured {tool} per-rule budget: {len(budget)} rules, {sum(counts.values())} errors total"
    )


def cmd_check(tool: str, counts: Mapping[str, int]) -> None:
    budget = json.loads(budget_path(tool).read_text())
    breaches = evaluate(counts, budget)
    if not breaches:
        print(f"OK: every rule is within its {tool} ceiling ({sum(counts.values())} errors total)")
        return
    print(f"FAIL: {tool} errors exceed the per-rule ceiling:")
    for breach in breaches:
        print(f"  {breach.code}: {breach.total} errors over cap {breach.cap}")
    print(
        f"Resolve the new errors, or run 'make lint-{tool}-budget-update' if the ceiling should move."
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", choices=("mypy", "basedpyright"), required=True)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    counts = count_errors(sys.stdin.read(), args.tool)
    cmd_update(args.tool, counts) if args.update else cmd_check(args.tool, counts)


if __name__ == "__main__":
    main()
