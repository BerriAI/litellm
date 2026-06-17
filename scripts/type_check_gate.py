#!/usr/bin/env python3
"""Per-rule count gate for basedpyright.

basedpyright's ``--outputjson`` is reduced to a count of errors per *rule*
(``reportAny``, ``reportArgumentType``, ...) and checked against a committed
budget of the form ``{rule: {baseline, slack}}``, the same shape as
``ruff-strict-budget.json``. A rule fails when its codebase-wide total exceeds
``baseline + slack``. Counts ignore file, line, and column, so a violation
moving anywhere in the tree is invisible; only the per-rule total moves the
needle.

Unlike ``ruff_strict_gate.py`` this does *not* re-run the tool on the merge base
to compute a delta: a second basedpyright pass is minutes and gigabytes, whereas
ruff is milliseconds. The committed budget is the baseline instead -- exactly
how the previous per-file gate worked -- so keep it fresh with ``--update``
(ratchet), which re-captures every rule's count from the current tree while
preserving each rule's slack. Tool output is read from stdin, so the caller
decides how to invoke basedpyright (and from which cwd).

``--outputjson`` is used rather than text diagnostics because the latter wrap
across lines, leaving the ``(reportRule)`` on a continuation line away from the
``- error:`` marker, so line parsing mis-attributes ~60% of errors -- the JSON
carries an unambiguous ``rule`` field.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent

# Bucket for a basedpyright diagnostic with no `rule`. Counted so it's gated.
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


def count_basedpyright(payload: str) -> dict[str, int]:
    """Count in-repo basedpyright errors per rule from `--outputjson`. Warnings
    and information are ignored; only `severity == "error"` is gated."""
    try:
        data = json.loads(payload or "{}")
    except json.JSONDecodeError as exc:
        sys.stderr.write(
            f"basedpyright did not emit valid JSON ({exc}); it likely crashed or "
            f"printed text before the JSON. First 500 chars of its output:\n"
            f"{payload[:500]}\n"
        )
        raise SystemExit(1) from exc
    counts: Counter[str] = Counter()
    for diag in data.get("generalDiagnostics", []):
        if diag.get("severity") != "error":
            continue
        if _to_repo_relative(diag.get("file", "")) is None:
            continue
        counts[diag.get("rule") or UNCODED] += 1
    return dict(counts)


def evaluate(
    counts: Mapping[str, int], budget: Mapping[str, Mapping[str, int]]
) -> list[Breach]:
    breaches = []
    for code, total in counts.items():
        spec = budget.get(code)
        cap = spec["baseline"] + spec["slack"] if spec else DEFAULT_SLACK
        if total > cap:
            breaches.append(Breach(code, total, cap))
    return sorted(breaches)


def is_vacuous_run(
    counts: Mapping[str, int], budget: Mapping[str, Mapping[str, int]]
) -> bool:
    """True when nothing was parsed but the budget expects errors -- the
    signature of a type checker that crashed or produced no output. The CI pipe
    swallows the tool's exit code (`tool || true`), so without this guard an
    empty run would clear every ceiling and pass silently."""
    return not counts and any(spec["baseline"] for spec in budget.values())


BUDGET_PATH = REPO_ROOT / "basedpyright-code-budget.json"


def cmd_update(counts: Mapping[str, int]) -> None:
    existing = json.loads(BUDGET_PATH.read_text()) if BUDGET_PATH.exists() else {}
    budget = {
        code: {
            "baseline": count,
            "slack": (
                existing[code]["slack"] if code in existing else _seed_slack(count)
            ),
        }
        for code, count in sorted(counts.items())
    }
    BUDGET_PATH.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n")
    print(
        f"Re-captured basedpyright per-rule budget: {len(budget)} rules, {sum(counts.values())} errors total"
    )


def cmd_check(counts: Mapping[str, int]) -> None:
    budget = json.loads(BUDGET_PATH.read_text())
    if is_vacuous_run(counts, budget):
        expected = sum(spec["baseline"] for spec in budget.values())
        print(
            f"FAIL: basedpyright produced no errors, but {BUDGET_PATH.name} expects "
            f"~{expected}. The type checker almost certainly crashed or emitted "
            f"nothing; refusing to certify a vacuous run."
        )
        raise SystemExit(1)
    breaches = evaluate(counts, budget)
    if not breaches:
        print(
            f"OK: every rule is within its basedpyright ceiling ({sum(counts.values())} errors total)"
        )
        return
    print("FAIL: basedpyright errors exceed the per-rule ceiling:")
    for breach in breaches:
        print(f"  {breach.code}: {breach.total} errors over cap {breach.cap}")
    print(
        "Resolve the new errors, or run 'make lint-basedpyright-budget-update' if the ceiling should move."
    )
    raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    counts = count_basedpyright(sys.stdin.read())
    cmd_update(counts) if args.update else cmd_check(counts)


if __name__ == "__main__":
    main()
