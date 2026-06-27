#!/usr/bin/env python3
"""Non-gating ratchet guard: budget baselines and ceilings may only fall, never rise.

Every `*-budget.json` file (ruff-strict, type-discipline, basedpyright-code) is a
one-way ratchet: each rule's ceiling is `baseline + slack`, and both the recorded
`baseline` (the live violation count) and that ceiling are meant to be driven DOWN
over time. This check compares every budget file against its own content at the
merge-base with the target branch and fails (exits 1, red) if:

  * a rule's ceiling (`baseline + slack`) went up,
  * a rule's `baseline` went up, even if `slack` was lowered to keep the ceiling
    flat (a higher baseline bakes in more accepted debt and must be acknowledged),
  * a rule was dropped from a budget (its ceiling effectively became infinite), or
  * an entire budget file was deleted.

New rules and lowered/equal baselines and ceilings are fine.

This is deliberately NOT a gating check. It should turn the run red so that a
loosening is impossible to miss in review, but it must stay OUT of the
branch-protection required-checks list: a justified bump (e.g. banning a new API,
which mechanically raises a baseline) can then still be merged by a human who has
seen the red and accepted it.

Usage:
    python scripts/budget_ratchet_check.py [--base REF] [budget.json ...]

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "origin/litellm_internal_staging"
DEFAULT_BUDGETS: tuple[str, ...] = (
    "ruff-strict-budget.json",
    "type-discipline-budget.json",
    "basedpyright-code-budget.json",
)


class Regression(NamedTuple):
    budget: str
    rule: str
    detail: str


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)


def _merge_base(base: str) -> str:
    """The common ancestor of `base` and HEAD, so unrelated base drift is ignored."""
    proc = _run(["git", "merge-base", base, "HEAD"])
    return proc.stdout.strip() or base


def _load_head(rel: str) -> dict | None:
    path = REPO_ROOT / rel
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _ref_is_commit(ref: str) -> bool:
    return (
        _run(
            ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]
        ).returncode
        == 0
    )


def _load_base(rel: str, ref: str) -> dict | None:
    """Budget content at `ref`, or None when the file did not exist there.

    `ref` is verified as a real commit by the caller, so a non-zero `git show` here means
    the path was absent at that commit, not that the ref itself is unresolvable.
    """
    proc = _run(["git", "show", f"{ref}:{rel}"])
    if proc.returncode != 0:
        return None
    return json.loads(proc.stdout)


def _baselines(budget: dict) -> dict[str, int]:
    """Map each rule to its recorded baseline; skip malformed specs."""
    return {
        rule: int(spec.get("baseline", 0))
        for rule, spec in budget.items()
        if isinstance(spec, dict)
    }


def _caps(budget: dict) -> dict[str, int]:
    """Map each rule to its ceiling (baseline + slack); skip malformed specs."""
    return {
        rule: int(spec.get("baseline", 0)) + int(spec.get("slack", 0))
        for rule, spec in budget.items()
        if isinstance(spec, dict)
    }


def _regression_detail(
    rule: str,
    base_caps: dict[str, int],
    head_caps: dict[str, int],
    base_baselines: dict[str, int],
    head_baselines: dict[str, int],
) -> str | None:
    """Why `rule` regressed vs base, or None when it held flat or fell.

    A dropped rule is terminal; otherwise a raised ceiling and a raised baseline are
    independent loosenings (the latter catches a baseline bump masked by a slack cut),
    so both reasons are reported when both apply.
    """
    base_cap = base_caps[rule]
    if rule not in head_caps:
        return f"rule dropped (ceiling {base_cap} -> removed)"
    reasons = tuple(
        message
        for raised, message in (
            (
                head_caps[rule] > base_cap,
                f"ceiling raised {base_cap} -> {head_caps[rule]}",
            ),
            (
                head_baselines[rule] > base_baselines[rule],
                f"baseline raised {base_baselines[rule]} -> {head_baselines[rule]}",
            ),
        )
        if raised
    )
    return "; ".join(reasons) or None


def regressions_for(rel: str, base: dict | None, head: dict | None) -> list[Regression]:
    if base is None:
        return []  # new budget file: nothing to ratchet against yet
    if head is None:
        return [Regression(rel, "*", "budget file was deleted (every ceiling removed)")]

    base_caps, head_caps = _caps(base), _caps(head)
    base_baselines, head_baselines = _baselines(base), _baselines(head)
    return [
        Regression(rel, rule, detail)
        for rule in sorted(base_caps)
        if (
            detail := _regression_detail(
                rule, base_caps, head_caps, base_baselines, head_baselines
            )
        )
        is not None
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("budgets", nargs="*", help="budget files to check")
    args = parser.parse_args()
    budgets = args.budgets or list(DEFAULT_BUDGETS)

    ref = _merge_base(args.base)
    if not _ref_is_commit(ref):
        print(
            f"FAIL: base ref {ref!r} does not resolve to a commit, so the ratchet has nothing "
            f"to compare against; refusing to pass vacuously (check the --base / BASE_SHA value)",
            file=sys.stderr,
        )
        return 1

    regressions: list[Regression] = []
    checked: list[str] = []
    for rel in budgets:
        base = _load_base(rel, ref)
        head = _load_head(rel)
        if base is None and head is None:
            continue
        if base is None:
            print(f"skip {rel}: new file (no base at {args.base} to ratchet against)")
            continue
        checked.append(rel)
        regressions.extend(regressions_for(rel, base, head))

    if regressions:
        print(
            f"FAIL: budget baseline(s)/ceiling(s) loosened vs base {args.base} (merge-base {ref[:12]}):"
        )
        for reg in regressions:
            print(f"  {reg.budget}  {reg.rule}: {reg.detail}")
        print(
            "Budgets are one-way ratchets and may only go down or stay flat. This "
            "check is non-gating: if the increase is justified (e.g. a newly banned "
            "API), a human can merge over the red after acknowledging it."
        )
        return 1

    suffix = f" ({', '.join(checked)})" if checked else ""
    print(f"OK: no budget ceiling increased vs base {args.base}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
