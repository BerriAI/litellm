#!/usr/bin/env python3
"""Total-count gate for the TQ* rules in scripts/check_test_quality.py.

Sibling of scripts/type_discipline_gate.py, pointed at the test tree instead of
the package. Each rule listed in test-quality-budget.json has a hard ``limit``.
The gate counts each rule across the whole `tests` tree and fails when a rule is
both over its limit and higher than the base it merges into, so a change is
blamed for the violations it adds, never for drift that already exists in the
base.

Every rule is seeded at exactly its count on the day the gate landed, so the
suite's existing debt is grandfathered and any net-new violation trips the gate
immediately. ``--update`` ratchets a limit down by the violations this branch
fixed relative to its branch point (the merge-base), so the ceilings only ever
fall. Base counts are measured with the *current* checker, so a rule introduced
on this branch is counted at the base too and ratchets like every other one.

Only ever falling is not the same as always falling, so the gate enforces the
second half: a branch that clears violations and leaves the ceiling above its
new count fails, naming the rules and telling the author to run
``make lint-budget-update``. Without that, a removed violation could come back
later under a ceiling nobody lowered. Drift already in the base is never
blamed, so this fires only on the branch that did the clearing.

The deliberate difference from its sibling: this gate has no headroom anywhere.
Type discipline seeded LIT010/LIT011 at 1.5x to leave room for an in-flight
sweep; a test-quality violation has no such transition to absorb, so the line is
today's count and the only legal direction is down.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
CHECKER: Final = REPO_ROOT / "scripts" / "check_test_quality.py"
BUDGET_PATH: Final = REPO_ROOT / "test-quality-budget.json"
TARGET: Final = "tests"
DEFAULT_BASE: Final = "origin/litellm_internal_staging"

_HUNK: Final = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_FILE_HEADER: Final = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_LINE: Final = re.compile(r"^(?P<file>.+?):(?P<line>\d+): (?P<code>TQ\d+) ")


class Violation(NamedTuple):
    file: str
    line: int
    code: str


class Breach(NamedTuple):
    rule: str
    total: int
    cap: int
    added: int


def _run(cmd: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    proc: Final = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def resolve_base_point(base_ref: str, cwd: Path = REPO_ROOT) -> str:
    """The snapshot commit base counts are measured at: merge-base(base_ref, HEAD),
    made aware of an in-progress merge. Mid-merge, HEAD is still the pre-merge tip,
    so its merge-base is the old branch point and every violation the base gained
    since then would be blamed on this change."""
    head_point: Final = _run(["git", "merge-base", base_ref, "HEAD"], cwd=cwd).strip()
    if not head_point:
        return base_ref
    merge_head: Final = _run(["git", "rev-parse", "--verify", "--quiet", "MERGE_HEAD"], cwd=cwd).strip()
    if not merge_head:
        return head_point
    merge_point: Final = _run(["git", "merge-base", base_ref, merge_head], cwd=cwd).strip()
    if not merge_point:
        return head_point
    older: Final = _run(["git", "merge-base", head_point, merge_point], cwd=cwd).strip()
    return merge_point if older == head_point else head_point


def _check(root: Path, checker: Path) -> tuple[Violation, ...]:
    # macOS tempfile dirs (/var/...) resolve to /private/var/..., so relative_to needs both sides resolved.
    resolved: Final = root.resolve()
    out: Final = _run([sys.executable, str(checker), str(resolved / TARGET)], cwd=resolved)
    return tuple(
        Violation(
            (resolved / match.group("file")).resolve().relative_to(resolved).as_posix(),
            int(match.group("line")),
            match.group("code"),
        )
        for line in out.splitlines()
        if (match := _LINE.match(line)) is not None
    )


def head_violations() -> tuple[Violation, ...]:
    return _check(REPO_ROOT, CHECKER)


def count_by_rule(violations: Sequence[Violation]) -> Mapping[str, int]:
    return MappingProxyType(dict(Counter(v.code for v in violations)))


def base_counts(ref: str) -> Mapping[str, int]:
    """Rule counts at `ref`, measured with the *current* rule logic rather than
    whatever the checker looked like at that commit."""
    parent: Final = Path(tempfile.mkdtemp(prefix="tq_base_"))
    worktree: Final = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref])
        (worktree / "scripts").mkdir(parents=True, exist_ok=True)
        checker: Final = worktree / "scripts" / "check_test_quality.py"
        shutil.copy(CHECKER, checker)
        return count_by_rule(_check(worktree, checker))
    finally:
        # Teardown must never raise, or it masks the real error when the body failed.
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)


def over_ceiling(head: Mapping[str, int], budget: Mapping[str, Mapping[str, int]]) -> frozenset[str]:
    """Rules whose head count already exceeds their limit. When none are, the base
    comparison cannot change the verdict and the base worktree scan is skipped."""
    return frozenset(
        rule for rule, spec in budget.items() if head.get(rule, 0) > spec["limit"]
    )


def unratcheted(
    head: Mapping[str, int],
    base: Mapping[str, int],
    budget: Mapping[str, Mapping[str, int]],
) -> tuple[Breach, ...]:
    """Rules this branch cleared without lowering the ceiling behind them. Requires
    both `head < base`, so drift already in the base is never blamed on this change,
    and `head < limit`, so a ceiling already at the count is left alone."""
    return tuple(sorted(
        Breach(rule, head.get(rule, 0), spec["limit"], head.get(rule, 0) - base.get(rule, 0))
        for rule, spec in budget.items()
        if head.get(rule, 0) < base.get(rule, 0) and head.get(rule, 0) < spec["limit"]
    ))


def evaluate(
    head: Mapping[str, int],
    base: Mapping[str, int],
    budget: Mapping[str, Mapping[str, int]],
) -> tuple[Breach, ...]:
    return tuple(sorted(
        Breach(rule, head.get(rule, 0), spec["limit"], head.get(rule, 0) - base.get(rule, 0))
        for rule, spec in budget.items()
        if head.get(rule, 0) > spec["limit"] and head.get(rule, 0) > base.get(rule, 0)
    ))


def _hunk_lines(body: str) -> frozenset[int]:
    return frozenset(
        line
        for match in _HUNK.finditer(body)
        for start in (int(match.group(1)),)
        for line in range(start, start + (int(match.group(2)) if match.group(2) is not None else 1))
    )


def parse_changed_lines(diff_text: str) -> Mapping[str, frozenset[int]]:
    """Each file in the diff mapped to the line numbers it adds. Splitting on the
    `+++ b/` headers keeps this a pure expression: `split` hands back
    [preamble, path, body, path, body, ...], so each file's hunks are already
    grouped with it."""
    parts: Final = _FILE_HEADER.split(diff_text)
    return MappingProxyType({
        path: _hunk_lines(body)
        for path, body in zip(parts[1::2], parts[2::2])
    })


def introduced(
    violations: Sequence[Violation], changed: Mapping[str, frozenset[int]]
) -> tuple[Violation, ...]:
    return tuple(v for v in violations if v.line in changed.get(v.file, frozenset()))


def touches_measured_tree(base_point: str) -> bool:
    """Whether this branch changed anything that can move a count. A branch that
    touches neither the test tree nor the checker cannot have cleared a violation,
    so the base scan is skipped and the gate stays cheap on the common change."""
    changed: Final = _run(
        ["git", "diff", "--name-only", base_point, "--", TARGET, str(CHECKER.relative_to(REPO_ROOT))]
    )
    return bool(changed.strip())


def cmd_check(base: str) -> None:
    budget: Final = json.loads(BUDGET_PATH.read_text())
    head: Final = head_violations()
    head_counts: Final = count_by_rule(head)
    base_point: Final = resolve_base_point(base)
    if not over_ceiling(head_counts, budget) and not touches_measured_tree(base_point):
        print(f"OK: every TQ rule is within its test-suite ceiling (base {base})")
        return
    base_at_point: Final = base_counts(base_point)
    stale: Final = unratcheted(head_counts, base_at_point, budget)
    if stale:
        print(f"FAIL: TQ-rule limits were left above the count this branch reached (base {base}):")
        for breach in stale:
            print(
                f"  {breach.rule}: this branch cleared {-breach.added} down to {breach.total}, "
                f"but the limit is still {breach.cap}"
            )
        print(
            "Run `make lint-budget-update` and commit the lowered limits, so the "
            "violations you cleared cannot come back under a ceiling nobody moved."
        )
        raise SystemExit(1)
    breaches: Final = evaluate(head_counts, base_at_point, budget)
    if not breaches:
        print(f"OK: every TQ rule is within its test-suite ceiling (base {base})")
        return
    new: Final = introduced(
        head,
        parse_changed_lines(
            _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])
        ),
    )
    print(f"FAIL: TQ-rule totals exceed their limit (base {base}):")
    for breach in breaches:
        print(
            f"  {breach.rule}: total {breach.total} over limit {breach.cap} "
            f"(this change added {breach.added})"
        )
        for violation in sorted(v for v in new if v.code == breach.rule):
            print(f"    {violation.file}:{violation.line}")
    print(
        "Fix the new violations, or give each one a reason "
        "(`# test-quality-ok: <reason>`), or remove an equal number elsewhere; "
        "the ceiling is the limit in test-quality-budget.json. "
        "Run `python scripts/check_test_quality.py tests/` to see every finding."
    )
    raise SystemExit(1)


def ratcheted_budget(
    budget: Mapping[str, Mapping[str, int]],
    current: Mapping[str, int],
    base: Mapping[str, int],
) -> Mapping[str, Mapping[str, int]]:
    """Each rule's limit lowered by the violations `current` fixed vs `base`. The drop
    is clamped to what was actually cleared, so a limit only ever falls."""
    return MappingProxyType({
        rule: {"limit": max(0, spec["limit"] - max(0, base.get(rule, 0) - current.get(rule, 0)))}
        for rule, spec in sorted(budget.items())
    })


def cmd_update(base_ref: str = DEFAULT_BASE) -> None:
    """Ratchet each rule's limit down by the violations this branch fixed."""
    budget: Final = json.loads(BUDGET_PATH.read_text())
    base_point: Final = resolve_base_point(base_ref)
    updated: Final = ratcheted_budget(
        budget, count_by_rule(head_violations()), base_counts(base_point)
    )
    BUDGET_PATH.write_text(json.dumps(dict(updated), indent=2, sort_keys=True) + "\n")
    cleared: Final = sum(budget[rule]["limit"] - updated[rule]["limit"] for rule in updated)
    print(f"Ratcheted TQ-rule limits down by {cleared} violations this branch fixed")


def cmd_seed() -> None:
    """Write the budget from the working tree's current counts. Used once, to land
    the gate; afterwards `--update` is the only thing that may move a limit."""
    counts: Final = count_by_rule(head_violations())
    BUDGET_PATH.write_text(
        json.dumps({rule: {"limit": counts[rule]} for rule in sorted(counts)}, indent=2) + "\n"
    )
    print(f"Seeded {BUDGET_PATH.name} at " + ", ".join(f"{r}={counts[r]}" for r in sorted(counts)))


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--seed", action="store_true")
    args: Final = parser.parse_args()
    from gate_slot_lock import held_slot

    with held_slot():
        if args.seed:
            cmd_seed()
        elif args.update:
            cmd_update(args.base)
        else:
            cmd_check(args.base)


if __name__ == "__main__":
    main()
