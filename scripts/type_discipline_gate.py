#!/usr/bin/env python3
"""Delta-vs-base gate for the LIT* rules in scripts/check_type_discipline.py.

Sibling of scripts/ruff_strict_gate.py. Each rule is counted across the whole
`litellm` tree at HEAD and at the merge-base with the branch this change merges
into, and the gate fails only when a rule grew past the merge-base count plus
its headroom in HEADROOM, so a change is blamed for the violations it adds,
never for drift that already exists in the base. There is no committed budget:
the merge-base count is the ceiling, so it moves only when the base branch does.

Every rule the checker emits is gated: LIT001 (mutable collection in any
annotation), LIT002 (mutable-collection construction), LIT003/LIT004 (noqa /
pyright-mypy ignore without codes or reason), LIT005 (`*-ok` suppression
without a reason), LIT006 (cast), LIT007 (TypeGuard/TypeIs), LIT008
(`**kwargs`), LIT009 (inert `# type: ignore`, dead syntax while
enableTypeIgnoreComments is false), LIT010 (assignment without a Final
declaration; suppress deliberate rebinding with `# rebind-ok: <reason>`),
LIT011 (parameter rebinding or in-place mutation), and LIT012 (TypedDict field
without a `ReadOnly[...]` qualifier; suppress with `# writable-ok: <reason>`).
Every rule has zero headroom except LIT010 and LIT011, which keep the pool
their seeding left them: a change may add that many before the gate trips.

The merge-base counts come from scripts/lint_base_counts.py: the disk cache,
then the CI artifact published for that commit, then a pass of the current
checker over a detached worktree at the merge-base, so a rule change on this
branch is measured on both sides. ``--emit-counts-dir`` writes HEAD's counts
as the file that artifact is built from.
"""

import argparse
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

from lint_base_counts import (
    Checker,
    base_counts_cached,
    emit_counts,
    evaluate,
    head_sha,
    resolve_base_point,
    sha256_of,
)

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
CHECKER: Final = REPO_ROOT / "scripts" / "check_type_discipline.py"
TARGET: Final = "litellm"
HEADROOM: Final[Mapping[str, int]] = MappingProxyType({"LIT010": 44, "LIT011": 5})

_HUNK: Final = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_LINE: Final = re.compile(r"^(?P<file>.+?):(?P<line>\d+): (?P<code>LIT\d+) ")


class Violation(NamedTuple):
    file: str
    line: int
    code: str


def _run(cmd: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    proc: Final = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def checker_identity() -> Checker:
    return Checker("type-discipline", (sha256_of(CHECKER),))


def _check(root: Path, checker: Path) -> list[Violation]:
    # Resolve root first: on macOS tempfile dirs (/var/...) resolve to /private/var/...,
    # and the checker prints already-resolved absolute paths, so relative_to would fail.
    root = root.resolve()
    out = _run([sys.executable, str(checker), str(root / TARGET)], cwd=root)
    found = []
    for line in out.splitlines():
        m = _LINE.match(line)
        if m is None:
            continue
        name = Path(m.group("file"))
        full = name if name.is_absolute() else root / name
        rel = full.resolve().relative_to(root).as_posix()
        found.append(Violation(rel, int(m.group("line")), m.group("code")))
    return found


def head_violations() -> list[Violation]:
    return _check(REPO_ROOT, CHECKER)


def count_by_rule(violations: Sequence[Violation]) -> dict[str, int]:
    return dict(Counter(v.code for v in violations))


def base_counts(ref: str) -> dict[str, int]:
    parent: Final = Path(tempfile.mkdtemp(prefix="lit_base_"))
    worktree: Final = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref])
        # Measure the base with the *current* rule logic, not whatever shipped at base.
        (worktree / "scripts").mkdir(parents=True, exist_ok=True)
        checker: Final = worktree / "scripts" / "check_type_discipline.py"
        shutil.copy(CHECKER, checker)
        return count_by_rule(_check(worktree, checker))
    finally:
        # Best-effort teardown: cleanup must never raise, or it masks the real error when
        # the body (or the `worktree add` itself) failed. rmtree is already best-effort.
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)


def parse_changed_lines(diff_text: str) -> dict[str, set[int]]:
    changed: dict[str, set[int]] = {}
    path = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif path and (match := _HUNK.match(line)):
            start = int(match.group(1))
            count = int(match.group(2)) if match.group(2) is not None else 1
            changed.setdefault(path, set()).update(range(start, start + count))
    return changed


def introduced(violations: Sequence[Violation], changed: Mapping[str, set[int]]) -> list[Violation]:
    return [v for v in violations if v.line in changed.get(v.file, set())]


def cmd_check(base: str) -> None:
    head: Final = head_violations()
    base_point: Final = resolve_base_point(base)
    breaches: Final = evaluate(
        count_by_rule(head), base_counts_cached(checker_identity(), base_point, base_counts), HEADROOM
    )
    if not breaches:
        print(f"OK: no LIT rule grew past its merge-base count (base {base})")
        return
    diff: Final = _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])
    new: Final = introduced(head, parse_changed_lines(diff))
    print(f"FAIL: LIT-rule totals grew past their merge-base count (base {base}):")
    for breach in breaches:
        print(f"  {breach.rule}: total {breach.total} over ceiling {breach.cap} (this change added {breach.added})")
        for violation in sorted(v for v in new if v.code == breach.rule):
            print(f"    {violation.file}:{violation.line}")
    print(
        "Remove the new violations, give each a reason (`# noqa: XXX  # <reason>`, "
        "`# pyright: ignore[rule]  # <reason>`, `# mutable-ok: <reason>`, `# cast-ok: <reason>`, "
        "`# guard-ok: <reason>`, `# kwargs-ok: <reason>`, `# rebind-ok: <reason>`, "
        "`# writable-ok: <reason>`), or remove an equal number elsewhere; the ceiling is the "
        "merge-base count plus the rule's headroom in scripts/type_discipline_gate.py."
    )
    raise SystemExit(1)


def main() -> None:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="Comparison ref (default: origin's current default branch)")
    parser.add_argument(
        "--emit-counts-dir",
        type=Path,
        help="Write HEAD's per-rule counts to this directory as a base-counts artifact instead of gating",
    )
    args: Final = parser.parse_args()
    from default_branch import resolve_base_ref
    from gate_slot_lock import held_slot

    if args.emit_counts_dir is not None:
        with held_slot():
            emit_counts(checker_identity(), count_by_rule(head_violations()), args.emit_counts_dir, head_sha())
        return
    base_ref: Final = resolve_base_ref(args.base, REPO_ROOT)
    with held_slot():
        cmd_check(base_ref)


if __name__ == "__main__":
    main()
