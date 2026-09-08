#!/usr/bin/env python3
"""Delta-vs-base gate for the TQ* rules in scripts/check_test_quality.py.

Sibling of scripts/type_discipline_gate.py, pointed at the test tree instead of
the package. Each rule is counted across the whole `tests` tree at HEAD and at
the merge-base with the branch this change merges into, and the gate fails only
when a rule grew past the merge-base count, so a change is blamed for the
violations it adds, never for drift that already exists in the base. There is
no committed budget: the merge-base count is the ceiling, so it moves only when
the base branch does.

The deliberate difference from its sibling: this gate has no headroom anywhere.
Type discipline keeps the LIT010/LIT011 pool its seeding left; a test-quality
violation has no such transition to absorb, so the line is the merge-base count
and the only legal direction is down.

The merge-base counts come from scripts/lint_base_counts.py: the disk cache,
then the CI artifact published for that commit, then a pass of the current
checker over a detached worktree at the merge-base, so a rule introduced on
this branch is counted at the base too. ``--emit-counts-dir`` writes HEAD's
counts as the file that artifact is built from.
"""

from __future__ import annotations

import argparse
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import FrameType, MappingProxyType
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
CHECKER: Final = REPO_ROOT / "scripts" / "check_test_quality.py"
TARGET: Final = "tests"
HEADROOM: Final[Mapping[str, int]] = MappingProxyType({})
TERMINATION_SIGNALS: Final = (signal.SIGTERM, signal.SIGHUP)

_HUNK: Final = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_FILE_HEADER: Final = re.compile(r"^\+\+\+ b/(.+)$", re.MULTILINE)
_LINE: Final = re.compile(r"^(?P<file>.+?):(?P<line>\d+): (?P<code>TQ\d+) ")


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
    return Checker("test-quality", (sha256_of(CHECKER),))


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


def _exit_on_termination(signum: int, _frame: FrameType | None) -> None:
    raise SystemExit(128 + signum)


def _install_termination_handlers() -> None:
    for termination in TERMINATION_SIGNALS:
        if signal.getsignal(termination) == signal.SIG_DFL:
            signal.signal(termination, _exit_on_termination)


def base_counts(ref: str, repo_root: Path = REPO_ROOT, checker: Path = CHECKER) -> Mapping[str, int]:
    """Rule counts at `ref`, measured with the *current* rule logic rather than
    whatever the checker looked like at that commit."""
    _install_termination_handlers()
    parent: Final = Path(tempfile.mkdtemp(prefix="tq_base_"))
    worktree: Final = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref], cwd=repo_root)
        (worktree / "scripts").mkdir(parents=True, exist_ok=True)
        base_checker: Final = worktree / "scripts" / "check_test_quality.py"
        shutil.copy(checker, base_checker)
        return count_by_rule(_check(worktree, base_checker))
    finally:
        # Teardown must never raise, or it masks the real error when the body failed.
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=repo_root, capture_output=True, text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)


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


def cmd_check(base: str) -> None:
    head: Final = head_violations()
    base_point: Final = resolve_base_point(base)
    breaches: Final = evaluate(
        count_by_rule(head), base_counts_cached(checker_identity(), base_point, base_counts), HEADROOM
    )
    if not breaches:
        print(f"OK: no TQ rule grew past its merge-base count (base {base})")
        return
    diff: Final = _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])
    new: Final = introduced(head, parse_changed_lines(diff))
    print(f"FAIL: TQ-rule totals grew past their merge-base count (base {base}):")
    for breach in breaches:
        print(f"  {breach.rule}: total {breach.total} over ceiling {breach.cap} (this change added {breach.added})")
        for violation in sorted(v for v in new if v.code == breach.rule):
            print(f"    {violation.file}:{violation.line}")
    print(
        "Fix the new violations, or give each one a reason (`# test-quality-ok: <reason>`), or remove an "
        "equal number elsewhere; the ceiling is the merge-base count. Run "
        "`python scripts/check_test_quality.py tests/` to see every finding."
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
