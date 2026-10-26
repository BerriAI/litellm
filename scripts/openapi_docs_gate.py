#!/usr/bin/env python3
"""Delta-vs-base gate for the OA* rules in scripts/check_openapi_docs.py.

Sibling of scripts/test_quality_gate.py, pointed at the proxy's OpenAPI document
instead of a source tree. Each rule is counted over every POST operation the proxy
serves at HEAD and at the merge-base with the branch this change merges into, and the
gate fails only when a rule grew past the merge-base count, so a new endpoint has to
ship with a request schema, a description and a 200 schema while the undocumented
routes that already exist are only ever allowed to shrink.

The merge-base counts come from scripts/lint_base_counts.py: the disk cache, then the
CI artifact published for that commit, then a pass of the current checker over a
detached worktree at the merge-base with that tree first on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import os
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
CHECKER: Final = REPO_ROOT / "scripts" / "check_openapi_docs.py"
TARGET: Final = "litellm"
TERMINATION_SIGNALS: Final = (signal.SIGTERM, signal.SIGHUP)

_LINE: Final = re.compile(r"^(?P<method>[A-Z]+) (?P<path>\S+) (?P<code>OA\d+)$")
_ADDED_LINE: Final = re.compile(r"^\+(?!\+\+)(?P<text>.*)$", re.MULTILINE)


class Violation(NamedTuple):
    method: str
    path: str
    code: str


def _run(cmd: Sequence[str], cwd: Path = REPO_ROOT, env: Mapping[str, str] | None = None) -> str:
    proc: Final = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def checker_identity(checker: Path = CHECKER) -> Checker:
    return Checker("openapi-docs", (sha256_of(checker),))


def parse_violations(out: str) -> tuple[Violation, ...]:
    return tuple(
        Violation(match.group("method"), match.group("path"), match.group("code"))
        for line in out.splitlines()
        if (match := _LINE.match(line)) is not None
    )


def _check(root: Path, checker: Path) -> tuple[Violation, ...]:
    resolved: Final = root.resolve()
    env: Final = {**os.environ, "PYTHONPATH": os.pathsep.join((str(resolved), os.environ.get("PYTHONPATH", "")))}
    return parse_violations(_run([sys.executable, str(checker)], cwd=resolved, env=env))


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
    """Rule counts at `ref`, measured with the *current* checker over the base tree."""
    _install_termination_handlers()
    parent: Final = Path(tempfile.mkdtemp(prefix="oa_base_"))
    worktree: Final = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref], cwd=repo_root)
        (worktree / "scripts").mkdir(parents=True, exist_ok=True)
        base_checker: Final = worktree / "scripts" / "check_openapi_docs.py"
        shutil.copy(checker, base_checker)
        return count_by_rule(_check(worktree, base_checker))
    finally:
        # Teardown must never raise, or it masks the real error when the body failed.
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=repo_root, capture_output=True, text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)


def added_text(diff_text: str) -> str:
    return "\n".join(match.group("text") for match in _ADDED_LINE.finditer(diff_text))


def introduced(violations: Sequence[Violation], added: str) -> tuple[Violation, ...]:
    """The violations whose route path is quoted on a line this change added. A new
    route is registered with its path as a string literal, so this names the endpoint
    the change brought in without needing the base document's own violation list."""
    return tuple(v for v in violations if f'"{v.path}"' in added or f"'{v.path}'" in added)


def cmd_check(base: str) -> None:
    head: Final = head_violations()
    base_point: Final = resolve_base_point(base)
    breaches: Final = evaluate(count_by_rule(head), base_counts_cached(checker_identity(), base_point, base_counts))
    if not breaches:
        print(f"OK: no OA rule grew past its merge-base count (base {base})")
        return
    diff: Final = _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])
    new: Final = introduced(head, added_text(diff))
    print(f"FAIL: OA-rule totals grew past their merge-base count (base {base}):")
    for breach in breaches:
        print(f"  {breach.rule}: total {breach.total} over ceiling {breach.ceiling} (this change added {breach.added})")
        for violation in sorted(v for v in new if v.code == breach.rule):
            print(f"    {violation.method} {violation.path}")
    print(
        "Give every new POST route a request body schema (a typed body parameter, or "
        "`openapi_extra={'requestBody': inline_request_body(Model, example)}` when the handler reads "
        "`request.body()`), a docstring, and a 200 schema (`response_model=` or `responses={200: {'model': ...}}`), "
        "or document an equal number of existing routes; the ceiling is the merge-base count. Run "
        "`python scripts/check_openapi_docs.py` to see every finding."
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
