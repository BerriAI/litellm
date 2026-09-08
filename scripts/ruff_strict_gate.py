#!/usr/bin/env python3
"""Delta-vs-base gate for the strict ruff rules in ruff-strict.toml.

Each rule is counted across the whole tree at HEAD and at the merge-base with
the branch this change merges into, and the gate fails only when a rule grew
past the merge-base count plus its headroom in HEADROOM (none today), so a
change is blamed for the violations it adds, never for drift that already sits
in the base. There is no committed budget: the merge-base count is the
ceiling, so it moves only when the base branch does.

The merge-base counts come from scripts/lint_base_counts.py: the disk cache,
then the CI artifact published for that commit, then a ruff pass over a
detached worktree at the merge-base under the current ruff configs.
``--emit-counts-dir`` writes HEAD's counts as the file that artifact is built
from.
"""

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
STRICT_CONFIG: Final = REPO_ROOT / "ruff-strict.toml"
BASE_CONFIG: Final = REPO_ROOT / "ruff.toml"
TARGET: Final = "litellm"
HEADROOM: Final[Mapping[str, int]] = MappingProxyType({})

_HUNK: Final = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


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


def ruff_version() -> str:
    return _run(["ruff", "--version"]).strip()


def checker_identity() -> Checker:
    return Checker("ruff-strict", (sha256_of(STRICT_CONFIG), sha256_of(BASE_CONFIG), ruff_version()))


def _ruff_json(cwd: Path, config: Path) -> list:
    raw = _run(
        ["ruff", "check", TARGET, "--config", str(config), "--output-format", "json"],
        cwd=cwd,
    )
    return json.loads(raw or "[]")


def head_violations() -> list[Violation]:
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


def count_by_rule(violations: Sequence[Violation]) -> dict[str, int]:
    return dict(Counter(v.code for v in violations))


def base_counts(ref: str) -> dict[str, int]:
    parent: Final = Path(tempfile.mkdtemp(prefix="ruff_base_"))
    worktree: Final = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref])
        shutil.copy(BASE_CONFIG, worktree / BASE_CONFIG.name)
        shutil.copy(STRICT_CONFIG, worktree / STRICT_CONFIG.name)
        items: Final = _ruff_json(worktree, worktree / STRICT_CONFIG.name)
        return dict(Counter(item["code"] for item in items))
    finally:
        _run(["git", "worktree", "remove", "--force", str(worktree)])
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
        print(f"OK: no strict rule grew past its merge-base count (base {base})")
        return
    diff: Final = _run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])
    new: Final = introduced(head, parse_changed_lines(diff))
    print(f"FAIL: strict-rule totals grew past their merge-base count (base {base}):")
    for breach in breaches:
        print(f"  {breach.rule}: total {breach.total} over ceiling {breach.cap} (this change added {breach.added})")
        for violation in sorted(v for v in new if v.code == breach.rule):
            print(f"    {violation.file}:{violation.line}")
    print(
        "Reduce the new violations or remove an equal number elsewhere; the ceiling is the "
        "merge-base count plus the rule's headroom in scripts/ruff_strict_gate.py."
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
