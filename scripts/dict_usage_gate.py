#!/usr/bin/env python3
"""Total-count gate for the DICT001 rule in scripts/check_dict_usage.py.

Sibling of scripts/type_discipline_gate.py, with the same blame model:
dict-usage-budget.json holds a hard ``limit`` per rule, the checker counts
inferred-mutable-dict usage across the whole `litellm` tree, and the gate
fails only when a rule is both over its limit and higher than the base it
merges into, so a change is blamed for the violations it adds, never for
drift that already exists in the base. ``--update`` ratchets a limit down by
the violations this branch fixed relative to its branch point, with the same
seeded-rule passthrough as the LIT gate.

Because DICT001 comes from mypy type inference rather than a per-file AST
walk, the measurement environment matters the same way it does for
basedpyright: which third-party packages are importable changes what mypy can
infer. Every pass therefore runs from the gate-owned ``.venv-typecheck``
environment that scripts/type_check_gate.py provisions (same frozen sync,
same generated Prisma client), and base counts are cached in the shared
lint-cache directory under a ``dict-usage-base-`` prefix keyed by merge-base,
the basedpyright environment fingerprints, and the checker's own bytes, so a
rule-logic change re-keys the cache and stale counts are never matched. The
head checker is copied into the base worktree so the base is judged by
today's rule logic, and the head pyrightconfig.json is copied alongside it so
both passes target one Python version.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Final

SCRIPTS_DIR: Final = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import type_check_gate as typecheck_env
from type_discipline_gate import (
    Violation,
    evaluate,
    introduced,
    over_ceiling,
    parse_changed_lines,
    ratcheted_budget,
    resolve_base_point,
)

REPO_ROOT: Final = SCRIPTS_DIR.parent
CHECKER: Final = SCRIPTS_DIR / "check_dict_usage.py"
BUDGET_PATH: Final = REPO_ROOT / "dict-usage-budget.json"
TARGET: Final = "litellm"
DEFAULT_BASE: Final = "origin/litellm_internal_staging"
CACHE_PREFIX: Final = "dict-usage-base-"

_LINE: Final = re.compile(r"^(?P<file>.+?):(?P<line>\d+): (?P<code>DICT\d+) ")


def _run(cmd: list[str], cwd: Path = REPO_ROOT) -> str:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode not in (0, 1):
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"{cmd[0]} exited {proc.returncode}")
    return proc.stdout


def checker_python() -> Path:
    return typecheck_env.TYPECHECK_ENV_DIR / "bin" / "python"


def parse_violations(output: str, root: Path) -> list[Violation]:
    found: Final[list[Violation]] = []
    for line in output.splitlines():
        match = _LINE.match(line)
        if match is None:
            continue
        name = Path(match.group("file"))
        full = name if name.is_absolute() else root / name
        rel = full.resolve().relative_to(root).as_posix()
        found.append(Violation(rel, int(match.group("line")), match.group("code")))
    return found


def _check(root: Path, checker: Path) -> list[Violation]:
    resolved: Final = root.resolve()
    output: Final = _run([str(checker_python()), str(checker), str(resolved / TARGET)], cwd=resolved)
    return parse_violations(output, resolved)


def head_violations() -> list[Violation]:
    return _check(REPO_ROOT, CHECKER)


def count_by_rule(violations: list[Violation]) -> dict[str, int]:
    return dict(Counter(v.code for v in violations))


def base_counts(ref: str) -> dict[str, int]:
    parent = Path(tempfile.mkdtemp(prefix="dict_base_"))
    worktree = parent / "wt"
    try:
        _run(["git", "worktree", "add", "--detach", str(worktree), ref])
        (worktree / "scripts").mkdir(parents=True, exist_ok=True)
        checker = worktree / "scripts" / CHECKER.name
        shutil.copy(CHECKER, checker)
        shutil.copy(typecheck_env.PYRIGHT_CONFIG, worktree / "pyrightconfig.json")
        return count_by_rule(_check(worktree, checker))
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(parent, ignore_errors=True)


def cache_fingerprints(checker: Path = CHECKER) -> tuple[str, ...]:
    return (
        *typecheck_env.environment_fingerprints(),
        "checker:" + hashlib.sha256(checker.read_bytes()).hexdigest(),
    )


def base_counts_cached(base_point: str, cache_dir: Path | None = None) -> dict[str, int]:
    directory: Final = typecheck_env.default_cache_dir() if cache_dir is None else cache_dir
    path: Final = typecheck_env.cache_path(directory, base_point, cache_fingerprints(), CACHE_PREFIX)
    cached: Final = typecheck_env.load_cached_counts(path)
    if cached is not None:
        return cached
    counts: Final = base_counts(base_point)
    if counts:
        typecheck_env.store_counts(directory, path, base_point, counts, CACHE_PREFIX)
    return counts


def cmd_check(base: str) -> None:
    budget = json.loads(BUDGET_PATH.read_text())
    head = head_violations()
    head_counts = count_by_rule(head)
    if typecheck_env.is_vacuous_run(head_counts, budget):
        expected = sum(spec["limit"] for spec in budget.values())
        print(
            f"FAIL: the dict-usage checker reported nothing, but {BUDGET_PATH.name} "
            f"allows up to ~{expected}. The mypy pass almost certainly emitted "
            f"nothing; refusing to certify a vacuous run."
        )
        raise SystemExit(1)
    if not over_ceiling(head_counts, budget):
        print(f"OK: every DICT rule is within its codebase ceiling (base {base})")
        return
    base_point = resolve_base_point(base)
    base_totals = base_counts_cached(base_point)
    if typecheck_env.is_vacuous_run(base_totals, budget):
        print(
            f"FAIL: the dict-usage checker reported nothing for the base tree at "
            f"{base_point[:12]}, so every violation would look freshly added. The "
            f"base pass almost certainly crashed; refusing to blame this change."
        )
        raise SystemExit(1)
    breaches = evaluate(head_counts, base_totals, budget)
    if not breaches:
        print(f"OK: every DICT rule is within its codebase ceiling (base {base})")
        return
    new = introduced(
        head,
        parse_changed_lines(_run(["git", "diff", base_point, "--unified=0", "--no-color", "--", TARGET])),
    )
    print(f"FAIL: DICT-rule totals exceed their limit (base {base}):")
    for breach in breaches:
        print(f"  {breach.rule}: total {breach.total} over limit {breach.cap} (this change added {breach.added})")
        for violation in sorted(v for v in new if v.code == breach.rule):
            print(f"    {violation.file}:{violation.line}")
    print(
        "Route the value through a typed seam instead of a bare mutable dict "
        "(a pydantic model, a ReadOnly TypedDict, Mapping/MappingProxyType), or "
        "remove an equal number of violations elsewhere; the ceiling is the limit "
        "in dict-usage-budget.json."
    )
    raise SystemExit(1)


def _base_budget_rules(base_point: str) -> frozenset[str]:
    proc = subprocess.run(
        ["git", "show", f"{base_point}:{BUDGET_PATH.name}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return frozenset()
    return frozenset(json.loads(proc.stdout))


def cmd_update(base_ref: str = DEFAULT_BASE) -> None:
    """Ratchet each rule's limit down by the violations this branch fixed.

    The working-tree count is compared against a checker pass over a detached
    worktree at the branch point (the merge-base with `base_ref`), so a branch's
    fixes tighten its own ceilings by exactly what they cleared since it
    diverged. Rules seeded on this branch pass through untouched, exactly as in
    scripts/type_discipline_gate.py.
    """
    budget = json.loads(BUDGET_PATH.read_text())
    base_point = resolve_base_point(base_ref)
    seeded = frozenset(budget) - _base_budget_rules(base_point)
    updated = ratcheted_budget(budget, count_by_rule(head_violations()), base_counts_cached(base_point), seeded)
    BUDGET_PATH.write_text(json.dumps(updated, indent=2, sort_keys=True) + "\n")
    cleared = sum(budget[rule]["limit"] - updated[rule]["limit"] for rule in updated)
    print(f"Ratcheted DICT-rule limits down by {cleared} violations this branch fixed")
    if seeded:
        print("Left untouched (seeded on this branch, absent from the base budget): " + ", ".join(sorted(seeded)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--update", action="store_true")
    args = parser.parse_args()
    typecheck_env.ensure_typecheck_env()
    cmd_update(args.base) if args.update else cmd_check(args.base)


if __name__ == "__main__":
    main()
