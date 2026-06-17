#!/usr/bin/env python3
"""Any-discipline gate: per-file budget on values typed `Any`.

Where ruff, `mypy --strict`, and even basedpyright's `reportAny` stop short, this
catches the case that actually bites: a *union* hiding an `Any`. For example
`re.Match.group()` -> `str | Any`, `json.loads()` -> `Any`, and bare `list`/`dict`
-> `list[Any]`/`dict[..., Any]`. Any value whose inferred type *contains* `Any`
(recursively, through unions / generics / tuples) is reported.

Scope: per-file grandfathering
------------------------------
litellm already contains a large amount of pre-existing `Any` (a single legacy
file can have >100 findings), and a whole-tree scan would have to re-export types
for litellm's entire import closure on every run (~2 min, ~3 GB). So the gate
grandfathers each file at its current count in `any-discipline-budget.json` and
gives it ~25% headroom: a file fails once its `Any`-tainted values exceed
`baseline + ceil(baseline * 0.25)`. A file with no entry is budgeted at zero, so
a brand new file must be `Any`-free, and a legacy file may absorb a little drift
before it has to be cleaned. Cold files the change doesn't touch are left to the
ratchet (run `--update`); like the mypy/basedpyright budgets, the gate only
re-counts the files a change actually edits, so CI stays fast.

How it works
------------
It loads `litellm/mypy.ini` (the same config `make lint-mypy` uses, so findings
match what developers already see), builds the target files with mypy asking for
its exported expression->type map, and walks each file's AST applying a recursive
"contains Any" predicate -- the test `mypy --disallow-any-expr` uses internally
but applies inconsistently (python/mypy#12856).

mypy only re-exports types for modules it re-type-checks, so for each target we
invalidate just its cached hash (deps stay warm) to force a fast re-check against
a persisted incremental cache (.mypy_cache_any).

Rules
-----
Codes share the `LIT***` namespace with `scripts/check_type_discipline.py` (PR
#30500), which owns LIT001/002/003/004/006/007/008. This gate claims the rest:
LIT009  A value expression's inferred type is, or contains, `Any`.
        Suppress with `# any-ok: <reason>` on the offending line.
LIT005  An `# any-ok` suppression without a reason (the shared
        suppression-needs-a-reason code, same as `# cast-ok` / `# guard-ok`).
LIT000  Setup failure: mypy could not build, or a target file could not be read.

`Any`s produced purely by an already-reported error, and the special-form /
implementation-artifact internal `Any`s, are ignored. A bound method *reference*
whose signature mentions `Any` is not flagged -- only the value its call produces.

Usage
-----
    # gate mode (CI / pre-push): budget the changed files under litellm/
    uv run --no-sync python scripts/check_any_discipline.py --changed --base origin/litellm_internal_staging

    # ratchet: re-capture every file's baseline from the current tree
    uv run --no-sync python scripts/check_any_discipline.py --update

    # whole-file spot-check (no budget), paths relative to repo root
    uv run --no-sync python scripts/check_any_discipline.py litellm/budget_manager.py

Exit code 1 if a file is over budget (or has a malformed any-ok), 2 on a
setup/usage error.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tokenize
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

try:
    from mypy import build
    from mypy.config_parser import parse_config_file
    from mypy.find_sources import create_source_list
    from mypy.fscache import FileSystemCache
    from mypy.modulefinder import BuildSource
    from mypy.nodes import AssignmentStmt, Expression, NameExpr, Node
    from mypy.options import Options
    from mypy.types import (
        AnyType,
        CallableType,
        Instance,
        Overloaded,
        TupleType,
        Type,
        TypeOfAny,
        UnionType,
        get_proper_type,
    )
except ImportError:  # pragma: no cover - environment guard
    sys.stderr.write(
        "check_any_discipline: mypy is not importable in this interpreter.\n"
        "Run it through the project environment, e.g.\n"
        "    uv run --no-sync python scripts/check_any_discipline.py --changed\n"
    )
    raise SystemExit(2)


REPO_ROOT = Path(__file__).resolve().parent.parent
LITELLM_DIR = REPO_ROOT / "litellm"
MYPY_INI = LITELLM_DIR / "mypy.ini"
CACHE_DIR = REPO_ROOT / ".mypy_cache_any"
BUDGET_PATH = REPO_ROOT / "any-discipline-budget.json"
PY_TAG = f"{sys.version_info.major}.{sys.version_info.minor}"
DEFAULT_BASE = "origin/litellm_internal_staging"

# Headroom each grandfathered file gets over its baseline before the gate trips.
HEADROOM_RATIO = 0.25

# `--update` re-counts every file. Type-checking each module also preserves its
# AST and exports its types, so we re-check in bounded batches against a warmed
# cache rather than holding all ~2k modules in memory at once.
UPDATE_BATCH = 150

MIN_REASON_LEN = 3
ANY_OK_RE = re.compile(r"#\s*any-ok(?::\s*(?P<reason>.*))?")

# Files allowed to surface `Any` (the typed/untyped boundary). A finding is
# skipped if any fragment below is a substring of the file's posix path. Keep
# this tight -- prefer a line-level `# any-ok: <reason>` over a blanket exemption.
BOUNDARY_PATHS: frozenset[str] = frozenset()

# `Any` kinds that are not actionable: produced by an already-reported error, or
# an internal placeholder that never corresponds to a concrete runtime value.
# NOTE: `special_form` is deliberately NOT here. In mypy 1.19 the `Any` in
# typeshed unions like `re.Match.group() -> str | Any` is tagged `special_form`,
# and that union is the headline case this gate exists to catch.
_HARMLESS_ANY = frozenset(
    kind
    for kind in (
        TypeOfAny.from_error,
        getattr(TypeOfAny, "implementation_artifact", None),
    )
    if kind is not None
)

# AST attributes that point OUTSIDE the syntactic subtree (a RefExpr's resolved
# definition, a node's TypeInfo). Skipping exactly these two makes a generic
# child-walk equivalent to mypy's TraverserVisitor -- validated to the node
# against ExtendedTraverserVisitor across the full grammar (see commit notes).
_NON_SYNTACTIC_ATTRS = frozenset({"node", "info"})


class Violation(NamedTuple):
    path: Path
    line: int
    col: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message}"


def cap_for(baseline: int) -> int:
    """The most `Any`-tainted values a file may hold: baseline plus ~25%."""
    return baseline + math.ceil(baseline * HEADROOM_RATIO)


# --------------------------------------------------------------------------- #
# The "contains Any" predicate
# --------------------------------------------------------------------------- #


def contains_any(t: Type, _seen: set[int] | None = None) -> bool:
    """True if a *value* of type ``t`` carries `Any` anywhere meaningful."""
    seen = _seen if _seen is not None else set()
    p = get_proper_type(t)
    if id(p) in seen:
        return False
    seen.add(id(p))

    # A function/method *reference* whose signature mentions Any is not itself an
    # unsafe value -- only its eventual call result is. Don't recurse into it.
    if isinstance(p, (CallableType, Overloaded)):
        return False
    if isinstance(p, AnyType):
        return p.type_of_any not in _HARMLESS_ANY
    if isinstance(p, UnionType):
        return any(contains_any(item, seen) for item in p.items)
    if isinstance(p, Instance):
        return any(contains_any(arg, seen) for arg in p.args)
    if isinstance(p, TupleType):
        return any(contains_any(item, seen) for item in p.items)
    return False


# --------------------------------------------------------------------------- #
# Generic, leak-free AST walk (works under a mypyc-compiled mypy, which forbids
# subclassing TraverserVisitor)
# --------------------------------------------------------------------------- #


def _walk_file(tree: Node) -> tuple[list[Expression], set[int]]:
    """Return (every Expression in `tree`, ids of simple assignment-target names).

    The walk follows only syntactic children (every attribute except the two
    non-syntactic back-references), so it never escapes the module. Simple
    ``x = <expr>`` name targets are collected separately so we don't double-report
    the assigned name as an echo of an Any rvalue.
    """
    exprs: list[Expression] = []
    skip_lvalues: set[int] = set()
    stack: list[object] = [tree]
    seen: set[int] = set()
    while stack:
        n = stack.pop()
        if isinstance(n, Node):
            if id(n) in seen:
                continue
            seen.add(id(n))
            if isinstance(n, Expression):
                exprs.append(n)
            if isinstance(n, AssignmentStmt):
                for lvalue in n.lvalues:
                    if isinstance(lvalue, NameExpr):
                        skip_lvalues.add(id(lvalue))
            for name in dir(n):
                if name.startswith("__") or name in _NON_SYNTACTIC_ATTRS:
                    continue
                try:
                    val = getattr(n, name)
                except Exception:
                    continue
                if callable(val):
                    continue
                if isinstance(val, (Node, list, tuple)):
                    stack.append(val)
        elif isinstance(n, (list, tuple)):
            stack.extend(n)
    return exprs, skip_lvalues


def find_any_in_tree(tree: Node, idmap: dict[int, Type]) -> list[tuple[int, int, str]]:
    exprs, skip_lvalues = _walk_file(tree)
    findings: list[tuple[int, int, str]] = []
    for expr in exprs:
        if id(expr) in skip_lvalues:
            continue
        t = idmap.get(id(expr))
        if t is not None and contains_any(t):
            findings.append((expr.line, expr.column, str(get_proper_type(t))))

    out: list[tuple[int, int, str]] = []
    seen_pos: set[tuple[int, int]] = set()
    for line, col, typ in sorted(findings):
        if line < 1 or (line, col) in seen_pos:
            continue
        seen_pos.add((line, col))
        out.append((line, col, typ))
    return out


# --------------------------------------------------------------------------- #
# Comment scanning (LIT005 + any-ok suppression)
# --------------------------------------------------------------------------- #


def _reason_ok(reason: str | None) -> bool:
    return reason is not None and len(reason.strip()) >= MIN_REASON_LEN


def scan_any_ok(
    path: Path, source: str
) -> tuple[frozenset[int], tuple[Violation, ...]]:
    """Return (lines with a valid any-ok suppression, LIT005 violations)."""
    try:
        tokens = tokenize.generate_tokens(
            iter(source.splitlines(keepends=True)).__next__
        )
        comments = tuple(
            (t.start[0], t.string) for t in tokens if t.type == tokenize.COMMENT
        )
    except tokenize.TokenError:
        return frozenset(), ()

    ok_lines: set[int] = set()
    violations: list[Violation] = []
    for line, text in comments:
        m = ANY_OK_RE.search(text)
        if m is None:
            continue
        if _reason_ok(m.group("reason")):
            ok_lines.add(line)
        else:
            violations.append(
                Violation(
                    path,
                    line,
                    0,
                    "LIT005",
                    "any-ok requires a reason: `# any-ok: <reason>`",
                )
            )
    return frozenset(ok_lines), tuple(violations)


# --------------------------------------------------------------------------- #
# mypy build (parity with `make lint-mypy`) + forced target re-check
# --------------------------------------------------------------------------- #


def _build_options() -> Options:
    opts = Options()
    if MYPY_INI.exists():
        parse_config_file(opts, lambda: None, str(MYPY_INI), sys.stdout, sys.stderr)
    opts.export_types = True
    opts.preserve_asts = True
    opts.incremental = True
    opts.cache_dir = str(CACHE_DIR)
    opts.show_traceback = False
    return opts


def _meta_path(module: str) -> Path:
    return CACHE_DIR / PY_TAG / (module.replace(".", os.sep) + ".meta.json")


def _force_recheck(sources: Sequence[BuildSource]) -> None:
    """Invalidate each target's cached entry so mypy re-type-checks (and thus
    re-exports types + preserves the AST for) exactly these modules, while their
    dependencies stay warm. A missing entry is a cold build for that module.

    mypy trusts a cache entry whenever the source mtime matches the cached one
    (it never re-hashes on that fast path), so we must break BOTH: zero the
    cached mtime to force a re-hash, and corrupt the cached hash so the re-hash
    mismatches and the module is treated as changed."""
    for src in sources:
        if not src.module:
            continue
        meta = _meta_path(src.module)
        if not meta.exists():
            continue
        try:
            data = json.loads(meta.read_text())
            data["hash"] = "0" * 40
            data["mtime"] = 0
            meta.write_text(json.dumps(data))
        except (OSError, ValueError):
            continue


def _warm_cache(rel_paths: Sequence[str]) -> Violation | None:
    """Populate `.mypy_cache_any` for the whole closure without retaining ASTs or
    exported types (neither affects cache validity, see mypy's
    OPTIONS_AFFECTING_CACHE). Run before a batched `--update` so each batch then
    re-checks only its own targets against a warm cache, bounding peak memory."""
    prev_cwd = Path.cwd()
    os.chdir(LITELLM_DIR)
    try:
        opts = _build_options()
        opts.export_types = False
        opts.preserve_asts = False
        fscache = FileSystemCache()
        sources = create_source_list(list(rel_paths), opts, fscache)
        try:
            build.build(sources, options=opts, fscache=fscache)
        except build.CompileError as exc:
            joined = "; ".join(exc.messages[:3]) or "blocking error"
            return Violation(
                Path(rel_paths[0]), 0, 0, "LIT000", f"mypy could not build: {joined}"
            )
    finally:
        os.chdir(prev_cwd)
    return None


def check_files(rel_paths: Sequence[str]) -> tuple[Violation, ...]:
    """`rel_paths` are relative to the litellm package dir (the build cwd)."""
    prev_cwd = Path.cwd()
    os.chdir(LITELLM_DIR)
    try:
        opts = _build_options()
        fscache = FileSystemCache()
        sources = create_source_list(list(rel_paths), opts, fscache)
        _force_recheck(sources)
        try:
            res = build.build(sources, options=opts, fscache=fscache)
        except build.CompileError as exc:
            joined = "; ".join(exc.messages[:3]) or "blocking error"
            return (
                Violation(
                    Path(rel_paths[0]),
                    0,
                    0,
                    "LIT000",
                    f"mypy could not build: {joined}",
                ),
            )
        idmap = {id(expr): t for expr, t in res.types.items()}
        # Resolve trees to absolute source paths while cwd is the build dir, since
        # mypy stores the paths it was given (relative to this cwd).
        trees: dict[str, Node] = {}
        for state in res.graph.values():
            if state.path and state.tree is not None:
                trees[os.path.realpath(state.path)] = state.tree
    finally:
        os.chdir(prev_cwd)

    out: list[Violation] = []
    for rel in rel_paths:
        abs_path = (LITELLM_DIR / rel).resolve()
        report_path = abs_path.relative_to(REPO_ROOT)
        if _is_boundary(report_path):
            continue
        try:
            source = abs_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            out.append(
                Violation(report_path, 0, 0, "LIT000", f"could not read file: {exc}")
            )
            continue

        ok_lines, ok_violations = scan_any_ok(report_path, source)
        out.extend(ok_violations)
        tree = trees.get(os.path.realpath(abs_path))
        if tree is None:
            continue
        for line, col, typ in find_any_in_tree(tree, idmap):
            if line in ok_lines:
                continue
            out.append(
                Violation(
                    report_path,
                    line,
                    col,
                    "LIT009",
                    f"value type contains Any -> {typ}",
                )
            )
    return tuple(out)


# --------------------------------------------------------------------------- #
# Budget (per-file grandfathering) + file selection + driver
# --------------------------------------------------------------------------- #


def _is_boundary(path: Path) -> bool:
    posix = path.as_posix()
    return any(frag in posix for frag in BOUNDARY_PATHS)


def _git(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.splitlines()


def load_budget() -> dict[str, int]:
    if not BUDGET_PATH.exists():
        return {}
    return json.loads(BUDGET_PATH.read_text())


def lit009_counts(violations: Iterable[Violation]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for v in violations:
        if v.code == "LIT009":
            counts[v.path.as_posix()] += 1
    return dict(counts)


def budget_breaches(
    counts: Mapping[str, int], budget: Mapping[str, int]
) -> list[tuple[str, int, int]]:
    """Files whose `Any` count clears their ceiling. A file with no budget entry
    is grandfathered at zero, so any `Any` in it breaches."""
    breaches = [
        (path, total, cap_for(budget.get(path, 0)))
        for path, total in counts.items()
        if total > cap_for(budget.get(path, 0))
    ]
    return sorted(breaches)


def changed_files(base: str) -> list[str] | None:
    """Repo-relative `.py` paths under litellm/ that the working tree changed vs
    the merge-base with `base` (committed-on-branch + unstaged + untracked). None
    if git is unavailable / not a repo."""
    try:
        merge_base = _git("merge-base", base, "HEAD")
        point = merge_base[0].strip() if merge_base else base
        tracked = _git("diff", "--name-only", "--diff-filter=d", point, "--", "litellm")
        untracked = _git("ls-files", "--others", "--exclude-standard", "--", "litellm")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return _py_under_litellm([*tracked, *untracked])


def all_tracked_files() -> list[str]:
    return _py_under_litellm(_git("ls-files", "--", "litellm"))


def _py_under_litellm(names: Iterable[str]) -> list[str]:
    out = {
        name for name in names if name.endswith(".py") and (REPO_ROOT / name).exists()
    }
    return sorted(out)


def _to_litellm_relative(paths: Iterable[Path]) -> list[str]:
    rels: list[str] = []
    for p in sorted(paths):
        try:
            rels.append(p.resolve().relative_to(LITELLM_DIR).as_posix())
        except ValueError:
            continue
    return rels


def _setup_errors(violations: Sequence[Violation]) -> tuple[Violation, ...]:
    return tuple(v for v in violations if v.code == "LIT000")


def _batched(items: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def cmd_update() -> int:
    rel_paths = _to_litellm_relative(
        (REPO_ROOT / name).resolve() for name in all_tracked_files()
    )
    if not rel_paths:
        print("check_any_discipline: no Python files under litellm/", file=sys.stderr)
        return 2
    warm_error = _warm_cache(rel_paths)
    if warm_error is not None:
        print(warm_error.render(), file=sys.stderr)
        return 2

    counts: dict[str, int] = {}
    for batch in _batched(rel_paths, UPDATE_BATCH):
        violations = check_files(batch)
        errors = _setup_errors(violations)
        if errors:
            for v in errors:
                print(v.render(), file=sys.stderr)
            return 2
        counts.update(lit009_counts(violations))
    budget = dict(sorted(counts.items()))
    BUDGET_PATH.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n")
    print(
        f"Re-captured Any-discipline budget: {len(budget)} file(s) hold "
        f"{sum(budget.values())} grandfathered Any-typed value(s)"
    )
    return 0


def _report_breaches(
    breaches: Sequence[tuple[str, int, int]],
    per_file: dict[str, list[Violation]],
    hard: Sequence[Violation],
) -> None:
    if breaches:
        print("FAIL: file(s) over their Any-discipline budget:")
        for path, total, cap in breaches:
            print(f"  {path}: {total} Any-typed value(s) over ceiling {cap}")
            for v in sorted(per_file.get(path, [])):
                print(f"    {v.line}:{v.col} {v.message}")
        print(
            "Give the new values a concrete type, annotate `# any-ok: <reason>`, or "
            "run 'make lint-any-budget-update' if the ceiling should move."
        )
    for v in sorted(hard):
        print(v.render(), file=sys.stderr)


def cmd_check(base: str) -> int:
    files = changed_files(base)
    if files is None:
        print(
            "check_any_discipline: not a git repository; nothing to check",
            file=sys.stderr,
        )
        return 0
    if not files:
        print("OK: no changed Python files under litellm/ to check")
        return 0

    rel_paths = _to_litellm_relative((REPO_ROOT / name).resolve() for name in files)
    violations = check_files(rel_paths)
    errors = _setup_errors(violations)
    if errors:
        for v in errors:
            print(v.render(), file=sys.stderr)
        return 2

    budget = load_budget()
    counts = lit009_counts(violations)
    per_file: dict[str, list[Violation]] = {}
    for v in violations:
        if v.code == "LIT009":
            per_file.setdefault(v.path.as_posix(), []).append(v)

    breaches = budget_breaches(counts, budget)
    hard = tuple(v for v in violations if v.code == "LIT005")

    if not breaches and not hard:
        print(f"OK: {len(files)} changed file(s) under litellm/ are within budget")
        return 0
    _report_breaches(breaches, per_file, hard)
    return 1


def cmd_paths(paths: Sequence[str]) -> int:
    rel_paths = _to_litellm_relative((REPO_ROOT / p).resolve() for p in paths)
    if not rel_paths:
        print("OK: no Python files under litellm/ to check")
        return 0
    violations = check_files(rel_paths)
    for v in sorted(violations):
        print(v.render())
    if any(v.code == "LIT000" for v in violations):
        return 2
    return 1 if violations else 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Any-discipline gate (per-file grandfathering)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="explicit files (repo-root relative); whole-file, no budget",
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="budget the changed files under litellm/ vs --base",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="re-capture every file's baseline into any-discipline-budget.json",
    )
    parser.add_argument("--base", default=os.environ.get("ANY_GATE_BASE", DEFAULT_BASE))
    args = parser.parse_args(list(argv))

    if args.update:
        return cmd_update()
    if args.changed:
        return cmd_check(args.base)
    if args.paths:
        return cmd_paths(args.paths)
    parser.error("pass --changed, --update, or explicit file paths")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
