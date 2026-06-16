#!/usr/bin/env python3
"""Any-discipline gate: fail when a *changed* file holds a value typed `Any`.

Where ruff, `mypy --strict`, and even basedpyright's `reportAny` stop short, this
catches the case that actually bites: a *union* hiding an `Any`. For example
`re.Match.group()` -> `str | Any`, `json.loads()` -> `Any`, and bare `list`/`dict`
-> `list[Any]`/`dict[..., Any]`. Any value whose inferred type *contains* `Any`
(recursively, through unions / generics / tuples) is reported.

Scope: changed-only, changed-lines
----------------------------------
litellm already contains a large amount of pre-existing `Any` (a single legacy
file can have >100 findings), and a whole-tree scan would have to re-export types
for litellm's entire import closure on every run (~2 min, ~3 GB). So this gate is
*changed-only* and reports a finding only on a line that the diff against
`--base` actually adds or edits (untracked files count as wholly new). A brand
new file is therefore checked in full, while editing a legacy file only requires
*your* lines to be clean -- you can't introduce an `X | Any`, but you aren't
forced to clean the file's existing debt. This mirrors how `ruff_strict_gate.py`
blames a change only for the violations it introduces; cold legacy code is left
to the ratchet gates (mypy/basedpyright/ruff budgets).

How it works
------------
It loads `litellm/mypy.ini` (the same config `make lint-mypy` uses, so findings
match what developers already see), builds the changed files with mypy asking for
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
    # gate mode (CI / pre-push): check changed lines under litellm/
    uv run --no-sync python scripts/check_any_discipline.py --changed --base origin/litellm_internal_staging

    # whole-file spot-check (no line filter), paths relative to repo root
    uv run --no-sync python scripts/check_any_discipline.py litellm/budget_manager.py

Exit code 1 if any Any-tainted value is found, 2 on a setup/usage error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tokenize
from collections.abc import Iterable, Sequence
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
PY_TAG = f"{sys.version_info.major}.{sys.version_info.minor}"
DEFAULT_BASE = "origin/litellm_internal_staging"

MIN_REASON_LEN = 3
ANY_OK_RE = re.compile(r"#\s*any-ok(?::\s*(?P<reason>.*))?")
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

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
# File selection (changed-only, changed-lines) + driver
# --------------------------------------------------------------------------- #


class _AllLines:
    """Sentinel: a wholly new / untracked file -- every line is in scope.

    A distinct object, not None, so that `line_map.get(path)` returning None for
    a path absent from the map is never mistaken for "whole file in scope"."""


# A changed file's in-scope lines: a specific set, or every line.
LineScope = set[int] | _AllLines
ALL_LINES = _AllLines()


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


def _parse_added_lines(diff_text: str) -> dict[str, set[int]]:
    """Map repo-relative path -> set of new-file line numbers the diff adds/edits."""
    changed: dict[str, set[int]] = {}
    path: str | None = None
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif path and (m := _HUNK_RE.match(line)):
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            if count:
                changed.setdefault(path, set()).update(range(start, start + count))
    return changed


def changed_line_map(base: str) -> dict[str, LineScope] | None:
    """Repo-relative `.py` path under litellm/ -> changed line numbers (or
    ALL_LINES for untracked files). Compares the working tree to the merge-base
    with `base`, so it covers committed-on-branch + unstaged edits. None if git
    is unavailable / not a repo."""
    try:
        merge_base = _git("merge-base", base, "HEAD")
        point = merge_base[0].strip() if merge_base else base
        diff = "\n".join(
            _git(
                "diff",
                "--unified=0",
                "--no-color",
                "--diff-filter=d",
                point,
                "--",
                "litellm",
            )
        )
        untracked = _git("ls-files", "--others", "--exclude-standard", "--", "litellm")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

    out: dict[str, LineScope] = {}
    for name, lines in _parse_added_lines(diff).items():
        if name.endswith(".py") and (REPO_ROOT / name).exists():
            out[name] = lines
    for name in untracked:
        if name.endswith(".py") and (REPO_ROOT / name).exists():
            out[name] = ALL_LINES
    return out


def _to_litellm_relative(paths: Iterable[Path]) -> list[str]:
    rels: list[str] = []
    for p in sorted(paths):
        try:
            rels.append(p.resolve().relative_to(LITELLM_DIR).as_posix())
        except ValueError:
            continue
    return rels


def _in_scope(v: Violation, line_map: dict[str, LineScope] | None) -> bool:
    """A finding survives if line filtering is off (explicit paths), it's a build
    error, or its line is one the diff added/edited."""
    if line_map is None or v.code == "LIT000":
        return True
    lines = line_map.get(v.path.as_posix())
    return lines is ALL_LINES or (lines is not None and v.line in lines)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Any-discipline gate (changed-only, changed-lines)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="explicit files (repo-root relative); whole-file, no line filter",
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="check changed lines under litellm/ vs --base",
    )
    parser.add_argument("--base", default=os.environ.get("ANY_GATE_BASE", DEFAULT_BASE))
    args = parser.parse_args(list(argv))

    line_map: dict[str, LineScope] | None = None
    if args.changed:
        line_map = changed_line_map(args.base)
        if line_map is None:
            print(
                "check_any_discipline: not a git repository; nothing to check",
                file=sys.stderr,
            )
            return 0
        rel_paths = _to_litellm_relative(
            (REPO_ROOT / name).resolve() for name in line_map
        )
    elif args.paths:
        rel_paths = _to_litellm_relative((REPO_ROOT / p).resolve() for p in args.paths)
    else:
        parser.error("pass --changed or explicit file paths")
        return 2

    if not rel_paths:
        print("OK: no changed Python lines under litellm/ to check")
        return 0

    violations = tuple(v for v in check_files(rel_paths) if _in_scope(v, line_map))

    for v in sorted(violations):
        print(v.render())

    if violations:
        n = len(violations)
        print(
            f"\nFAIL: {n} Any-discipline violation(s) on changed lines.\n"
            "Give the value a concrete type, or annotate the line `# any-ok: <reason>`.",
            file=sys.stderr,
        )
        return 1
    print(
        f"OK: {len(rel_paths)} changed file(s) under litellm/ have no Any-typed values on changed lines"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
