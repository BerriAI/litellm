#!/usr/bin/env python3
"""Coarse-type discipline gate: fail when a changed file exceeds its budget.

Two coarse value types are caught, each with its own per-file budget:

`Any` (LIT009)
    Where ruff, `mypy --strict`, and even basedpyright's `reportAny` stop short,
    this catches the case that actually bites: a *union* hiding an `Any`. For
    example `re.Match.group()` -> `str | Any`, `json.loads()` -> `Any`, and bare
    `list`/`dict` -> `list[Any]`/`dict[..., Any]`. `Any` is unsound -- it silently
    disables checking -- so any value whose inferred type *contains* it is flagged.

`object` (LIT010)
    `object` is sound but coarse: you cannot do anything with an `object` value
    until you narrow it. A *bare* `object` at an untyped boundary (e.g. a
    `def f(x: object)` you immediately validate) is the disciplined alternative to
    `Any` -- but `object` buried in a container (`dict[str, object]`,
    `list[object]`) hands back an unnarrowable value on every read, so any value
    whose inferred type is, or contains, `object` is flagged the same way.

Scope: changed files, per-file budget
-------------------------------------
litellm carries a large amount of pre-existing coarse typing (a single legacy
file can have >100 `Any` findings). Rather than force every touched line clean
(the original changed-lines rule, which tripped on merely *editing* a legacy
`X | Any` line), each gate grandfathers per file: a budget file records every
file's current count, and a file fails only when its count exceeds
`baseline + slack`, where `slack` is 50% headroom (rounded up). New or unbudgeted
files have baseline 0, so they stay airtight.

Only *changed* files (vs the merge-base with `--base`) are re-type-checked -- an
unchanged file's count can't move from edits this branch didn't make -- so the
per-PR cost equals re-checking just those files, exactly like the original
changed-lines gate. The whole-tree scan needed to (re)capture the budgets
(~2 min, ~3 GB) runs only under `--update`.

Each budget is a one-way ratchet (the same `{baseline, slack}` shape as the
ruff / mypy / basedpyright budgets) guarded by `scripts/budget_ratchet_check.py`:
a file's ceiling may fall but never rise. Drive a file's count down and rerun
`--update` (`make lint-any-budget-update`) to lock in the lower ceiling.

How it works
------------
It loads `litellm/mypy.ini` (the same config `make lint-mypy` uses, so findings
match what developers already see), builds the changed files with mypy asking for
its exported expression->type map, and walks each file's AST applying a recursive
"contains a coarse leaf" predicate -- for `Any`, the test `mypy --disallow-any-expr`
uses internally but applies inconsistently (python/mypy#12856); for `object`, the
same walk asking instead whether a leaf is `builtins.object`.

mypy only re-exports types for modules it re-type-checks, so for each target we
invalidate just its cached hash (deps stay warm) to force a fast re-check against
a persisted incremental cache (.mypy_cache_any).

Rules
-----
Codes share the `LIT***` namespace with `scripts/check_type_discipline.py` (PR
#30500), which owns LIT001/002/003/004/006/007/008. This gate claims the rest:
LIT009  A value expression's inferred type is, or contains, `Any`. Budgeted
        per file (a file fails when its count exceeds `baseline + slack`).
        Suppress an individual line with `# any-ok: <reason>`.
LIT010  A value expression's inferred type is, or contains, `object`. Budgeted
        per file in its own budget. Suppress an individual line with
        `# object-ok: <reason>`.
LIT005  An `# any-ok` / `# object-ok` suppression without a reason (the shared
        suppression-needs-a-reason code, same as `# cast-ok` / `# guard-ok`).
LIT000  Setup failure: mypy could not build, or a target file could not be read.

`Any`s produced purely by an already-reported error, and the special-form /
implementation-artifact internal `Any`s, are ignored. A bound method *reference*
whose signature mentions a coarse type is not flagged -- only the value its call
produces.

Usage
-----
    # gate mode (CI / pre-push): per-file budgets on changed files
    uv run --no-sync python scripts/check_any_discipline.py --changed --base origin/litellm_internal_staging

    # re-capture the per-file budgets across the whole tree (ratchet)
    uv run --no-sync python scripts/check_any_discipline.py --update

    # whole-file spot-check (no budget, no line filter), paths relative to repo root
    uv run --no-sync python scripts/check_any_discipline.py litellm/budget_manager.py

Exit code 1 if a file is over budget (or a hard rule trips), 2 on a setup error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tokenize
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import NamedTuple

try:
    from mypy import build
    from mypy.config_parser import parse_config_file
    from mypy.find_sources import create_source_list
    from mypy.fscache import FileSystemCache
    from mypy.modulefinder import BuildSource
    from mypy.nodes import AssignmentStmt, Expression, NameExpr, Node, TempNode
    from mypy.options import Options
    from mypy.types import (
        AnyType,
        CallableType,
        Instance,
        Overloaded,
        ProperType,
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
ANY_BUDGET_PATH = REPO_ROOT / "any-discipline-budget.json"
OBJECT_BUDGET_PATH = REPO_ROOT / "object-discipline-budget.json"

# Kept as the public name the budget tests + Any-centric callers reference.
BUDGET_PATH = ANY_BUDGET_PATH

MIN_REASON_LEN = 3
_HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _ok_re(keyword: str) -> re.Pattern[str]:
    return re.compile(rf"#\s*{re.escape(keyword)}(?::\s*(?P<reason>.*))?")


ANY_OK_RE = _ok_re("any-ok")
OBJECT_OK_RE = _ok_re("object-ok")

# Files allowed to surface a coarse type (the typed/untyped boundary). A finding
# is skipped if any fragment below is a substring of the file's posix path. Keep
# this tight -- prefer a line-level `# any-ok` / `# object-ok: <reason>` over a
# blanket exemption.
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

# Awaitable / coroutine / generator instances carry synthetic `Any` in their
# send (and, for coroutines, yield) protocol slots: `async def f() -> float`
# produces `Coroutine[Any, Any, float]`, so the bare call expression `f()` would
# be flagged even though the awaited value is a clean `float`. Only the args that
# hold a value the caller observes (the awaited result, the yielded item) are
# meaningful; a real coarse type there -- e.g. a coroutine that returns `Any` --
# is still caught because that index is still checked.
_SYNTHETIC_SEND_YIELD_VALUE_ARGS: dict[str, tuple[int, ...]] = {
    "typing.Coroutine": (2,),
    "typing.Generator": (0, 2),
    "typing.AsyncGenerator": (0,),
}


class Violation(NamedTuple):
    path: Path
    line: int
    col: int
    code: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.code} {self.message}"


# --------------------------------------------------------------------------- #
# The "contains a coarse leaf" predicate
# --------------------------------------------------------------------------- #


# Recursive type aliases (e.g. a JSON-like `T = Union[..., list[T], dict[str, T]]`)
# make `get_proper_type` yield a fresh object at every unfold, so an id()-based
# cycle guard never trips and a naive recursion overflows the stack. We walk
# iteratively and cap the depth: a real coarse leaf lives at shallow depth in the
# alias's definition, so a deep alias that has not produced one by `_MAX_DEPTH`
# never will. (The changed-lines gate never hit this; a whole-tree scan does.)
_MAX_DEPTH = 100


def _is_actionable_any(p: ProperType) -> bool:
    """A real, unsound `Any` -- not one mypy emitted purely from an error."""
    return isinstance(p, AnyType) and p.type_of_any not in _HARMLESS_ANY


def _is_object(p: ProperType) -> bool:
    """The bare top type `object` (not a subclass -- `str` is not flagged)."""
    return isinstance(p, Instance) and p.type.fullname == "builtins.object"


def _contains(t: Type, is_hit: Callable[[ProperType], bool]) -> bool:
    """True if a *value* of type ``t`` has, anywhere meaningful, a proper-type leaf
    satisfying ``is_hit``. Walks unions / generics / tuples iteratively, sharing one
    depth cap, cycle guard, function-reference skip, and synthetic send/yield
    handling across the `Any` and `object` predicates."""
    seen: set[int] = set()
    stack: list[tuple[Type, int]] = [(t, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > _MAX_DEPTH:
            continue
        p = get_proper_type(cur)
        if id(p) in seen:
            continue
        seen.add(id(p))

        # A function/method *reference* whose signature mentions a coarse type is
        # not itself an unsafe value -- only its eventual call result is.
        if isinstance(p, (CallableType, Overloaded)):
            continue
        if is_hit(p):
            return True
        if isinstance(p, UnionType):
            stack.extend((item, depth + 1) for item in p.items)
        elif isinstance(p, Instance):
            value_arg_indices = _SYNTHETIC_SEND_YIELD_VALUE_ARGS.get(p.type.fullname)
            if value_arg_indices is None:
                stack.extend((arg, depth + 1) for arg in p.args)
            else:
                stack.extend(
                    (p.args[index], depth + 1)
                    for index in value_arg_indices
                    if index < len(p.args)
                )
        elif isinstance(p, TupleType):
            stack.extend((item, depth + 1) for item in p.items)
    return False


def contains_any(t: Type) -> bool:
    """True if a *value* of type ``t`` carries `Any` anywhere meaningful."""
    return _contains(t, _is_actionable_any)


def contains_object(t: Type) -> bool:
    """True if a *value* of type ``t`` is, or carries, `object` anywhere meaningful."""
    return _contains(t, _is_object)


class Discipline(NamedTuple):
    """One coarse-type rule: its code, the noun for its messages, the per-file
    budget it ratchets, the `# <keyword>: <reason>` line suppression, and the
    proper-type leaf test that defines it."""

    code: str
    noun: str
    budget_path: Path
    ok_keyword: str
    ok_re: re.Pattern[str]
    is_hit: Callable[[ProperType], bool]

    def message(self, rendered_type: str) -> str:
        return f"value type contains {self.noun} -> {rendered_type}"

    def reasonless_message(self) -> str:
        return f"{self.ok_keyword} requires a reason: `# {self.ok_keyword}: <reason>`"


DISCIPLINES: tuple[Discipline, ...] = (
    Discipline(
        "LIT009", "Any", ANY_BUDGET_PATH, "any-ok", ANY_OK_RE, _is_actionable_any
    ),
    Discipline(
        "LIT010", "object", OBJECT_BUDGET_PATH, "object-ok", OBJECT_OK_RE, _is_object
    ),
)
_DISCIPLINE_BY_CODE: dict[str, Discipline] = {d.code: d for d in DISCIPLINES}


# --------------------------------------------------------------------------- #
# Generic, leak-free AST walk (works under a mypyc-compiled mypy, which forbids
# subclassing TraverserVisitor)
# --------------------------------------------------------------------------- #


def _walk_file(tree: Node) -> tuple[list[Expression], set[int]]:
    """Return (every Expression in `tree`, ids of simple assignment-target names).

    The walk follows only syntactic children (every attribute except the two
    non-syntactic back-references), so it never escapes the module. Simple
    ``x = <expr>`` name targets are collected separately so we don't double-report
    the assigned name as an echo of a coarse rvalue.
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


def find_coarse_in_tree(
    tree: Node, idmap: dict[int, Type]
) -> list[tuple[int, int, str, str]]:
    """Return (line, col, code, rendered-type) for every value expression whose
    inferred type trips a discipline's predicate. A single expression can trip more
    than one discipline (e.g. ``dict[object, Any]`` is both LIT009 and LIT010)."""
    exprs, skip_lvalues = _walk_file(tree)
    findings: list[tuple[int, int, str, str]] = []
    for expr in exprs:
        # A TempNode is mypy's synthetic placeholder for a position with no real
        # expression -- e.g. the rvalue of an annotation-only `field: T` in a
        # TypedDict / class body, whose `special_form` `Any` is not a value the
        # author wrote. It never corresponds to a runtime value, so skip it.
        if id(expr) in skip_lvalues or isinstance(expr, TempNode):
            continue
        t = idmap.get(id(expr))
        if t is None:
            continue
        rendered = str(get_proper_type(t))
        for disc in DISCIPLINES:
            if _contains(t, disc.is_hit):
                findings.append((expr.line, expr.column, disc.code, rendered))

    out: list[tuple[int, int, str, str]] = []
    seen_pos: set[tuple[int, int, str]] = set()
    for line, col, code, typ in sorted(findings):
        if line < 1 or (line, col, code) in seen_pos:
            continue
        seen_pos.add((line, col, code))
        out.append((line, col, code, typ))
    return out


# --------------------------------------------------------------------------- #
# Comment scanning (LIT005 + any-ok / object-ok suppression)
# --------------------------------------------------------------------------- #


def _reason_ok(reason: str | None) -> bool:
    return reason is not None and len(reason.strip()) >= MIN_REASON_LEN


def scan_suppressions(
    path: Path, source: str
) -> tuple[dict[int, frozenset[str]], tuple[Violation, ...]]:
    """Return (line -> codes suppressed there, LIT005 violations).

    `# any-ok: <reason>` suppresses LIT009 on its line; `# object-ok: <reason>`
    suppresses LIT010. A suppression without a reason is itself a LIT005."""
    try:
        tokens = tokenize.generate_tokens(
            iter(source.splitlines(keepends=True)).__next__
        )
        comments = tuple(
            (t.start[0], t.string) for t in tokens if t.type == tokenize.COMMENT
        )
    except tokenize.TokenError:
        return {}, ()

    suppressed: dict[int, set[str]] = {}
    violations: list[Violation] = []
    for line, text in comments:
        for disc in DISCIPLINES:
            m = disc.ok_re.search(text)
            if m is None:
                continue
            if _reason_ok(m.group("reason")):
                suppressed.setdefault(line, set()).add(disc.code)
            else:
                violations.append(
                    Violation(path, line, 0, "LIT005", disc.reasonless_message())
                )
    frozen = {line: frozenset(codes) for line, codes in suppressed.items()}
    return frozen, tuple(violations)


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

        suppressed, ok_violations = scan_suppressions(report_path, source)
        out.extend(ok_violations)
        tree = trees.get(os.path.realpath(abs_path))
        if tree is None:
            continue
        for line, col, code, typ in find_coarse_in_tree(tree, idmap):
            if code in suppressed.get(line, frozenset()):
                continue
            out.append(
                Violation(
                    report_path, line, col, code, _DISCIPLINE_BY_CODE[code].message(typ)
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
    return lines is ALL_LINES or (isinstance(lines, set) and v.line in lines)


# --------------------------------------------------------------------------- #
# Per-file budgets (one-way ratchet, 50% headroom; ratchet-checked), one per
# discipline (Any, object)
# --------------------------------------------------------------------------- #


def _slack_for(baseline: int) -> int:
    """50% headroom, rounded up so even a 1-finding file gets a little room."""
    return (baseline + 1) // 2


def _ceiling(spec: dict[str, int]) -> int:
    """A file's ceiling: ``baseline + slack`` (0 for an absent/empty entry)."""
    return int(spec.get("baseline", 0)) + int(spec.get("slack", 0))


def load_budget(path: Path) -> dict[str, dict[str, int]]:
    """Read a budget file ({path: {baseline, slack}}); {} if absent/malformed."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_budget(path: Path, counts: dict[str, int]) -> None:
    """Write a fresh budget from per-file counts, with 50% headroom each.

    Files with zero findings are omitted: an absent entry means baseline 0, so a
    file's first coarse value always trips the gate until it is deliberately
    baselined."""
    budget = {
        file: {"baseline": n, "slack": _slack_for(n)}
        for file, n in counts.items()
        if n > 0
    }
    path.write_text(json.dumps(budget, indent=2, sort_keys=True) + "\n")


def counts_for_code(violations: Iterable[Violation], code: str) -> dict[str, int]:
    """Count `code` findings per repo-relative file path."""
    counts: dict[str, int] = {}
    for v in violations:
        if v.code == code:
            key = v.path.as_posix()
            counts[key] = counts.get(key, 0) + 1
    return counts


def lit009_counts(violations: Iterable[Violation]) -> dict[str, int]:
    """Count LIT009 (Any-typed value) findings per repo-relative file path."""
    return counts_for_code(violations, "LIT009")


def all_litellm_py_files() -> list[str] | None:
    """Every tracked ``.py`` under litellm/, as litellm-package-relative paths;
    None if git is unavailable / not a repo (mirrors ``changed_line_map``)."""
    try:
        tracked = _git("ls-files", "--", "litellm")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return _to_litellm_relative(
        REPO_ROOT / name for name in tracked if name.endswith(".py")
    )


def update_budget(
    list_files: Callable[[], list[str] | None] = all_litellm_py_files,
) -> int:
    """Whole-tree scan: recapture every file's count into each discipline's budget."""
    rel_paths = list_files()
    if rel_paths is None:
        print(
            "check_any_discipline: not a git repository; cannot capture the budget",
            file=sys.stderr,
        )
        return 2
    if not rel_paths:
        print("check_any_discipline: no litellm/*.py files found", file=sys.stderr)
        return 2
    violations = check_files(rel_paths)
    build_errors = [v for v in violations if v.code == "LIT000"]
    if build_errors:
        for v in build_errors:
            print(v.render(), file=sys.stderr)
        print(
            "FAIL: mypy could not build the tree; budgets left unchanged.",
            file=sys.stderr,
        )
        return 2
    for disc in DISCIPLINES:
        counts = counts_for_code(violations, disc.code)
        save_budget(disc.budget_path, counts)
        print(
            f"Wrote {disc.budget_path.name}: "
            f"{sum(1 for n in counts.values() if n > 0)} file(s), "
            f"{sum(counts.values())} {disc.noun}-typed value(s) baselined "
            "(50% headroom each)."
        )
    return 0


def _report_over_budget(
    disc: Discipline,
    path: str,
    count: int,
    spec: dict[str, int] | None,
    findings: list[Violation],
    line_map: dict[str, LineScope],
) -> None:
    """Print one over-budget file plus the findings on its changed lines."""
    ceiling = _ceiling(spec or {})
    if spec:
        why = f"baseline {spec['baseline']} + 50% slack {spec['slack']} = ceiling {ceiling}"
    else:
        why = f"no budget entry -> baseline 0 (a new/unbudgeted file must be {disc.noun}-free)"
    print(f"{path}: {count} {disc.noun}-typed value(s) total, over budget ({why})")
    # Surface the findings on changed lines first: the ones this branch most
    # likely just added, and the cheapest path back under the ceiling.
    scope = line_map.get(path)
    for v in sorted(findings):
        if scope is ALL_LINES or (isinstance(scope, set) and v.line in scope):
            print(f"  changed-line {disc.noun}  {v.line}:{v.col} {v.message}")


class _OverBudget(NamedTuple):
    disc: Discipline
    path: str
    count: int
    spec: dict[str, int] | None


def run_gate(base: str) -> int:
    """Gate changed files under litellm/ against the committed per-file budgets."""
    line_map = changed_line_map(base)
    if line_map is None:
        print(
            "check_any_discipline: not a git repository; nothing to check",
            file=sys.stderr,
        )
        return 0
    rel_paths = _to_litellm_relative((REPO_ROOT / name).resolve() for name in line_map)
    if not rel_paths:
        print("OK: no changed Python files under litellm/ to check")
        return 0

    violations = check_files(rel_paths)

    # Hard rules, independent of any budget: a build/read failure (always), and a
    # reasonless `# any-ok` / `# object-ok` on a line this branch touched.
    hard = sorted(
        v
        for v in violations
        if v.code == "LIT000" or (v.code == "LIT005" and _in_scope(v, line_map))
    )

    # Per-file budget per discipline: a changed file fails when its total count of
    # that code exceeds its ceiling. Unchanged files keep their committed baseline.
    findings_by: dict[tuple[str, str], list[Violation]] = {}
    for v in violations:
        if v.code in _DISCIPLINE_BY_CODE:
            findings_by.setdefault((v.code, v.path.as_posix()), []).append(v)
    over_budget = [
        _OverBudget(disc, path, count, budget.get(path))
        for disc in DISCIPLINES
        for budget in (load_budget(disc.budget_path),)
        for path, count in sorted(counts_for_code(violations, disc.code).items())
        if count > _ceiling(budget.get(path, {}))
    ]

    if not hard and not over_budget:
        print(
            f"OK: {len(rel_paths)} changed file(s) under litellm/ are within their "
            "Any/object budgets"
        )
        return 0

    for v in hard:
        print(v.render())
    for ob in over_budget:
        _report_over_budget(
            ob.disc,
            ob.path,
            ob.count,
            ob.spec,
            findings_by.get((ob.disc.code, ob.path), []),
            line_map,
        )

    print(
        f"\nFAIL: {len(hard)} hard violation(s), {len(over_budget)} file(s) over a budget.\n"
        "Give the new values concrete types (validate untyped input with Pydantic) to get back\n"
        "under the file's ceiling, or annotate a genuine boundary line `# any-ok` / `# object-ok: <reason>`.\n"
        "Re-baseline with `make lint-any-budget-update` only to lock in a reduction.",
        file=sys.stderr,
    )
    return 1


def spot_check(rel_paths: Sequence[str]) -> int:
    """Explicit-paths mode: report every finding in the files (no budget)."""
    violations = sorted(check_files(rel_paths))
    for v in violations:
        print(v.render())
    if violations:
        print(
            f"\nFAIL: {len(violations)} coarse-type discipline finding(s).",
            file=sys.stderr,
        )
        return 1
    print(f"OK: {len(rel_paths)} file(s) have no Any/object-typed values")
    return 0


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Coarse-type discipline gate (changed files, per-file Any/object budgets)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="explicit files (repo-root relative); whole-file spot-check, no budget",
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="gate changed files under litellm/ vs --base against the per-file budgets",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="recapture the whole-tree per-file budgets (any/object-discipline-budget.json)",
    )
    parser.add_argument("--base", default=os.environ.get("ANY_GATE_BASE", DEFAULT_BASE))
    args = parser.parse_args(list(argv))

    if args.update:
        return update_budget()
    if args.changed:
        return run_gate(args.base)
    if args.paths:
        rel_paths = _to_litellm_relative((REPO_ROOT / p).resolve() for p in args.paths)
        if not rel_paths:
            print("check_any_discipline: no litellm/*.py paths given", file=sys.stderr)
            return 2
        return spot_check(rel_paths)
    parser.error("pass --changed, --update, or explicit file paths")
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
