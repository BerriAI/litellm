#!/usr/bin/env python3
"""Report SQL `IN (...)` lists whose length nothing bounds.

Postgres caps a prepared statement at 32,767 bind parameters and a membership filter
binds one per value, so a list built from table data breaks once the table outgrows
the cap (LIT-7535). Reported, across litellm/ and enterprise/:

  prisma   a dict literal with an `"in"` / `"not_in"` key whose value has no fixed
           size. A list, tuple or set written out in full passes, as does a name the
           module binds once to a tuple, frozenset or constant; any other name, a call,
           a comprehension or a starred display does not.
  raw-sql  a string literal whose `IN (` is followed by a value spliced in at runtime:
           an f-string or format slot, a `%s`, or the end of the literal itself.
           `IN (SELECT ...)`, `IN ($1, $2)` and `= ANY($1::text[])` pass.

The Prisma engine chunks `create_many` on its own but sends an `update_many` /
`delete_many` filter whole. Record a real bound with `# bounded-ok: <reason>` on the
reported line or alone on the line above; the reason is required. Warning only: every
finding prints as `path:line: kind message` and the exit code is 0.

Usage: python check_unbounded_in_lists.py [files-or-dirs...]   (default: litellm enterprise)
"""

from __future__ import annotations

import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS: Final = ("litellm", "enterprise")

MEMBERSHIP_KEYS: Final = frozenset({"in", "not_in", "notIn"})
CONSTANT_WRAPPERS: Final = frozenset({"list", "tuple", "sorted", "frozenset", "set"})
FREEZING_WRAPPERS: Final = frozenset({"tuple", "frozenset"})
MIN_REASON_LEN: Final = 3

MARKER: Final = re.compile(r"#\s*bounded-ok(?::[ \t]*(?P<reason>[^#]*))?")
# The text right after `IN (` is where a runtime value lands: an f-string or format
# slot (`{x}`, never the escaped `{{`), a `%` slot, or the end of the literal itself.
SPLICED_IN: Final = re.compile(r"\bIN\s*\(\s*(?:\{(?!\{)|%s\b|%\(|$)", re.IGNORECASE)
CLOSING_QUOTES: Final = re.compile(r"(?:\"\"\"|'''|\"|')$")


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    line: int
    kind: str
    message: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.kind} {self.message}"


@dataclass(frozen=True, slots=True)
class Marker:
    reason: str
    standalone: bool

    @property
    def valid(self) -> bool:
        return len(self.reason) >= MIN_REASON_LEN


@dataclass(frozen=True, slots=True)
class Markers:
    by_line: Mapping[int, Marker]

    def exempt(self, line: int) -> bool:
        """A marker on the line itself, or alone on the line above it, speaks for it."""
        same: Final = self.by_line.get(line)
        above: Final = self.by_line.get(line - 1)
        return (same is not None and same.valid) or (above is not None and above.standalone and above.valid)


def read_markers(source: str) -> Markers:
    try:
        tokens: Final = tuple(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return Markers({})
    return Markers(
        {
            token.start[0]: Marker(
                reason=(match.group("reason") or "").strip(),
                standalone=not token.line[: token.start[1]].strip(),
            )
            for token in tokens
            if token.type == tokenize.COMMENT
            for match in (MARKER.search(token.string),)
            if match is not None
        }
    )


def has_fixed_size(value: ast.expr, constants: frozenset[str]) -> bool:
    """Whether the value's length is visible in the source rather than decided at runtime."""
    match value:
        case ast.List(elts=elts) | ast.Tuple(elts=elts) | ast.Set(elts=elts):
            return not any(isinstance(elt, ast.Starred) for elt in elts)
        case ast.Constant():
            return True
        case ast.Name(id=name):
            return name in constants
        case ast.Call(func=ast.Name(id=wrapper), args=[argument], keywords=[]) if wrapper in CONSTANT_WRAPPERS:
            return has_fixed_size(argument, constants)
        case _:
            return False


def _module_binding(stmt: ast.stmt) -> tuple[tuple[str, ast.expr], ...]:
    match stmt:
        case ast.Assign(targets=[ast.Name(id=name)], value=value):
            return ((name, value),)
        case ast.AnnAssign(target=ast.Name(id=name), value=ast.expr() as value):
            return ((name, value),)
        case _:
            return ()


def _stays_fixed(value: ast.expr, constants: frozenset[str]) -> bool:
    """has_fixed_size, less the shapes a later append or extend could grow."""
    match value:
        case ast.Tuple(elts=elts):
            return not any(isinstance(elt, ast.Starred) for elt in elts)
        case ast.Constant():
            return True
        case ast.Name(id=name):
            return name in constants
        case ast.Call(func=ast.Name(id=wrapper), args=[argument], keywords=[]) if wrapper in FREEZING_WRAPPERS:
            return has_fixed_size(argument, constants)
        case _:
            return False


def module_constants(tree: ast.Module) -> frozenset[str]:
    """Module-level names bound exactly once to a frozen value of fixed size, in binding
    order so one constant may be built from another. Casing plays no part: an ALL_CAPS
    name that is imported or filled at runtime is as unbounded as any other."""
    bound: Final = tuple(binding for stmt in tree.body for binding in _module_binding(stmt))
    names: Final = tuple(name for name, _ in bound)
    rebound: Final = frozenset(name for name in names if names.count(name) > 1)

    def fold(constants: frozenset[str], binding: tuple[str, ast.expr]) -> frozenset[str]:
        name, value = binding
        return constants | {name} if name not in rebound and _stays_fixed(value, constants) else constants

    return reduce(fold, bound, frozenset())


def prisma_findings(path: Path, tree: ast.Module) -> Iterator[Finding]:
    constants: Final = module_constants(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and key.value in MEMBERSHIP_KEYS):
                continue
            if has_fixed_size(value, constants):
                continue
            yield Finding(
                path,
                key.lineno,
                "prisma",
                f'`"{key.value}"` filter over `{ast.unparse(value)}` has no written bound: it binds one '
                f"parameter per value and Postgres caps a statement at 32,767. Chunk the list, or record the "
                f"bound with `# bounded-ok: <reason>`",
            )


def _literal_body(lines: tuple[bytes, ...], node: ast.expr) -> str | None:
    """The literal's source text with its closing quotes removed, so a literal that
    ends right after `IN (` reads as an open list rather than as `IN ('`. Column
    offsets count UTF-8 bytes, so the slice is taken on the encoded lines."""
    end_line: Final = node.end_lineno
    end_col: Final = node.end_col_offset
    if end_line is None or end_col is None:
        return None
    first: Final = node.lineno - 1
    last: Final = end_line - 1
    segment: Final = (
        lines[first][node.col_offset : end_col]
        if first == last
        else b"".join((lines[first][node.col_offset :], *lines[first + 1 : last], lines[last][:end_col]))
    )
    return CLOSING_QUOTES.sub("", segment.decode("utf-8", errors="replace"))


def _fstring_part_ids(tree: ast.AST) -> frozenset[int]:
    """ids() of the literal pieces inside f-strings, which the enclosing JoinedStr already covers."""
    return frozenset(
        id(part)
        for node in ast.walk(tree)
        if isinstance(node, ast.JoinedStr)
        for value in node.values
        for part in (
            (value,)
            if isinstance(value, ast.Constant)
            else tuple(ast.walk(value.format_spec))
            if isinstance(value, ast.FormattedValue) and value.format_spec is not None
            else ()
        )
    )


def raw_sql_findings(path: Path, source: str, tree: ast.AST) -> Iterator[Finding]:
    parts: Final = _fstring_part_ids(tree)
    lines: Final = tuple(source.encode("utf-8").splitlines(keepends=True))
    for node in ast.walk(tree):
        is_text = isinstance(node, ast.JoinedStr) or (isinstance(node, ast.Constant) and isinstance(node.value, str))
        if not is_text or id(node) in parts:
            continue
        body = _literal_body(lines, node)
        match = None if body is None else SPLICED_IN.search(body)
        if body is None or match is None:
            continue
        in_line = node.lineno + body[: match.start()].count("\n")
        where = "" if in_line == node.lineno else f" (the `IN (` is on line {in_line})"
        yield Finding(
            path,
            node.lineno,
            "raw-sql",
            f"`IN (` takes a list spliced in at runtime{where}: it binds one parameter per value and Postgres "
            f"caps a statement at 32,767. Pass the list as one array parameter (`= ANY($1::text[])`), chunk it, "
            f"or record the bound with `# bounded-ok: <reason>`",
        )


def marker_findings(path: Path, markers: Markers) -> Iterator[Finding]:
    for line, marker in sorted(markers.by_line.items()):
        if not marker.valid:
            yield Finding(
                path, line, "marker", "`# bounded-ok` needs a reason naming the bound: `# bounded-ok: <reason>`"
            )


def check_file(path: Path) -> tuple[Finding, ...]:
    try:
        source: Final = path.read_text(encoding="utf-8")
        tree: Final = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return (Finding(path, getattr(exc, "lineno", None) or 0, "unreadable", str(exc)),)
    markers: Final = read_markers(source)
    return (
        *marker_findings(path, markers),
        *(
            finding
            for finding in (*prisma_findings(path, tree), *raw_sql_findings(path, source, tree))
            if not markers.exempt(finding.line)
        ),
    )


def collect_paths(raw: Iterable[str]) -> Iterator[Path]:
    for item in raw:
        path = Path(item)
        if path.is_dir():
            yield from sorted(path.rglob("*.py"))
        elif path.suffix == ".py":
            yield path


def scan(paths: Iterable[Path]) -> tuple[Finding, ...]:
    return tuple(sorted((f for path in paths for f in check_file(path)), key=lambda f: (str(f.path), f.line, f.kind)))


def main(argv: Iterable[str]) -> int:
    targets: Final = tuple(argv) or DEFAULT_TARGETS
    findings: Final = scan(collect_paths(targets))
    for finding in findings:
        print(finding.render())
    counts: Final = {
        kind: sum(1 for f in findings if f.kind == kind) for kind in ("prisma", "raw-sql", "marker", "unreadable")
    }
    summary: Final = ", ".join(f"{count} {kind}" for kind, count in counts.items() if count)
    print(f"\n{len(findings)} unbounded IN list(s) ({summary or 'none'}); warning only, not blocking.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
