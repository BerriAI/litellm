#!/usr/bin/env python3
"""Report SQL `IN (...)` lists whose length nothing bounds.

Postgres refuses a prepared statement with more than 32,767 bind parameters, and a
membership filter binds one parameter per value, so an `IN` list built from table
data stops working the day the table outgrows the cap. The budget reset job did
exactly that (LIT-7535, litellm#40564): one `UPDATE ... WHERE user_id IN (...)`
naming every end user on a budget was rejected once a budget held 33,000 of them,
and because the job reset every due budget in one transaction the rejection rolled
all of them back, on every tick, until an operator split the population.

Two shapes are reported, across litellm/ and enterprise/:

  prisma   a dict literal with an `"in"` / `"not_in"` key whose value has no fixed
           size, such as `{"user_id": {"in": user_ids}}`. A list, tuple or set written
           out in full has the length it shows, so `["a", "b"]` and `[user_id]` pass,
           while a name, a call, a comprehension or a starred display does not. A name
           bound once at module level to such a value passes too, bare or wrapped in
           list/tuple/sorted/frozenset/set; an imported name does not, whatever its
           casing, since its size is not visible from here.
  raw-sql  a string literal whose `IN (` is followed by a value spliced in at
           runtime: an f-string `IN ({placeholders})`, a `{}` or `%s` slot for
           `.format` / `%`, or a literal that closes right after `IN (` so something
           gets concatenated on. `IN (SELECT ...)`, `IN ($1, $2)` and
           `= ANY($1::text[])` bind a fixed number of parameters and pass.

The Prisma engine (5.4.2, pinned by prisma 0.11.0) splits `create_many` rows across
statements to stay under the cap, and chunks a `find_many` `in` in a way that still
breaks with two large lists or extra criteria (prisma/prisma#21802), but sends an
`update_many` / `delete_many` filter as one statement, which is the write that froze
the reset job. Filters are mostly built away from the call that sends them, so every
membership filter is reported rather than only the ones a write can be seen to use.

A list with a real bound records it on the reported line, or alone on the line above:

    where={"team_id": {"in": page_team_ids}}  # bounded-ok: one page of at most 100 ids

The reason is required, and a marker without one is reported as its own finding. A
list that grows with a table needs chunking instead (the PTU rollup prune splits its
ids at `_PRUNE_ID_CHUNK_SIZE`), or a raw statement that takes the whole list as one
array parameter (`= ANY($1::text[])`).

This check only warns: it prints every finding as `path:line: kind message` and
exits 0, so the output is the inventory of lists still waiting for a bound.

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


def module_constants(tree: ast.Module) -> frozenset[str]:
    """Module-level names bound exactly once to a value of fixed size, in binding order
    so one constant may be built from another. Casing plays no part: an ALL_CAPS name
    that is imported or filled at runtime is as unbounded as any other."""
    bound: Final = tuple(binding for stmt in tree.body for binding in _module_binding(stmt))
    names: Final = tuple(name for name, _ in bound)
    rebound: Final = frozenset(name for name in names if names.count(name) > 1)

    def fold(constants: frozenset[str], binding: tuple[str, ast.expr]) -> frozenset[str]:
        name, value = binding
        return constants | {name} if name not in rebound and has_fixed_size(value, constants) else constants

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
