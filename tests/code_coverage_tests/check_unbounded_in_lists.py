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
`delete_many` filter whole. `litellm.repositories.chunked_in` chunks an `in` list for
find_many / count / update_many / delete_many and is itself exempt. Record a real bound with
`# bounded-ok: <reason>` on the reported line or alone on the line above; the reason is required.

Findings that predate the check are grandfathered in `unbounded_in_baseline.txt`, keyed by
path, enclosing function or class, kind, field and occurrence within that scope, so moving
code up or down a file keeps its entry. The run fails on a finding the baseline lacks and on
a baseline entry no finding matches, so the baseline only shrinks. `--update-baseline`
rewrites the entries under the scanned targets.

Usage: python check_unbounded_in_lists.py [--update-baseline] [--baseline FILE] [files-or-dirs...]
       (default targets: litellm enterprise)
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Final

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS: Final = ("litellm", "enterprise")
DEFAULT_BASELINE: Final = Path(__file__).resolve().with_name("unbounded_in_baseline.txt")
EXEMPT_PATHS: Final = frozenset({"litellm/repositories/chunked_in.py"})
MODULE_SCOPE: Final = "<module>"
BASELINE_HEADER: Final = (
    "# Grandfathered findings of check_unbounded_in_lists.py: path::scope::kind::subject::occurrence.\n"
    "# Fix a site and delete its line; regenerate with `check_unbounded_in_lists.py --update-baseline`.\n"
)

MEMBERSHIP_KEYS: Final = frozenset({"in", "not_in", "notIn"})
TYPED_DICT_MODULES: Final = frozenset({"typing", "typing_extensions"})
CONSTANT_WRAPPERS: Final = frozenset({"list", "tuple", "sorted", "frozenset", "set"})
FREEZING_WRAPPERS: Final = frozenset({"tuple", "frozenset"})
MIN_REASON_LEN: Final = 3

MARKER: Final = re.compile(r"#\s*bounded-ok(?::[ \t]*(?P<reason>[^#]*))?")
# The text right after `IN (` is where a runtime value lands: an f-string or format
# slot (`{x}`, never the escaped `{{`), a `%` slot, or the end of the literal itself.
SPLICED_IN: Final = re.compile(r"\bIN\s*\(\s*(?:\{(?!\{)|%s\b|%\(|$)", re.IGNORECASE)
IN_OPERAND: Final = re.compile(r"(\S+)\s+(?:NOT\s+)?$", re.IGNORECASE)
STRING_PREFIX_AND_QUOTES: Final = re.compile(r"^[rbfuRBFU]{0,2}(?=[\"'])|[\\\"']")
CLOSING_QUOTES: Final = re.compile(r"(?:\"\"\"|'''|\"|')$")


@dataclass(frozen=True, slots=True)
class Finding:
    path: Path
    line: int
    kind: str
    message: str
    scope: str = MODULE_SCOPE
    subject: str = ""

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


def _fixed_element(element: ast.expr, constants: frozenset[str]) -> bool:
    match element:
        case ast.Starred(value=value):
            return has_fixed_size(value, constants)
        case _:
            return True


def has_fixed_size(value: ast.expr, constants: frozenset[str]) -> bool:
    """Whether the value's length is visible in the source rather than decided at runtime."""
    match value:
        case ast.List(elts=elts) | ast.Tuple(elts=elts) | ast.Set(elts=elts):
            return all(_fixed_element(elt, constants) for elt in elts)
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
            return all(_fixed_element(elt, constants) for elt in elts)
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


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int
    qualname: str


def _spans(node: ast.AST, prefix: str) -> Iterator[Span]:
    for child in ast.iter_child_nodes(node):
        match child:
            case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
                yield Span(child.lineno, child.end_lineno or child.lineno, prefix + name)
                yield from _spans(child, f"{prefix}{name}.")
            case _:
                yield from _spans(child, prefix)


def scope_finder(tree: ast.AST) -> Callable[[int], str]:
    """The innermost function or class around a line, dotted like a qualname, else `<module>`."""
    spans: Final = tuple(_spans(tree, ""))

    def scope_of(line: int) -> str:
        enclosing: Final = tuple(span for span in spans if span.start <= line <= span.end)
        return max(enclosing, key=lambda span: (span.start, -span.end)).qualname if enclosing else MODULE_SCOPE

    return scope_of


def _field_name(key: ast.expr) -> str:
    match key:
        case ast.Constant(value=str(name)):
            return name
        case _:
            return f"[{ast.unparse(key)}]"


def _field_bindings(node: ast.AST) -> Iterator[tuple[str, ast.expr]]:
    """Where a dict literal is written as a field's filter: `{field: {...}}`, `where[field] = {...}`
    or `Filter(field={...})`. A computed field reads as `[expr]`."""
    match node:
        case ast.Dict(keys=keys, values=values):
            yield from ((_field_name(key), value) for key, value in zip(keys, values) if key is not None)
        case ast.Assign(targets=[ast.Subscript(slice=key)], value=value):
            yield (_field_name(key), value)
        case ast.Call(keywords=keywords):
            yield from ((keyword.arg, keyword.value) for keyword in keywords if keyword.arg is not None)
        case _:
            return


def _filtered_fields(tree: ast.AST) -> Mapping[int, str]:
    """id() of each dict literal written as a field's filter, mapped to that field."""
    return {
        id(value): field
        for node in ast.walk(tree)
        for field, value in _field_bindings(node)
        if isinstance(value, ast.Dict)
    }


def _is_typed_dict(func: ast.expr) -> bool:
    match func:
        case ast.Name(id="TypedDict"):
            return True
        case ast.Attribute(value=ast.Name(id=module), attr="TypedDict"):
            return module in TYPED_DICT_MODULES
        case _:
            return False


def _typed_dict_field_map(node: ast.AST) -> ast.expr | None:
    """The field map of a functional `TypedDict("Name", {...})`, whose keys are field names, not filters."""
    match node:
        case ast.Call(func=func, args=[_, fields, *_]) if _is_typed_dict(func):
            return fields
        case ast.Call(func=func, keywords=keywords) if _is_typed_dict(func):
            return next((keyword.value for keyword in keywords if keyword.arg == "fields"), None)
        case _:
            return None


def _typed_dict_field_maps(tree: ast.AST) -> frozenset[int]:
    """id() of each dict literal passed as a functional TypedDict's field map."""
    return frozenset(id(fields) for fields in map(_typed_dict_field_map, ast.walk(tree)) if fields is not None)


def _prisma_advice(key: str) -> str:
    if key == "in":
        return (
            "Chunk it with `litellm.repositories.chunked_in` (find_many_in / count_in / update_many_in / "
            "delete_many_in)"
        )
    return "A negated list cannot be chunked: use `<> ALL($1::text[])` in raw SQL or a relation filter"


def prisma_findings(path: Path, tree: ast.Module) -> Iterator[Finding]:
    constants: Final = module_constants(tree)
    scope_of: Final = scope_finder(tree)
    fields: Final = _filtered_fields(tree)
    typed_dict_field_maps: Final = _typed_dict_field_maps(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict) or id(node) in typed_dict_field_maps:
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
                f"parameter per value and Postgres caps a statement at 32,767. {_prisma_advice(key.value)}, "
                f"or record the bound with `# bounded-ok: <reason>`",
                scope=scope_of(key.lineno),
                subject=f"{fields.get(id(node), '?')}.{key.value}",
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
    scope_of: Final = scope_finder(tree)
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
        operand = IN_OPERAND.search(body[: match.start()])
        yield Finding(
            path,
            node.lineno,
            "raw-sql",
            f"`IN (` takes a list spliced in at runtime{where}: it binds one parameter per value and Postgres "
            f"caps a statement at 32,767. Pass the list as one array parameter (`= ANY($1::text[])`, or "
            f"`<> ALL($1::text[])` for `NOT IN`), or record the bound with `# bounded-ok: <reason>`",
            scope=scope_of(node.lineno),
            subject=f"{STRING_PREFIX_AND_QUOTES.sub('', operand.group(1)) if operand else '?'}.IN",
        )


def marker_findings(path: Path, markers: Markers, scope_of: Callable[[int], str]) -> Iterator[Finding]:
    for line, marker in sorted(markers.by_line.items()):
        if not marker.valid:
            yield Finding(
                path,
                line,
                "marker",
                "`# bounded-ok` needs a reason naming the bound: `# bounded-ok: <reason>`",
                scope=scope_of(line),
                subject="bounded-ok",
            )


def check_file(path: Path) -> tuple[Finding, ...]:
    try:
        source: Final = path.read_text(encoding="utf-8")
        tree: Final = ast.parse(source, filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return (Finding(path, getattr(exc, "lineno", None) or 0, "unreadable", str(exc)),)
    markers: Final = read_markers(source)
    return (
        *marker_findings(path, markers, scope_finder(tree)),
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


def repo_relative(path: Path) -> str:
    resolved: Final = path.resolve()
    return resolved.relative_to(REPO_ROOT).as_posix() if resolved.is_relative_to(REPO_ROOT) else resolved.as_posix()


def scan(paths: Iterable[Path]) -> tuple[Finding, ...]:
    return tuple(
        sorted(
            (f for path in paths if repo_relative(path) not in EXEMPT_PATHS for f in check_file(path)),
            key=lambda f: (str(f.path), f.line, f.kind),
        )
    )


def identify(findings: tuple[Finding, ...]) -> Mapping[str, Finding]:
    """Each finding keyed by `path scope kind subject occurrence`, the occurrence counting the
    earlier findings in the same file that share the rest of the key. No line number goes in,
    so code shifting up or down leaves the key alone."""
    ordered: Final = sorted(findings, key=lambda f: (str(f.path), f.line))
    keys: Final = tuple(f"{repo_relative(f.path)} {f.scope} {f.kind} {f.subject or '-'}" for f in ordered)
    return {f"{key} {keys[:index].count(key)}": finding for index, (key, finding) in enumerate(zip(keys, ordered))}


def read_baseline(path: Path) -> frozenset[str]:
    if not path.exists():
        return frozenset()
    return frozenset(
        stripped
        for line in path.read_text(encoding="utf-8").splitlines()
        for stripped in (line.strip(),)
        if stripped and not stripped.startswith("#")
    )


def covered_by(targets: tuple[str, ...]) -> Callable[[str], bool]:
    """Whether a baseline entry's file lies under one of the scanned targets."""
    roots: Final = tuple(repo_relative(Path(target)) for target in targets)

    def covers(entry: str) -> bool:
        entry_path: Final = entry.split(" ", 1)[0]
        return any(entry_path == root or entry_path.startswith(f"{root}/") for root in roots)

    return covers


@dataclass(frozen=True, slots=True)
class Options:
    targets: tuple[str, ...]
    baseline: Path
    update_baseline: bool


def parse_options(argv: Iterable[str]) -> Options:
    parser: Final = argparse.ArgumentParser(description="Fail on SQL IN lists with no written bound.")
    parser.add_argument("targets", nargs="*", default=list(DEFAULT_TARGETS))
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--update-baseline", action="store_true")
    namespace: Final = parser.parse_args(list(argv))
    return Options(
        targets=tuple(str(target) for target in namespace.targets),
        baseline=Path(str(namespace.baseline)),
        update_baseline=bool(namespace.update_baseline),
    )


def main(argv: Iterable[str]) -> int:
    options: Final = parse_options(argv)
    findings: Final = scan(collect_paths(options.targets))
    current: Final = identify(findings)
    baseline: Final = read_baseline(options.baseline)
    covers: Final = covered_by(options.targets)
    if options.update_baseline:
        entries: Final = sorted({*(entry for entry in baseline if not covers(entry)), *current})
        options.baseline.write_text(BASELINE_HEADER + "".join(f"{entry}\n" for entry in entries), encoding="utf-8")
        print(f"Wrote {len(entries)} baseline entries to {options.baseline}")
        return 0
    new: Final = tuple(finding for key, finding in current.items() if key not in baseline)
    stale: Final = sorted(entry for entry in baseline if covers(entry) and entry not in current)
    for finding in new:
        print(finding.render())
    for entry in stale:
        print(f"{options.baseline}: stale entry `{entry}`: no finding matches it any more, delete the line")
    counts: Final = {
        kind: sum(1 for f in findings if f.kind == kind) for kind in ("prisma", "raw-sql", "marker", "unreadable")
    }
    summary: Final = ", ".join(f"{count} {kind}" for kind, count in counts.items() if count)
    print(
        f"\n{len(findings)} unbounded IN list(s) ({summary or 'none'}): {len(findings) - len(new)} baselined, "
        f"{len(new)} new, {len(stale)} stale baseline entries."
    )
    return 1 if new or stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
