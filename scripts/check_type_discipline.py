#!/usr/bin/env python3
"""Type-discipline checker: the rules ruff can't enforce.
 
Rules
-----
LIT001  Coarse builtin annotation (dict/list/set, bare or parameterized) at an
        interface: function parameters, return types, or class-body attributes.
        Locals are intentionally not checked.
        Suppress with `# coarse-ok: <reason>` on the offending line.
LIT002  Retired: annotation *presence* on *args/**kwargs is ruff's job (ANN002/ANN003).
        The code is left unused (not reused) so older messages stay unambiguous.
LIT003  noqa suppression without rule codes or without a reason.
        Required shape: `# noqa: TID251  # <reason>`
LIT004  type/pyright/mypy ignore without bracketed codes or without a reason.
        Required shape: `# pyright: ignore[reportArgumentType]  # <reason>`
LIT005  A `# coarse-ok` / `# cast-ok` / `# guard-ok` / `# kwargs-ok` suppression
        without a reason.
LIT006  `cast(...)` call. typing.cast is an unchecked assertion (the moral equivalent
        of TypeScript's `as`); it lies to the type checker with zero runtime guarantee.
        Validate into a concrete frozen type at the boundary instead.
        Suppress with `# cast-ok: <reason>` on the call's first line.
LIT007  `TypeGuard[...]` / `TypeIs[...]` annotation. The narrowing predicate's body is
        never verified by the checker, so a wrong guard silently corrupts types.
        Prefer parsing into a concrete type. Suppress with `# guard-ok: <reason>`.
LIT008  `**kwargs` parameter. The keyword contract is erased and everything it carries
        is effectively Any. ruff can force it to be typed (ANN003) but can't ban the
        syntax. Declare explicit keyword params, or accept one frozen payload. `*args`,
        by contrast, is fine when typed (it's just a tuple). Suppress: `# kwargs-ok: <reason>`.
 
Usage
-----
    python check_type_discipline.py litellm/ tests/
    python check_type_discipline.py --changed-only file1.py file2.py
 
Exit code 1 if any violation is found. Stdlib only.
"""
 
from __future__ import annotations
 
import ast
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator, Mapping, Sequence
from typing import NamedTuple
 
BANNED_BUILTINS = frozenset({"dict", "list", "set"})
UNSAFE_GUARDS = frozenset({"TypeGuard", "TypeIs"})
MIN_REASON_LEN = 3
 
NOQA_RE = re.compile(
    r"#\s*noqa"
    r"(?P<colon>:\s*(?P<codes>[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*))?"
    r"(?P<rest>.*)",
    re.IGNORECASE,
)
IGNORE_RE = re.compile(
    r"#\s*(?:type|pyright|mypy):\s*ignore(?P<codes>\[[^\]]*\])?(?P<rest>.*)"
)
COARSE_OK_RE = re.compile(r"#\s*coarse-ok(?::\s*(?P<reason>.*))?")
CAST_OK_RE = re.compile(r"#\s*cast-ok(?::\s*(?P<reason>.*))?")
GUARD_OK_RE = re.compile(r"#\s*guard-ok(?::\s*(?P<reason>.*))?")
KWARGS_OK_RE = re.compile(r"#\s*kwargs-ok(?::\s*(?P<reason>.*))?")

# Suppression tokens that must each carry a reason (LIT005).
OK_SUPPRESSIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("coarse-ok", COARSE_OK_RE),
    ("cast-ok", CAST_OK_RE),
    ("guard-ok", GUARD_OK_RE),
    ("kwargs-ok", KWARGS_OK_RE),
) # coarse-ok: <reason>
 
 
class Violation(NamedTuple):
    path: Path
    line: int
    code: str
    message: str
 
    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"
 
 
@dataclass(frozen=True, slots=True)
class Comments:
    """Per-line comment text, plus the lines carrying each valid `*-ok` suppression."""

    # Mapping, not dict: keys are line numbers, genuinely dynamic. Read-only downstream.
    by_line: Mapping[int, str]
    coarse_ok_lines: frozenset[int]
    cast_ok_lines: frozenset[int]
    guard_ok_lines: frozenset[int]
    kwargs_ok_lines: frozenset[int]
 
 
# --------------------------------------------------------------------------- #
# Comment scanning (LIT003 / LIT004 / LIT005)
# --------------------------------------------------------------------------- #
 
 
def _reason_of(rest: str) -> str:
    return rest.strip().lstrip("#-").strip()

 
def _valid_ok(regex: re.Pattern[str], text: str) -> bool:
    """True iff `text` carries this suppression with a reason of usable length."""
    m = regex.search(text)
    return bool(m) and len((m.group("reason") or "").strip()) >= MIN_REASON_LEN


def _comment_violations(path: Path, line_no: int, text: str) -> Iterator[Violation]:
    """Pure: all LIT003/004/005 findings for one comment."""
    for token, regex in OK_SUPPRESSIONS:
        m = regex.search(text)
        if m and len((m.group("reason") or "").strip()) < MIN_REASON_LEN:
            yield Violation(path, line_no, "LIT005", f"{token} requires a reason: `# {token}: <reason>`")
 
    m = NOQA_RE.search(text)
    if m:
        if not m.group("codes"):
            yield Violation(path, line_no, "LIT003", "noqa requires rule codes: `# noqa: XXX123  # <reason>`")
        elif len(_reason_of(m.group("rest"))) < MIN_REASON_LEN:
            yield Violation(path, line_no, "LIT003", "noqa requires a reason: `# noqa: XXX123  # <reason>`")
 
    m = IGNORE_RE.search(text)
    if m:
        codes = m.group("codes")
        if not codes or codes == "[]":
            yield Violation(path, line_no, "LIT004",
                            "ignore requires codes: `# pyright: ignore[ruleName]  # <reason>`")
        elif len(_reason_of(m.group("rest"))) < MIN_REASON_LEN:
            yield Violation(path, line_no, "LIT004",
                            "ignore requires a reason: `# pyright: ignore[ruleName]  # <reason>`")
 
 
def scan_comments(path: Path, source: str) -> tuple[Comments, tuple[Violation, ...]]:
    try:
        tokens = tokenize.generate_tokens(iter(source.splitlines(keepends=True)).__next__)
        comment_toks = tuple((t.start[0], t.string) for t in tokens if t.type == tokenize.COMMENT)
    except tokenize.TokenError:
        return Comments({}, frozenset(), frozenset(), frozenset(), frozenset()), ()

    def _lines_with(regex: re.Pattern[str]) -> frozenset[int]:
        return frozenset(line for line, text in comment_toks if _valid_ok(regex, text))

    return (
        Comments(
            by_line={line: text for line, text in comment_toks},
            coarse_ok_lines=_lines_with(COARSE_OK_RE),
            cast_ok_lines=_lines_with(CAST_OK_RE),
            guard_ok_lines=_lines_with(GUARD_OK_RE),
            kwargs_ok_lines=_lines_with(KWARGS_OK_RE),
        ),
        tuple(v for line, text in comment_toks for v in _comment_violations(path, line, text)),
    )
 
 
# --------------------------------------------------------------------------- #
 
 
def banned_names_in(annotation: ast.expr) -> Iterator[str]:
    """Yield banned builtin names anywhere inside an annotation expression.
 
    Handles nesting (Optional[dict[str, str]], tuple[list[int], ...]) and
    string forward references.
    """
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in BANNED_BUILTINS:
            yield node.id
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                inner = ast.parse(node.value, mode="eval").body
            except SyntaxError:
                continue
            yield from banned_names_in(inner)
 
 
def _coarse(path: Path, line: int, name: str, where: str) -> Violation:
    return Violation(
        path, line, "LIT001",
        f"coarse `{name}` annotation in {where}; use a frozen dataclass, "
        f"NamedTuple, ReadOnly TypedDict, tuple[X, ...], frozenset[X], or Sequence[X] "
        f"(suppress: `# coarse-ok: <reason>`)",
    )
 
 
def _annotation_violations(
    path: Path, annotation: ast.expr | None, line: int, where: str, ok_lines: frozenset[int]
) -> Iterator[Violation]:
    if annotation is None or line in ok_lines:
        return
    yield from (_coarse(path, line, name, where) for name in banned_names_in(annotation))
 
 
def _function_violations(
    path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef, comments: Comments
) -> Iterator[Violation]:
    coarse_ok = comments.coarse_ok_lines
    args = node.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        yield from _annotation_violations(
            path, arg.annotation, arg.lineno, f"parameter `{arg.arg}` of `{node.name}`", coarse_ok
        )

    # *args is allowed when typed (it's just a tuple); ruff ANN002 forces the
    # annotation, so here we only add the LIT001 coarse check on the element type.
    if args.vararg is not None:
        yield from _annotation_violations(
            path, args.vararg.annotation, args.vararg.lineno, f"`*args` of `{node.name}`", coarse_ok
        )

    # **kwargs is banned outright (LIT008): it erases the keyword contract and forces
    # Any-typing on everything it carries. ruff can require it be typed (ANN003) but
    # cannot ban the syntax, so this rule does.
    if args.kwarg is not None and args.kwarg.lineno not in comments.kwargs_ok_lines:
        yield Violation(
            path, args.kwarg.lineno, "LIT008",
            f"`**{args.kwarg.arg}` is banned: it erases the keyword contract and forces "
            f"Any-typing; declare explicit keyword parameters, or accept one frozen payload "
            f"(frozen dataclass / NamedTuple / ReadOnly TypedDict) "
            f"(suppress: `# kwargs-ok: <reason>`)",
        )

    if node.returns is not None:
        yield from _annotation_violations(
            path, node.returns, node.returns.lineno, f"return type of `{node.name}`", coarse_ok
        )
 
 
def _class_violations(path: Path, node: ast.ClassDef, ok_lines: frozenset[int]) -> Iterator[Violation]:
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign):
            target = stmt.target.id if isinstance(stmt.target, ast.Name) else "<attr>"
            yield from _annotation_violations(
                path, stmt.annotation, stmt.lineno,
                f"attribute `{target}` of class `{node.name}`", ok_lines,
            )
 
 
def iter_interface_violations(path: Path, tree: ast.AST, comments: Comments) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _function_violations(path, node, comments)
        elif isinstance(node, ast.ClassDef):
            yield from _class_violations(path, node, comments.coarse_ok_lines)


# --------------------------------------------------------------------------- #
# Unchecked casts (LIT006) and unverified narrowing predicates (LIT007)
# --------------------------------------------------------------------------- #


def _is_cast_call(node: ast.Call) -> bool:
    """`cast(...)` or `typing.cast(...)`, however the name was imported/aliased.

    Name-based like BANNED_BUILTINS: a stray method called `.cast()` is a rare
    false positive, suppressible with `# cast-ok: <reason>`.
    """
    func = node.func
    return (isinstance(func, ast.Name) and func.id == "cast") or (
        isinstance(func, ast.Attribute) and func.attr == "cast"
    )


def iter_cast_violations(path: Path, tree: ast.AST, comments: Comments) -> Iterator[Violation]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_cast_call(node) and node.lineno not in comments.cast_ok_lines:
            yield Violation(
                path, node.lineno, "LIT006",
                "cast() is an unchecked assertion (the type checker takes it on faith); "
                "validate into a frozen dataclass/NamedTuple/ReadOnly TypedDict at the "
                "boundary instead (suppress: `# cast-ok: <reason>`)",
            )


def iter_guard_violations(path: Path, tree: ast.AST, comments: Comments) -> Iterator[Violation]:
    # The `from typing import TypeGuard` line is an ast.alias, not a Name/Attribute,
    # so only *uses* (e.g. `-> TypeGuard[int]`) are flagged here; ruff bans the import.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Name, ast.Attribute)):
            continue
        name = node.id if isinstance(node, ast.Name) else node.attr
        if name in UNSAFE_GUARDS and node.lineno not in comments.guard_ok_lines:
            yield Violation(
                path, node.lineno, "LIT007",
                f"`{name}` narrowing predicate: the checker never verifies the body, so a "
                f"wrong guard silently corrupts types; parse into a concrete type instead "
                f"(suppress: `# guard-ok: <reason>`)",
            )
 
 
# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
 
 
def check_file(path: Path) -> tuple[Violation, ...]:
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return (Violation(path, 0, "LIT000", f"could not read file: {exc}"),)
 
    comments, violations = scan_comments(path, source)
 
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return (*violations, Violation(path, exc.lineno or 0, "LIT000", f"syntax error: {exc.msg}"))
 
    return (
        *violations,
        *iter_interface_violations(path, tree, comments),
        *iter_cast_violations(path, tree, comments),
        *iter_guard_violations(path, tree, comments),
    )
 
 
def collect_paths(raw: Iterable[str]) -> Iterator[Path]:
    for item in raw:
        p = Path(item)
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
        elif p.suffix == ".py":
            yield p
 
 
def main(argv: Sequence[str]) -> int:
    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        print("usage: check_type_discipline.py <files-or-dirs>...", file=sys.stderr)
        return 2
 
    violations = sorted(v for path in collect_paths(paths) for v in check_file(path))
    for v in violations:
        print(v.render())
 
    if violations:
        print(f"\n{len(violations)} violation(s).", file=sys.stderr)
        return 1
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
 