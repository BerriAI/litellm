#!/usr/bin/env python3
"""Type-discipline checker: the rules ruff can't enforce.
 
Rules
-----
LIT001  Mutable collection in a type annotation, anywhere it appears: function
        parameters, return types, class attributes, locals, and module globals.
        Covers the builtins (dict/list/set, bare or parameterized), their typing
        aliases (Dict/List/...), the collections concretes (deque/defaultdict/...),
        and the mutable ABCs (MutableMapping/MutableSequence/MutableSet). A mutable
        collection lets whoever holds it grow or rewrite it after the fact; annotate
        a read-only view instead (Mapping/Sequence/AbstractSet/tuple[X, ...]/
        frozenset[X], or a frozen dataclass / NamedTuple / ReadOnly TypedDict) and
        build it functionally (comprehension / map, not append-in-a-loop).
        Suppress with `# mutable-ok: <reason>` on the offending line.
LIT002  Mutable-collection *construction*: a list/dict/set literal or comprehension, or
        a call to a mutable constructor (list/dict/set/deque/defaultdict/Counter/...).
        Catches the unannotated seed-then-mutate pattern LIT001 cannot see (`acc = []`).
        Build the value in one shot and freeze it: a `tuple`/`frozenset` wrapping a
        generator (`tuple(f(x) for x in xs)`), a tuple literal, or a frozen dataclass /
        NamedTuple / ReadOnly TypedDict. Generator expressions and `tuple`/`frozenset`
        calls are not construction and pass. Annotation-internal lists (`Callable[[int],
        str]`) are exempt. Suppress with `# mutable-ok: <reason>`.
LIT003  noqa suppression without rule codes or without a reason.
        Required shape: `# noqa: TID251  # <reason>`
LIT004  type/pyright/mypy ignore without bracketed codes or without a reason.
        Required shape: `# pyright: ignore[reportArgumentType]  # <reason>`
LIT005  A `# mutable-ok` / `# cast-ok` / `# guard-ok` / `# kwargs-ok`
        suppression without a reason.
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

LIT000  Setup failure: a target file could not be read, or contains a syntax error.
        Reported as a violation rather than crashing the run.
 
Usage
-----
    python check_type_discipline.py litellm/ tests/

Exit code 1 if any violation is found. Stdlib only.
"""
 
from __future__ import annotations
 
import ast
import io
import re
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterable, Iterator, Sequence
from typing import NamedTuple
 
# Mutable collection types, banned in *every* annotation. Name-based, so `dict`,
# `typing.Dict`, `collections.deque`, and `collections.abc.MutableMapping` all match
# however they were imported. The read-only interfaces (Mapping, Sequence, the
# immutable AbstractSet / `abc.Set`, Collection) and the immutable concretes (tuple,
# frozenset) are the escape hatch and are deliberately absent -- as is the bare name
# `Set`, which collides with the read-only `collections.abc.Set`.
MUTABLE_COLLECTIONS = frozenset((
    "dict", "list", "set",
    "Dict", "List", "DefaultDict", "OrderedDict", "Counter", "Deque", "ChainMap",
    "deque", "defaultdict",
    "MutableMapping", "MutableSequence", "MutableSet",
))

# Callables whose result is a fresh *mutable* collection (LIT002). `tuple` and
# `frozenset` are deliberately absent -- they are the wrappers you reach for, and
# a generator expression fed to them is the blessed one-shot build.
MUTABLE_CONSTRUCTORS = frozenset((
    "dict", "list", "set",
    "deque", "defaultdict", "OrderedDict", "Counter", "ChainMap",
))
# A *qualified* call (`x.deque()`) counts as construction only for names that are rarely
# method names; `dict`/`list`/`set` are dropped here because `.dict()` / `.set()` / `.list()`
# are common methods (e.g. pydantic's `model.dict()`), not collection construction. A
# qualified `collections.deque(...)` still counts.
QUALIFIED_CONSTRUCTORS = MUTABLE_CONSTRUCTORS - frozenset(("dict", "list", "set"))
UNSAFE_GUARDS = frozenset(("TypeGuard", "TypeIs"))
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
MUTABLE_OK_RE = re.compile(r"#\s*mutable-ok(?::\s*(?P<reason>.*))?")
CAST_OK_RE = re.compile(r"#\s*cast-ok(?::\s*(?P<reason>.*))?")
GUARD_OK_RE = re.compile(r"#\s*guard-ok(?::\s*(?P<reason>.*))?")
KWARGS_OK_RE = re.compile(r"#\s*kwargs-ok(?::\s*(?P<reason>.*))?")

# Suppression tokens that must each carry a reason (LIT005).
OK_SUPPRESSIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mutable-ok", MUTABLE_OK_RE),
    ("cast-ok", CAST_OK_RE),
    ("guard-ok", GUARD_OK_RE),
    ("kwargs-ok", KWARGS_OK_RE),
)
 
 
class Violation(NamedTuple):
    path: Path
    line: int
    code: str
    message: str
 
    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.code} {self.message}"
 
 
@dataclass(frozen=True, slots=True)
class Comments:
    """The lines carrying each valid `*-ok` suppression."""

    mutable_ok_lines: frozenset[int]
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
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        comment_toks = tuple((t.start[0], t.string) for t in tokens if t.type == tokenize.COMMENT)
    except (tokenize.TokenError, SyntaxError):
        # tokenize raises TokenError (EOF mid-construct) or a SyntaxError subclass
        # (IndentationError / TabError) on malformed source; defer to ast.parse below,
        # which re-raises and is reported as LIT000 rather than crashing the run.
        return Comments(frozenset(), frozenset(), frozenset(), frozenset()), ()

    def _lines_with(regex: re.Pattern[str]) -> frozenset[int]:
        return frozenset(line for line, text in comment_toks if _valid_ok(regex, text))

    return (
        Comments(
            mutable_ok_lines=_lines_with(MUTABLE_OK_RE),
            cast_ok_lines=_lines_with(CAST_OK_RE),
            guard_ok_lines=_lines_with(GUARD_OK_RE),
            kwargs_ok_lines=_lines_with(KWARGS_OK_RE),
        ),
        tuple(v for line, text in comment_toks for v in _comment_violations(path, line, text)),
    )
 
 
# --------------------------------------------------------------------------- #
 
 
def mutable_names_in(annotation: ast.expr) -> Iterator[str]:
    """Yield mutable-collection names anywhere inside an annotation expression.

    Matches bare names (`dict`, `MutableMapping`) and dotted access (`typing.Dict`,
    `collections.deque`, `collections.abc.MutableMapping`), descends through nesting
    (`Mapping[str, list[int]]`, `tuple[set[int], ...]`) and string forward references.
    """
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in MUTABLE_COLLECTIONS:
            yield node.id
        elif isinstance(node, ast.Attribute) and node.attr in MUTABLE_COLLECTIONS:
            yield node.attr
        elif isinstance(node, ast.Constant):
            value: object = node.value  # forward references arrive as string constants
            if isinstance(value, str):
                try:
                    inner = ast.parse(value, mode="eval").body
                except SyntaxError:
                    continue
                yield from mutable_names_in(inner)
 
 
def _mutable_ann(path: Path, line: int, name: str, where: str) -> Violation:
    return Violation(
        path, line, "LIT001",
        f"mutable `{name}` in {where}: a mutable collection can be grown or rewritten "
        f"by whoever holds it. Annotate a read-only view -- Mapping[...], Sequence[...], "
        f"AbstractSet[...], tuple[X, ...], frozenset[X], or a frozen dataclass / "
        f"NamedTuple / ReadOnly TypedDict -- and build it functionally, not by "
        f"append-in-a-loop (suppress: `# mutable-ok: <reason>`)",
    )


def _annotation_violations(
    path: Path, annotation: ast.expr | None, line: int, where: str, ok_lines: frozenset[int]
) -> Iterator[Violation]:
    if annotation is None or line in ok_lines:
        return
    yield from (_mutable_ann(path, line, name, where) for name in mutable_names_in(annotation))
 
 
def _function_violations(
    path: Path, node: ast.FunctionDef | ast.AsyncFunctionDef, comments: Comments
) -> Iterator[Violation]:
    mutable_ok = comments.mutable_ok_lines
    args = node.args
    for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
        yield from _annotation_violations(
            path, arg.annotation, arg.lineno, f"parameter `{arg.arg}` of `{node.name}`", mutable_ok
        )

    # *args is allowed when typed (it's just a tuple); ruff ANN002 forces the
    # annotation, so here we only add the LIT001 mutable-collection check on the element type.
    if args.vararg is not None:
        yield from _annotation_violations(
            path, args.vararg.annotation, args.vararg.lineno, f"`*args` of `{node.name}`", mutable_ok
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
            path, node.returns, node.returns.lineno, f"return type of `{node.name}`", mutable_ok
        )
 
 
def iter_annotation_violations(path: Path, tree: ast.AST, comments: Comments) -> Iterator[Violation]:
    # Every annotation is in scope: signatures (params / *args / return) plus every
    # `x: T` -- class attribute, local, or module global. The latter three are all
    # ast.AnnAssign, so one walk covers them; only the signature annotations (which
    # are not AnnAssign) need the dedicated helper.
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield from _function_violations(path, node, comments)
        elif isinstance(node, ast.AnnAssign):
            target = node.target.id if isinstance(node.target, ast.Name) else "<target>"
            yield from _annotation_violations(
                path, node.annotation, node.lineno,
                f"the type of `{target}`", comments.mutable_ok_lines,
            )


# --------------------------------------------------------------------------- #
# Unchecked casts (LIT006) and unverified narrowing predicates (LIT007)
# --------------------------------------------------------------------------- #


def _is_cast_call(node: ast.Call) -> bool:
    """`cast(...)` or `typing.cast(...)`, however the name was imported/aliased.

    Name-based like MUTABLE_COLLECTIONS: a stray method called `.cast()` is a rare
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
    # TypeGuard/TypeIs are legal only as a function's return annotation (`-> TypeGuard[int]`),
    # so the walk is confined to `node.returns`; a runtime name that merely happens to read
    # `TypeGuard` is not a narrowing predicate. ruff bans the import; this flags the use.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.returns is None:
            continue
        for sub in ast.walk(node.returns):
            name = (
                sub.id if isinstance(sub, ast.Name)
                else sub.attr if isinstance(sub, ast.Attribute)
                else None
            )
            if name in UNSAFE_GUARDS and sub.lineno not in comments.guard_ok_lines:
                yield Violation(
                    path, sub.lineno, "LIT007",
                    f"`{name}` narrowing predicate: the checker never verifies the body, so a "
                    f"wrong guard silently corrupts types; parse into a concrete type instead "
                    f"(suppress: `# guard-ok: <reason>`)",
                )
 
 
# --------------------------------------------------------------------------- #
# Mutable-collection construction (LIT002)
# --------------------------------------------------------------------------- #


def _annotations_of(node: ast.AST) -> tuple[ast.expr | None, ...]:
    """The annotation expressions a node carries (signatures and `x: T`)."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        params = (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg)
        return (*(p.annotation for p in params if p is not None), node.returns)
    if isinstance(node, ast.AnnAssign):
        return (node.annotation,)
    return ()


def _annotation_node_ids(tree: ast.AST) -> frozenset[int]:
    """ids() of every node living inside an annotation.

    A list display inside an annotation (`Callable[[int], str]`) is type syntax,
    not construction, so the LIT002 walk must skip those subtrees.
    """
    return frozenset(
        id(sub)
        for node in ast.walk(tree)
        for ann in _annotations_of(node)
        if ann is not None
        for sub in ast.walk(ann)
    )


def _construction_kind(node: ast.expr) -> str | None:
    """Human label if `node` builds a mutable collection, else None."""
    if isinstance(node, ast.List):
        return "list literal"
    if isinstance(node, ast.ListComp):
        return "list comprehension"
    if isinstance(node, ast.Set):
        return "set literal"
    if isinstance(node, ast.SetComp):
        return "set comprehension"
    if isinstance(node, ast.Dict):
        return "dict literal"
    if isinstance(node, ast.DictComp):
        return "dict comprehension"
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name) and func.id in MUTABLE_CONSTRUCTORS:
            return f"`{func.id}()` constructor"
        if isinstance(func, ast.Attribute) and func.attr in QUALIFIED_CONSTRUCTORS:
            return f"`{func.attr}()` constructor"
    return None


def iter_construction_violations(path: Path, tree: ast.AST, comments: Comments) -> Iterator[Violation]:
    in_annotation = _annotation_node_ids(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.expr) or id(node) in in_annotation:
            continue
        kind = _construction_kind(node)
        if kind is None or node.lineno in comments.mutable_ok_lines:
            continue
        yield Violation(
            path, node.lineno, "LIT002",
            f"mutable {kind}: this builds a collection that can be grown or rewritten. "
            f"Build it in one shot and freeze it -- a tuple/frozenset wrapping a generator "
            f"(`tuple(f(x) for x in xs)`), a tuple literal, or a frozen dataclass / NamedTuple "
            f"/ ReadOnly TypedDict (suppress: `# mutable-ok: <reason>`)",
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
        *iter_annotation_violations(path, tree, comments),
        *iter_cast_violations(path, tree, comments),
        *iter_guard_violations(path, tree, comments),
        *iter_construction_violations(path, tree, comments),
    )
 
 
def collect_paths(raw: Iterable[str]) -> Iterator[Path]:
    for item in raw:
        p = Path(item)
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
        elif p.suffix == ".py":
            yield p
 
 
def main(argv: Sequence[str]) -> int:
    paths = tuple(a for a in argv if not a.startswith("-"))
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
 