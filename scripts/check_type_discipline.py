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
        generator (`tuple(f(x) for x in xs)`), a tuple literal, a frozen dataclass /
        NamedTuple, a TypedDict-annotated dict literal, or (if it really must be
        dynamic) a MappingProxyType wrapping a dict literal or comprehension. Generator
        expressions and freezing-wrapper calls (`tuple(...)`, `frozenset(...)`,
        `MappingProxyType(...)`) are not construction and pass, as does the value passed
        directly to a wrapper: it is frozen before it can escape, though anything
        mutable nested inside it still counts. Annotation-internal lists
        (`Callable[[int], str]`) are exempt. A dict literal whose assignment is
        annotated with a TypedDict (`x: Final[MyTD] = {...}`; bare `x: Final = {...}`
        does not qualify) is a fixed-shape build basedpyright checks key-by-key against
        fields LIT012 keeps ReadOnly, not a growable accumulator, so it is exempt along
        with the dict literals nested in it (nested TypedDict fields); any other
        construction inside still counts. Detection is name-based: Final/ClassVar/
        Optional (and Annotated's first argument) unwrap, a PEP 604 union
        (`MyTD | None`) qualifies through either arm, and any remaining named head
        outside the mutable collections and Mapping/Any/object is taken to be a
        TypedDict, since a dict literal assigned to any other named type would not
        survive basedpyright. Suppress with `# mutable-ok: <reason>`.
LIT003  noqa suppression without rule codes or without a reason.
        Required shape: `# noqa: TID251  # <reason>`
LIT004  pyright/mypy ignore without bracketed codes or without a reason.
        Required shape: `# pyright: ignore[reportArgumentType]  # <reason>`
LIT005  A `# mutable-ok` / `# cast-ok` / `# guard-ok` / `# kwargs-ok` /
        `# rebind-ok` / `# writable-ok` suppression without a reason.
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
LIT009  `# type: ignore` in any shape (bare, with codes, with a reason).
        pyrightconfig.json sets enableTypeIgnoreComments to false, so basedpyright
        never honors it: the comment is inert dead syntax that suppresses nothing.
        Use `# pyright: ignore[ruleName]  # <reason>` instead.
LIT010  Variable assignment without a `Final` declaration. Every local and module-level
        name bound by `=` (plain, augmented, or annotated) must be declared Final --
        `x: Final = ...`, `x: Final[T] = ...`, or a bare `x: Final[T]` declaration
        followed by a single deferred assignment (basedpyright polices reassignment
        and allows exactly one). A name that is never Final is an open invitation to
        rebind it later. Unpacking (`a, b = ...`) and walrus targets cannot carry
        Final, so their first binding is implicitly final: a later `=`-form binding
        of the name trips as an ordinary unannotated assignment, and a later unpack
        or walrus binding trips as a re-bind. A `global`/`nonlocal` statement counts
        as the name's first binding in that scope, so the same re-bind logic applies
        under it. Exempt: `for`/`with`/`except`/import/def bindings (including as
        re-binds -- a distinct statement form re-using a name is out of scope),
        assignments inside a `for`/`while` body (pyright forbids Final in a loop;
        this exemption applies everywhere, including under `global`), valueless
        declarations (`x: int` binds nothing), dunder names, `_`, class bodies,
        `TypeAlias` declarations, and module-level names in `litellm/__init__.py`:
        that namespace is the SDK's runtime-settable config surface (users follow
        the documented `litellm.api_key = ...` pattern and the proxy rebinds these
        via setattr), and the package ships py.typed, so a Final there turns every
        documented downstream assignment into a mypy error. Suppress a deliberately
        rebindable name with `# rebind-ok: <reason>` on each offending line.
LIT011  Function-argument mutation: a parameter that is re-bound (`param = ...`,
        `param += ...`, a `for`/`with`/unpacking/walrus target, `del param`, or a
        re-bind in a nested function under `nonlocal param`) or mutated in place
        (`param.attr = ...`, `param[k] = ...`, `del param[k]`, a `for`/`with`
        target like `for param.attr in ...`). Re-binding silently detaches the name
        from what the caller passed; in-place stores rewrite the caller's object at
        a distance. Bind a new name / build a new value instead. Lambda parameters
        count (a walrus can re-bind them). A function's decorators, defaults, and
        annotations are evaluated in the enclosing scope and are attributed there.
        `self`/`cls` are exempt from the in-place-store check (methods own their
        instance), not from re-binding. Method-call mutation (`param.append(x)`) is
        out of reach without type information; LIT001/LIT002 keep mutable collections
        off signatures instead. Suppress with `# rebind-ok: <reason>`.
LIT012  TypedDict field without a `ReadOnly[...]` qualifier. A writable key lets any
        holder of the payload rewrite it after construction; qualify every field with
        `ReadOnly[...]` (PEP 705), which nests freely with Required/NotRequired/
        Annotated in any order. Detection is name-based, like MUTABLE_COLLECTIONS:
        a class is a TypedDict when `TypedDict` appears among its bases or when it
        inherits, transitively within the same module, from a class that has it;
        the functional form (`X = TypedDict("X", {...})`) is checked too. A base
        imported from another module is out of reach without import resolution.
        Suppress with `# writable-ok: <reason>`.

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
import os
import re
import sys
import tokenize
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from collections.abc import Iterable, Iterator, Mapping, Sequence
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
FREEZING_WRAPPERS = frozenset(("tuple", "frozenset", "MappingProxyType"))
# Wrappers unwrapped when deciding whether an assignment's annotation names a
# TypedDict (the LIT002 dict-literal exemption); bare, they name no type. Annotated
# is handled separately: only its first argument is type syntax.
TYPEDDICT_ANNOTATION_WRAPPERS = frozenset(("Final", "ClassVar", "Optional"))
# Heads that can type a dict literal without being a TypedDict. Every other named
# head counts as one: a dict literal assigned to any other named type would not
# survive basedpyright, which is the second gate behind this name-based check.
NON_TYPEDDICT_HEADS = MUTABLE_COLLECTIONS | frozenset(("Mapping", "Any", "object"))
UNSAFE_GUARDS = frozenset(("TypeGuard", "TypeIs"))
READONLY_QUALIFIER = "ReadOnly"
# Qualifiers ReadOnly may nest under, in any order (PEP 705); for Annotated only the
# first argument is type syntax, the rest is metadata and never qualifies the field.
FIELD_QUALIFIER_WRAPPERS = frozenset(("Required", "NotRequired", "Annotated"))
TYPEDDICT_BASE = "TypedDict"
MIN_REASON_LEN = 3
 
NOQA_RE = re.compile(
    r"#\s*noqa"
    r"(?P<colon>:\s*(?P<codes>[A-Z]+[0-9]+(?:\s*,\s*[A-Z]+[0-9]+)*))?"
    r"(?P<rest>.*)",
    re.IGNORECASE,
)
TYPE_IGNORE_RE = re.compile(r"#\s*type:\s*ignore\b")
IGNORE_RE = re.compile(
    r"#\s*(?:pyright|mypy):\s*ignore(?P<codes>\[[^\]]*\])?(?P<rest>.*)"
)
MUTABLE_OK_RE = re.compile(r"#\s*mutable-ok(?::\s*(?P<reason>.*))?")
CAST_OK_RE = re.compile(r"#\s*cast-ok(?::\s*(?P<reason>.*))?")
GUARD_OK_RE = re.compile(r"#\s*guard-ok(?::\s*(?P<reason>.*))?")
KWARGS_OK_RE = re.compile(r"#\s*kwargs-ok(?::\s*(?P<reason>.*))?")
REBIND_OK_RE = re.compile(r"#\s*rebind-ok(?::\s*(?P<reason>.*))?")
WRITABLE_OK_RE = re.compile(r"#\s*writable-ok(?::\s*(?P<reason>.*))?")

# Suppression tokens that must each carry a reason (LIT005).
OK_SUPPRESSIONS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mutable-ok", MUTABLE_OK_RE),
    ("cast-ok", CAST_OK_RE),
    ("guard-ok", GUARD_OK_RE),
    ("kwargs-ok", KWARGS_OK_RE),
    ("rebind-ok", REBIND_OK_RE),
    ("writable-ok", WRITABLE_OK_RE),
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
    rebind_ok_lines: frozenset[int]
    writable_ok_lines: frozenset[int]
 
 
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
 
    if TYPE_IGNORE_RE.search(text):
        yield Violation(path, line_no, "LIT009",
                        "`# type: ignore` is inert (enableTypeIgnoreComments is false, so "
                        "basedpyright never honors it); use `# pyright: ignore[ruleName]  # <reason>`")

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
        return Comments(frozenset(), frozenset(), frozenset(), frozenset(), frozenset(), frozenset()), ()

    def _lines_with(regex: re.Pattern[str]) -> frozenset[int]:
        return frozenset(line for line, text in comment_toks if _valid_ok(regex, text))

    return (
        Comments(
            mutable_ok_lines=_lines_with(MUTABLE_OK_RE),
            cast_ok_lines=_lines_with(CAST_OK_RE),
            guard_ok_lines=_lines_with(GUARD_OK_RE),
            kwargs_ok_lines=_lines_with(KWARGS_OK_RE),
            rebind_ok_lines=_lines_with(REBIND_OK_RE),
            writable_ok_lines=_lines_with(WRITABLE_OK_RE),
        ),
        tuple(v for line, text in comment_toks for v in _comment_violations(path, line, text)),
    )
 
 
# --------------------------------------------------------------------------- #
 
 
def _head_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _is_literal_subscript(node: ast.AST) -> bool:
    if not isinstance(node, ast.Subscript):
        return False
    base: Final = node.value
    return (isinstance(base, ast.Name) and base.id == "Literal") or (
        isinstance(base, ast.Attribute) and base.attr == "Literal"
    )


def mutable_names_in(annotation: ast.AST) -> Iterator[str]:
    """Yield mutable-collection names anywhere inside an annotation expression.

    Matches bare names (`dict`, `MutableMapping`) and dotted access (`typing.Dict`,
    `collections.deque`, `collections.abc.MutableMapping`), descends through nesting
    (`Mapping[str, list[int]]`, `tuple[set[int], ...]`) and string forward references.
    Skips `Literal[...]` subtrees: their string arguments are values, not forward
    references, so `Literal["list"]` is not the `list` type.
    """
    if _is_literal_subscript(annotation):
        return
    if isinstance(annotation, ast.Name) and annotation.id in MUTABLE_COLLECTIONS:
        yield annotation.id
    elif isinstance(annotation, ast.Attribute) and annotation.attr in MUTABLE_COLLECTIONS:
        yield annotation.attr
    elif isinstance(annotation, ast.Constant):
        value: object = annotation.value  # forward references arrive as string constants
        if isinstance(value, str):
            try:
                inner = ast.parse(value, mode="eval").body
            except SyntaxError:
                return
            yield from mutable_names_in(inner)
    for child in ast.iter_child_nodes(annotation):
        yield from mutable_names_in(child)
 
 
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


def _is_freezing_wrapper(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id in FREEZING_WRAPPERS
    return (
        isinstance(func, ast.Attribute)
        and func.attr == "MappingProxyType"
        and isinstance(func.value, ast.Name)
        and func.value.id == "types"
    )


def _frozen_argument_ids(tree: ast.AST) -> frozenset[int]:
    """ids() of every expression passed directly to a freezing wrapper.

    `MappingProxyType({...})`, `frozenset({...})`, and `tuple([...])` freeze their
    argument before it can escape, so the literal inside is a one-shot build, not a
    mutable value anyone can grow later. Only the argument itself is exempt; a
    mutable collection nested inside it still trips LIT002. Only bare names (plus
    `types.MappingProxyType`) qualify, so an unrelated method that happens to share
    a wrapper's name cannot exempt its argument.
    """
    return frozenset(
        id(node.args[0])
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and len(node.args) == 1 and _is_freezing_wrapper(node.func)
    )


def _is_typeddict_annotation(annotation: ast.expr) -> bool:
    """True iff the annotation names a TypedDict, by the name-based heuristic.

    Final/ClassVar/Optional unwrap (as does Annotated's first argument, the only
    one that is type syntax), a PEP 604 union qualifies through either arm, string
    forward references are parsed, and whatever named head remains counts as a
    TypedDict unless it is a mutable collection or Mapping/Any/object -- the heads
    that can type a dict literal without being one. Bare wrappers
    (`x: Final = ...`) name no type and never qualify.
    """
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            inner = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False
        return _is_typeddict_annotation(inner)
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        return _is_typeddict_annotation(annotation.left) or _is_typeddict_annotation(annotation.right)
    if isinstance(annotation, ast.Subscript):
        head = _head_name(annotation.value)
        if head in TYPEDDICT_ANNOTATION_WRAPPERS:
            return _is_typeddict_annotation(annotation.slice)
        if head == "Annotated":
            first = annotation.slice.elts[0] if isinstance(annotation.slice, ast.Tuple) and annotation.slice.elts else None
            return first is not None and _is_typeddict_annotation(first)
        return head is not None and head not in NON_TYPEDDICT_HEADS
    name = _head_name(annotation)
    return (
        name is not None
        and name not in NON_TYPEDDICT_HEADS
        and name not in TYPEDDICT_ANNOTATION_WRAPPERS
        and name != "Annotated"
    )


def _typeddict_build_ids(tree: ast.AST) -> frozenset[int]:
    """ids() of every dict literal built under a TypedDict-annotated assignment.

    `x: Final[MyTD] = {...}` is a fixed-shape build: basedpyright checks each key
    against the declared fields, which LIT012 keeps ReadOnly, so nothing here is
    the seed-then-mutate accumulator LIT002 hunts. Dict literals nested in the
    value (nested TypedDict fields) share the exemption; any other construction
    inside it still counts, and a bare `x: Final = {...}` stays flagged.
    """
    return frozenset(
        id(sub)
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.value, ast.Dict)
        and _is_typeddict_annotation(node.annotation)
        for sub in ast.walk(node.value)
        if isinstance(sub, ast.Dict)
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
    frozen_arguments = _frozen_argument_ids(tree)
    typeddict_builds = _typeddict_build_ids(tree)
    for node in ast.walk(tree):
        if (
            not isinstance(node, ast.expr)
            or id(node) in in_annotation
            or id(node) in frozen_arguments
            or id(node) in typeddict_builds
        ):
            continue
        kind = _construction_kind(node)
        if kind is None or node.lineno in comments.mutable_ok_lines:
            continue
        yield Violation(
            path, node.lineno, "LIT002",
            f"mutable {kind}: this builds a collection that can be grown or rewritten. "
            f"Build it in one shot and freeze it -- a tuple/frozenset wrapping a generator "
            f"(`tuple(f(x) for x in xs)`), a tuple literal, a frozen dataclass / NamedTuple, "
            f"a TypedDict-annotated dict literal (`x: Final[MyTD] = {{...}}`), or (if it "
            f"really must be dynamic) a MappingProxyType wrapping a dict literal or "
            f"comprehension (suppress: `# mutable-ok: <reason>`)",
        )
 
 
# --------------------------------------------------------------------------- #
# Final-annotation discipline (LIT010) and argument immutability (LIT011)
# --------------------------------------------------------------------------- #

CONSTANT_DECLARATIONS = frozenset(("Final", "TypeAlias"))
CONFIG_SURFACE_PARTS = ("litellm", "__init__.py")
SELF_PARAMS = frozenset(("self", "cls"))
NESTED_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
ASSIGN_FORMS = frozenset(("assign", "annassign", "aug"))
IMPLICIT_FINAL_FORMS = frozenset(("unpack", "walrus"))
SCOPE_STATEMENT_FORMS = frozenset(("global", "nonlocal"))


class Binding(NamedTuple):
    name: str
    line: int
    form: str
    in_loop: bool


def _declares_constant(annotation: ast.expr) -> bool:
    root = annotation.value if isinstance(annotation, ast.Subscript) else annotation
    if isinstance(root, ast.Name):
        return root.id in CONSTANT_DECLARATIONS
    return isinstance(root, ast.Attribute) and root.attr in CONSTANT_DECLARATIONS


def _defaults(args: ast.arguments) -> tuple[ast.expr, ...]:
    return (*args.defaults, *(d for d in args.kw_defaults if d is not None))


def _annotations(args: ast.arguments) -> tuple[ast.expr, ...]:
    return tuple(
        p.annotation
        for p in (*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg)
        if p is not None and p.annotation is not None
    )


def _scope_body(scope: ast.AST) -> tuple[ast.AST, ...]:
    match scope:
        case ast.FunctionDef(body=body) | ast.AsyncFunctionDef(body=body):
            return tuple(body)
        case ast.Lambda(body=body):
            return (body,)
        case _:
            return tuple(ast.iter_child_nodes(scope))


def _enclosing_scope_parts(node: ast.AST) -> tuple[ast.expr, ...]:
    """The pieces of a nested scope's statement that Python evaluates in the scope
    DEFINING it: decorators, parameter defaults, annotations, class bases."""
    match node:
        case ast.FunctionDef() | ast.AsyncFunctionDef():
            returns = (node.returns,) if node.returns is not None else ()
            return (*node.decorator_list, *_defaults(node.args), *_annotations(node.args), *returns)
        case ast.Lambda():
            return _defaults(node.args)
        case ast.ClassDef():
            return (*node.decorator_list, *node.bases, *(k.value for k in node.keywords))
        case _:
            return ()


def _walk_scope(scope: ast.AST, in_loop: bool = False) -> Iterator[tuple[ast.AST, bool]]:
    """Depth-first over everything this scope evaluates, nested scopes excluded.

    A nested function's decorators, defaults, and annotations belong to THIS scope
    (Python evaluates them at def time, here), while its body does not; conversely
    this scope's own defaults/decorators belong to its parent and are skipped.
    `in_loop` marks nodes living under a `for`/`while` of this scope: basedpyright
    rejects a Final assignment there ("cannot be assigned within a loop"), so LIT010
    must not demand one.
    """
    for child in _scope_body(scope):
        yield from _walk_within(child, in_loop)


def _walk_within(node: ast.AST, in_loop: bool) -> Iterator[tuple[ast.AST, bool]]:
    yield node, in_loop
    if isinstance(node, NESTED_SCOPES):
        for part in _enclosing_scope_parts(node):
            yield from _walk_within(part, in_loop)
        return
    deeper = in_loop or isinstance(node, (ast.For, ast.AsyncFor, ast.While))
    for child in ast.iter_child_nodes(node):
        yield from _walk_within(child, deeper)


def _stored_targets(target: ast.expr) -> Iterator[ast.expr]:
    """The individual store sites inside an assignment target, unpacking flattened."""
    if isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _stored_targets(elt)
    elif isinstance(target, ast.Starred):
        yield from _stored_targets(target.value)
    else:
        yield target


def _bound_names(target: ast.expr) -> Iterator[tuple[str, int]]:
    for t in _stored_targets(target):
        if isinstance(t, ast.Name):
            yield t.id, t.lineno


def _node_bindings(node: ast.AST, in_loop: bool) -> Iterator[Binding]:
    match node:
        case ast.Assign(targets=targets):
            for target in targets:
                if isinstance(target, ast.Name):
                    yield Binding(target.id, target.lineno, "assign", in_loop)
                else:
                    yield from (Binding(n, line, "unpack", in_loop) for n, line in _bound_names(target))
        case ast.AnnAssign(target=ast.Name(id=name, lineno=line), annotation=annotation, value=value):
            if _declares_constant(annotation):
                yield Binding(name, line, "declared", in_loop)
            elif value is not None:
                yield Binding(name, line, "annassign", in_loop)
        case ast.AugAssign(target=ast.Name(id=name, lineno=line)):
            yield Binding(name, line, "aug", in_loop)
        case ast.For(target=target) | ast.AsyncFor(target=target):
            yield from (Binding(n, line, "other", in_loop) for n, line in _bound_names(target))
        case ast.withitem(optional_vars=ast.expr() as optional_vars):
            yield from (Binding(n, line, "other", in_loop) for n, line in _bound_names(optional_vars))
        case ast.ExceptHandler(name=str(name)):
            yield Binding(name, node.lineno, "other", in_loop)
        case ast.NamedExpr(target=ast.Name(id=name, lineno=line)):
            yield Binding(name, line, "walrus", in_loop)
        case ast.Import(names=aliases):
            yield from (
                Binding((a.asname or a.name).partition(".")[0], node.lineno, "other", in_loop)
                for a in aliases
            )
        case ast.ImportFrom(names=aliases):
            yield from (
                Binding(a.asname or a.name, node.lineno, "other", in_loop)
                for a in aliases
                if a.name != "*"
            )
        case ast.Delete(targets=targets):
            yield from (
                Binding(t.id, t.lineno, "other", in_loop) for t in targets if isinstance(t, ast.Name)
            )
        case ast.FunctionDef(name=name) | ast.AsyncFunctionDef(name=name) | ast.ClassDef(name=name):
            yield Binding(name, node.lineno, "other", in_loop)
        case ast.Global(names=names):
            yield from (Binding(name, node.lineno, "global", in_loop) for name in names)
        case ast.Nonlocal(names=names):
            yield from (Binding(name, node.lineno, "nonlocal", in_loop) for name in names)
        case ast.MatchAs(name=str(name)) | ast.MatchStar(name=str(name)):
            yield Binding(name, node.lineno, "other", in_loop)
        case ast.MatchMapping(rest=str(rest)):
            yield Binding(rest, node.lineno, "other", in_loop)
        case _:
            pass


def scope_bindings(scope: ast.AST) -> tuple[Binding, ...]:
    """Every name-binding event in `scope` itself.

    Comprehension targets bind their own scope and never surface here; a walrus
    inside a comprehension binds the enclosing scope (PEP 572) and does.
    """
    return tuple(b for node, in_loop in _walk_scope(scope) for b in _node_bindings(node, in_loop))


def iter_scopes(tree: ast.AST) -> Iterator[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node


def _function_params(node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) -> frozenset[str]:
    a = node.args
    return frozenset(
        p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs, a.vararg, a.kwarg) if p is not None
    )


def _exempt_final_name(name: str) -> bool:
    return name == "_" or (name.startswith("__") and name.endswith("__"))


def _first_binding_index(bindings: Sequence[Binding]) -> Mapping[str, int]:
    return {b.name: i for i, b in reversed(tuple(enumerate(bindings)))}


def _is_config_surface(path: Path) -> bool:
    """The SDK's runtime-settable config module: `litellm.<name> = ...` is the
    documented way to configure the library and the proxy rebinds these names via
    setattr, so with py.typed shipped a Final here breaks downstream mypy runs."""
    return path.parts[-2:] == CONFIG_SURFACE_PARTS


def iter_final_violations(path: Path, tree: ast.AST, comments: Comments) -> Iterator[Violation]:
    for scope in iter_scopes(tree):
        if isinstance(scope, ast.Module) and _is_config_surface(path):
            continue
        params = (
            _function_params(scope)
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
            else frozenset()
        )
        bindings = scope_bindings(scope)
        declared = frozenset(b.name for b in bindings if b.form == "declared")
        first = _first_binding_index(bindings)
        for i, b in enumerate(bindings):
            if b.name in declared or b.name in params or b.in_loop:
                continue
            if _exempt_final_name(b.name) or b.line in comments.rebind_ok_lines:
                continue
            if b.form in ASSIGN_FORMS:
                yield Violation(
                    path, b.line, "LIT010",
                    f"`{b.name}` is assigned without a Final declaration, leaving it open to "
                    f"rebinding: annotate `{b.name}: Final = ...` (or `Final[T]`, or a bare "
                    f"`{b.name}: Final[T]` declaration with a single deferred assignment); "
                    f"implicit type aliases must use `{b.name}: TypeAlias = ...` instead, since "
                    f"a Final alias no longer resolves in type expressions; if rebinding is the "
                    f"point, suppress with `# rebind-ok: <reason>`",
                )
            elif b.form in IMPLICIT_FINAL_FORMS and i > first[b.name]:
                yield Violation(
                    path, b.line, "LIT010",
                    f"`{b.name}` is re-bound here after an earlier binding: unpacking and "
                    f"walrus targets cannot carry Final, so their names are implicitly final; "
                    f"bind a fresh name instead, or suppress with `# rebind-ok: <reason>`",
                )


def _mutation_sites(scope: ast.AST) -> Iterator[tuple[str, int]]:
    for node, _in_loop in _walk_scope(scope):
        match node:
            case ast.Assign(targets=targets) | ast.Delete(targets=targets):
                stored = tuple(targets)
            case ast.AnnAssign(target=target) | ast.AugAssign(target=target):
                stored = (target,)
            case ast.For(target=target) | ast.AsyncFor(target=target):
                stored = (target,)
            case ast.withitem(optional_vars=ast.expr() as optional_vars):
                stored = (optional_vars,)
            case _:
                continue
        for target in stored:
            for site in _stored_targets(target):
                root = site
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                if isinstance(root, ast.Name) and root is not site:
                    yield root.id, site.lineno


class _EnclosingFunction(NamedTuple):
    name: str
    params: frozenset[str]


def _iter_param_scopes(
    node: ast.AST, enclosing: tuple[_EnclosingFunction, ...] = ()
) -> Iterator[tuple[ast.AST, tuple[_EnclosingFunction, ...]]]:
    for child in ast.iter_child_nodes(node):
        match child:
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                yield child, enclosing
                yield from _iter_param_scopes(
                    child, (*enclosing, _EnclosingFunction(child.name, _function_params(child)))
                )
            case ast.Lambda():
                yield child, enclosing
                yield from _iter_param_scopes(child, enclosing)
            case _:
                yield from _iter_param_scopes(child, enclosing)


def _param_owners(
    scope: ast.AST, bindings: Sequence[Binding], enclosing: Sequence[_EnclosingFunction]
) -> Mapping[str, str]:
    own_name = (
        scope.name if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<lambda>"
    )
    nonlocal_params = {
        b.name: owner.name
        for b in bindings
        if b.form == "nonlocal"
        for owner in (next((s for s in reversed(tuple(enclosing)) if b.name in s.params), None),)
        if owner is not None
    }
    return {**{p: own_name for p in _function_params(scope)}, **nonlocal_params}


def iter_param_violations(path: Path, tree: ast.AST, comments: Comments) -> Iterator[Violation]:
    for scope, enclosing in _iter_param_scopes(tree):
        bindings = scope_bindings(scope)
        owners = _param_owners(scope, bindings, enclosing)
        if not owners:
            continue
        for b in bindings:
            if b.form in SCOPE_STATEMENT_FORMS or b.name not in owners:
                continue
            if b.line in comments.rebind_ok_lines:
                continue
            yield Violation(
                path, b.line, "LIT011",
                f"parameter `{b.name}` of `{owners[b.name]}` is re-bound: the name silently "
                f"detaches from what the caller passed; bind a new name instead "
                f"(suppress: `# rebind-ok: <reason>`)",
            )
        for name, line in _mutation_sites(scope):
            if name not in owners or name in SELF_PARAMS or line in comments.rebind_ok_lines:
                continue
            yield Violation(
                path, line, "LIT011",
                f"parameter `{name}` of `{owners[name]}` is mutated in place: the caller's "
                f"object is rewritten at a distance; build and return a new value instead "
                f"(suppress: `# rebind-ok: <reason>`)",
            )


# --------------------------------------------------------------------------- #
# Writable TypedDict fields (LIT012)
# --------------------------------------------------------------------------- #


def _base_names(cls: ast.ClassDef) -> frozenset[str]:
    """The names of a class's bases; a subscripted base (`Foo[int]`) counts as `Foo`."""
    return frozenset(
        name
        for base in cls.bases
        for name in (_head_name(base.value if isinstance(base, ast.Subscript) else base),)
        if name is not None
    )


def _typeddict_classes(tree: ast.AST) -> tuple[ast.ClassDef, ...]:
    """ClassDefs that are TypedDicts: `TypedDict` among the bases, or -- transitively,
    within this module -- a base that is itself one of these classes. A base defined
    in another module is invisible here; that subclass goes unchecked."""
    classes = tuple(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    bases_of = {cls.name: _base_names(cls) for cls in classes}

    def expand(known: frozenset[str]) -> frozenset[str]:
        grown = known | frozenset(name for name, bases in bases_of.items() if bases & known)
        return grown if grown == known else expand(grown)

    names = expand(frozenset((TYPEDDICT_BASE,)))
    return tuple(cls for cls in classes if cls.name in names)


def _has_readonly_qualifier(annotation: ast.expr) -> bool:
    """True iff the annotation is `ReadOnly[...]`, possibly nested under
    Required/NotRequired/Annotated (in any order) or a string forward reference."""
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            inner = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return False
        return _has_readonly_qualifier(inner)
    if not isinstance(annotation, ast.Subscript):
        return False
    name = _head_name(annotation.value)
    if name == READONLY_QUALIFIER:
        return True
    if name not in FIELD_QUALIFIER_WRAPPERS:
        return False
    if name == "Annotated":
        if isinstance(annotation.slice, ast.Tuple) and annotation.slice.elts:
            return _has_readonly_qualifier(annotation.slice.elts[0])
        return False
    return _has_readonly_qualifier(annotation.slice)


class _Field(NamedTuple):
    owner: str
    name: str
    annotation: ast.expr
    line: int


def _class_fields(cls: ast.ClassDef) -> Iterator[_Field]:
    for stmt in cls.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            yield _Field(cls.name, stmt.target.id, stmt.annotation, stmt.lineno)


def _functional_fields(tree: ast.AST) -> Iterator[_Field]:
    """Fields of the functional form: `X = TypedDict("X", {"field": type, ...})`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _head_name(node.func) != TYPEDDICT_BASE:
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Dict):
            continue
        first = node.args[0]
        owner = first.value if isinstance(first, ast.Constant) and isinstance(first.value, str) else "<TypedDict>"
        for key, value in zip(node.args[1].keys, node.args[1].values):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                yield _Field(owner, key.value, value, value.lineno)


def iter_typeddict_violations(path: Path, tree: ast.AST, comments: Comments) -> Iterator[Violation]:
    fields = (
        *(f for cls in _typeddict_classes(tree) for f in _class_fields(cls)),
        *_functional_fields(tree),
    )
    for field in fields:
        if _has_readonly_qualifier(field.annotation) or field.line in comments.writable_ok_lines:
            continue
        yield Violation(
            path, field.line, "LIT012",
            f"TypedDict field `{field.name}` of `{field.owner}` is writable: any holder "
            f"of the payload can rewrite the key after construction. Qualify it as "
            f"`ReadOnly[...]` (PEP 705; nests freely with Required/NotRequired/Annotated) "
            f"(suppress: `# writable-ok: <reason>`)",
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
        *iter_final_violations(path, tree, comments),
        *iter_param_violations(path, tree, comments),
        *iter_typeddict_violations(path, tree, comments),
    )
 
 
def collect_paths(raw: Iterable[str]) -> Iterator[Path]:
    for item in raw:
        p = Path(item)
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
        elif p.suffix == ".py":
            yield p
 
 
PARALLEL_MIN_PATHS = 200
MAX_WORKERS = 8


def _worker_count(path_count: int) -> int:
    """1 when the run is too small to repay process startup, else one worker per
    core up to MAX_WORKERS."""
    if path_count < PARALLEL_MIN_PATHS:
        return 1
    return max(1, min(os.cpu_count() or 1, MAX_WORKERS))


def scan_paths(paths: Sequence[Path]) -> tuple[Violation, ...]:
    """check_file over every path. Pure per-file work, so it fans out across
    processes; callers sort, which is what keeps output order stable."""
    workers = _worker_count(len(paths))
    if workers == 1:
        return tuple(v for path in paths for v in check_file(path))
    with Pool(workers) as pool:
        return tuple(v for found in pool.imap_unordered(check_file, paths, chunksize=32) for v in found)


def main(argv: Sequence[str]) -> int:
    paths = tuple(a for a in argv if not a.startswith("-"))
    if not paths:
        print("usage: check_type_discipline.py <files-or-dirs>...", file=sys.stderr)
        return 2
 
    targets = tuple(collect_paths(paths))
    violations = sorted(scan_paths(targets))
    for v in violations:
        print(v.render())
 
    if violations:
        print(f"\n{len(violations)} violation(s).", file=sys.stderr)
        return 1
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
 