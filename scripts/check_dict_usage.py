#!/usr/bin/env python3
"""DICT001: expressions whose mypy-inferred type is a bare mutable dict.

LIT001 bans `dict` in annotations and LIT002 bans dict construction, but a
value can still flow in as an *inferred* `dict[...]` from an untyped helper, a
third-party API return, or a `.copy()` of another dict, and every read, call,
or subscript on it is mutable-dict usage neither AST rule can see. This
checker is the inference-powered complement: it runs mypy's build API over the
tree with ``export_types`` and flags the expression forms through which a dict
value is used (names, attribute reads, calls, subscripts, conditionals,
awaits, and walrus expressions) whenever the expression's proper type is
exactly ``builtins.dict`` or one of the `collections` dict variants.
TypedDicts, ``Mapping``, and ``MappingProxyType`` are distinct types and never
match, so the sanctioned immutable alternatives stay silent. Dict literals are
DictExpr nodes, which LIT002 already gates, so they are not re-flagged here.

The AST walk is hand-rolled instead of subclassing
``mypy.traverser.TraverserVisitor`` because mypy ships compiled with mypyc and
interpreted classes cannot inherit from compiled traits: the subclass
definition succeeds but instantiating it raises ``TypeError``. ``CHILD_ATTRS``
therefore maps every syntactic node class to the child attributes it owns,
``LEAF_NODES`` lists the classes with no syntactic children (semantic-only
nodes included, since semantic links like ``.node``/``.info``/``.analyzed``
must not be followed: they cross file boundaries and would double-count), and
a class in neither map fails loudly so a mypy upgrade that adds a node form
cannot be silently skipped. tests/test_litellm/test_check_dict_usage.py pins
the classification of every concrete node class.

Usage: python scripts/check_dict_usage.py [target-dir ...]
Targets default to the repo's `litellm` tree. Output lines look like
``path:line: DICT001 expression is typed as mutable builtins.dict``, matching
the shape scripts/dict_usage_gate.py parses.
"""

import json
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final, NamedTuple

from mypy import build
from mypy.find_sources import create_source_list
from mypy.nodes import (
    Argument,
    AssertStmt,
    AssertTypeExpr,
    AssignmentExpr,
    AssignmentStmt,
    AwaitExpr,
    Block,
    BreakStmt,
    BytesExpr,
    CallExpr,
    CastExpr,
    ClassDef,
    ComparisonExpr,
    ComplexExpr,
    ConditionalExpr,
    Context,
    ContinueStmt,
    Decorator,
    DelStmt,
    DictExpr,
    DictionaryComprehension,
    EllipsisExpr,
    EnumCallExpr,
    Expression,
    ExpressionStmt,
    FakeExpression,
    FakeInfo,
    FloatExpr,
    ForStmt,
    FuncDef,
    GeneratorExpr,
    GlobalDecl,
    IfStmt,
    Import,
    ImportAll,
    ImportFrom,
    IndexExpr,
    IntExpr,
    LambdaExpr,
    ListComprehension,
    ListExpr,
    MatchStmt,
    MemberExpr,
    MypyFile,
    NamedTupleExpr,
    NameExpr,
    NewTypeExpr,
    NonlocalDecl,
    OperatorAssignmentStmt,
    OpExpr,
    OverloadedFuncDef,
    ParamSpecExpr,
    PassStmt,
    PlaceholderNode,
    PromoteExpr,
    RaiseStmt,
    ReturnStmt,
    RevealExpr,
    SetComprehension,
    SetExpr,
    SliceExpr,
    StarExpr,
    StrExpr,
    SuperExpr,
    TempNode,
    TemplateStrExpr,
    TryStmt,
    TupleExpr,
    TypeAlias,
    TypeAliasExpr,
    TypeAliasStmt,
    TypeApplication,
    TypedDictExpr,
    TypeFormExpr,
    TypeInfo,
    TypeVarExpr,
    TypeVarTupleExpr,
    UnaryExpr,
    Var,
    WhileStmt,
    WithStmt,
    YieldExpr,
    YieldFromExpr,
)
from mypy.options import Options
from mypy.patterns import (
    AsPattern,
    ClassPattern,
    MappingPattern,
    OrPattern,
    SequencePattern,
    SingletonPattern,
    StarredPattern,
    ValuePattern,
)
from mypy.types import Instance, Type, get_proper_type

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
PYRIGHT_CONFIG: Final = REPO_ROOT / "pyrightconfig.json"
DEFAULT_TARGET: Final = "litellm"
FALLBACK_PYTHON_VERSION: Final = (3, 12)

DICT_FULLNAMES: Final = frozenset(
    (
        "builtins.dict",
        "collections.defaultdict",
        "collections.OrderedDict",
        "collections.Counter",
        "collections.ChainMap",
    )
)

FLAGGED_NODES: Final = (
    NameExpr,
    MemberExpr,
    CallExpr,
    IndexExpr,
    ConditionalExpr,
    AwaitExpr,
    AssignmentExpr,
)

CHILD_ATTRS: Final[dict[type[Context], tuple[str, ...]]] = {
    MypyFile: ("defs",),
    Block: ("body",),
    ExpressionStmt: ("expr",),
    AssignmentStmt: ("lvalues", "rvalue"),
    OperatorAssignmentStmt: ("lvalue", "rvalue"),
    WhileStmt: ("expr", "body", "else_body"),
    ForStmt: ("index", "expr", "body", "else_body"),
    ReturnStmt: ("expr",),
    AssertStmt: ("expr", "msg"),
    DelStmt: ("expr",),
    IfStmt: ("expr", "body", "else_body"),
    RaiseStmt: ("expr", "from_expr"),
    TryStmt: ("body", "types", "vars", "handlers", "else_body", "finally_body"),
    WithStmt: ("expr", "target", "body"),
    MatchStmt: ("subject", "patterns", "guards", "bodies"),
    TypeAliasStmt: ("name", "value"),
    FuncDef: ("arguments", "body"),
    LambdaExpr: ("arguments", "body"),
    OverloadedFuncDef: ("items", "impl"),
    Decorator: ("func", "decorators"),
    ClassDef: ("defs", "base_type_exprs", "metaclass", "decorators", "keywords"),
    Argument: ("initializer",),
    StarExpr: ("expr",),
    MemberExpr: ("expr",),
    CallExpr: ("callee", "args"),
    YieldFromExpr: ("expr",),
    YieldExpr: ("expr",),
    AwaitExpr: ("expr",),
    IndexExpr: ("base", "index"),
    UnaryExpr: ("expr",),
    AssignmentExpr: ("target", "value"),
    OpExpr: ("left", "right"),
    ComparisonExpr: ("operands",),
    SliceExpr: ("begin_index", "end_index", "stride"),
    CastExpr: ("expr",),
    AssertTypeExpr: ("expr",),
    RevealExpr: ("expr",),
    SuperExpr: ("call",),
    ListExpr: ("items",),
    DictExpr: ("items",),
    TupleExpr: ("items",),
    SetExpr: ("items",),
    GeneratorExpr: ("left_expr", "indices", "sequences", "condlists"),
    ListComprehension: ("generator",),
    SetComprehension: ("generator",),
    DictionaryComprehension: ("key", "value", "indices", "sequences", "condlists"),
    ConditionalExpr: ("cond", "if_expr", "else_expr"),
    TypeApplication: ("expr",),
    TemplateStrExpr: ("items",),
    AsPattern: ("pattern", "name"),
    OrPattern: ("patterns",),
    ValuePattern: ("expr",),
    SequencePattern: ("patterns",),
    StarredPattern: ("capture",),
    MappingPattern: ("keys", "values", "rest"),
    ClassPattern: ("class_ref", "positionals", "keyword_values"),
}

LEAF_NODES: Final = frozenset(
    (
        NameExpr,
        IntExpr,
        StrExpr,
        BytesExpr,
        FloatExpr,
        ComplexExpr,
        EllipsisExpr,
        Import,
        ImportFrom,
        ImportAll,
        PassStmt,
        BreakStmt,
        ContinueStmt,
        GlobalDecl,
        NonlocalDecl,
        TempNode,
        Var,
        FakeExpression,
        PlaceholderNode,
        TypeVarExpr,
        ParamSpecExpr,
        TypeVarTupleExpr,
        TypeAliasExpr,
        NamedTupleExpr,
        TypedDictExpr,
        EnumCallExpr,
        NewTypeExpr,
        PromoteExpr,
        TypeFormExpr,
        TypeInfo,
        FakeInfo,
        TypeAlias,
        SingletonPattern,
    )
)


class Violation(NamedTuple):
    path: str
    line: int
    type_fullname: str


def _child_nodes(value: object) -> Iterator[Context]:
    if isinstance(value, Context):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _child_nodes(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _child_nodes(item)


def iter_nodes(root: MypyFile) -> Iterator[Context]:
    stack: Final[list[Context]] = [root]
    while stack:
        node = stack.pop()
        yield node
        attrs = CHILD_ATTRS.get(type(node))
        if attrs is None:
            if type(node) in LEAF_NODES:
                continue
            raise SystemExit(
                f"check_dict_usage: unmapped mypy node class "
                f"{type(node).__module__}.{type(node).__qualname__}; classify it in "
                f"CHILD_ATTRS or LEAF_NODES"
            )
        for attr in attrs:
            stack.extend(_child_nodes(getattr(node, attr)))


def dict_fullname(typ: Type | None) -> str | None:
    if typ is None:
        return None
    proper: Final = get_proper_type(typ)
    if isinstance(proper, Instance) and proper.type.fullname in DICT_FULLNAMES:
        return proper.type.fullname
    return None


def module_violations(tree: MypyFile, path: str, types: dict[Expression, Type]) -> list[Violation]:
    found: Final[list[Violation]] = []
    for node in iter_nodes(tree):
        if not isinstance(node, FLAGGED_NODES) or node.line <= 0:
            continue
        fullname = dict_fullname(types.get(node))
        if fullname is not None:
            found.append(Violation(path, node.line, fullname))
    return found


def parse_python_version(config_text: str) -> tuple[int, int] | None:
    try:
        config = json.loads(config_text)
    except ValueError:
        return None
    version: Final = config.get("pythonVersion") if isinstance(config, dict) else None
    if not isinstance(version, str):
        return None
    major, _, minor = version.partition(".")
    if not (major.isdigit() and minor.isdigit()):
        return None
    return (int(major), int(minor))


def configured_python_version() -> tuple[int, int]:
    try:
        text = PYRIGHT_CONFIG.read_text()
    except OSError:
        return FALLBACK_PYTHON_VERSION
    return parse_python_version(text) or FALLBACK_PYTHON_VERSION


def build_options(python_version: tuple[int, int]) -> Options:
    options: Final = Options()
    options.export_types = True
    options.preserve_asts = True
    options.ignore_errors = True
    options.follow_imports = "normal"
    options.ignore_missing_imports = True
    options.check_untyped_defs = True
    options.incremental = False
    options.python_version = python_version
    return options


def _swallow_errors(path: str | None, lines: list[str], is_serious: bool) -> None:
    pass


def collect_violations(targets: Sequence[Path], python_version: tuple[int, int]) -> list[Violation]:
    roots: Final = tuple(target.resolve() for target in targets)
    options: Final = build_options(python_version)
    sources: Final = create_source_list([str(root) for root in roots], options)
    result: Final = build.build(sources, options, flush_errors=_swallow_errors)
    found: Final[list[Violation]] = []
    for state in result.graph.values():
        if state.tree is None or not state.path:
            continue
        path = Path(state.path).resolve()
        if not any(root == path or root in path.parents for root in roots):
            continue
        found.extend(module_violations(state.tree, path.as_posix(), result.types))
    return sorted(found)


def main(argv: Sequence[str]) -> int:
    raw_targets: Final = tuple(arg for arg in argv if not arg.startswith("-"))
    targets: Final = tuple(Path(raw) for raw in (raw_targets or (str(REPO_ROOT / DEFAULT_TARGET),)))
    violations: Final = collect_violations(targets, configured_python_version())
    for violation in violations:
        print(f"{violation.path}:{violation.line}: DICT001 expression is typed as mutable {violation.type_fullname}")
    if violations:
        print(f"\n{len(violations)} DICT001 violation(s)", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
