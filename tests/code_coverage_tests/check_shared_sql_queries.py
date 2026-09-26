#!/usr/bin/env python3
"""Hold every proxy SQL statement to the shared query file pattern.

See litellm/proxy/db/queries/AGENTS.md. Each file there is loaded by its domain package,
used by Python, and declared in litellm-rust/crates/db/src/queries/<domain>.rs so sqlx
checks it against the Prisma-migrated schema. What has not moved yet is ratcheted: inline
SQL handed to a raw Prisma call and Prisma ORM table calls may only go down, so lower the
ceiling in the same change that moves some. A raw call whose SQL genuinely cannot be static
carries a `# dynamic-sql-ok: <reason>` marker instead.
"""

import ast
import io
import re
import sys
import tokenize
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, NamedTuple, TypeAlias

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
QUERIES_DIR: Final = REPO_ROOT / "litellm" / "proxy" / "db" / "queries"
RUST_QUERIES_DIR: Final = REPO_ROOT / "litellm-rust" / "crates" / "db" / "src" / "queries"
SCANNED_DIRS: Final = (REPO_ROOT / "litellm", REPO_ROOT / "enterprise")
QUERIES_MODULE: Final = "litellm.proxy.db.queries"
ORM_OPERATIONS: Final = frozenset(
    {
        "find_unique",
        "find_unique_or_raise",
        "find_first",
        "find_first_or_raise",
        "find_many",
        "create",
        "create_many",
        "update",
        "update_many",
        "upsert",
        "delete",
        "delete_many",
        "count",
        "group_by",
    }
)
RUST_QUERY_FILE: Final = re.compile(r'"\.\./\.\./\.\./litellm/proxy/db/queries/(\w+)/(\w+)\.sql"')
DYNAMIC_SQL_MARKER: Final = re.compile(r"#\s*dynamic-sql-ok:\s*\S")
INLINE_RAW_SQL_CEILING: Final = 193
ORM_CALL_CEILING: Final = 385

Scope: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
Function: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef


class SqlSlot(NamedTuple):
    position: int
    keyword: str


RawMethods: TypeAlias = MappingProxyType[str, SqlSlot]

PRISMA_RAW_METHODS: Final[RawMethods] = MappingProxyType(
    {name: SqlSlot(0, "query") for name in ("query_raw", "execute_raw", "query_first")}
)


def _parameters(scope: Scope) -> frozenset[str]:
    arguments: Final = scope.args
    variadic: Final = tuple(arg for arg in (arguments.vararg, arguments.kwarg) if arg is not None)
    return frozenset(arg.arg for arg in chain(arguments.posonlyargs, arguments.args, arguments.kwonlyargs, variadic))


def _parameters_inside(node: ast.AST, enclosing: frozenset[str]) -> frozenset[str]:
    return _parameters(node) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda) else enclosing


def _calls_in_scope(node: ast.AST, parameters: frozenset[str]) -> Iterator[tuple[ast.Call, frozenset[str]]]:
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.Call):
            yield child, parameters
        yield from _calls_in_scope(child, _parameters_inside(child, parameters))


def _shared_query_names(module: ast.Module) -> frozenset[str]:
    imports: Final = (
        node
        for node in ast.walk(module)
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(f"{QUERIES_MODULE}.")
    )
    return frozenset(alias.asname or alias.name for alias in chain.from_iterable(node.names for node in imports))


def _method_name(call: ast.Call) -> str | None:
    match call.func:
        case ast.Attribute(attr=name) | ast.Name(id=name):
            return name
        case _:
            return None


def _sql_argument(call: ast.Call, slot: SqlSlot) -> ast.expr | None:
    if len(call.args) > slot.position:
        return call.args[slot.position]
    return next((keyword.value for keyword in call.keywords if keyword.arg == slot.keyword), None)


def _raw_slot(call: ast.Call, methods: RawMethods) -> SqlSlot | None:
    name: Final = _method_name(call)
    return methods.get(name) if name is not None else None


def _positional_parameters(function: Function) -> tuple[str, ...]:
    names: Final = tuple(arg.arg for arg in (*function.args.posonlyargs, *function.args.args))
    return names[1:] if names[:1] in (("self",), ("cls",)) else names


def _raw_sql_arguments(function: Function, methods: RawMethods) -> Iterator[ast.expr | None]:
    calls: Final = (node for node in ast.walk(function) if isinstance(node, ast.Call))
    return (_sql_argument(call, slot) for call in calls if (slot := _raw_slot(call, methods)) is not None)


def _forwarded_slot(function: Function, methods: RawMethods) -> SqlSlot | None:
    positional: Final = _positional_parameters(function)
    forwarded: Final = (
        argument.id
        for argument in _raw_sql_arguments(function, methods)
        if isinstance(argument, ast.Name) and argument.id in positional
    )
    name: Final = next(forwarded, None)
    return SqlSlot(positional.index(name), name) if name is not None else None


def _raw_methods(modules: tuple[ast.Module, ...], known: RawMethods = PRISMA_RAW_METHODS) -> RawMethods:
    functions: Final = (
        node
        for node in chain.from_iterable(ast.walk(module) for module in modules)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name not in known
    )
    found: Final = {
        function.name: slot for function in functions if (slot := _forwarded_slot(function, known)) is not None
    }
    return _raw_methods(modules, MappingProxyType({**known, **found})) if found else known


def _is_inline(argument: ast.expr | None, known_names: frozenset[str]) -> bool:
    match argument:
        case None | ast.Starred():
            return False
        case ast.Name(id=name):
            return name not in known_names
        case _:
            return True


def _is_orm_call(call: ast.Call) -> bool:
    match call.func:
        case ast.Attribute(attr=operation, value=ast.Attribute(attr=table)) if operation in ORM_OPERATIONS:
            return table in ("table", "deleted_table") or table.startswith("litellm_")
        case _:
            return False


def _marked_lines(source: str) -> frozenset[int]:
    tokens: Final = tokenize.generate_tokens(io.StringIO(source).readline)
    return frozenset(
        token.start[0] for token in tokens if token.type == tokenize.COMMENT and DYNAMIC_SQL_MARKER.match(token.string)
    )


def _is_marked(call: ast.Call, marked: frozenset[int]) -> bool:
    return any(line in marked for line in range(call.lineno, (call.end_lineno or call.lineno) + 1))


def _location(path: Path, call: ast.Call) -> str:
    return f"{path.relative_to(REPO_ROOT)}:{call.lineno}"


def _inline_raw_sql_calls(source: Source, methods: RawMethods) -> Iterator[str]:
    shared: Final = _shared_query_names(source.module)
    marked: Final = _marked_lines(source.text)
    raw_calls: Final = (
        (call, parameters, slot)
        for call, parameters in _calls_in_scope(source.module, frozenset())
        if (slot := _raw_slot(call, methods)) is not None and not _is_marked(call, marked)
    )
    return (
        _location(source.path, call)
        for call, parameters, slot in raw_calls
        if _is_inline(_sql_argument(call, slot), parameters | shared)
    )


def _orm_calls(source: Source) -> Iterator[str]:
    return (
        _location(source.path, node)
        for node in ast.walk(source.module)
        if isinstance(node, ast.Call) and _is_orm_call(node)
    )


class Source(NamedTuple):
    path: Path
    text: str
    module: ast.Module


def _source(path: Path) -> Source:
    text: Final = path.read_text(encoding="utf-8")
    return Source(path, text, ast.parse(text, filename=str(path)))


def _python_files() -> Iterator[Path]:
    return (
        path
        for path in chain.from_iterable(root.rglob("*.py") for root in SCANNED_DIRS)
        if "__pycache__" not in path.parts
    )


def _loaded_constants(domain: Path) -> dict[str, str]:
    module: Final = ast.parse((domain / "__init__.py").read_text(encoding="utf-8"))
    loads: Final = (
        node
        for node in module.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value, ast.Call)
        and _method_name(node.value) == "load"
    )
    return {
        node.value.args[1].value: node.target.id
        for node in loads
        if isinstance(node.value, ast.Call)
        and isinstance(node.target, ast.Name)
        and isinstance(node.value.args[1], ast.Constant)
        and isinstance(node.value.args[1].value, str)
    }


def _imported_names(modules: tuple[ast.Module, ...], domain: str) -> frozenset[str]:
    imports: Final = (
        node
        for node in chain.from_iterable(ast.walk(module) for module in modules)
        if isinstance(node, ast.ImportFrom) and node.module == f"{QUERIES_MODULE}.{domain}"
    )
    return frozenset(alias.name for alias in chain.from_iterable(node.names for node in imports))


def _rust_declarations() -> frozenset[tuple[str, str]]:
    return frozenset(
        chain.from_iterable(
            (
                (path.stem, stem)
                for domain, stem in RUST_QUERY_FILE.findall(path.read_text(encoding="utf-8"))
                if domain == path.stem
            )
            for path in RUST_QUERIES_DIR.glob("*.rs")
        )
    )


def _domain_errors(domain: Path, declared: frozenset[tuple[str, str]], users: tuple[ast.Module, ...]) -> Iterator[str]:
    constants: Final = _loaded_constants(domain)
    imported: Final = _imported_names(users, domain.name)
    for sql in sorted(domain.glob("*.sql")):
        where: Final = sql.relative_to(REPO_ROOT)
        constant: Final = constants.get(sql.stem)
        if constant != sql.stem.upper():
            yield f"{where} must be loaded in its package as {sql.stem.upper()}: Final = load(__name__, {sql.stem!r})"
        if constant is not None and constant not in imported:
            yield f"{where} is loaded as {constant} but nothing imports it"
        if (domain.name, sql.stem) not in declared:
            yield f"{where} has no declare_queries! entry in {RUST_QUERIES_DIR.relative_to(REPO_ROOT)}/{domain.name}.rs"


def _layout_errors() -> Iterator[str]:
    return (
        f"{sql.relative_to(REPO_ROOT)} must live in a domain package, {QUERIES_DIR.relative_to(REPO_ROOT)}/<domain>/"
        for sql in QUERIES_DIR.glob("*.sql")
    )


def _ceiling_error(what: str, locations: tuple[str, ...], ceiling: int, name: str) -> tuple[str, ...]:
    if len(locations) == ceiling:
        return ()
    return (
        f"{len(locations)} {what}, but {name} is {ceiling}. Move new SQL into "
        f"{QUERIES_DIR.relative_to(REPO_ROOT)}/<domain>/<name>.sql, or lower the ceiling to match if you moved some",
        *locations,
    )


def main() -> int:
    sources: Final = tuple(_source(path) for path in _python_files())
    methods: Final = _raw_methods(tuple(source.module for source in sources))
    users: Final = tuple(source.module for source in sources if QUERIES_DIR not in source.path.parents)
    declared: Final = _rust_declarations()
    domains: Final = tuple(sorted(path for path in QUERIES_DIR.iterdir() if (path / "__init__.py").is_file()))
    errors: Final = (
        *_layout_errors(),
        *chain.from_iterable(_domain_errors(domain, declared, users) for domain in domains),
        *_ceiling_error(
            "raw Prisma SQL calls take inline SQL",
            tuple(chain.from_iterable(_inline_raw_sql_calls(source, methods) for source in sources)),
            INLINE_RAW_SQL_CEILING,
            "INLINE_RAW_SQL_CEILING",
        ),
        *_ceiling_error(
            "Prisma ORM table calls remain",
            tuple(chain.from_iterable(_orm_calls(source) for source in sources)),
            ORM_CALL_CEILING,
            "ORM_CALL_CEILING",
        ),
    )
    sys.stderr.write("".join(f"{line}\n" for line in errors))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
