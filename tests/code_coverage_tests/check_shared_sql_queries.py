#!/usr/bin/env python3
"""Ratchet raw SQL toward the shared query files.

A query in litellm/proxy/db/queries/<name>.sql is executed by Python through Prisma and
compile-checked by litellm-rust/crates/db against the Prisma-migrated schema, so every
file there must be named by a sqlx query_file! in that crate. SQL handed inline to a raw
Prisma call is checked by nothing, so the number of those calls may only go down: lower
INLINE_RAW_SQL_CEILING in the same change that moves one into a shared file.
"""

import ast
import re
import sys
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import Final, TypeAlias

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
QUERIES_DIR: Final = REPO_ROOT / "litellm" / "proxy" / "db" / "queries"
RUST_DEFINITIONS_DIR: Final = REPO_ROOT / "litellm-rust" / "crates" / "db" / "src"
SCANNED_DIRS: Final = (REPO_ROOT / "litellm", REPO_ROOT / "enterprise")
QUERIES_MODULE: Final = "litellm.proxy.db.queries"
RAW_SQL_METHODS: Final = frozenset(
    {"query_raw", "execute_raw", "query_first", "_query_first_with_cached_plan_fallback"}
)
RUST_QUERY_FILE: Final = re.compile(r'"(?:\.\./)+litellm/proxy/db/queries/(\w+)\.sql"')
INLINE_RAW_SQL_CEILING: Final = 135

Scope: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda


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
        node for node in ast.walk(module) if isinstance(node, ast.ImportFrom) and node.module == QUERIES_MODULE
    )
    return frozenset(alias.asname or alias.name for alias in chain.from_iterable(node.names for node in imports))


def _method_name(call: ast.Call) -> str | None:
    match call.func:
        case ast.Attribute(attr=name) | ast.Name(id=name):
            return name
        case _:
            return None


def _sql_argument(call: ast.Call) -> ast.expr | None:
    if call.args:
        return call.args[0]
    return next((keyword.value for keyword in call.keywords if keyword.arg == "query"), None)


def _is_inline(argument: ast.expr | None, known_names: frozenset[str]) -> bool:
    match argument:
        case None | ast.Starred():
            return False
        case ast.Name(id=name):
            return name not in known_names
        case _:
            return True


def _inline_raw_sql_calls(path: Path) -> Iterator[str]:
    module: Final = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    shared: Final = _shared_query_names(module)
    raw_calls: Final = (
        (call, parameters)
        for call, parameters in _calls_in_scope(module, frozenset())
        if _method_name(call) in RAW_SQL_METHODS
    )
    return (
        f"{path.relative_to(REPO_ROOT)}:{call.lineno}"
        for call, parameters in raw_calls
        if _is_inline(_sql_argument(call), parameters | shared)
    )


def _python_files() -> Iterator[Path]:
    return (
        path
        for path in chain.from_iterable(root.rglob("*.py") for root in SCANNED_DIRS)
        if "__pycache__" not in path.parts
    )


def _queries_without_rust_definition() -> tuple[str, ...]:
    defined: Final = frozenset(
        chain.from_iterable(
            RUST_QUERY_FILE.findall(path.read_text(encoding="utf-8")) for path in RUST_DEFINITIONS_DIR.rglob("*.rs")
        )
    )
    return tuple(sorted(path.stem for path in QUERIES_DIR.glob("*.sql") if path.stem not in defined))


def main() -> int:
    inline_calls: Final = tuple(chain.from_iterable(_inline_raw_sql_calls(path) for path in _python_files()))
    undefined: Final = _queries_without_rust_definition()
    ceiling_errors: Final = (
        ()
        if len(inline_calls) == INLINE_RAW_SQL_CEILING
        else (
            f"{len(inline_calls)} raw Prisma SQL calls take inline SQL, but INLINE_RAW_SQL_CEILING is "
            f"{INLINE_RAW_SQL_CEILING}. Move new SQL into {QUERIES_DIR.relative_to(REPO_ROOT)}/<name>.sql, "
            "or lower the ceiling to match if you moved some there",
        )
    )
    definition_errors: Final = tuple(
        f"{QUERIES_DIR.relative_to(REPO_ROOT)}/{name}.sql has no sqlx::query_file! in "
        f"{RUST_DEFINITIONS_DIR.relative_to(REPO_ROOT)}"
        for name in undefined
    )
    errors: Final = ceiling_errors + definition_errors
    listing: Final = inline_calls if ceiling_errors else ()
    sys.stderr.write("".join(f"{line}\n" for line in errors + listing))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
