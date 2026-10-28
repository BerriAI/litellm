import ast
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from litellm.integrations.otel.model.payloads import ServiceSpanData
from litellm.integrations.otel.model import spans as spans_mod
from litellm.integrations.otel.model.spans import (
    _POSTGRES_OPERATION_BY_CALL_TYPE,
    PRISMA_RELATIONS,
    service_span_name,
)
from litellm.proxy.db.prisma_query_span import UNKNOWN_PRISMA_QUERY, parse_prisma_query, sql_operation

_REPO: Final = Path(__file__).resolve().parents[4]
_SOURCE_ROOTS: Final = ("litellm", "enterprise", "litellm-proxy-extras")
_RAW_METHODS: Final = frozenset({"query_first", "query_raw", "execute_raw"})
_MODEL_METHODS: Final = frozenset(
    {
        "find_unique",
        "find_unique_or_raise",
        "find_first",
        "find_first_or_raise",
        "find_many",
        "count",
        "group_by",
        "create",
        "create_many",
        "update",
        "update_many",
        "delete",
        "delete_many",
        "upsert",
    }
)
_MODEL_BY_ACCESSOR: Final[Mapping[str, str]] = {relation.lower(): relation for relation in PRISMA_RELATIONS}
_GENERIC_CRUD_HELPERS: Final = frozenset({"get_data", "get_generic_data", "insert_data", "update_data", "delete_data"})
_TRANSACTION_BODIES: Final[Mapping[str, str]] = {"litellm/proxy/db/baseline_accounting.py": "baseline_accounting"}
_RENDERED_NAME: Final = re.compile(
    r"postgres\.(select|insert|update|delete|upsert|ddl|set|transaction|lock) .+|postgres\.ping"
)


def _engine_payload(root_field: str, sql: str | None = None) -> str:
    selection: Final = f'queryRaw(query: "{sql}", parameters: "[]")' if sql is not None else root_field
    return (
        f'{{"query": "mutation {{ result: {selection} }}"}}'
        if sql is not None
        else f'{{"query": "query {{ result: {root_field}(where: {{token: \\"x\\"}}) {{ token }} }}"}}'
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        (
            _engine_payload("findUniqueLiteLLM_VerificationToken"),
            ("find_unique", "select", "LiteLLM_VerificationToken"),
        ),
        (_engine_payload("createOneLiteLLM_SpendLogs"), ("create", "insert", "LiteLLM_SpendLogs")),
        (_engine_payload("findFirstLiteLLM_UserTableOrThrow"), ("find_first", "select", "LiteLLM_UserTable")),
        (
            '{"query": "mutation { result: queryRaw(query: \\"SELECT * FROM \\\\\\"LiteLLM_UserTable\\\\\\" WHERE user_id = $1\\", parameters: \\"[]\\") }"}',
            ("query_raw", "select", "LiteLLM_UserTable"),
        ),
        (
            '{"query": "mutation { result: executeRaw(query: \\"SET LOCAL statement_timeout = 5000\\", parameters: \\"[]\\") }"}',
            ("execute_raw", "set", "statement_timeout"),
        ),
        (
            '{"query": "mutation { result: queryRaw(query: \\"SELECT to_regclass($1) IS NOT NULL AS present\\", parameters: \\"[]\\") }"}',
            ("query_raw", "select", "pg_catalog"),
        ),
        (
            '{"query": "mutation { result: queryRaw(query: \\"SELECT 1\\", parameters: \\"[]\\") }"}',
            ("query_raw", "ping", None),
        ),
    ],
)
def test_the_engine_names_a_round_trip_from_its_payload_without_copying_sql_text(
    content: str, expected: tuple[str, str | None, str | None]
) -> None:
    query = parse_prisma_query(content)
    assert (query.call_type, query.operation, query.table) == expected
    assert query.table is None or " " not in query.table


def test_a_payload_the_parser_does_not_know_stays_the_legacy_function_named_span() -> None:
    assert parse_prisma_query("not json at all") is UNKNOWN_PRISMA_QUERY
    assert parse_prisma_query('{"query": "mutation { result: somethingNew(x: 1) }"}') is UNKNOWN_PRISMA_QUERY
    rendered = service_span_name(ServiceSpanData(service_name="postgres", call_type=UNKNOWN_PRISMA_QUERY.call_type))
    assert rendered == "postgres prisma_query"


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ('SELECT 1 FROM "LiteLLM_VerificationTokenView" LIMIT 1', ("select", "LiteLLM_VerificationTokenView")),
        (
            '\n  WITH keys AS (SELECT * FROM "LiteLLM_VerificationToken") SELECT 1',
            ("select", "LiteLLM_VerificationToken"),
        ),
        ('INSERT INTO "LiteLLM_DailyUserSpend" (id) VALUES ($1)', ("insert", "LiteLLM_DailyUserSpend")),
        ("SET LOCAL lock_timeout = 1000", ("set", "lock_timeout")),
        ("SELECT set_config('lock_timeout', $1::text, true)", ("set", "lock_timeout")),
        ("SELECT COUNT(*) FROM pg_stat_activity", ("select", "pg_catalog")),
        ("SELECT 1", ("ping", None)),
        ("SELECT current_setting('transaction_read_only') AS transaction_read_only", ("select", "pg_catalog")),
        ('REFRESH MATERIALIZED VIEW "MonthlyGlobalSpend"', ("ddl", "MonthlyGlobalSpend")),
        (
            'WITH team_rows AS (UPDATE "LiteLLM_TeamTable" SET models = $1 RETURNING team_id) SELECT team_id FROM team_rows',
            ("update", "LiteLLM_TeamTable"),
        ),
        ('LOCK TABLE "LiteLLM_LensIngestionKey" IN EXCLUSIVE MODE', ("lock", "LiteLLM_LensIngestionKey")),
        ("BEGIN", (None, None)),
    ],
)
def test_sql_operation_is_the_leading_verb_and_the_first_schema_relation(
    sql: str, expected: tuple[str | None, str | None]
) -> None:
    assert sql_operation(sql) == expected


@dataclass(frozen=True, slots=True)
class _PrismaCallSite:
    location: str
    method: str
    owner: str
    rendered: str | None


@dataclass(frozen=True, slots=True)
class _Module:
    path: Path
    tree: ast.Module
    constants: Mapping[str, ast.expr]

    def ancestors(self, node: ast.AST) -> tuple[ast.AST, ...]:
        parent_of: Final = _parent_map(self.tree)
        chain: Final = [node]
        while (parent := parent_of.get(id(chain[-1]))) is not None:
            chain.append(parent)
        return tuple(chain[1:])


_PARENTS: Final[dict[int, Mapping[int, ast.AST]]] = {}  # mutable-ok: per-tree parent map memo


def _parent_map(tree: ast.Module) -> Mapping[int, ast.AST]:
    if id(tree) not in _PARENTS:
        _PARENTS[id(tree)] = {
            id(child): node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)
        }  # comprehension-ok: parent links
    return _PARENTS[id(tree)]


def _modules() -> Iterator[_Module]:
    for root in _SOURCE_ROOTS:
        for path in sorted((_REPO / root).rglob("*.py")):
            if "tests" in path.parts or "node_modules" in path.parts:
                continue
            tree: Final = ast.parse(path.read_text(encoding="utf-8"))
            yield _Module(path, tree, _assignments(tree.body))


def _imported_module(module: _Module, name: str) -> Path | None:
    for node in module.tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and any(alias.name == name for alias in node.names):
            return _REPO / (node.module.replace(".", "/") + ".py")
    return None


def _assignments(body: list[ast.stmt]) -> Mapping[str, ast.expr]:
    return {
        target.id: node.value
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
        for target in (node.targets if isinstance(node, ast.Assign) else (node.target,))
        if isinstance(target, ast.Name)
    }  # comprehension-ok: constants by name


def _mapping_values(expr: ast.expr | None) -> ast.expr | None:
    """The dict a ``Mapping`` constant was built from, through ``MappingProxyType(...)``."""
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "MappingProxyType"
        and expr.args
    ):
        return expr.args[0]
    return expr if isinstance(expr, (ast.Dict, ast.DictComp)) else None


def _returned_text(function_name: str, module: _Module) -> ast.expr | None:
    """What a module-level SQL builder returns, when its body is one ``return`` of a string expression."""
    for node in module.tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            returns: Final = [stmt for stmt in ast.walk(node) if isinstance(stmt, ast.Return)]
            return returns[0].value if len(returns) == 1 else None
    return None


_DYNAMIC: Final = " ? "


def _fragment(value: ast.expr, module: _Module, depth: int) -> str:
    """One f-string piece: literal text, a module constant spliced in, or a runtime placeholder."""
    spliced: Final = (
        _sql_text(value.value, module, depth + 1)
        if isinstance(value, ast.FormattedValue)
        else _sql_text(value, module, depth)
    )
    return _DYNAMIC if spliced is None or _ALTERNATIVE in spliced else spliced


def _sql_text(expr: ast.expr | None, module: _Module, depth: int = 0) -> str | None:
    if expr is None or depth > 3:
        return None
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.JoinedStr):
        return "".join(_fragment(value, module, depth) for value in expr.values)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left: Final = _sql_text(expr.left, module, depth)
        return left if left is not None else _sql_text(expr.right, module, depth)
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Attribute)
        and expr.func.attr in {"format", "strip", "lstrip"}
    ):
        return _sql_text(expr.func.value, module, depth)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Attribute) and expr.func.attr == "dedent":
        return _sql_text(expr.args[0], module, depth) if expr.args else None
    if isinstance(expr, ast.IfExp):
        branches: Final = (_sql_text(expr.body, module, depth), _sql_text(expr.orelse, module, depth))
        return branches[0] if branches[0] == branches[1] or None in branches else _multi(branches)
    if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name):
        return _sql_text(_mapping_values(module.constants.get(expr.value.id)), module, depth + 1)
    if (
        isinstance(expr, ast.Call)
        and isinstance(expr.func, ast.Name)
        and expr.func.id == "MappingProxyType"
        and expr.args
    ):
        return _sql_text(expr.args[0], module, depth)
    if isinstance(expr, ast.Dict):
        values: Final = tuple(_sql_text(value, module, depth) for value in expr.values)
        return _multi(values) if values and None not in values else None
    if isinstance(expr, ast.DictComp):
        return _sql_text(expr.value, module, depth)
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        returned: Final = _returned_text(expr.func.id, module)
        return _sql_text(returned, module, depth + 1) if returned is not None else None
    if isinstance(expr, ast.Name):
        if expr.id in module.constants:
            return _sql_text(module.constants[expr.id], module, depth + 1)
        source: Final = _imported_module(module, expr.id)
        if source is None or not source.exists():
            return None
        imported: Final = ast.parse(source.read_text(encoding="utf-8"))
        imported_module: Final = _Module(source, imported, _assignments(imported.body))
        return _sql_text(imported_module.constants.get(expr.id), imported_module, depth + 1)
    return None


_ALTERNATIVE: Final = "\x1f"


def _multi(texts: tuple[str | None, ...]) -> str:
    return _ALTERNATIVE.join(text for text in texts if text is not None)


def _parameters(parents: tuple[ast.AST, ...]) -> frozenset[str]:
    function: Final = next((p for p in parents if isinstance(p, (ast.AsyncFunctionDef, ast.FunctionDef))), None)
    if function is None:
        return frozenset()
    return frozenset(arg.arg for arg in (*function.args.args, *function.args.kwonlyargs))


def _argument_for(call: ast.Call, function: ast.AsyncFunctionDef | ast.FunctionDef, parameter: str) -> ast.expr | None:
    positional: Final = tuple(arg.arg for arg in function.args.args)
    by_keyword: Final = next((k.value for k in call.keywords if k.arg == parameter), None)
    if by_keyword is not None or parameter not in positional:
        return by_keyword
    index: Final = positional.index(parameter)
    return call.args[index] if index < len(call.args) else None


def _parameter_site(
    parameter: str, module: _Module, parents: tuple[ast.AST, ...], method: str, location: str
) -> _PrismaCallSite:
    """A statement that arrives as a parameter: a ``query_raw`` forwarder adds no round trip of its
    own, any other helper is named by what its callers in the module hand it."""
    function: Final = next(p for p in parents if isinstance(p, (ast.AsyncFunctionDef, ast.FunctionDef)))
    if function.name in _RAW_METHODS:
        return _PrismaCallSite(location, method, "forwarder", f"(callers of {function.name})")
    callers: Final = tuple(
        node
        for node in ast.walk(module.tree)
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(function.name)
    )
    sites: Final = tuple(_caller_site(call, function, parameter, module, method, location) for call in callers)
    names: Final = tuple(site.rendered for site in sites)
    owners: Final = ", ".join(sorted({site.owner for site in sites}))
    rendered: Final = " | ".join(sorted(set(names))) if names and None not in names else None  # pyright: ignore[reportArgumentType]  # None filtered above
    return _PrismaCallSite(location, method, f"{owners} via {function.name} callers", rendered)


def _caller_site(
    call: ast.Call,
    function: ast.AsyncFunctionDef | ast.FunctionDef,
    parameter: str,
    module: _Module,
    method: str,
    location: str,
) -> _PrismaCallSite:
    parents: Final = module.ancestors(call)
    wrapped: Final = _wrapper_site(call, parents, method, location, module)
    if wrapped is not None:
        return wrapped
    text: Final = _sql_text(_argument_for(call, function, parameter), _scope(module, parents))
    return _PrismaCallSite(location, method, "engine", _render_statements(method, text) if text is not None else None)


def _render_statements(method: str, text: str) -> str | None:
    names: Final = tuple(_render_statement(method, alternative) for alternative in text.split(_ALTERNATIVE))
    return " | ".join(sorted(set(names))) if None not in names else None  # pyright: ignore[reportArgumentType]  # None filtered above


def _render_statement(method: str, text: str) -> str | None:
    verb, target = sql_operation(text)
    if verb is not None and target is None and verb != "ping" and _DYNAMIC in text:
        return f"postgres.{verb} {{relation built at runtime}}"
    return _render(method, target, verb) if verb is not None else None


def _render(call_type: str, table: str | None, operation: str | None = None) -> str:
    metadata: Final = {
        key: value for key, value in (("table_name", table), ("db_operation", operation)) if value is not None
    }
    return service_span_name(ServiceSpanData(service_name="postgres", call_type=call_type, event_metadata=metadata))


def _wrapper_site(
    call: ast.Call, parents: tuple[ast.AST, ...], method: str, location: str, module: _Module
) -> _PrismaCallSite | None:
    for parent in parents:
        items: Final = parent.items if isinstance(parent, (ast.AsyncWith, ast.With)) else ()
        for item in items:
            context: Final = item.context_expr
            if isinstance(context, ast.Call) and isinstance(context.func, ast.Name) and context.func.id == "db_span":
                return _wrapped_by(context, method, location, "db_span", _scope(module, parents))
            if (
                isinstance(context, ast.Call)
                and isinstance(context.func, ast.Name)
                and context.func.id == "_spend_update_tx"
            ):
                call_type: Final = context.args[2] if len(context.args) > 2 else ast.Constant("commit_spend_updates")
                spend_tx: Final = ast.Call(func=ast.Name("db_span"), args=[call_type, context.args[1]], keywords=[])
                return _wrapped_by(spend_tx, method, location, "_spend_update_tx", _scope(module, parents))
        if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name) and parent.func.id == "db_spanned":
            return _wrapped_by(parent, method, location, "db_spanned", _scope(module, parents))
        if isinstance(parent, (ast.AsyncFunctionDef, ast.FunctionDef)):
            decorators: Final = tuple(
                decorator.id for decorator in parent.decorator_list if isinstance(decorator, ast.Name)
            )
            if "log_db_metrics" in decorators:
                return _decorated_site(parent.name, method, location)
    return None


def _wrapped_by(wrapper: ast.Call, method: str, location: str, owner: str, scope: _Module) -> _PrismaCallSite:
    call_type: Final = _sql_text(wrapper.args[0], scope)
    table_expr: Final = wrapper.args[1] if len(wrapper.args) > 1 else None
    if call_type is None:
        return _PrismaCallSite(location, method, owner, None)
    if isinstance(table_expr, ast.Constant) and table_expr.value is None:
        return _PrismaCallSite(location, method, owner, _render(call_type, None))
    table: Final = _sql_text(table_expr, scope)
    if table is not None:
        return _PrismaCallSite(location, method, owner, _render(call_type, table))
    operation: Final = _POSTGRES_OPERATION_BY_CALL_TYPE.get(call_type)
    rendered: Final = f"postgres.{operation.verb} {{relation}}" if operation is not None else None
    return _PrismaCallSite(location, method, f"{owner}(bounded)", rendered)


def _decorated_site(function: str, method: str, location: str) -> _PrismaCallSite:
    if function in _GENERIC_CRUD_HELPERS:
        return _PrismaCallSite(location, method, "log_db_metrics(crud)", "postgres.{verb} {table_name}")
    operation: Final = _POSTGRES_OPERATION_BY_CALL_TYPE.get(function)
    return _PrismaCallSite(
        location, method, "log_db_metrics", _render(function, None) if operation is not None else None
    )


def _scope(module: _Module, parents: tuple[ast.AST, ...]) -> _Module:
    function: Final = next((p for p in parents if isinstance(p, (ast.AsyncFunctionDef, ast.FunctionDef))), None)
    if function is None:
        return module
    return _Module(module.path, module.tree, {**module.constants, **_assignments(function.body)})


def _engine_site(
    call: ast.Call, module: _Module, parents: tuple[ast.AST, ...], method: str, accessor: str | None, location: str
) -> _PrismaCallSite:
    if accessor is not None:
        return _PrismaCallSite(location, method, "engine", _render(method, _MODEL_BY_ACCESSOR.get(accessor)))
    scope: Final = _scope(module, parents)
    statement: Final = call.args[0] if call.args else next((k.value for k in call.keywords if k.arg == "query"), None)
    if isinstance(statement, ast.Name) and statement.id in _parameters(parents):
        return _parameter_site(statement.id, module, parents, method, location)
    rendered: Final = _render_statements(method, _sql_text(statement, scope) or "")
    relative: Final = str(module.path.relative_to(_REPO))
    if _RENDERED_NAME.fullmatch(rendered or "") is None and relative in _TRANSACTION_BODIES:
        owner: Final = _TRANSACTION_BODIES[relative]
        return _PrismaCallSite(location, method, f"transaction({owner})", _render(owner, None))
    return _PrismaCallSite(location, method, "engine", rendered)


def _accessor(receiver: ast.expr) -> str | None:
    if isinstance(receiver, ast.Attribute) and receiver.attr in _MODEL_BY_ACCESSOR:
        return receiver.attr
    return None


def _call_sites(module: _Module) -> Iterator[_PrismaCallSite]:
    def walk(node: ast.AST, parents: tuple[ast.AST, ...]) -> Iterator[_PrismaCallSite]:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                method: Final = child.func.attr
                accessor: Final = _accessor(child.func.value)
                if method in _RAW_METHODS or (method in _MODEL_METHODS and accessor is not None):
                    location: Final = f"{module.path.relative_to(_REPO)}:{child.lineno}"
                    yield _wrapper_site(child, parents, method, location, module) or _engine_site(
                        child, module, parents, method, accessor, location
                    )
            yield from walk(child, (child, *parents))

    yield from walk(module.tree, ())


def prisma_call_sites() -> tuple[_PrismaCallSite, ...]:
    return tuple(site for module in _modules() for site in _call_sites(module))  # comprehension-ok: flatten


def test_every_prisma_call_site_in_the_proxy_renders_a_bounded_postgres_span_name() -> None:
    """A raw ``query_raw``/``execute_raw``/``query_first`` or a direct model call that no producer
    wraps is named by the engine from its payload; this scan replays that naming (and the wrappers')
    statically so a new statement that would ship as a bare ``postgres.select`` or an unnamed
    ``postgres query_raw`` fails here rather than in a trace."""
    sites = prisma_call_sites()
    assert len(sites) >= 120, f"the scan lost the Prisma call sites: {len(sites)}"
    unresolved = [site for site in sites if site.rendered is None]
    assert unresolved == [], f"Prisma call sites whose span name cannot be resolved: {unresolved}"
    half_named = [
        site
        for site in sites
        if site.owner.startswith("engine")
        and any(_RENDERED_NAME.fullmatch(name) is None for name in (site.rendered or "").split(" | "))
    ]
    assert half_named == [], f"Prisma call sites that would ship a half-named or legacy span: {half_named}"


def test_every_model_in_the_prisma_schema_is_a_renderable_span_table() -> None:
    schema: Final = (_REPO / "schema.prisma").read_text()
    declared: Final = frozenset(re.findall(r"^model (\w+) \{", schema, re.MULTILINE))

    assert declared == spans_mod._PRISMA_MODELS
