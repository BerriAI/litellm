"""Generic list handling for `/management/v1` collection routes.

A resource declares a `ListSpec`; `build_query_plan` turns query parameters into a
`QueryPlan` or an RFC 9457 problem without touching a database, and `handle_list`
runs that plan through an injected `ListExecutor`. Keeping the planning pure is what
lets a caller assert the plan as a value instead of asserting against a live Prisma
client, and it keeps this module free of any database dependency.

A plan's `where` is a tuple of frozen `Predicate`s rather than a backend-shaped
mapping, so the framework never has to know which query builder executes it and a
planned predicate cannot be rewritten afterwards. `where_sql` renders one for a
raw-SQL executor with every caller-supplied value bound to a placeholder.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial, reduce
from math import ceil
from typing import Final, Generic, Literal, Protocol, TypeVar

from fastapi import Request
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.management_v1.common import (
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    build_list_links,
    escape_like,
    unknown_query_param_problem,
)
from litellm.types.proxy.management_endpoints.management_v1 import (
    ListMeta,
    ListResponse,
    ProblemDetail,
)

ComparisonOp = Literal["eq", "gte", "lte", "gt", "lt", "contains", "not"]
# `is_null` is not in the design doc's operator set. It is here because there is no
# other way to ask for "max_budget IS NULL", and a table that renders nulls as
# "Unlimited" has to be able to filter on them.
FilterOp = ComparisonOp | Literal["in", "is_null"]

FilterType = type[str] | type[int] | type[float] | type[datetime]
FilterValue = str | int | float | datetime

PAGE_PARAM: Final = "page"
PAGE_SIZE_PARAM: Final = "page_size"
SORT_PARAM: Final = "sort"
SEARCH_PARAM: Final = "q"

TRow = TypeVar("TRow")
TRow_co = TypeVar("TRow_co", covariant=True)
TOut = TypeVar("TOut")

_FILTER_OP_ADAPTER: Final[TypeAdapter[FilterOp]] = TypeAdapter(FilterOp)


@dataclass(frozen=True, slots=True)
class Compare:
    """`field <op> value`."""

    field: str
    op: ComparisonOp
    value: FilterValue


@dataclass(frozen=True, slots=True)
class Within:
    """`field IN (values)`."""

    field: str
    values: tuple[FilterValue, ...]


@dataclass(frozen=True, slots=True)
class IsNull:
    """`field IS NULL`, or `IS NOT NULL` when negated."""

    field: str
    negated: bool


@dataclass(frozen=True, slots=True)
class AnyOf:
    """Disjunction of its clauses. `?q=` is the only producer today."""

    clauses: tuple["Predicate", ...]


Predicate = Compare | Within | IsNull | AnyOf


@dataclass(frozen=True, slots=True)
class FilterSpec:
    type: FilterType
    ops: frozenset[FilterOp]


@dataclass(frozen=True, slots=True)
class SortKey:
    field: str
    descending: bool


@dataclass(frozen=True, slots=True)
class ScopeAll:
    """The caller may read every row of the resource."""


@dataclass(frozen=True, slots=True)
class ScopeWhere:
    """The caller may read the rows matching every predicate in `where`."""

    where: tuple[Predicate, ...]


@dataclass(frozen=True, slots=True)
class ScopeDenied:
    """The caller may read no rows at all, and should be told so rather than shown an empty page."""

    reason: str


Scope = ScopeAll | ScopeWhere | ScopeDenied


@dataclass(frozen=True, slots=True)
class ListSpec(Generic[TRow, TOut]):
    resource: str
    sortable: frozenset[str]
    searchable: frozenset[str]
    filters: Mapping[str, FilterSpec]
    default_sort: tuple[SortKey, ...]
    default_page_size: int
    max_page_size: int
    scope: Callable[[UserAPIKeyAuth], Scope]
    serialize: Callable[[TRow], TOut]
    tiebreaker: str

    def __post_init__(self) -> None:
        """A malformed spec is a programming error at import time, so this raises rather than
        returning a problem: there is no request in flight and no caller to answer."""
        if not 1 <= self.default_page_size <= self.max_page_size:
            raise ValueError(
                f"{self.resource}: default_page_size must be between 1 and max_page_size "
                f"({self.max_page_size}), got {self.default_page_size}. A default above the cap "
                f"would serve more rows than the resource allows whenever page_size is omitted."
            )
        if not self.tiebreaker:
            raise ValueError(f"{self.resource}: tiebreaker is required; it is the final sort key on every query.")
        undeclared: Final = tuple(sorted(frozenset(key.field for key in self.default_sort) - self.sortable))
        if undeclared:
            raise ValueError(f"{self.resource}: default_sort orders by non-sortable field(s): {', '.join(undeclared)}.")
        non_text: Final = tuple(
            sorted(field for field, spec in self.filters.items() if "contains" in spec.ops and spec.type is not str)
        )
        if non_text:
            raise ValueError(
                f"{self.resource}: contains renders as ILIKE and is only meaningful on text columns, "
                f"but is declared on: {', '.join(non_text)}."
            )


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """`where` is an implicit AND, ordered scope-first; `order` always ends with the spec's tiebreaker."""

    where: tuple[Predicate, ...]
    order: tuple[SortKey, ...]
    skip: int
    take: int


class ListExecutor(Protocol[TRow_co]):
    """The database half of a list, injected so this module never imports Prisma."""

    async def count(self, where: tuple[Predicate, ...]) -> int: ...

    async def find_many(self, plan: QueryPlan) -> Sequence[TRow_co]: ...


def order_by_sql(order: tuple[SortKey, ...]) -> str:
    """`ORDER BY` body for a plan, NULLS LAST in both directions.

    Postgres sorts nulls last ascending but first descending, so an unqualified flip of
    the sort direction drags every "Unlimited" row to the top of the table. Every field
    reaching here is either a member of `ListSpec.sortable` (caller-supplied sort is
    checked against it, `default_sort` at construction) or the spec's `tiebreaker`, so
    these are developer-declared column names, never caller-controlled text.
    """
    return ", ".join(f'"{key.field}" {"DESC" if key.descending else "ASC"} NULLS LAST' for key in order)


def _sql_operator(op: ComparisonOp) -> str:
    match op:
        case "eq":
            return "="
        case "not":
            return "<>"
        case "gte":
            return ">="
        case "lte":
            return "<="
        case "gt":
            return ">"
        case "lt":
            return "<"
        case "contains":
            return "ILIKE"
        case _:
            assert_never(op)


def _placeholder(index: int, value: FilterValue) -> str:
    """`$n`, cast when the bind is a datetime.

    Binds cross into the query engine as JSON, so a datetime arrives as text and
    Postgres refuses `timestamp >= text` outright. Prisma stores DateTime as a naive
    `TIMESTAMP(3)` holding UTC, so the bind is read as an instant and then dropped to
    naive UTC to match the column, the same cast `/spend/logs/ui` applies.
    """
    return f"${index}::timestamptz AT TIME ZONE 'UTC'" if isinstance(value, datetime) else f"${index}"


def _render(predicate: Predicate, index: int) -> tuple[str, tuple[object, ...]]:
    match predicate:
        case IsNull(field=field, negated=negated):
            return f'"{field}" IS {"NOT NULL" if negated else "NULL"}', ()
        case Within(field=field, values=values):
            placeholders: Final = ", ".join(_placeholder(index + offset, value) for offset, value in enumerate(values))
            return f'"{field}" IN ({placeholders})', values
        case AnyOf(clauses=clauses):
            rendered, params = _render_all(clauses, index)
            return f"({' OR '.join(rendered)})", params
        case Compare(field=field, op="contains", value=value):
            return f"\"{field}\" ILIKE ${index} ESCAPE '\\'", (f"%{escape_like(str(value))}%",)
        case Compare(field=field, op=op, value=value):
            return f'"{field}" {_sql_operator(op)} {_placeholder(index, value)}', (value,)
        case _:
            assert_never(predicate)


def _render_one(
    rendered: tuple[tuple[str, ...], tuple[object, ...]],
    predicate: Predicate,
    first_index: int,
) -> tuple[tuple[str, ...], tuple[object, ...]]:
    """Append one predicate, numbering it after the binds already consumed."""
    clauses, params = rendered
    clause, clause_params = _render(predicate, first_index + len(params))
    return (*clauses, clause), (*params, *clause_params)


def _render_all(predicates: tuple[Predicate, ...], index: int) -> tuple[tuple[str, ...], tuple[object, ...]]:
    """Render every predicate, numbering placeholders continuously across them.

    Folded rather than self-recursive: walking a predicate list is a running index, and
    recursing per predicate grew the stack with the filter count for nothing. `_render`
    still re-enters here for `AnyOf`, whose clauses are plain `Compare`s from `?q=`, so
    that nesting is one level deep and cannot be driven deeper by a caller.
    """
    return reduce(partial(_render_one, first_index=index), predicates, ((), ()))


def where_sql(where: tuple[Predicate, ...], first_index: int = 1) -> tuple[str, tuple[object, ...]]:
    """`WHERE` body and its bind parameters, numbered from `first_index`.

    Returns `("", ())` when there is nothing to filter on. Every caller-supplied value
    becomes a `$n` placeholder rather than being written into the SQL text; only column
    names reach the text, and those come from the spec's own declarations.
    """
    clauses, params = _render_all(where, first_index)
    return " AND ".join(clauses), params


def _problem(slug: str, title: str, status: int, detail: str, allowed: tuple[str, ...] | None = None) -> ProblemDetail:
    return ProblemDetail(
        type=f"{PROBLEM_TYPE_BASE}{slug}",
        title=title,
        status=status,
        detail=detail,
        allowed=sorted(allowed) if allowed is not None else None,
    )


def _invalid(detail: str) -> ProblemDetail:
    return _problem("invalid-query-parameter", "Invalid query parameter", 400, detail)


def _parse_filter_key(name: str) -> tuple[str, FilterOp] | None:
    """`filter[max_budget][gte]` -> `("max_budget", "gte")`; bare `filter[status]` -> `("status", "eq")`."""
    if not name.startswith("filter[") or not name.endswith("]"):
        return None
    field, separator, raw_op = name[len("filter[") : -1].partition("][")
    if not separator:
        return field, "eq"
    try:
        return field, _FILTER_OP_ADAPTER.validate_python(raw_op)
    except ValidationError:
        return None


def _is_known_param(spec: ListSpec[TRow, TOut], name: str) -> bool:
    if name in (PAGE_PARAM, PAGE_SIZE_PARAM):
        return True
    if name == SORT_PARAM:
        return bool(spec.sortable)
    if name == SEARCH_PARAM:
        return bool(spec.searchable)
    parsed: Final = _parse_filter_key(name)
    return parsed is not None and parsed[0] in spec.filters


def _allowed_params(spec: ListSpec[TRow, TOut]) -> tuple[str, ...]:
    return tuple(
        sorted(
            (PAGE_PARAM, PAGE_SIZE_PARAM)
            + ((SORT_PARAM,) if spec.sortable else ())
            + ((SEARCH_PARAM,) if spec.searchable else ())
            + tuple(
                f"filter[{field}]" if op == "eq" else f"filter[{field}][{op}]"
                for field, filter_spec in spec.filters.items()
                for op in filter_spec.ops
            )
        )
    )


def _parse_positive_int(name: str, raw: str) -> int | ProblemDetail:
    try:
        value: Final = int(raw)
    except ValueError:
        return _invalid(f"'{name}' must be an integer.")
    if value < 1:
        return _invalid(f"'{name}' must be 1 or greater.")
    return value


def _parse_page(params: Mapping[str, str]) -> int | ProblemDetail:
    raw: Final = params.get(PAGE_PARAM)
    return 1 if raw is None else _parse_positive_int(PAGE_PARAM, raw)


def _parse_page_size(spec: ListSpec[TRow, TOut], params: Mapping[str, str]) -> int | ProblemDetail:
    raw: Final = params.get(PAGE_SIZE_PARAM)
    if raw is None:
        return spec.default_page_size
    value: Final = _parse_positive_int(PAGE_SIZE_PARAM, raw)
    if isinstance(value, ProblemDetail):
        return value
    return min(value, spec.max_page_size)


def _parse_sort(spec: ListSpec[TRow, TOut], params: Mapping[str, str]) -> tuple[SortKey, ...] | ProblemDetail:
    raw: Final = params.get(SORT_PARAM)
    if raw is None:
        return spec.default_sort
    segments: Final = tuple(segment.strip() for segment in raw.split(","))
    keys = tuple(SortKey(field=segment.removeprefix("-"), descending=segment.startswith("-")) for segment in segments)
    rejected: Final = tuple(sorted(frozenset(key.field for key in keys) - spec.sortable))
    if rejected:
        return _problem(
            "invalid-sort-field",
            "Invalid sort field",
            400,
            f"Cannot sort {spec.resource} by: {', '.join(repr(field) for field in rejected)}.",
            tuple(spec.sortable),
        )
    return keys


def _to_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _coerce(field: str, op: FilterOp, raw: str, target: FilterType) -> FilterValue | ProblemDetail:
    try:
        if target is str:
            return raw
        if target is int:
            return int(raw)
        if target is float:
            return float(raw)
        return _to_utc(datetime.fromisoformat(raw[:-1] + "+00:00" if raw.endswith("Z") else raw))
    except ValueError:
        return _invalid(f"'filter[{field}][{op}]' is not a valid {target.__name__}: {raw!r}.")


def _null_predicate(field: str, raw: str) -> Predicate | ProblemDetail:
    if raw.lower() == "true":
        return IsNull(field=field, negated=False)
    if raw.lower() == "false":
        return IsNull(field=field, negated=True)
    return _invalid(f"'filter[{field}][is_null]' must be 'true' or 'false'.")


def _within_predicate(field: str, raw: str, target: FilterType) -> Predicate | ProblemDetail:
    coerced: Final = tuple(_coerce(field, "in", item.strip(), target) for item in raw.split(","))
    problems: Final = tuple(item for item in coerced if isinstance(item, ProblemDetail))
    if problems:
        return problems[0]
    return Within(field=field, values=tuple(item for item in coerced if not isinstance(item, ProblemDetail)))


def _parse_filter(field: str, op: FilterOp, raw: str, filter_spec: FilterSpec) -> Predicate | ProblemDetail:
    if op not in filter_spec.ops:
        return _problem(
            "unsupported-filter-operator",
            "Unsupported filter operator",
            400,
            f"Operator '{op}' is not supported on '{field}'.",
            tuple(filter_spec.ops),
        )
    if op == "is_null":
        return _null_predicate(field, raw)
    if op == "in":
        return _within_predicate(field, raw, filter_spec.type)
    value: Final = _coerce(field, op, raw, filter_spec.type)
    if isinstance(value, ProblemDetail):
        return value
    return Compare(field=field, op=op, value=value)


def _parse_filters(spec: ListSpec[TRow, TOut], params: Mapping[str, str]) -> tuple[Predicate, ...] | ProblemDetail:
    keys: Final = tuple(
        (name, parsed)
        for name in sorted(params)
        if (parsed := _parse_filter_key(name)) is not None and parsed[0] in spec.filters
    )
    parsed = tuple(_parse_filter(field, op, params[name], spec.filters[field]) for name, (field, op) in keys)
    problems: Final = tuple(item for item in parsed if isinstance(item, ProblemDetail))
    if problems:
        return problems[0]
    return tuple(item for item in parsed if not isinstance(item, ProblemDetail))


def _search_predicate(spec: ListSpec[TRow, TOut], params: Mapping[str, str]) -> Predicate | None:
    raw: Final = params.get(SEARCH_PARAM)
    if not raw:
        return None
    return AnyOf(clauses=tuple(Compare(field=field, op="contains", value=raw) for field in sorted(spec.searchable)))


def _scope_predicates(scope: Scope) -> tuple[Predicate, ...] | ProblemDetail:
    match scope:
        case ScopeAll():
            return ()
        case ScopeWhere(where=where):
            return where
        case ScopeDenied(reason=reason):
            return _problem("forbidden", "Forbidden", 403, reason)
        case _:
            assert_never(scope)


def build_query_plan(
    spec: ListSpec[TRow, TOut],
    params: Mapping[str, str],
    caller: UserAPIKeyAuth,
) -> QueryPlan | ProblemDetail:
    """Turn query parameters into a plan, or into the problem that explains why they are not one."""
    scope_predicates: Final = _scope_predicates(spec.scope(caller))
    if isinstance(scope_predicates, ProblemDetail):
        return scope_predicates

    unknown: Final = tuple(sorted(name for name in params if not _is_known_param(spec, name)))
    if unknown:
        return unknown_query_param_problem(unknown=unknown, allowed=_allowed_params(spec))

    page: Final = _parse_page(params)
    if isinstance(page, ProblemDetail):
        return page

    page_size: Final = _parse_page_size(spec, params)
    if isinstance(page_size, ProblemDetail):
        return page_size

    sort: Final = _parse_sort(spec, params)
    if isinstance(sort, ProblemDetail):
        return sort

    filters: Final = _parse_filters(spec, params)
    if isinstance(filters, ProblemDetail):
        return filters

    search: Final = _search_predicate(spec, params)
    return QueryPlan(
        # Scope first: conjuncts a caller filter sits behind and cannot replace.
        where=scope_predicates + filters + ((search,) if search is not None else ()),
        # Ordering by an all-null column without a unique final key lets Postgres return
        # the same row on two different pages.
        order=sort + (SortKey(field=spec.tiebreaker, descending=False),),
        skip=(page - 1) * page_size,
        take=page_size,
    )


def _duplicate_params(request: Request) -> tuple[str, ...]:
    names: Final = tuple(name for name, _ in request.query_params.multi_items())
    return tuple(sorted(frozenset(name for name in names if names.count(name) > 1)))


async def handle_list(
    spec: ListSpec[TRow, TOut],
    executor: ListExecutor[TRow],
    request: Request,
    caller: UserAPIKeyAuth,
) -> ListResponse[TOut]:
    """Plan, execute, count, serialize, envelope. Failures reach the client as RFC 9457 problems."""
    plan: Final = build_query_plan(spec=spec, params=request.query_params, caller=caller)
    if isinstance(plan, ProblemDetail):
        raise ManagementProblem(plan)

    # Checked here rather than in build_query_plan because a Mapping[str, str] cannot
    # represent a repeat: query_params.get() silently keeps the last one, so ?page=1&page=999
    # would page from 999 without the caller ever being told which value won.
    duplicates: Final = _duplicate_params(request)
    if duplicates:
        raise ManagementProblem(
            _problem(
                "duplicate-query-parameter",
                "Duplicate query parameter",
                400,
                f"Repeated query parameter(s): {', '.join(duplicates)}. Each may appear once; "
                f"use a comma-separated list for multiple sort keys or filter values.",
            )
        )

    total_count: Final = await executor.count(plan.where)
    rows: Final = await executor.find_many(plan)
    total_pages: Final = ceil(total_count / plan.take)
    page: Final = plan.skip // plan.take + 1
    return ListResponse[TOut](
        data=tuple(spec.serialize(row) for row in rows),
        meta=ListMeta(
            total_count=total_count,
            page=page,
            page_size=plan.take,
            total_pages=total_pages,
        ),
        links=build_list_links(request=request, page=page, total_pages=total_pages),
    )
