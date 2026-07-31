"""Generic list handling for `/management/v1` collection routes.

A resource declares a `ListSpec`; `build_query_plan` turns query parameters into a
`QueryPlan` or an RFC 9457 problem without touching a database, and `handle_list`
runs that plan through an injected `ListExecutor`. Keeping the planning pure is what
lets a caller assert the plan as a value instead of asserting against a live Prisma
client, and it keeps this module free of any database dependency.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil
from typing import Generic, Literal, Protocol, TypeVar

from fastapi import Request
from pydantic import TypeAdapter, ValidationError
from typing_extensions import assert_never

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.management_v1.common import (
    PROBLEM_TYPE_BASE,
    ManagementProblem,
    build_list_links,
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

PAGE_PARAM = "page"
PAGE_SIZE_PARAM = "page_size"
SORT_PARAM = "sort"
SEARCH_PARAM = "q"

TRow = TypeVar("TRow")
TRow_co = TypeVar("TRow_co", covariant=True)
TOut = TypeVar("TOut")

_FILTER_OP_ADAPTER: TypeAdapter[FilterOp] = TypeAdapter(FilterOp)


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
    """The caller may read the rows matching `where`."""

    where: Mapping[str, object]


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


@dataclass(frozen=True, slots=True)
class QueryPlan:
    """`where` conjuncts are ordered scope-first; `order` always ends with the spec's tiebreaker."""

    where: Mapping[str, object]
    order: tuple[SortKey, ...]
    skip: int
    take: int


class ListExecutor(Protocol[TRow_co]):
    """The database half of a list, injected so this module never imports Prisma."""

    async def count(self, where: Mapping[str, object]) -> int: ...

    async def find_many(self, plan: QueryPlan) -> Sequence[TRow_co]: ...


def order_by_sql(order: tuple[SortKey, ...]) -> str:
    """`ORDER BY` body for a plan, NULLS LAST in both directions.

    Postgres sorts nulls last ascending but first descending, so an unqualified flip of
    the sort direction drags every "Unlimited" row to the top of the table. Field names
    come from `ListSpec.sortable`/`tiebreaker` and are validated against it before they
    reach here, so they are never caller-controlled text.
    """
    return ", ".join(f'"{key.field}" {"DESC" if key.descending else "ASC"} NULLS LAST' for key in order)


def _problem(slug: str, title: str, status: int, detail: str, allowed: list[str] | None = None) -> ProblemDetail:
    return ProblemDetail(
        type=f"{PROBLEM_TYPE_BASE}{slug}",
        title=title,
        status=status,
        detail=detail,
        allowed=allowed,
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
    parsed = _parse_filter_key(name)
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
        value = int(raw)
    except ValueError:
        return _invalid(f"'{name}' must be an integer.")
    if value < 1:
        return _invalid(f"'{name}' must be 1 or greater.")
    return value


def _parse_page(params: Mapping[str, str]) -> int | ProblemDetail:
    raw = params.get(PAGE_PARAM)
    return 1 if raw is None else _parse_positive_int(PAGE_PARAM, raw)


def _parse_page_size(spec: ListSpec[TRow, TOut], params: Mapping[str, str]) -> int | ProblemDetail:
    raw = params.get(PAGE_SIZE_PARAM)
    if raw is None:
        return spec.default_page_size
    value = _parse_positive_int(PAGE_SIZE_PARAM, raw)
    if isinstance(value, ProblemDetail):
        return value
    return min(value, spec.max_page_size)


def _parse_sort(spec: ListSpec[TRow, TOut], params: Mapping[str, str]) -> tuple[SortKey, ...] | ProblemDetail:
    raw = params.get(SORT_PARAM)
    if raw is None:
        return spec.default_sort
    segments = tuple(segment.strip() for segment in raw.split(","))
    keys = tuple(
        SortKey(field=segment[1:] if segment.startswith("-") else segment, descending=segment.startswith("-"))
        for segment in segments
    )
    rejected = tuple(sorted({key.field for key in keys} - spec.sortable))
    if rejected:
        return _problem(
            "invalid-sort-field",
            "Invalid sort field",
            400,
            f"Cannot sort {spec.resource} by: {', '.join(repr(field) for field in rejected)}.",
            sorted(spec.sortable),
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


def _null_fragment(field: str, raw: str) -> Mapping[str, object] | ProblemDetail:
    if raw.lower() == "true":
        return {field: None}
    if raw.lower() == "false":
        return {field: {"not": None}}
    return _invalid(f"'filter[{field}][is_null]' must be 'true' or 'false'.")


def _in_fragment(field: str, raw: str, target: FilterType) -> Mapping[str, object] | ProblemDetail:
    coerced = tuple(_coerce(field, "in", item.strip(), target) for item in raw.split(","))
    problems = tuple(item for item in coerced if isinstance(item, ProblemDetail))
    if problems:
        return problems[0]
    return {field: {"in": tuple(item for item in coerced if not isinstance(item, ProblemDetail))}}


def _comparison_fragment(field: str, op: ComparisonOp, value: FilterValue) -> Mapping[str, object]:
    match op:
        case "eq":
            return {field: value}
        case "not":
            return {field: {"not": value}}
        case "contains":
            return {field: {"contains": value, "mode": "insensitive"}}
        case "gte" | "lte" | "gt" | "lt":
            return {field: {op: value}}
        case _:
            assert_never(op)


def _parse_filter(field: str, op: FilterOp, raw: str, filter_spec: FilterSpec) -> Mapping[str, object] | ProblemDetail:
    if op not in filter_spec.ops:
        return _problem(
            "unsupported-filter-operator",
            "Unsupported filter operator",
            400,
            f"Operator '{op}' is not supported on '{field}'.",
            sorted(filter_spec.ops),
        )
    if op == "is_null":
        return _null_fragment(field, raw)
    if op == "in":
        return _in_fragment(field, raw, filter_spec.type)
    value = _coerce(field, op, raw, filter_spec.type)
    if isinstance(value, ProblemDetail):
        return value
    return _comparison_fragment(field, op, value)


def _parse_filters(
    spec: ListSpec[TRow, TOut], params: Mapping[str, str]
) -> tuple[Mapping[str, object], ...] | ProblemDetail:
    keys = tuple(
        (name, parsed)
        for name in sorted(params)
        if (parsed := _parse_filter_key(name)) is not None and parsed[0] in spec.filters
    )
    fragments = tuple(_parse_filter(field, op, params[name], spec.filters[field]) for name, (field, op) in keys)
    problems = tuple(fragment for fragment in fragments if isinstance(fragment, ProblemDetail))
    if problems:
        return problems[0]
    return tuple(fragment for fragment in fragments if not isinstance(fragment, ProblemDetail))


def _search_fragment(spec: ListSpec[TRow, TOut], params: Mapping[str, str]) -> Mapping[str, object] | None:
    raw = params.get(SEARCH_PARAM)
    if not raw:
        return None
    return {
        "OR": tuple({field: {"contains": raw, "mode": "insensitive"}} for field in sorted(spec.searchable)),
    }


def _scope_clauses(scope: Scope) -> tuple[Mapping[str, object], ...] | ProblemDetail:
    match scope:
        case ScopeAll():
            return ()
        case ScopeWhere(where=where):
            return (where,)
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
    scope_clauses = _scope_clauses(spec.scope(caller))
    if isinstance(scope_clauses, ProblemDetail):
        return scope_clauses

    unknown = tuple(sorted(name for name in params if not _is_known_param(spec, name)))
    if unknown:
        return unknown_query_param_problem(unknown=unknown, allowed=_allowed_params(spec))

    page = _parse_page(params)
    if isinstance(page, ProblemDetail):
        return page

    page_size = _parse_page_size(spec, params)
    if isinstance(page_size, ProblemDetail):
        return page_size

    sort = _parse_sort(spec, params)
    if isinstance(sort, ProblemDetail):
        return sort

    filters = _parse_filters(spec, params)
    if isinstance(filters, ProblemDetail):
        return filters

    search = _search_fragment(spec, params)
    # Scope first: a conjunct a caller filter cannot reach, let alone replace.
    clauses = scope_clauses + filters + ((search,) if search is not None else ())
    return QueryPlan(
        where={"AND": clauses} if clauses else {},
        # Ordering by an all-null column without a unique final key lets Postgres return
        # the same row on two different pages.
        order=sort + (SortKey(field=spec.tiebreaker, descending=False),),
        skip=(page - 1) * page_size,
        take=page_size,
    )


async def handle_list(
    spec: ListSpec[TRow, TOut],
    executor: ListExecutor[TRow],
    request: Request,
    caller: UserAPIKeyAuth,
) -> ListResponse[TOut]:
    """Plan, execute, count, serialize, envelope. Failures reach the client as RFC 9457 problems."""
    plan = build_query_plan(spec=spec, params=request.query_params, caller=caller)
    if isinstance(plan, ProblemDetail):
        raise ManagementProblem(plan)

    total_count = await executor.count(plan.where)
    rows = await executor.find_many(plan)
    total_pages = ceil(total_count / plan.take)
    page = plan.skip // plan.take + 1
    return ListResponse[TOut](
        data=[spec.serialize(row) for row in rows],
        meta=ListMeta(
            total_count=total_count,
            page=page,
            page_size=plan.take,
            total_pages=total_pages,
        ),
        links=build_list_links(request=request, page=page, total_pages=total_pages),
    )
