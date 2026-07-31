"""Generic paging/sorting/filtering contract for `/management/v1` entity lists.

Prisma-free by construction: a route declares a `ListSpec` and injects a
`ListExecutor` that owns the table, so the parsing, scoping and envelope rules
stay in one place and every entity list answers the same way.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar
from urllib.parse import urlencode

from fastapi import Request
from pydantic import JsonValue

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.management_endpoints.management_v1.common import (
    PROBLEM_TYPE_BASE,
    ManagementProblem,
)
from litellm.types.proxy.management_endpoints.management_v1 import (
    ListLinks,
    ListMeta,
    ListResponse,
    ProblemDetail,
)

FilterOp: TypeAlias = Literal["eq", "in", "gte", "lte", "contains", "is_null"]
FilterType: TypeAlias = Literal["string", "number", "datetime"]

# Quoted so the recursive alias parses under the repo's 3.10 floor, where neither
# the `type` statement nor a forward reference inside a `|` expression exists.
WhereLeaf: TypeAlias = "str | int | float | bool | datetime | None"
WhereValue: TypeAlias = "WhereLeaf | Sequence[WhereLeaf] | Where | Sequence[Where]"
Where: TypeAlias = "Mapping[str, WhereValue]"
OrderBy: TypeAlias = "Sequence[Mapping[str, Literal['asc', 'desc']]]"

RowT = TypeVar("RowT")

PAGINATION_PARAMS = frozenset({"page", "page_size", "sort", "q"})


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
    """The caller may read every row."""


@dataclass(frozen=True, slots=True)
class ScopeWhere:
    """The caller may read only rows matching `where`."""

    where: Where


@dataclass(frozen=True, slots=True)
class ScopeDenied:
    """The caller may not read the collection at all."""

    detail: str


Scope: TypeAlias = "ScopeAll | ScopeWhere | ScopeDenied"


class ListExecutor(Protocol, Generic[RowT]):
    """The table half of a list, injected so the framework never imports Prisma."""

    async def count(self, where: Where) -> int: ...

    async def find_many(self, where: Where, order: OrderBy, skip: int, take: int) -> Sequence[RowT]: ...


@dataclass(frozen=True, slots=True)
class ListSpec(Generic[RowT]):
    resource: str
    sortable: frozenset[str]
    searchable: frozenset[str]
    filters: Mapping[str, FilterSpec]
    default_sort: tuple[SortKey, ...]
    default_page_size: int
    max_page_size: int
    scope: Callable[[UserAPIKeyAuth], Scope]
    serialize: Callable[[RowT], Mapping[str, JsonValue]]
    tiebreaker: str


@dataclass(frozen=True, slots=True)
class QueryPlan:
    where: Where
    order: OrderBy
    skip: int
    take: int
    page: int
    page_size: int


def _problem(slug: str, title: str, detail: str, allowed: Sequence[str] | None = None) -> ManagementProblem:
    return ManagementProblem(
        ProblemDetail(
            type=f"{PROBLEM_TYPE_BASE}{slug}",
            title=title,
            status=400,
            detail=detail,
            allowed=list(allowed) if allowed is not None else None,
        )
    )


def _allowed_params(filters: Mapping[str, FilterSpec]) -> frozenset[str]:
    return PAGINATION_PARAMS | frozenset(
        f"filter[{field}][{op}]" for field, filter_spec in filters.items() for op in filter_spec.ops
    )


def _reject_unknown_params(request: Request, filters: Mapping[str, FilterSpec]) -> None:
    allowed = _allowed_params(filters)
    unknown = tuple(sorted(name for name in request.query_params if name not in allowed))
    if not unknown:
        return
    raise _problem(
        "unknown-query-parameter",
        "Unknown query parameter",
        f"Unrecognized query parameter(s): {', '.join(unknown)}.",
        sorted(allowed),
    )


def _positive_int(raw: str | None, default: int, name: str) -> int:
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise _problem("invalid-query-parameter", "Invalid query parameter", f"{name} must be an integer.")
    if value < 1:
        raise _problem("invalid-query-parameter", "Invalid query parameter", f"{name} must be at least 1.")
    return value


def _parse_sort(raw: str | None, sortable: frozenset[str], default_sort: tuple[SortKey, ...]) -> tuple[SortKey, ...]:
    if raw is None:
        return default_sort
    keys = tuple(
        SortKey(field=token.removeprefix("-"), descending=token.startswith("-"))
        for token in (part.strip() for part in raw.split(","))
        if token
    )
    unknown = tuple(key.field for key in keys if key.field not in sortable)
    if unknown:
        raise _problem(
            "invalid-sort-field",
            "Invalid sort field",
            f"Cannot sort on: {', '.join(unknown)}.",
            sorted(sortable),
        )
    return keys or default_sort


def _order_by(keys: Sequence[SortKey], tiebreaker: str) -> OrderBy:
    tail = () if any(key.field == tiebreaker for key in keys) else (SortKey(field=tiebreaker, descending=False),)
    return tuple({key.field: ("desc" if key.descending else "asc")} for key in (*keys, *tail))


def _coerce(value: str, filter_type: FilterType, param: str) -> WhereLeaf:
    if filter_type == "number":
        try:
            return float(value)
        except ValueError:
            raise _problem("invalid-filter-value", "Invalid filter value", f"{param} must be a number.")
    if filter_type == "datetime":
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            raise _problem("invalid-filter-value", "Invalid filter value", f"{param} must be an ISO-8601 timestamp.")
    return value


def _bool(value: str, param: str) -> bool:
    if value.lower() in ("true", "1"):
        return True
    if value.lower() in ("false", "0"):
        return False
    raise _problem("invalid-filter-value", "Invalid filter value", f"{param} must be true or false.")


def _condition(field: str, op: FilterOp, raw: str, filter_type: FilterType, param: str) -> Where:
    if op == "is_null":
        return {field: None} if _bool(raw, param) else {field: {"not": None}}
    if op == "in":
        return {field: {"in": tuple(_coerce(part, filter_type, param) for part in raw.split(",") if part)}}
    if op == "contains":
        return {field: {"contains": raw, "mode": "insensitive"}}
    if op == "eq":
        return {field: _coerce(raw, filter_type, param)}
    return {field: {op: _coerce(raw, filter_type, param)}}


def _filter_conditions(request: Request, filters: Mapping[str, FilterSpec]) -> tuple[Where, ...]:
    return tuple(
        _condition(field, op, request.query_params[f"filter[{field}][{op}]"], spec.type, f"filter[{field}][{op}]")
        for field, spec in filters.items()
        for op in sorted(spec.ops)
        if f"filter[{field}][{op}]" in request.query_params
    )


def _search_condition(raw: str | None, searchable: frozenset[str]) -> tuple[Where, ...]:
    if not raw or not searchable:
        return ()
    return ({"OR": tuple({field: {"contains": raw, "mode": "insensitive"}} for field in sorted(searchable))},)


def build_query_plan(request: Request, spec: ListSpec[RowT], scope: Scope) -> QueryPlan:
    """Turn the query string into the executor's arguments, or raise a 400 problem.

    `scope` is derived from the caller, never from the query string, and is ANDed
    with the caller's filters so a filter can only ever narrow what they may read.
    """
    _reject_unknown_params(request, spec.filters)

    page = _positive_int(request.query_params.get("page"), 1, "page")
    page_size = min(
        _positive_int(request.query_params.get("page_size"), spec.default_page_size, "page_size"),
        spec.max_page_size,
    )
    keys = _parse_sort(request.query_params.get("sort"), spec.sortable, spec.default_sort)

    scope_conditions: tuple[Where, ...] = (scope.where,) if isinstance(scope, ScopeWhere) else ()
    conditions = (
        scope_conditions
        + _filter_conditions(request, spec.filters)
        + _search_condition(request.query_params.get("q"), spec.searchable)
    )

    return QueryPlan(
        where={"AND": conditions} if conditions else {},
        order=_order_by(keys, spec.tiebreaker),
        skip=(page - 1) * page_size,
        take=page_size,
        page=page,
        page_size=page_size,
    )


def _page_url(request: Request, page: int) -> str:
    others = tuple((key, value) for key, value in request.query_params.multi_items() if key != "page")
    return f"{request.url.path}?{urlencode((*others, ('page', page)))}"


def _links(request: Request, page: int, last_page: int) -> ListLinks:
    return ListLinks(
        self_link=_page_url(request, page),
        first=_page_url(request, 1),
        prev=_page_url(request, page - 1) if page > 1 else None,
        next=_page_url(request, page + 1) if page < last_page else None,
        last=_page_url(request, last_page),
    )


async def handle_list(
    request: Request,
    spec: ListSpec[RowT],
    executor: ListExecutor[RowT],
    caller: UserAPIKeyAuth,
) -> ListResponse:
    """Serve one page of `spec.resource` under the caller's scope."""
    scope = spec.scope(caller)
    if isinstance(scope, ScopeDenied):
        raise ManagementProblem(
            ProblemDetail(
                type=f"{PROBLEM_TYPE_BASE}forbidden",
                title="Forbidden",
                status=403,
                detail=scope.detail,
            )
        )

    plan = build_query_plan(request, spec, scope)
    total_count = await executor.count(plan.where)
    rows = await executor.find_many(where=plan.where, order=plan.order, skip=plan.skip, take=plan.take)
    total_pages = math.ceil(total_count / plan.page_size)

    return ListResponse(
        data=tuple(spec.serialize(row) for row in rows),
        meta=ListMeta(
            page=plan.page,
            page_size=plan.page_size,
            total_count=total_count,
            total_pages=total_pages,
        ),
        links=_links(request, plan.page, max(total_pages, 1)),
    )
