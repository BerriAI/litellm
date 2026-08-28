"""An in-memory `ListExecutor`, for list resources whose rows are computed rather than queried.

Answers the same `QueryPlan` a SQL executor would render through `where_sql` / `order_by_sql`,
so a filter or a sort means the same thing on either. `enrich_page` runs on the page slice and
never on the whole match set.
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from typing import Final, Generic, TypeAlias, TypeVar

from typing_extensions import assert_never

from litellm.proxy.list_api.list_framework import (
    AnyOf,
    Compare,
    ComparisonOp,
    FilterValue,
    IsNull,
    Predicate,
    QueryPlan,
    SortKey,
    Within,
)

TRow: Final = TypeVar("TRow")

Cell: TypeAlias = str | int | float | datetime | None
# A tuple-valued cell is a row's repeated field (a model group's providers, say). A predicate
# holds against it when it holds against any one element, the way an SQL join would answer.
Cells: TypeAlias = Mapping[str, Cell | tuple[Cell, ...]]


def _sign(cell: Cell, value: FilterValue) -> int | None:
    """None when the two values are not orderable against each other."""
    if isinstance(cell, str) and isinstance(value, str):
        return (cell > value) - (cell < value)
    if isinstance(cell, datetime) and isinstance(value, datetime):
        return (cell > value) - (cell < value)
    if isinstance(cell, (int, float)) and isinstance(value, (int, float)):
        return (cell > value) - (cell < value)
    return None


def _matches(cell: Cell, op: ComparisonOp, value: FilterValue) -> bool:
    """SQL's three-valued logic: a NULL cell satisfies no comparison, only `is_null`."""
    if cell is None:
        return False
    sign: Final = _sign(cell, value)
    match op:
        case "eq":
            return cell == value
        case "not":
            return cell != value
        case "contains":
            return str(value).casefold() in str(cell).casefold()
        case "gt":
            return sign is not None and sign > 0
        case "gte":
            return sign is not None and sign >= 0
        case "lt":
            return sign is not None and sign < 0
        case "lte":
            return sign is not None and sign <= 0
        case _:
            assert_never(op)


def _any_cell(cells: Cells, name: str, matches: Callable[[Cell], bool]) -> bool:
    cell: Final = cells.get(name)
    if isinstance(cell, tuple):
        return any(matches(item) for item in cell)
    return matches(cell)


def _holds(predicate: Predicate, cells: Cells) -> bool:
    match predicate:
        case Compare(field=name, op=op, value=value):
            return _any_cell(cells, name, lambda cell: _matches(cell, op, value))
        case Within(field=name, values=values):
            return _any_cell(cells, name, lambda cell: cell is not None and cell in values)
        case IsNull(field=name, negated=negated):
            return _any_cell(cells, name, lambda cell: (cell is None) != negated)
        case AnyOf(clauses=clauses):
            return any(_holds(clause, cells) for clause in clauses)
        case _:
            assert_never(predicate)


def _sort_key(cells: Cells, key: SortKey) -> tuple[bool, Cell | tuple[Cell, ...]]:
    """NULLS LAST in both directions, matching `order_by_sql`.

    The placeholder standing in for a null is only ever compared against another null's,
    because the rank ahead of it already separates nulls from the rest.
    """
    cell: Final = cells.get(key.field)
    return (cell is None) != key.descending, 0 if cell is None else cell


def _ordered(
    matched: Sequence[tuple[Cells, TRow]],
    order: tuple[SortKey, ...],
) -> Sequence[tuple[Cells, TRow]]:
    """Least significant key first: Python's sort is stable, so the most significant pass wins."""
    return reduce(
        lambda rows, key: sorted(rows, key=lambda pair: _sort_key(pair[0], key), reverse=key.descending),
        reversed(order),
        matched,
    )


async def _unchanged(rows: Sequence[TRow]) -> Sequence[TRow]:
    return rows


@dataclass(frozen=True, slots=True)
class InMemoryListExecutor(Generic[TRow]):
    """`cells` projects a row down to the values the spec's filters, search and sort read, so a
    plan can be applied without this module knowing the row type."""

    rows: Sequence[TRow]
    cells: Callable[[TRow], Cells]
    enrich_page: Callable[[Sequence[TRow]], Awaitable[Sequence[TRow]]] = _unchanged

    def _matching(self, where: tuple[Predicate, ...]) -> Sequence[tuple[Cells, TRow]]:
        return tuple(
            (cells, row)
            for cells, row in ((self.cells(row), row) for row in self.rows)
            if all(_holds(predicate, cells) for predicate in where)
        )

    async def count(self, where: tuple[Predicate, ...]) -> int:
        return len(self._matching(where))

    async def find_many(self, plan: QueryPlan) -> Sequence[TRow]:
        page: Final = _ordered(self._matching(plan.where), plan.order)[plan.skip : plan.skip + plan.take]
        return await self.enrich_page(tuple(row for _, row in page))
