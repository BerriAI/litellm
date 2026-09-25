"""
Prisma `{"in": [...]}` filters whose value list may outgrow Postgres's bind-parameter cap.

A membership filter binds one parameter per value and Postgres caps a statement at 32,767,
so each operation here splits the deduplicated values into chunks of `IN_LIST_CHUNK_SIZE`,
runs them one after another (a transaction handle works as `table`), and combines the
results. An empty list returns without querying.

`not_in` cannot be chunked: a row must be outside every chunk at once. Such sites need
`<> ALL($1::text[])` in raw SQL or a relation filter instead.
"""

from collections.abc import Awaitable, Callable, Hashable, Iterable, Mapping
from itertools import accumulate, repeat, takewhile
from typing import Final, Literal, TypeAlias, TypeVar

from litellm.repositories.prisma_protocols import CountTable, DeleteManyTable, FindManyTable, UpdateManyTable

IN_LIST_CHUNK_SIZE: Final = 5_000
LOGICAL_KEYS: Final = frozenset({"AND", "OR", "NOT"})

RowT: Final = TypeVar("RowT")
ResultT: Final = TypeVar("ResultT")

Atomicity: TypeAlias = Literal["caller_transaction", "per_chunk_ok"]
"""More than `IN_LIST_CHUNK_SIZE` values means more than one statement. `caller_transaction`
states `table` is a transaction handle, so the chunks commit together; `per_chunk_ok` states
the caller accepts earlier chunks staying applied when a later one fails."""


class SameFieldFilterError(ValueError):
    pass


def _as_clauses(value: object) -> tuple[object, ...]:
    match value:
        case list() | tuple():
            return tuple(value)  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]  # filters nest arbitrary data
        case _:
            return (value,)


def _logical_clauses(clause: object) -> tuple[object, ...]:
    match clause:
        case Mapping():
            return tuple(nested for key in LOGICAL_KEYS if key in clause for nested in _as_clauses(clause[key]))  # pyright: ignore[reportUnknownArgumentType]  # filters nest arbitrary data
        case _:
            return ()


def _filters_field(where: Mapping[str, object], field: str) -> bool:
    """Whether `field` is filtered in `where` or in any AND / OR / NOT clause under it, walked level by level."""
    levels: Final = accumulate(
        repeat(None),
        lambda level, _: tuple(nested for clause in level for nested in _logical_clauses(clause)),
        initial=(where,),
    )
    return any(isinstance(clause, Mapping) and field in clause for level in takewhile(bool, levels) for clause in level)


def _chunk_filter(field: str, chunk: tuple[Hashable, ...], where: Mapping[str, object] | None) -> Mapping[str, object]:
    membership: Final = {field: {"in": chunk}}  # mutable-ok: prisma's query builder only accepts dict filters
    if where is None:
        return membership
    return {"AND": (dict(where), membership)}  # mutable-ok: prisma's query builder only accepts dict filters


async def _each_chunk(
    field: str,
    values: Iterable[Hashable],
    where: Mapping[str, object] | None,
    run: Callable[[Mapping[str, object]], Awaitable[ResultT]],
) -> tuple[ResultT, ...]:
    if where is not None and _filters_field(where, field):
        raise SameFieldFilterError(f"`where` already filters `{field}`; fold that condition into the values instead")
    unique: Final = tuple(dict.fromkeys(values))
    starts: Final = range(0, len(unique), IN_LIST_CHUNK_SIZE)
    return tuple(
        [await run(_chunk_filter(field, unique[start : start + IN_LIST_CHUNK_SIZE], where)) for start in starts]
    )


async def find_many_in(
    table: FindManyTable[RowT], field: str, values: Iterable[Hashable], *, where: Mapping[str, object] | None = None
) -> tuple[RowT, ...]:
    """Rows in chunk order. No take/skip/cursor/order/distinct: none of them survive a split."""
    pages: Final = await _each_chunk(field, values, where, lambda chunk: table.find_many(where=chunk))
    return tuple(row for page in pages for row in page)


async def count_in(
    table: CountTable, field: str, values: Iterable[Hashable], *, where: Mapping[str, object] | None = None
) -> int:
    return sum(await _each_chunk(field, values, where, lambda chunk: table.count(where=chunk)))


async def update_many_in(
    table: UpdateManyTable,
    field: str,
    values: Iterable[Hashable],
    *,
    data: Mapping[str, object],
    atomicity: Atomicity,
    where: Mapping[str, object] | None = None,
) -> int:
    payload: Final = dict(data)  # mutable-ok: prisma's query builder only accepts dict payloads
    return sum(await _each_chunk(field, values, where, lambda chunk: table.update_many(data=payload, where=chunk)))


async def delete_many_in(
    table: DeleteManyTable,
    field: str,
    values: Iterable[Hashable],
    *,
    atomicity: Atomicity,
    where: Mapping[str, object] | None = None,
) -> int:
    return sum(await _each_chunk(field, values, where, lambda chunk: table.delete_many(where=chunk)))
