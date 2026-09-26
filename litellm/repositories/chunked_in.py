"""
Prisma `{"in": [...]}` filters whose value list may outgrow Postgres's bind-parameter cap.

A membership filter binds one parameter per value and Postgres caps a statement at 32,767,
so each operation here splits the deduplicated values into chunks of `chunk_size` values
(`IN_LIST_CHUNK_SIZE` by default, at most `MAX_IN_LIST_CHUNK_SIZE` so the rest of the filter
keeps headroom under the cap), runs them one after another (a transaction handle works as `table`), and combines the
results. An empty list returns without querying.

`not_in` cannot be chunked: a row must be outside every chunk at once. Such sites need
`<> ALL($1::text[])` in raw SQL or a relation filter instead.
"""

from collections.abc import Awaitable, Callable, Hashable, Iterable, Mapping
from itertools import accumulate, chain, repeat, takewhile
from typing import Final, Literal, TypeAlias, TypeVar

from litellm.repositories.prisma_protocols import CountTable, DeleteManyTable, FindManyTable, UpdateManyTable

IN_LIST_CHUNK_SIZE: Final = 5_000
MAX_IN_LIST_CHUNK_SIZE: Final = 30_000
LOGICAL_KEYS: Final = frozenset({"AND", "OR", "NOT"})

RowT: Final = TypeVar("RowT")
ResultT: Final = TypeVar("ResultT")

Atomicity: TypeAlias = Literal["caller_transaction", "per_chunk_ok"]
"""More than `chunk_size` values means more than one statement. `caller_transaction`
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
            return tuple(chain.from_iterable(_as_clauses(clause[key]) for key in LOGICAL_KEYS if key in clause))  # pyright: ignore[reportUnknownArgumentType]  # filters nest arbitrary data
        case _:
            return ()


def _filters_field(where: Mapping[str, object], field: str) -> bool:
    """Whether `field` is filtered in `where` or in any AND / OR / NOT clause under it, walked level by level."""
    levels: Final = accumulate(
        repeat(None),
        lambda level, _: tuple(chain.from_iterable(map(_logical_clauses, level))),
        initial=(where,),
    )
    return any(
        isinstance(clause, Mapping) and field in clause for clause in chain.from_iterable(takewhile(bool, levels))
    )


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
    chunk_size: int,
) -> tuple[ResultT, ...]:
    if not 1 <= chunk_size <= MAX_IN_LIST_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be between 1 and {MAX_IN_LIST_CHUNK_SIZE:,}, got {chunk_size}")
    if where is not None and _filters_field(where, field):
        raise SameFieldFilterError(f"`where` already filters `{field}`; fold that condition into the values instead")
    unique: Final = tuple(dict.fromkeys(values))
    starts: Final = range(0, len(unique), chunk_size)
    return tuple([await run(_chunk_filter(field, unique[start : start + chunk_size], where)) for start in starts])


async def find_many_in(
    table: FindManyTable[RowT],
    field: str,
    values: Iterable[Hashable],
    *,
    where: Mapping[str, object] | None = None,
    chunk_size: int = IN_LIST_CHUNK_SIZE,
) -> tuple[RowT, ...]:
    """Rows in chunk order. No take/skip/cursor/order/distinct: none of them survive a split."""
    pages: Final = await _each_chunk(field, values, where, lambda chunk: table.find_many(where=chunk), chunk_size)
    return tuple(chain.from_iterable(pages))


async def count_in(
    table: CountTable,
    field: str,
    values: Iterable[Hashable],
    *,
    where: Mapping[str, object] | None = None,
    chunk_size: int = IN_LIST_CHUNK_SIZE,
) -> int:
    return sum(await _each_chunk(field, values, where, lambda chunk: table.count(where=chunk), chunk_size))


async def update_many_in(
    table: UpdateManyTable,
    field: str,
    values: Iterable[Hashable],
    *,
    data: Mapping[str, object],
    atomicity: Atomicity,
    where: Mapping[str, object] | None = None,
    chunk_size: int = IN_LIST_CHUNK_SIZE,
) -> int:
    payload: Final = dict(data)  # mutable-ok: prisma's query builder only accepts dict payloads
    return sum(
        await _each_chunk(field, values, where, lambda chunk: table.update_many(data=payload, where=chunk), chunk_size)
    )


async def delete_many_in(
    table: DeleteManyTable,
    field: str,
    values: Iterable[Hashable],
    *,
    atomicity: Atomicity,
    where: Mapping[str, object] | None = None,
    chunk_size: int = IN_LIST_CHUNK_SIZE,
) -> int:
    return sum(await _each_chunk(field, values, where, lambda chunk: table.delete_many(where=chunk), chunk_size))
