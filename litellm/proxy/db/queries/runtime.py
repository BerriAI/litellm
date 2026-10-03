from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Generic, Protocol, TypeVar

from pydantic import BaseModel, JsonValue

from litellm.proxy.db.queries.generated.models import DbFailure, NotFound

ParamsT = TypeVar("ParamsT", bound=BaseModel)
RowT = TypeVar("RowT", bound=BaseModel)


class Executor(Protocol):
    async def query(self, sql: str, arguments: tuple[JsonValue, ...]) -> Sequence[Mapping[str, object]] | DbFailure: ...

    async def execute(self, sql: str, arguments: tuple[JsonValue, ...]) -> int | DbFailure: ...


@dataclass(frozen=True, slots=True)
class ManyQuery(Generic[ParamsT, RowT]):
    sql: str
    arguments: tuple[str, ...]
    params: type[ParamsT]
    row: type[RowT]


@dataclass(frozen=True, slots=True)
class OptionalQuery(Generic[ParamsT, RowT]):
    sql: str
    arguments: tuple[str, ...]
    params: type[ParamsT]
    row: type[RowT]


@dataclass(frozen=True, slots=True)
class OneQuery(Generic[ParamsT, RowT]):
    sql: str
    arguments: tuple[str, ...]
    params: type[ParamsT]
    row: type[RowT]


@dataclass(frozen=True, slots=True)
class ExecuteQuery(Generic[ParamsT]):
    sql: str
    arguments: tuple[str, ...]
    params: type[ParamsT]


def _bind(arguments: tuple[str, ...], params: BaseModel) -> tuple[JsonValue, ...]:
    values: Final[dict[str, JsonValue]] = params.model_dump(mode="json")
    return tuple(values[name] for name in arguments)


async def _rows(
    executor: Executor, query: ManyQuery[ParamsT, RowT] | OptionalQuery[ParamsT, RowT] | OneQuery[ParamsT, RowT], params: ParamsT
) -> tuple[RowT, ...] | DbFailure:
    rows: Final = await executor.query(query.sql, _bind(query.arguments, params))
    if not isinstance(rows, Sequence):
        return rows
    return tuple(query.row.model_validate(row) for row in rows)


async def fetch_many(executor: Executor, query: ManyQuery[ParamsT, RowT], params: ParamsT) -> tuple[RowT, ...] | DbFailure:
    return await _rows(executor, query, params)


async def fetch_optional(
    executor: Executor, query: OptionalQuery[ParamsT, RowT], params: ParamsT
) -> RowT | None | DbFailure:
    rows: Final = await _rows(executor, query, params)
    if not isinstance(rows, tuple):
        return rows
    return rows[0] if rows else None


async def fetch_one(executor: Executor, query: OneQuery[ParamsT, RowT], params: ParamsT) -> RowT | DbFailure:
    rows: Final = await _rows(executor, query, params)
    if not isinstance(rows, tuple):
        return rows
    return rows[0] if rows else NotFound()


async def execute(executor: Executor, query: ExecuteQuery[ParamsT], params: ParamsT) -> int | DbFailure:
    return await executor.execute(query.sql, _bind(query.arguments, params))
