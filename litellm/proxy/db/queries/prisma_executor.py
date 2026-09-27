from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Protocol

from prisma.errors import RawQueryError
from pydantic import BaseModel, JsonValue

from litellm.proxy.db.queries.generated.models import DbFailure, ForeignKeyViolation, UniqueViolation, WriteConflict

FAILURES_BY_SQLSTATE: Final[Mapping[str, DbFailure]] = MappingProxyType(
    {
        "23505": UniqueViolation(),
        "23503": ForeignKeyViolation(),
        "40001": WriteConflict(),
        "40P01": WriteConflict(),
    }
)


class RawClient(Protocol):
    async def query_raw(self, query: str, *args: JsonValue) -> Sequence[Mapping[str, object]]: ...

    async def execute_raw(self, query: str, *args: JsonValue) -> int: ...


class _RawQueryMeta(BaseModel):
    code: str | None = None


def _failure(error: RawQueryError) -> DbFailure | None:
    meta: Final = _RawQueryMeta.model_validate(error.meta if isinstance(error.meta, Mapping) else {})
    return FAILURES_BY_SQLSTATE.get(meta.code or "")


@dataclass(frozen=True, slots=True)
class PrismaExecutor:
    client: RawClient

    async def query(self, sql: str, arguments: tuple[JsonValue, ...]) -> Sequence[Mapping[str, object]] | DbFailure:
        try:
            return await self.client.query_raw(sql, *arguments)
        except RawQueryError as error:
            failure: Final = _failure(error)
            if failure is None:
                raise
            return failure

    async def execute(self, sql: str, arguments: tuple[JsonValue, ...]) -> int | DbFailure:
        try:
            return await self.client.execute_raw(sql, *arguments)
        except RawQueryError as error:
            failure: Final = _failure(error)
            if failure is None:
                raise
            return failure
