"""An ``AsyncMock`` standing in for a ``prisma_client.db`` method that reached the engine,
marking the DB I/O witness the way ``_TrackedPrismaEngine`` does, so the producer under test
emits its service event."""

from typing import TypeVar
from unittest.mock import AsyncMock

from litellm.proxy.db.log_db_metrics import record_db_io

_T = TypeVar("_T")


def engine_call(return_value: _T | None = None) -> AsyncMock:
    async def run(*args: object, **kwargs: object) -> _T | None:
        record_db_io()
        return return_value

    return AsyncMock(side_effect=run)
