"""An ``AsyncMock`` standing in for a ``prisma_client.db`` method that reached the engine,
marking the DB I/O witness the way ``_TrackedPrismaEngine`` does, so the producer under test
emits its service event."""

from typing import Final, TypeVar
from unittest.mock import AsyncMock, MagicMock

from prisma.errors import DataError

from litellm.proxy.db.log_db_metrics import record_db_io

_T = TypeVar("_T")


def engine_call(return_value: _T | None = None) -> AsyncMock:
    async def run(*args: object, **kwargs: object) -> _T | None:
        record_db_io()
        return return_value

    return AsyncMock(side_effect=run)


class RecordingReadTransaction:
    """The transaction ``query_raw_within_statement_timeout`` opens on ``read_db``, keeping its ``SET LOCAL`` statements."""

    def __init__(self, query_raw: AsyncMock) -> None:
        self.statements: tuple[str, ...] = ()
        self.query_raw: AsyncMock = query_raw

    async def execute_raw(self, sql: str) -> int:
        self.statements = (*self.statements, sql)
        return 0

    async def __aenter__(self) -> "RecordingReadTransaction":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass


def statement_timeout_reads(prisma: MagicMock, query_raw: AsyncMock) -> RecordingReadTransaction:
    """Make ``prisma.read_db.tx()`` hand out one recording transaction whose raw reads answer with ``query_raw``."""
    transaction: Final = RecordingReadTransaction(query_raw)
    prisma.read_db.tx = MagicMock(return_value=transaction)
    return transaction


def statement_timeout_error() -> DataError:
    """The error prisma raises when Postgres cancels a statement under ``statement_timeout`` (SQLSTATE 57014)."""
    message: Final = "canceling statement due to statement timeout"
    return DataError(
        data={
            "user_facing_error": {
                "is_panic": False,
                "message": f"Error querying the database: {message}",
                "meta": {"code": "57014", "message": message},
            }
        }
    )


class _DelegatingReadTransaction:
    """``read_db.tx()`` of a hand-written prisma fake: raw reads go to the fake's ``db.query_raw``."""

    def __init__(self, db: object) -> None:
        self.db: Final = db
        self.statements: tuple[str, ...] = ()

    async def execute_raw(self, sql: str) -> int:
        self.statements = (*self.statements, sql)
        return 0

    async def query_raw(self, sql: str, *args: object) -> object:
        return await getattr(self.db, "query_raw")(sql, *args)

    async def __aenter__(self) -> "_DelegatingReadTransaction":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        pass


class ReadTargetOf:
    """What a prisma fake returns from ``read_db``: ``tx()`` opens a transaction over the fake's ``db``."""

    def __init__(self, db: object) -> None:
        self.db: Final = db

    def tx(self, **kwargs: object) -> _DelegatingReadTransaction:
        return _DelegatingReadTransaction(self.db)
