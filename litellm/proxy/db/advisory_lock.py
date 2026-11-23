import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Generic, Protocol, TypeVar

from litellm._logging import verbose_proxy_logger

ADVISORY_LOCK_SQL: Final = "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtext($1))"

_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class AdvisoryLockUnavailable:
    pass


@dataclass(frozen=True, slots=True)
class UncommittedResult(Generic[_Result]):
    value: _Result


class AdvisoryLockTransaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...


class _TransactionalDb(Protocol):
    def tx(self, *, timeout: timedelta) -> AbstractAsyncContextManager[AdvisoryLockTransaction]: ...


@dataclass(frozen=True, slots=True)
class PrismaAdvisoryLock:
    db: _TransactionalDb
    timeout: timedelta

    async def while_held(
        self, key: str, action: Callable[[AdvisoryLockTransaction], Awaitable[_Result]]
    ) -> _Result | UncommittedResult[_Result] | AdvisoryLockUnavailable:
        finished: Final[asyncio.Future[_Result]] = asyncio.get_running_loop().create_future()
        try:
            async with self.db.tx(timeout=self.timeout) as transaction:
                await transaction.query_raw(ADVISORY_LOCK_SQL, key)
                finished.set_result(await action(transaction))
        except Exception as exc:  # noqa: BLE001  # callers fall back to work that needs no lock
            return _unfinished(finished, exc)
        return finished.result()


def _unfinished(
    finished: asyncio.Future[_Result], error: Exception
) -> UncommittedResult[_Result] | AdvisoryLockUnavailable:
    if finished.done():
        verbose_proxy_logger.error(
            "The work done under the Postgres advisory lock was not committed (%s)", type(error).__name__
        )
        return UncommittedResult(finished.result())
    verbose_proxy_logger.warning(
        "Could not hold the Postgres advisory lock (%s), so the work that needs it is skipped", type(error).__name__
    )
    return AdvisoryLockUnavailable()
