import asyncio
from collections.abc import AsyncGenerator, Generator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, contextmanager
from contextvars import ContextVar
from datetime import timedelta
from threading import BoundedSemaphore
from typing import Final, Protocol

from prisma.engine.errors import EngineRequestError
from prisma.errors import RawQueryError, TransactionExpiredError, TransactionNotStartedError
from pydantic import BaseModel

from litellm.litellm_core_utils.pending_operations import defer_until_operations_finish


class AnalyticsTransaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]: ...


class AnalyticsDatabase(Protocol):
    def tx(self, *, max_wait: timedelta, timeout: timedelta) -> AbstractAsyncContextManager[AnalyticsTransaction]: ...


class AnalyticsCapacityError(RuntimeError):
    pass


class _Admission(BaseModel, frozen=True):
    acquired: bool


class _PostgresError(BaseModel, frozen=True):
    code: str = ""


_FOREGROUND: Final = BoundedSemaphore(2)
_BACKGROUND: Final = BoundedSemaphore(1)
_DEADLINE: Final = ContextVar[float | None]("analytics_deadline", default=None)
_CONFIGURE: Final = """
SELECT
    set_config('statement_timeout', LEAST(NULLIF(
        EXTRACT(epoch FROM current_setting('statement_timeout')::interval)*1000,0),$2::int)::text,true),
    set_config('lock_timeout', LEAST(NULLIF(
        EXTRACT(epoch FROM current_setting('lock_timeout')::interval)*1000,0),100,$2::int)::text,true),
    set_config('work_mem', LEAST(pg_size_bytes(current_setting('work_mem')),1048576)::text || 'B',true),
    CASE WHEN current_setting('hash_mem_multiplier',true) IS NOT NULL
         THEN set_config('hash_mem_multiplier','1',true) END,
    set_config('jit','off',true),
    set_config('max_parallel_workers_per_gather','0',true),
    CASE WHEN $1::boolean THEN pg_try_advisory_xact_lock(1876539401,2)
         WHEN pg_try_advisory_xact_lock(1876539401,0) THEN true
         ELSE pg_try_advisory_xact_lock(1876539401,1) END AS acquired
"""


@contextmanager
def analytics_deadline(deadline: float) -> Generator[None, None, None]:
    current: Final = _DEADLINE.get()
    token: Final = _DEADLINE.set(deadline if current is None else min(current, deadline))
    try:
        yield
    finally:
        _DEADLINE.reset(token)


def _remaining_seconds() -> float | None:
    deadline: Final = _DEADLINE.get()
    if deadline is None:
        return None
    remaining: Final = deadline - asyncio.get_running_loop().time()
    if remaining <= 0:
        raise AnalyticsCapacityError("Analytics work budget is exhausted")
    return remaining


def _milliseconds(limit: int) -> int:
    remaining: Final = _remaining_seconds()
    return limit if remaining is None else min(limit, max(1, int(remaining * 1000)))


@asynccontextmanager
async def _bounded_transaction(
    manager: AbstractAsyncContextManager[AnalyticsTransaction],
) -> AsyncGenerator[AnalyticsTransaction, None]:
    transaction: Final = await manager.__aenter__()
    try:
        yield transaction
    except BaseException as error:
        suppressed: Final = await asyncio.wait_for(
            manager.__aexit__(type(error), error, error.__traceback__), timeout=1
        )
        if not suppressed:
            raise
    else:
        await asyncio.wait_for(manager.__aexit__(None, None, None), timeout=1)


class BoundedAnalyticsDatabase:
    def __init__(self, database: AnalyticsDatabase, *, background: bool = False) -> None:
        self._database = database
        self._background = background
        self._admission = _BACKGROUND if background else _FOREGROUND

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if self._background:
            remaining: Final = _remaining_seconds()
            await asyncio.sleep(0.1 if remaining is None else min(0.1, remaining))
            _remaining_seconds()
        if not self._admission.acquire(blocking=False):
            raise AnalyticsCapacityError("Analytics capacity is busy. Try again shortly")
        with defer_until_operations_finish(self._admission.release):
            try:
                available: Final = _remaining_seconds()
                return await asyncio.wait_for(
                    self._query(query, args), timeout=6.25 if available is None else min(6.25, available)
                )
            except EngineRequestError as error:
                if error.response.status not in (503, 504):
                    raise
                raise AnalyticsCapacityError("Analytics exceeded its connection budget. Try again shortly") from error
            except RawQueryError as error:
                if _PostgresError.model_validate(error.meta or {}).code not in ("57014", "55P03"):
                    raise
                raise AnalyticsCapacityError("Analytics exceeded its query budget. Try again shortly") from error
            except (
                TransactionExpiredError,
                TransactionNotStartedError,
                asyncio.TimeoutError,
            ) as error:
                raise AnalyticsCapacityError("Analytics exceeded its connection budget. Try again shortly") from error

    async def _query(self, query: str, args: tuple[object, ...]) -> Sequence[Mapping[str, object]]:
        async with _bounded_transaction(
            self._database.tx(
                max_wait=timedelta(milliseconds=_milliseconds(250)),
                timeout=timedelta(milliseconds=_milliseconds(6000)),
            )
        ) as transaction:
            configuration: Final = await transaction.query_raw(_CONFIGURE, self._background, _milliseconds(5000))
            if not configuration or not _Admission.model_validate(configuration[0]).acquired:
                raise AnalyticsCapacityError("Analytics capacity is busy. Try again shortly")
            _remaining_seconds()
            return await transaction.query_raw(query, *args)
