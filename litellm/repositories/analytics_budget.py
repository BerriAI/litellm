from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from datetime import timedelta
from threading import BoundedSemaphore
from typing import Final, Protocol

from prisma.errors import RawQueryError, TransactionExpiredError, TransactionNotStartedError
from pydantic import BaseModel


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
_CONFIGURE: Final = """
SELECT
    set_config('statement_timeout', LEAST(NULLIF(
        EXTRACT(epoch FROM current_setting('statement_timeout')::interval)*1000,0),5000)::text,true),
    set_config('lock_timeout', LEAST(NULLIF(
        EXTRACT(epoch FROM current_setting('lock_timeout')::interval)*1000,0),100)::text,true),
    set_config('work_mem', LEAST(pg_size_bytes(current_setting('work_mem')),1048576)::text || 'B',true),
    set_config('max_parallel_workers_per_gather','0',true),
    CASE WHEN $1::boolean THEN pg_try_advisory_xact_lock(1876539401,2)
         WHEN pg_try_advisory_xact_lock(1876539401,0) THEN true
         ELSE pg_try_advisory_xact_lock(1876539401,1) END AS acquired
"""


class BoundedAnalyticsDatabase:
    def __init__(self, database: AnalyticsDatabase, *, background: bool = False) -> None:
        self._database = database
        self._background = background
        self._admission = _BACKGROUND if background else _FOREGROUND

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        if not self._admission.acquire(blocking=False):
            raise AnalyticsCapacityError("Analytics capacity is busy. Try again shortly")
        try:
            async with self._database.tx(
                max_wait=timedelta(milliseconds=250), timeout=timedelta(seconds=6)
            ) as transaction:
                configuration: Final = await transaction.query_raw(_CONFIGURE, self._background)
                if not configuration or not _Admission.model_validate(configuration[0]).acquired:
                    raise AnalyticsCapacityError("Analytics capacity is busy. Try again shortly")
                return await transaction.query_raw(query, *args)
        except RawQueryError as error:
            if _PostgresError.model_validate(error.meta or {}).code not in ("57014", "55P03"):
                raise
            raise AnalyticsCapacityError("Analytics exceeded its query budget. Try again shortly") from error
        except (TransactionExpiredError, TransactionNotStartedError) as error:
            raise AnalyticsCapacityError("Analytics exceeded its connection budget. Try again shortly") from error
        finally:
            self._admission.release()
