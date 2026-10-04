"""A ``ServiceTypes.DB`` event around Prisma I/O that ``@log_db_metrics`` cannot wrap.

The spend flush, the spend-log batch insert and the background jobs run raw
``prisma_client.db`` statements and transactions, often inside retry loops, so the
decorator (one event per decorated coroutine) cannot name the table each round
trip touches. ``db_span`` emits one success or failure event per round trip,
carrying the raw ``call_type`` for the metric labels and the Prisma model on
``table_name`` so OTel renders ``postgres.{verb} {table}``; ``db_spanned`` is the
same event around a thunk, for the retry helpers that take one. The outermost
producer owns the event: a ``db_span`` nested in another ``db_span`` or in a
decorated helper emits nothing, so one transaction stays one span, and a block
whose Prisma client never reached the engine emits nothing at all.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Final, TypeVar

from litellm._logging import verbose_proxy_logger
from litellm._service_logger import ServiceLogging, ServiceTypes
from litellm.proxy.db.log_db_metrics import _is_exception_related_to_db, claim_db_io, db_io_claimed

_T = TypeVar("_T")


def _service_logging() -> ServiceLogging | None:
    try:
        from litellm.proxy.proxy_server import proxy_logging_obj
    except ImportError:
        return None
    return proxy_logging_obj.service_logging_obj


def _event_metadata(table: str | None, operation: str | None) -> dict[str, str]:
    pairs: Final = (("table_name", table), ("db_operation", operation))
    return {key: value for key, value in pairs if value is not None}


async def _emit_failure(
    service_logging: ServiceLogging,
    call_type: str,
    event_metadata: Mapping[str, str],
    start_time: datetime,
    error: Exception,
) -> None:
    end_time: Final = datetime.now()
    try:
        await service_logging.async_service_failure_hook(
            error=error,
            service=ServiceTypes.DB,
            call_type=call_type,
            parent_otel_span=None,
            duration=(end_time - start_time).total_seconds(),
            start_time=start_time,
            end_time=end_time,
            event_metadata=dict(event_metadata),
        )
    except Exception as hook_error:
        verbose_proxy_logger.debug("db_span: failure hook raised for %s: %s", call_type, hook_error)


@asynccontextmanager
async def db_span(call_type: str, table: str | None, operation: str | None = None) -> AsyncGenerator[None]:
    if db_io_claimed():
        yield
        return
    service_logging: Final = _service_logging()
    start_time: Final = datetime.now()
    event_metadata: Final = _event_metadata(table, operation)
    with claim_db_io() as witness:
        try:
            yield
        except Exception as e:
            if service_logging is not None and _is_exception_related_to_db(e):
                await _emit_failure(service_logging, call_type, event_metadata, start_time, e)
            raise
    if service_logging is None or not witness.touched:
        return
    end_time_ok: Final = datetime.now()
    asyncio.create_task(
        service_logging.async_service_success_hook(
            service=ServiceTypes.DB,
            call_type=call_type,
            parent_otel_span=None,
            duration=(end_time_ok - start_time).total_seconds(),
            start_time=start_time,
            end_time=end_time_ok,
            event_metadata=event_metadata,
        )
    )


async def db_spanned(call_type: str, table: str | None, load: Callable[[], Awaitable[_T]]) -> _T:
    async with db_span(call_type, table):
        return await load()
