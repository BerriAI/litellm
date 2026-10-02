"""
Handles logging DB success/failure to ServiceLogger()

ServiceLogger() then sends DB logs to Prometheus, OTEL, Datadog etc
"""

import asyncio
from collections.abc import Callable
from contextvars import ContextVar
from datetime import datetime
from functools import wraps
from typing import Final

from litellm._service_logger import ServiceTypes
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs


def _safe_db_event_metadata(kwargs: dict) -> dict[str, str] | None:
    """Minimal, non-sensitive ``event_metadata`` for a DB service log.

    The raw ``kwargs``/``args`` carry live objects (Prisma client, OTel spans)
    and secrets (tokens), none of which belongs on a span — so we surface only
    the table name when present. Everything else is dropped.
    """
    table_name: Final = kwargs.get("table_name")
    return {"table_name": table_name} if isinstance(table_name, str) else None


class _DbIoWitness:
    """One per ``log_db_metrics`` activation; marked by the Prisma engine wrapper on every query.

    Marks propagate to the enclosing activation so a decorated function that reaches the
    database only through another decorated function still records its own event.
    """

    __slots__ = ("_parent", "_touched")

    def __init__(self, parent: "_DbIoWitness | None") -> None:
        self._parent: Final = parent
        self._touched = False

    @property
    def touched(self) -> bool:
        return self._touched

    def mark(self) -> None:
        self._touched = True
        if self._parent is not None:
            self._parent.mark()


_db_io_witness: Final[ContextVar["_DbIoWitness | None"]] = ContextVar("litellm_db_io_witness", default=None)


def record_db_io() -> None:
    """Tell the enclosing ``log_db_metrics`` activations that a real database round trip happened."""
    witness: Final = _db_io_witness.get()
    if witness is not None:
        witness.mark()


def log_db_metrics(func):
    """
    Decorator to log the duration of a DB related function to ServiceLogger()

    The DB success event is emitted only when the wrapped call performed database I/O
    (reported through ``record_db_io`` by the Prisma engine wrapper), so cache hits in
    cache-first helpers produce no ``postgres <fn>`` span or DB service metric

    Handles logging DB success/failure to ServiceLogger(), which logs to Prometheus, OTEL, Datadog

    When logging Failure it checks if the Exception is a PrismaError, httpx.ConnectError or httpx.TimeoutException and then logs that as a DB Service Failure

    Args:
        func: The function to be decorated

    Returns:
        Result from the decorated function

    Raises:
        Exception: If the decorated function raises an exception
    """

    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time: Final[datetime] = datetime.now()
        witness: Final = _DbIoWitness(parent=_db_io_witness.get())
        witness_token: Final = _db_io_witness.set(witness)

        try:
            result: Final = await func(*args, **kwargs)
            end_time: datetime = datetime.now()
            from litellm.proxy.proxy_server import proxy_logging_obj

            if "PROXY" not in func.__name__:
                if not witness.touched:
                    return result
                asyncio.create_task(
                    proxy_logging_obj.service_logging_obj.async_service_success_hook(
                        service=ServiceTypes.DB,
                        call_type=func.__name__,
                        parent_otel_span=kwargs.get("parent_otel_span", None),
                        duration=(end_time - start_time).total_seconds(),
                        start_time=start_time,
                        end_time=end_time,
                        event_metadata=_safe_db_event_metadata(kwargs),
                    )
                )
            elif (
                # in litellm custom callbacks kwargs is passed as arg[0]
                # https://docs.litellm.ai/docs/observability/custom_callback#callback-functions
                args is not None and len(args) > 1 and isinstance(args[1], dict)
            ):
                passed_kwargs: Final = args[1]
                parent_otel_span: Final = _get_parent_otel_span_from_kwargs(kwargs=passed_kwargs)
                if parent_otel_span is not None:
                    # No metadata dump: identity rides on Baggage, and the full
                    # request metadata (auth blob, response headers, tokens) must
                    # not land on a span.
                    asyncio.create_task(
                        proxy_logging_obj.service_logging_obj.async_service_success_hook(
                            service=ServiceTypes.BATCH_WRITE_TO_DB,
                            call_type=func.__name__,
                            parent_otel_span=parent_otel_span,
                            duration=0.0,
                            start_time=start_time,
                            end_time=end_time,
                            event_metadata=None,
                        )
                    )
            # end of logging to otel
            return result
        except Exception as e:
            end_time: datetime = datetime.now()
            await _handle_logging_db_exception(
                e=e,
                func=func,
                kwargs=kwargs,
                args=args,
                start_time=start_time,
                end_time=end_time,
            )
            raise e
        finally:
            _db_io_witness.reset(witness_token)

    return wrapper


def _is_exception_related_to_db(e: Exception) -> bool:
    """
    Returns True if the exception is related to the DB
    """

    import httpx
    from prisma.errors import PrismaError

    return isinstance(e, (PrismaError, httpx.ConnectError, httpx.TimeoutException))


async def _handle_logging_db_exception(
    e: Exception,
    func: Callable,
    kwargs: dict,
    args: tuple,
    start_time: datetime,
    end_time: datetime,
) -> None:
    from litellm.proxy.proxy_server import proxy_logging_obj

    # don't log this as a DB Service Failure, if the DB did not raise an exception
    if _is_exception_related_to_db(e) is not True:
        return

    await proxy_logging_obj.service_logging_obj.async_service_failure_hook(
        error=e,
        service=ServiceTypes.DB,
        call_type=func.__name__,
        parent_otel_span=kwargs.get("parent_otel_span"),
        duration=(end_time - start_time).total_seconds(),
        start_time=start_time,
        end_time=end_time,
        event_metadata=_safe_db_event_metadata(kwargs),
    )
