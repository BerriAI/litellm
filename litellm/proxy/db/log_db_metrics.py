"""
Handles logging DB success/failure to ServiceLogger()

ServiceLogger() then sends DB logs to Prometheus, OTEL, Datadog etc
"""

import asyncio
from datetime import datetime
from functools import wraps
from typing import Callable, Dict, Optional, Tuple

from litellm._service_logger import ServiceTypes
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs


def _safe_db_event_metadata(kwargs: Dict) -> Optional[Dict[str, str]]:
    """Minimal, non-sensitive ``event_metadata`` for a DB service log.

    The raw ``kwargs``/``args`` carry live objects (Prisma client, OTel spans)
    and secrets (tokens), none of which belongs on a span — so we surface only
    the table name when present. Everything else is dropped.
    """
    table_name = kwargs.get("table_name")
    return {"table_name": table_name} if isinstance(table_name, str) else None


def log_db_metrics(func):
    """
    Decorator to log the duration of a DB related function to ServiceLogger()

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
        start_time: datetime = datetime.now()

        try:
            result = await func(*args, **kwargs)
            end_time: datetime = datetime.now()
            from litellm.proxy.proxy_server import proxy_logging_obj

            if "PROXY" not in func.__name__:
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
                args is not None
                and len(args) > 1
                and isinstance(args[1], dict)
            ):
                passed_kwargs = args[1]
                parent_otel_span = _get_parent_otel_span_from_kwargs(
                    kwargs=passed_kwargs
                )
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

    return wrapper


_db_exception_types: Optional[Tuple[type, ...]] = None


def _db_exception_types_cached() -> Tuple[type, ...]:
    """Exception types that mark a DB failure, resolved once.

    ``prisma`` is an optional dependency, so a proxy without it must not re-run
    (and re-fail) ``from prisma.errors import PrismaError`` on every handled
    exception -- a failed import is never cached in ``sys.modules`` and re-runs
    the whole finder/loader machinery each call.
    """
    global _db_exception_types
    if _db_exception_types is None:
        import httpx

        types: Tuple[type, ...] = (httpx.ConnectError, httpx.TimeoutException)
        try:
            from prisma.errors import PrismaError

            types = (PrismaError, *types)
        except Exception:
            pass
        _db_exception_types = types
    return _db_exception_types


def _is_exception_related_to_db(e: Exception) -> bool:
    """
    Returns True if the exception is related to the DB
    """
    return isinstance(e, _db_exception_types_cached())


async def _handle_logging_db_exception(
    e: Exception,
    func: Callable,
    kwargs: Dict,
    args: Tuple,
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
