"""
Handles logging DB success/failure to ServiceLogger()

ServiceLogger() then sends DB logs to Prometheus, OTEL, Datadog etc
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm._service_logger import ServiceTypes
from litellm.litellm_core_utils.core_helpers import _get_parent_otel_span_from_kwargs
from litellm.proxy.db.db_pool_metrics import DBPoolMetricsSampler
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.proxy.db.db_pool_metrics import SupportsPoolSample

# One sampler per process. The pool it reads is per-process too, so there is
# nothing to key this by.
_pool_metrics_sampler: Final = DBPoolMetricsSampler()


def _prometheus_logger() -> "PrometheusLogger | None":
    """The active PrometheusLogger, or None when prometheus is not configured.

    Imports lazily. Bare ``import litellm`` does not load the prometheus
    integration, so hoisting this would make every proxy pay for a module that
    only prometheus deployments use.
    """
    from litellm.integrations.prometheus import PrometheusLogger

    return PrometheusLogger.get_instance()


def _resolve_pool_client() -> "SupportsPoolSample | None":
    from litellm.proxy.proxy_server import prisma_client

    return None if prisma_client is None else prisma_client.db


async def _sample_db_pool_metrics() -> None:
    """Publish a throttled reading of the connection pool, if one is due.

    Runs alongside real database work rather than on a timer: a scheduled
    exporter goes quiet exactly when the event loop is saturated, which is the
    window this metric exists to cover.

    Every step is inside the guard, including resolving the client and locating
    the logger. This runs off a database call that has already succeeded, so
    nothing here may turn a working query into a failed one.
    """
    try:
        update: Final = await _pool_metrics_sampler.maybe_sample(_resolve_pool_client)
        if update is None:
            return
        logger: Final = _prometheus_logger()
        if logger is not None:
            logger.record_db_pool_sample(update)
    except Exception as e:  # noqa: BLE001  # a metrics failure must never fail the database call it rides on
        verbose_proxy_logger.debug("db pool metrics publish failed: %s", e)


def _record_db_pool_timeout_if_exhausted(e: Exception) -> None:
    """Count a pool exhaustion, without ever displacing the error that caused it.

    The caller re-raises the original exception after this returns. Anything
    that escaped here would replace a P2024 with a metrics error, during the
    incident this counter exists to record.
    """
    try:
        if not PrismaDBExceptionHandler.is_connection_pool_timeout_error(e):
            return
        logger: Final = _prometheus_logger()
        if logger is not None:
            logger.record_db_pool_timeout()
    except Exception as metrics_error:  # noqa: BLE001  # counting an exhaustion must not mask the exhaustion itself
        verbose_proxy_logger.debug("db pool timeout metric failed: %s", metrics_error)


def _safe_db_event_metadata(kwargs: dict) -> dict[str, str] | None:
    """Minimal, non-sensitive ``event_metadata`` for a DB service log.

    The raw ``kwargs``/``args`` carry live objects (Prisma client, OTel spans)
    and secrets (tokens), none of which belongs on a span — so we surface only
    the table name when present. Everything else is dropped.
    """
    table_name: Final = kwargs.get("table_name")
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
        start_time: Final[datetime] = datetime.now()

        try:
            result: Final = await func(*args, **kwargs)
            end_time: datetime = datetime.now()
            from litellm.proxy.proxy_server import proxy_logging_obj

            # Dispatched, never awaited. Awaiting here would add a suspension
            # point after the query already succeeded, so a client disconnect in
            # that window would discard a completed result and skip the success
            # hook below. `is_due` keeps this to one task per sample interval.
            if _pool_metrics_sampler.is_due():
                asyncio.create_task(_sample_db_pool_metrics())

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

    # Counted before the DB-relatedness gate below: a pool timeout is the proxy
    # running out of connections, so it must be recorded whether or not the
    # failure is classified as a DB service failure.
    _record_db_pool_timeout_if_exhausted(e)

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
