"""
Tracks per-worker request pressure: how many HTTP requests are in flight, and
how many the proxy shed rather than served.

Counting shed responses here rather than at each limiter means no rejection path
can be missed, and the count is per worker for the same reason the in-flight
gauge is.
"""

import os
from typing import Any, Final

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger


class InFlightRequestsMiddleware:
    """
    ASGI middleware that increments a counter when a request arrives and
    decrements it when the response is sent (or an error occurs).

    The counter is class-level and therefore scoped to a single uvicorn worker
    process — exactly the per-pod granularity we want.

    Also updates the `litellm_in_flight_requests` Prometheus gauge if
    prometheus_client is installed. The gauge is lazily initialised on the
    first request so that PROMETHEUS_MULTIPROC_DIR is already set by the time
    we register the metric. Initialisation is attempted only once — if
    prometheus_client is absent the class remembers and never retries.
    """

    _in_flight: int = 0
    _gauge: Any | None = None
    _gauge_init_attempted: bool = False

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        InFlightRequestsMiddleware._in_flight += 1
        gauge: Final = InFlightRequestsMiddleware._get_gauge()
        if gauge is not None:
            gauge.inc()
        try:
            await self.app(scope, receive, send=_counting_send(send))
        finally:
            InFlightRequestsMiddleware._in_flight -= 1
            if gauge is not None:
                gauge.dec()

    @staticmethod
    def get_count() -> int:
        """Return the number of HTTP requests currently in-flight."""
        return InFlightRequestsMiddleware._in_flight

    @staticmethod
    def _get_gauge() -> Any | None:
        if InFlightRequestsMiddleware._gauge_init_attempted:
            return InFlightRequestsMiddleware._gauge
        InFlightRequestsMiddleware._gauge_init_attempted = True
        try:
            from prometheus_client import Gauge

            if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
                # livesum aggregates across all worker processes in the scrape response
                InFlightRequestsMiddleware._gauge = Gauge(
                    "litellm_in_flight_requests",
                    "Number of HTTP requests currently in-flight on this uvicorn worker",
                    multiprocess_mode="livesum",
                )
            else:
                InFlightRequestsMiddleware._gauge = Gauge(
                    "litellm_in_flight_requests",
                    "Number of HTTP requests currently in-flight on this uvicorn worker",
                )
        except Exception:
            InFlightRequestsMiddleware._gauge = None
        return InFlightRequestsMiddleware._gauge


# Statuses the proxy uses when it declines to serve. A response only counts once
# the request is also marked as shed by this proxy, since litellm forwards
# upstream 429s with the same status.
_SHED_STATUSES: Final = frozenset({429, 503})


def _record_shed_response(status: int) -> None:
    if status not in _SHED_STATUSES:
        return
    from litellm.proxy.common_utils.request_pressure_metrics import was_request_shed_by_proxy

    if not was_request_shed_by_proxy():
        return
    try:
        from litellm.integrations.prometheus import PrometheusLogger

        logger = PrometheusLogger.get_instance()
        if logger is not None:
            logger.record_request_shed(status)
    except Exception as e:  # noqa: BLE001  # counting a shed response must not break the response
        verbose_proxy_logger.debug("request shed metric failed: %s", e)


def _counting_send(send: Send) -> Send:
    async def wrapped(message: Message) -> None:
        if message["type"] == "http.response.start":
            _record_shed_response(int(message["status"]))
        await send(message)

    return wrapped


def get_in_flight_requests() -> int:
    """Module-level convenience wrapper used by the /health/backlog endpoint."""
    return InFlightRequestsMiddleware.get_count()
