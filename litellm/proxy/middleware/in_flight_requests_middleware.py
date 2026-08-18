"""
Tracks the number of HTTP requests currently in-flight on this uvicorn worker.

Used by /health/backlog to expose per-pod queue depth, and emitted as the
Prometheus gauge `litellm_in_flight_requests`.
"""

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final

from starlette.types import ASGIApp, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger

EMPTY_STATE: Final[Mapping[str, object]] = MappingProxyType({})  # mutable-ok: MappingProxyType needs a dict to wrap


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
            await self.app(scope, receive, send)
        finally:
            state: Final = scope.get("state", EMPTY_STATE)
            registry: Final = state.get("active_request_registry")
            try:
                if registry is not None:
                    await registry.remove(state.get("active_request_registry_id"))
            except BaseException:  # noqa: BLE001  # not Exception: CancelledError must still hit the finally below
                verbose_proxy_logger.exception("Failed to deregister an active request")
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


def get_in_flight_requests() -> int:
    """Module-level convenience wrapper used by the /health/backlog endpoint."""
    return InFlightRequestsMiddleware.get_count()
