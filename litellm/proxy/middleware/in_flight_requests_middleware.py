"""
Tracks the number of HTTP requests currently in-flight on this uvicorn worker.

Used by /health/backlog to expose per-pod queue depth, and emitted as the
Prometheus gauge `litellm_in_flight_requests`.
"""

import os
from typing import Optional

from starlette.types import ASGIApp, Receive, Scope, Send


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
    _gauge: Optional[object] = None
    _gauge_init_attempted: bool = False

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        InFlightRequestsMiddleware._in_flight += 1
        gauge = InFlightRequestsMiddleware._get_gauge()
        if gauge is not None:
            gauge.inc()  # type: ignore[union-attr]
        try:
            await self.app(scope, receive, send)
        finally:
            InFlightRequestsMiddleware._in_flight -= 1
            if gauge is not None:
                gauge.dec()  # type: ignore[union-attr]

    @staticmethod
    def get_count() -> int:
        """Return the number of HTTP requests currently in-flight."""
        return InFlightRequestsMiddleware._in_flight

    @staticmethod
    def _get_gauge() -> Optional[object]:
        if InFlightRequestsMiddleware._gauge_init_attempted:
            return InFlightRequestsMiddleware._gauge
        InFlightRequestsMiddleware._gauge_init_attempted = True
        try:
            from prometheus_client import Gauge

            kwargs = {}
            if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
                # livesum aggregates across all worker processes in the scrape response
                kwargs["multiprocess_mode"] = "livesum"
            InFlightRequestsMiddleware._gauge = Gauge(
                "litellm_in_flight_requests",
                "Number of HTTP requests currently in-flight on this uvicorn worker",
                **kwargs,
            )
        except Exception:
            InFlightRequestsMiddleware._gauge = None
        return InFlightRequestsMiddleware._gauge


def get_in_flight_requests() -> int:
    """Module-level convenience wrapper used by the /health/backlog endpoint."""
    return InFlightRequestsMiddleware.get_count()
