"""
Tracks the number of HTTP requests currently in-flight on this uvicorn worker.

Used by /health/backlog to expose per-pod queue depth, and emitted as the
Prometheus gauge `litellm_in_flight_requests`.
"""

import os
from typing import Optional

from starlette.types import ASGIApp, Receive, Scope, Send

_in_flight: int = 0

# Lazily created on first request so PROMETHEUS_MULTIPROC_DIR is already set
# by the time we register the metric.
_gauge: Optional[object] = None


def _get_gauge() -> Optional[object]:
    global _gauge
    if _gauge is not None:
        return _gauge
    try:
        from prometheus_client import Gauge

        kwargs = {}
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            # livesum aggregates across all worker processes in the scrape response
            kwargs["multiprocess_mode"] = "livesum"
        _gauge = Gauge(
            "litellm_in_flight_requests",
            "Number of HTTP requests currently in-flight on this uvicorn worker",
            **kwargs,
        )
    except Exception:
        pass
    return _gauge


def get_in_flight_requests() -> int:
    return _in_flight


class InFlightRequestsMiddleware:
    """
    ASGI middleware that increments a counter when a request arrives
    and decrements it when the response is sent (or an error occurs).

    The counter is module-level and therefore scoped to a single uvicorn
    worker process — exactly the per-pod granularity we want.

    Also updates the `litellm_in_flight_requests` Prometheus gauge if
    prometheus_client is installed.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        global _in_flight
        _in_flight += 1
        gauge = _get_gauge()
        if gauge is not None:
            gauge.inc()  # type: ignore[union-attr]
        try:
            await self.app(scope, receive, send)
        finally:
            _in_flight -= 1
            if gauge is not None:
                gauge.dec()  # type: ignore[union-attr]
