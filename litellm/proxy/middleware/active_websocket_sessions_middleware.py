import os
from collections.abc import Callable, Collection
from functools import cache
from typing import Final, Protocol

from starlette.routing import WebSocketRoute
from starlette.types import ASGIApp, Message, Receive, Scope, Send

UNMATCHED_ROUTE: Final = "unmatched"
ACTIVE_WEBSOCKET_SESSIONS_METRIC: Final = "litellm_active_websocket_sessions"


class _GaugeChild(Protocol):
    def inc(self, amount: float = 1) -> None: ...

    def dec(self, amount: float = 1) -> None: ...


class RouteGauge(Protocol):
    def labels(self, route: str) -> _GaugeChild: ...


class _AcceptTracker:
    __slots__ = ("_gauge", "_scope", "_send", "_session")

    def __init__(self, gauge: RouteGauge, scope: Scope, send: Send) -> None:
        self._gauge = gauge
        self._scope = scope
        self._send = send
        self._session: _GaugeChild | None = None

    async def send(self, message: Message) -> None:
        if message["type"] == "websocket.accept" and self._session is None:
            self._session = self._gauge.labels(route=_route_template(self._scope))
            self._session.inc()
        await self._send(message)

    def release(self) -> None:
        if self._session is not None:
            self._session.dec()


def _route_template(scope: Scope) -> str:
    route: Final = scope.get("route")
    return route.path if isinstance(route, WebSocketRoute) else UNMATCHED_ROUTE


class ActiveWebSocketSessionsMiddleware:
    def __init__(self, app: ASGIApp, gauge_factory: Callable[[], RouteGauge | None]) -> None:
        self.app = app
        self._gauge_factory = gauge_factory
        self._gauge: RouteGauge | None = None
        self._gauge_init_attempted = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        gauge: Final = self._get_gauge() if scope["type"] == "websocket" else None
        if gauge is None:
            await self.app(scope, receive, send)
            return
        tracker: Final = _AcceptTracker(gauge, scope, send)
        try:
            await self.app(scope, receive, tracker.send)
        finally:
            tracker.release()

    def _get_gauge(self) -> RouteGauge | None:
        if not self._gauge_init_attempted:
            self._gauge_init_attempted = True
            self._gauge = self._gauge_factory()
        return self._gauge


def create_prometheus_active_websocket_sessions_gauge(excluded_metrics: Collection[str]) -> RouteGauge | None:
    if ACTIVE_WEBSOCKET_SESSIONS_METRIC in excluded_metrics:
        return None
    return _process_wide_gauge()


@cache
def _process_wide_gauge() -> RouteGauge | None:
    try:
        from prometheus_client import Gauge
    except ImportError:
        return None
    description: Final = "Number of accepted WebSocket sessions currently open on this worker"
    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        return Gauge(ACTIVE_WEBSOCKET_SESSIONS_METRIC, description, labelnames=("route",), multiprocess_mode="livesum")
    return Gauge(ACTIVE_WEBSOCKET_SESSIONS_METRIC, description, labelnames=("route",))
