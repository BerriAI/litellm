"""
Counts HTTP requests to LLM inference, MCP, and A2A endpoints.

Feeds two independent sinks off one classification:

- ``GatewayRequestSink`` receives every classified request with its status and
  is the source of truth for SGR (successful gateway requests) on the admin UI.
  Not license-gated (see litellm.proxy.db.gateway_request_tracking). It is not
  told which deployment served the request: it persists its counts, so every
  dimension it takes has to be one the proxy chooses.
- ``BillingRecorder`` receives 2xx requests only and exports them for
  enterprise metering (see litellm.proxy.enterprise_billing.billing_metrics).

Both are injected. When neither is present the middleware is a transparent
pass-through.
"""

import threading
from collections.abc import Callable, Sequence
from enum import Enum
from typing import Final, Literal, Protocol, runtime_checkable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm._logging import verbose_proxy_logger
from litellm.rust_bridge._native import classify_billable_route


class BillableCategory(str, Enum):
    LLM = "llm"
    MCP = "mcp"
    A2A = "a2a"


@runtime_checkable
class BillingRecorder(Protocol):
    def record(self, *, category: BillableCategory, route: str, status_code: int, model_id: str | None) -> None: ...


@runtime_checkable
class GatewayRequestSink(Protocol):
    """
    Records every classified request, 2xx or not, for the SGR dashboard.

    Distinct from BillingRecorder on three counts: this is not license-gated,
    it is not restricted to 2xx, and it takes no model id. The deployment that
    served a request is deliberately not part of what it records, because the
    dashboard aggregates by route and a per-deployment dimension would only
    multiply the rows it has to sum back together.
    """

    def record(self, *, category: BillableCategory, route: str, status_code: int) -> None: ...


_MODEL_ID_HEADER: Final = b"x-litellm-model-id"


def classify_billable_request(path: str, method: str = "POST") -> tuple[BillableCategory, str] | None:
    """
    Map a request path to its (category, normalized route), or None if not billable.

    The route table lives in Rust (litellm-core-utils billable_route) and is
    shared with the Rust gateway's request volume middleware, so both sides of
    a gateway in front of the proxy count the same requests under the same
    routes.
    """
    classified: Final = classify_billable_route(path, method)
    if classified is None:
        return None
    category, route = classified
    return (BillableCategory(category), route)


def _extract_model_id(headers: Sequence[tuple[bytes, bytes]]) -> str | None:
    return next(
        (value.decode("latin-1") for name, value in headers if name.lower() == _MODEL_ID_HEADER and value),
        None,
    )


class _RequestLineFields(Protocol):
    def get(self, key: Literal["path", "method"], default: str, /) -> str: ...


class _ResponseStartFields(Protocol):
    def __getitem__(self, key: Literal["status"], /) -> int: ...

    def get(
        self, key: Literal["headers"], default: Sequence[tuple[bytes, bytes]], /
    ) -> Sequence[tuple[bytes, bytes]]: ...


def _request_line_fields(scope: _RequestLineFields) -> _RequestLineFields:
    return scope


def _response_start_fields(message: _ResponseStartFields) -> _ResponseStartFields:
    return message


class BillableRequestMetricsMiddleware:
    """
    Pure ASGI middleware that classifies each request once and fans the result
    out to the SGR sink (any status) and the billing recorder (2xx only).
    Modeled on InFlightRequestsMiddleware: it wraps `send`, reads the final
    status and the x-litellm-model-id header off the `http.response.start`
    message, and never blocks or fails the request path.
    """

    def __init__(
        self,
        app: ASGIApp,
        recorder: BillingRecorder | None = None,
        recorder_factory: Callable[[], BillingRecorder | None] | None = None,
        sink: GatewayRequestSink | None = None,
        sink_factory: Callable[[], GatewayRequestSink | None] | None = None,
    ) -> None:
        self.app = app
        self.recorder = recorder
        # The factory defers recorder construction to the first request, AFTER the
        # startup event has loaded the YAML config's environment_variables (license
        # and cert env vars). Building at import time captured recorder=None for
        # deployments configured that way. Resolved exactly once; the result
        # (including None) is cached.
        self._recorder_factory = recorder_factory
        self._resolved = recorder_factory is None
        self._resolve_lock = threading.Lock()
        # Resolved on the same schedule and for the same reason: the DB is not
        # connected at import time, so the sink cannot be built there either.
        self.sink = sink
        self._sink_factory = sink_factory
        self._sink_resolved = sink_factory is None
        self._sink_resolve_lock = threading.Lock()

    def _resolve_recorder(self) -> BillingRecorder | None:
        if self._resolved:
            return self.recorder
        # The lock keeps concurrent first requests from each building their own
        # MeterProvider (and leaking its background exporter thread).
        with self._resolve_lock:
            if not self._resolved:
                factory: Final = self._recorder_factory
                self.recorder = factory() if factory is not None else self.recorder
                self._resolved = True
        return self.recorder

    def _resolve_sink(self) -> GatewayRequestSink | None:
        if self._sink_resolved:
            return self.sink
        with self._sink_resolve_lock:
            if not self._sink_resolved:
                factory: Final = self._sink_factory
                self.sink = factory() if factory is not None else self.sink
                self._sink_resolved = True
        return self.sink

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        recorder: Final = self._resolve_recorder()
        sink: Final = self._resolve_sink()
        if recorder is None and sink is None:
            await self.app(scope, receive, send)
            return

        request_line: Final = _request_line_fields(scope)
        classification: Final = classify_billable_request(
            request_line.get("path", ""), request_line.get("method", "POST")
        )
        if classification is None:
            await self.app(scope, receive, send)
            return

        category, route = classification
        status_code = 0
        model_id: str | None = None

        async def send_wrapper(message: Message) -> None:
            nonlocal status_code, model_id
            if message["type"] == "http.response.start":
                response_start: Final = _response_start_fields(message)
                status_code = response_start["status"]
                model_id = _extract_model_id(response_start.get("headers", []))
            await send(message)

        await self.app(scope, receive, send_wrapper)

        if sink is not None:
            try:
                sink.record(category=category, route=route, status_code=status_code)
            except Exception:  # noqa: BLE001 -- metering must never fail a request that was already served
                verbose_proxy_logger.warning("gateway request metering failed for %s", route, exc_info=True)

        if recorder is not None and 200 <= status_code < 300:
            try:
                recorder.record(category=category, route=route, status_code=status_code, model_id=model_id)
            except Exception:  # noqa: BLE001 -- metering must never fail a request that was already served
                verbose_proxy_logger.warning("billable request metering failed for %s", route, exc_info=True)
