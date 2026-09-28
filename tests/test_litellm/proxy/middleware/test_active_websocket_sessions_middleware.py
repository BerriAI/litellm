import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Final

import pytest
from fastapi import FastAPI
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient
from starlette.websockets import WebSocket, WebSocketDisconnect

from litellm.proxy.middleware.active_websocket_sessions_middleware import (
    ActiveWebSocketSessionsMiddleware,
)


@dataclass(slots=True)
class _Child:
    gauge: "_FakeGauge"
    route: str

    def inc(self, amount: float = 1) -> None:
        self.gauge.values[self.route] += amount

    def dec(self, amount: float = 1) -> None:
        self.gauge.values[self.route] -= amount


@dataclass(slots=True)
class _FakeGauge:
    values: defaultdict[str, float] = field(default_factory=lambda: defaultdict(float))

    def labels(self, route: str) -> _Child:
        return _Child(self, route)


def _make_app(gauge: _FakeGauge | None, observed: list[dict[str, float]]) -> FastAPI:
    app: Final = FastAPI()

    def snapshot() -> None:
        observed.append(dict(gauge.values) if gauge else {})

    async def echo(websocket: WebSocket) -> None:
        await websocket.accept()
        snapshot()
        async for text in websocket.iter_text():
            await websocket.send_text(text)

    async def crash_after_accept(websocket: WebSocket) -> None:
        await websocket.accept()
        snapshot()
        raise RuntimeError("provider failed")

    async def reject(websocket: WebSocket) -> None:
        await websocket.close(code=1008)

    async def http() -> PlainTextResponse:
        snapshot()
        return PlainTextResponse("ok")

    app.add_api_websocket_route("/v1/realtime", echo)
    app.add_api_websocket_route("/openai/{endpoint:path}", echo)
    app.add_api_websocket_route("/crash", crash_after_accept)
    app.add_api_websocket_route("/reject", reject)
    app.add_api_route("/http", http)
    app.add_middleware(ActiveWebSocketSessionsMiddleware, gauge_factory=lambda: gauge)
    return app


def test_counts_open_session_and_releases_on_close() -> None:
    gauge: Final = _FakeGauge()
    observed: Final[list[dict[str, float]]] = []
    with TestClient(_make_app(gauge, observed)).websocket_connect("/v1/realtime") as ws:
        ws.send_text("hi")
        assert ws.receive_text() == "hi"
    assert observed == [{"/v1/realtime": 1}]
    assert gauge.values["/v1/realtime"] == 0


def test_concurrent_sessions_are_summed_per_route() -> None:
    gauge: Final = _FakeGauge()
    client: Final = TestClient(_make_app(gauge, []))
    with (
        client.websocket_connect("/v1/realtime"),
        client.websocket_connect("/v1/realtime"),
        client.websocket_connect("/openai/v1/realtime"),
    ):
        assert gauge.values == {"/v1/realtime": 2, "/openai/{endpoint:path}": 1}
    assert gauge.values == {"/v1/realtime": 0, "/openai/{endpoint:path}": 0}


def test_releases_session_when_handler_raises() -> None:
    gauge: Final = _FakeGauge()
    observed: Final[list[dict[str, float]]] = []
    with pytest.raises(RuntimeError, match="provider failed"):
        with TestClient(_make_app(gauge, observed)).websocket_connect("/crash") as ws:
            ws.receive_text()
    assert observed == [{"/crash": 1}]
    assert gauge.values["/crash"] == 0


def test_rejected_handshake_is_never_counted() -> None:
    gauge: Final = _FakeGauge()
    with pytest.raises(WebSocketDisconnect):
        with TestClient(_make_app(gauge, [])).websocket_connect("/reject"):
            pass
    assert gauge.values == {}


def test_http_requests_are_not_counted() -> None:
    gauge: Final = _FakeGauge()
    observed: Final[list[dict[str, float]]] = []
    assert TestClient(_make_app(gauge, observed)).get("/http").text == "ok"
    assert observed == [{}]
    assert gauge.values == {}


def test_passes_traffic_through_without_prometheus() -> None:
    with TestClient(_make_app(None, [])).websocket_connect("/v1/realtime") as ws:
        ws.send_text("hi")
        assert ws.receive_text() == "hi"


def test_gauge_factory_runs_once() -> None:
    calls: Final[list[None]] = []
    gauge: Final = _FakeGauge()

    def factory() -> _FakeGauge:
        calls.append(None)
        return gauge

    middleware: Final = ActiveWebSocketSessionsMiddleware(_make_app(None, []), gauge_factory=factory)

    async def noop_receive() -> dict[str, str]:
        return {"type": "http.disconnect"}

    async def noop_send(_message: object) -> None:
        return None

    async def run_twice() -> None:
        for _ in range(2):
            await middleware(
                {"type": "websocket", "path": "/reject", "headers": [], "query_string": b""},
                noop_receive,
                noop_send,
            )

    asyncio.run(run_twice())
    assert len(calls) == 1
