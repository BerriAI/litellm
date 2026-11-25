import asyncio
from typing import Final
from unittest.mock import MagicMock

import pytest
import uvicorn
from uvicorn._types import ASGIReceiveCallable, ASGISendCallable, Scope
from uvicorn.server import ServerState

from litellm.proxy.shutdown.recycling_http_protocol import RecyclingHTTPProtocol


@pytest.mark.asyncio
@pytest.mark.parametrize("shutdown_when", ("before request", "idle", "before headers", "after headers"))
async def test_recycling_http_worker_finishes_requests_and_signals_connection_close(shutdown_when: str) -> None:
    request_started: Final = asyncio.Event()
    finish_request: Final = asyncio.Event()

    async def app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        if shutdown_when == "before headers":
            request_started.set()
            await finish_request.wait()
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"4")]})
        request_started.set()
        await finish_request.wait()
        await send({"type": "http.response.body", "body": b"pong"})

    state: Final = ServerState()
    transport: Final = MagicMock(spec=asyncio.Transport, is_closing=MagicMock(return_value=False))
    config: Final = uvicorn.Config(app, log_config=None, ws="none")
    protocol: Final = RecyclingHTTPProtocol(config, state, {})
    protocol.connection_made(transport)
    request: Final = b"GET /health/liveliness HTTP/1.1\r\nHost: localhost\r\n\r\n"
    try:
        if shutdown_when in ("before request", "idle"):
            finish_request.set()
        if shutdown_when == "before request":
            protocol.shutdown()
            transport.close.assert_not_called()
        protocol.data_received(request)
        await request_started.wait()
        if shutdown_when == "idle":
            await asyncio.gather(*state.tasks)
        if shutdown_when != "before request":
            protocol.shutdown()
            transport.close.assert_not_called()
        finish_request.set()
        await asyncio.gather(*state.tasks)

        if shutdown_when in ("idle", "after headers"):
            first_response: Final = b"".join(call.args[0] for call in transport.write.call_args_list).lower()
            assert b"connection: close\r\n" not in first_response
            assert first_response.endswith(b"pong")
            transport.write.reset_mock()
            protocol.data_received(request)
            await asyncio.gather(*state.tasks)

        response: Final = b"".join(call.args[0] for call in transport.write.call_args_list).lower()
        assert b"connection: close\r\n" in response
        assert response.endswith(b"pong")
        transport.close.assert_called_once()
    finally:
        finish_request.set()
        await asyncio.gather(*state.tasks)
        protocol.connection_lost(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("server_shutdown_timeout", (3, 600))
async def test_recycling_http_worker_bounds_idle_drain_by_graceful_shutdown_timeout(
    monkeypatch: pytest.MonkeyPatch,
    server_shutdown_timeout: int,
) -> None:
    clock: Final = MagicMock(return_value=100.0)
    monkeypatch.setattr(asyncio.get_running_loop(), "time", clock)

    async def app(scope: Scope, receive: ASGIReceiveCallable, send: ASGISendCallable) -> None:
        await send({"type": "http.response.start", "status": 204})
        await send({"type": "http.response.body"})

    state: Final = ServerState()
    transport: Final = MagicMock(spec=asyncio.Transport, is_closing=MagicMock(return_value=False))
    config: Final = uvicorn.Config(app, log_config=None, ws="none", timeout_keep_alive=15, timeout_graceful_shutdown=3)
    protocol: Final = RecyclingHTTPProtocol(config, state, {})
    protocol.connection_made(transport)
    try:
        protocol.data_received(b"GET /health/liveliness HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await asyncio.gather(*state.tasks)
        config.timeout_graceful_shutdown = server_shutdown_timeout
        protocol.shutdown()
        protocol.data_received(b"GET /health/")
        transport.close.assert_not_called()

        clock.return_value = 103.0
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        transport.close.assert_called_once()
    finally:
        protocol.connection_lost(None)
