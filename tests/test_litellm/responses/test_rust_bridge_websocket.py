from __future__ import annotations

import pytest

from litellm.rust_bridge import configuration, responses_websocket
from litellm.rust_bridge.request import NativeRequestContext, NativeResponsesWebSocketRequest


class _FakeNativeConnection:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.closed = False

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def recv_text(self) -> str:
        return "response.completed"

    async def close(self) -> None:
        self.closed = True


class _ClosedNativeConnection:
    async def recv_text(self) -> None:
        return None


class _FakeNativeBridge:
    @classmethod
    async def connect(
        cls,
        request: NativeResponsesWebSocketRequest,
        *,
        options: object,
        context: NativeRequestContext,
        callback_adapter: object | None = None,
    ) -> _FakeNativeConnection:
        return _FakeNativeConnection()


@pytest.fixture(autouse=True)
def reset_responses_websocket():
    responses_websocket.set_rust_responses_websocket(connection=None, decline=None)
    configuration.reset_rust_configuration()
    responses_websocket.set_rust_responses_websocket(
        decline=lambda model, custom_llm_provider, *, context: (
            "unsupported feature"
            if any(getattr(context.capabilities, key) for key in ("stream", "has_agentic_hook", "has_custom_client"))
            or context.capabilities.request_format == "native"
            else None
        )
    )
    yield
    responses_websocket.set_rust_responses_websocket(connection=None, decline=None)
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
async def test_adapter_raises_clean_close_when_rust_connection_ends() -> None:
    adapter = responses_websocket._ConnectionAdapter(_ClosedNativeConnection())

    with pytest.raises(responses_websocket.ConnectionClosedOK):
        await adapter.recv()


@pytest.mark.asyncio
async def test_bridge_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration.rust(True)
    responses_websocket._RESPONSES_WEBSOCKET.override(None)

    assert (
        await responses_websocket.connect(
            url="wss://example.test/responses",
            headers={},
            timeout=None,
        )
        is None
    )


@pytest.mark.asyncio
async def test_enabled_bridge_connects_and_adapts_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configuration.rust(True)
    responses_websocket.set_rust_responses_websocket(connection=_FakeNativeBridge)

    connection = await responses_websocket.connect(
        url="wss://example.test/responses",
        headers={"Authorization": "Bearer key"},
        timeout=1.0,
    )

    assert connection is not None
    await connection.send("response.create")
    assert await connection.recv() == "response.completed"
    await connection.close()


class _FailingNativeBridge:
    @classmethod
    async def connect(
        cls,
        request: NativeResponsesWebSocketRequest,
        *,
        options: object,
        context: NativeRequestContext,
        callback_adapter: object | None = None,
    ) -> _FakeNativeConnection:
        raise RuntimeError("connection failed")


@pytest.mark.asyncio
async def test_connection_failure_does_not_authorize_python_fallback() -> None:
    configuration.rust(True)
    responses_websocket.set_rust_responses_websocket(connection=_FailingNativeBridge)

    with pytest.raises(RuntimeError, match="connection failed"):
        await responses_websocket.connect(url="wss://example.test/responses", headers={}, timeout=None)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("session_error", [False, True])
@pytest.mark.asyncio
async def test_connection_dispatch_cleans_up_without_reconnecting(native, session_error):
    from contextlib import asynccontextmanager

    configuration.rust(True)
    native_socket = _FakeNativeConnection()
    python_socket = _FakeNativeConnection()
    connections = []

    class Native:
        @classmethod
        async def connect(cls, request, *, options, context):
            connections.append("native")
            assert options.custom_llm_provider == "azure"
            return native_socket

    @asynccontextmanager
    async def python():
        connections.append("python")
        try:
            yield python_socket
        finally:
            await python_socket.close()

    responses_websocket.set_rust_responses_websocket(connection=Native)
    if not native:
        responses_websocket.set_rust_responses_websocket(
            decline=lambda model, custom_llm_provider, **features: "declined"
        )

    async def run():
        async with responses_websocket.open_connection(
            url="wss://example.test",
            headers={},
            timeout=1,
            model="test-model",
            provider="azure",
            fallback=python,
        ):
            if session_error:
                raise RuntimeError("session failed")

    if session_error:
        with pytest.raises(RuntimeError, match="session failed"):
            await run()
    else:
        await run()
    assert connections == ["native" if native else "python"]
    assert native_socket.closed == native
    assert python_socket.closed == (not native)


@pytest.mark.asyncio
async def test_missing_acceptance_export_uses_python_connection_once():
    from contextlib import asynccontextmanager

    calls = []
    socket = _FakeNativeConnection()

    @asynccontextmanager
    async def python():
        calls.append("python")
        try:
            yield socket
        finally:
            await socket.close()

    configuration.rust(True)
    responses_websocket.set_rust_responses_websocket(connection=_FailingNativeBridge)
    responses_websocket._PREFLIGHT.override(None)
    async with responses_websocket.open_connection(
        url="wss://example.test", headers={}, timeout=1, model="model", provider="openai", fallback=python
    ) as connection:
        assert connection is socket
    assert calls == ["python"]
    assert socket.closed
