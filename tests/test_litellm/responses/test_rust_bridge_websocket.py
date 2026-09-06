from __future__ import annotations

from typing import cast

import pytest

from litellm.rust_bridge import configuration, responses_websocket
from litellm.rust_bridge.callbacks import SessionCallbackHandle
from litellm.rust_bridge.request import NativeRequestContext, NativeResponsesWebSocketRequest
from tests.test_litellm._rust_bridge_utils import use_fake_native_bridge


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
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
async def test_adapter_raises_clean_close_when_rust_connection_ends() -> None:
    adapter = responses_websocket._ConnectionAdapter(_ClosedNativeConnection())

    with pytest.raises(responses_websocket.ConnectionClosedOK):
        await adapter.recv()


@pytest.mark.asyncio
async def test_bridge_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration.rust(True)
    use_fake_native_bridge(monkeypatch)

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
    use_fake_native_bridge(monkeypatch, ResponsesWebSocketConnection=_FakeNativeBridge)

    connection = await responses_websocket.connect(
        url="wss://example.test/responses",
        headers={"Authorization": "Bearer key"},
        timeout=1.0,
    )

    assert connection is not None
    await connection.send("response.create")
    assert await connection.recv() == "response.completed"
    await connection.close()


@pytest.mark.asyncio
async def test_connection_forwards_session_callback_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration.rust(True)
    received: list[object] = []
    callback_adapter = cast(SessionCallbackHandle, object())

    class Native:
        @classmethod
        async def connect(
            cls,
            request: NativeResponsesWebSocketRequest,
            *,
            options: object,
            context: NativeRequestContext,
            callback_adapter: object | None = None,
        ) -> _FakeNativeConnection:
            received.append(callback_adapter)
            return _FakeNativeConnection()

    use_fake_native_bridge(monkeypatch, ResponsesWebSocketConnection=Native)

    connection = await responses_websocket.connect(
        url="wss://example.test/responses",
        headers={},
        timeout=None,
        callback_adapter=callback_adapter,
    )

    assert connection is not None
    assert received == [callback_adapter]


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
async def test_connection_failure_does_not_authorize_python_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration.rust(True)
    use_fake_native_bridge(monkeypatch, ResponsesWebSocketConnection=_FailingNativeBridge)

    with pytest.raises(RuntimeError, match="connection failed"):
        await responses_websocket.connect(url="wss://example.test/responses", headers={}, timeout=None)


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("session_error", [False, True])
@pytest.mark.asyncio
async def test_connection_dispatch_cleans_up_without_reconnecting(monkeypatch, native, session_error):
    from contextlib import asynccontextmanager

    configuration.rust(True)
    native_socket = _FakeNativeConnection()
    python_socket = _FakeNativeConnection()
    connections = []

    class Native:
        @classmethod
        async def connect(cls, request, *, options, context, callback_adapter=None):
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

    if native:
        use_fake_native_bridge(monkeypatch, ResponsesWebSocketConnection=Native)
    else:
        use_fake_native_bridge(monkeypatch)

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
async def test_missing_acceptance_export_keeps_native_failure_terminal(monkeypatch: pytest.MonkeyPatch):
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
    use_fake_native_bridge(monkeypatch, ResponsesWebSocketConnection=_FailingNativeBridge)
    with pytest.raises(RuntimeError, match="connection failed"):
        async with responses_websocket.open_connection(
            url="wss://example.test", headers={}, timeout=1, model="model", provider="openai", fallback=python
        ):
            pass
    assert calls == []
    assert not socket.closed
