from __future__ import annotations

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.responses import websocket as responses_websocket


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
        *,
        url: str,
        headers: dict[str, str],
        timeout_seconds: float | None,
        custom_llm_provider: str | None,
    ) -> _FakeNativeConnection:
        return _FakeNativeConnection()


@pytest.fixture(autouse=True)
def reset_responses_websocket():
    responses_websocket.set_rust_responses_websocket(connection=None)
    configuration.reset_rust_configuration()
    yield
    responses_websocket.set_rust_responses_websocket(connection=None)
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
async def test_adapter_raises_clean_close_when_rust_connection_ends() -> None:
    adapter = responses_websocket._ConnectionAdapter(_ClosedNativeConnection())

    with pytest.raises(responses_websocket.ConnectionClosedOK):
        await adapter.recv()


@pytest.mark.asyncio
async def test_bridge_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    configuration.rust(True)
    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: None)

    assert (
        await responses_websocket.connect(
            url="wss://example.test/responses",
            custom_llm_provider="openai",
            model="test",
            headers={},
            timeout=None,
        )
        is None
    )


@pytest.mark.asyncio
async def test_enabled_bridge_connects_and_adapts_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses_websocket.set_rust_responses_websocket(connection=_FakeNativeBridge)
    configuration.rust(True)

    connection = await responses_websocket.connect(
        url="wss://example.test/responses",
        custom_llm_provider="openai",
        model="test",
        headers={"Authorization": "Bearer key"},
        timeout=1.0,
    )

    assert connection is not None
    await connection.send("response.create")
    assert await connection.recv() == "response.completed"
    await connection.close()


@pytest.mark.asyncio
async def test_disabled_websocket_does_not_connect() -> None:
    class UnexpectedConnection:
        @classmethod
        async def connect(cls, **kwargs: object) -> None:
            raise AssertionError("disabled Rust must not connect")

    responses_websocket.set_rust_responses_websocket(connection=UnexpectedConnection)
    configuration.rust(False)
    assert (
        await responses_websocket.connect(
            url="ws://127.0.0.1:1",
            headers={},
            timeout=0.1,
            custom_llm_provider="openai",
            model="test",
        )
        is None
    )


@pytest.mark.asyncio
async def test_native_websocket_decline_falls_back_but_connection_failure_does_not() -> None:
    from litellm.exceptions import APIError

    native = pytest.importorskip("litellm.rust_bridge._native")
    responses_websocket.set_rust_responses_websocket(connection=native.ResponsesWebSocketConnection)
    configuration.rust(True)
    assert (
        await responses_websocket.connect(
            url="ws://127.0.0.1:1",
            headers={},
            timeout=0.1,
            custom_llm_provider="azure",
            model="test",
        )
        is None
    )
    with pytest.raises(APIError):
        await responses_websocket.connect(
            url="ws://127.0.0.1:1",
            headers={},
            timeout=0.1,
            custom_llm_provider="openai",
            model="test",
        )
