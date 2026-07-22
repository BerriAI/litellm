from __future__ import annotations

import pytest

from litellm.llms.custom_httpx.llm_http_handler import _rust_responses_websocket_enabled
from litellm.rust_bridge import responses_websocket


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
    ) -> _FakeNativeConnection:
        return _FakeNativeConnection()


def test_rust_websocket_bridge_enabled_for_openai_only() -> None:
    assert _rust_responses_websocket_enabled("openai")
    assert not _rust_responses_websocket_enabled("anthropic")
    assert not _rust_responses_websocket_enabled(None)


@pytest.mark.asyncio
async def test_adapter_raises_clean_close_when_rust_connection_ends() -> None:
    adapter = responses_websocket._ConnectionAdapter(_ClosedNativeConnection())

    with pytest.raises(responses_websocket.ConnectionClosedOK):
        await adapter.recv()


@pytest.mark.asyncio
async def test_bridge_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(responses_websocket, "_STATE", responses_websocket._RustResponsesWebSocketState())
    monkeypatch.setattr(responses_websocket, "get_native_bridge", lambda: None)

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
