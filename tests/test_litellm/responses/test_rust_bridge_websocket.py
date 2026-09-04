from __future__ import annotations

import pytest

from litellm.rust_bridge import configuration, responses_websocket


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


@pytest.fixture(autouse=True)
def reset_responses_websocket():
    responses_websocket.set_rust_responses_websocket(connection=None)
    configuration.reset_rust_configuration()
    yield
    responses_websocket.set_rust_responses_websocket(connection=None)
    configuration.reset_rust_configuration()


@pytest.mark.asyncio
async def test_explicit_false_overrides_process_enable() -> None:
    configuration.rust(True)
    responses_websocket.set_rust_responses_websocket(connection=_FakeNativeBridge)

    assert await responses_websocket.connect(url="wss://example.test", headers={}, timeout=None, request_override=False) is None


@pytest.mark.asyncio
async def test_ineligible_provider_does_not_use_rust() -> None:
    configuration.rust(True)
    responses_websocket.set_rust_responses_websocket(connection=_FakeNativeBridge)

    assert await responses_websocket.connect(url="wss://example.test", headers={}, timeout=None, eligible=False) is None


@pytest.mark.asyncio
async def test_adapter_raises_clean_close_when_rust_connection_ends() -> None:
    adapter = responses_websocket._ConnectionAdapter(_ClosedNativeConnection())

    with pytest.raises(responses_websocket.ConnectionClosedOK):
        await adapter.recv()


@pytest.mark.asyncio
async def test_bridge_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    responses_websocket._RESPONSES_WEBSOCKET.override(None)

    assert (
        await responses_websocket.connect(
            url="wss://example.test/responses",
            headers={},
            timeout=None,
            request_override=True,
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
        request_override=True,
    )

    assert connection is not None
    await connection.send("response.create")
    assert await connection.recv() == "response.completed"
    await connection.close()
