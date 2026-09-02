from __future__ import annotations

import pytest

from litellm.llms.custom_httpx.llm_http_handler import _rust_responses_websocket_enabled
from litellm.rust_bridge import bindings, configuration, responses_websocket
from litellm.types.router import GenericLiteLLMParams


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


def test_rust_websocket_bridge_is_disabled_without_flag() -> None:
    assert not _rust_responses_websocket_enabled("openai", GenericLiteLLMParams())
    assert not _rust_responses_websocket_enabled("anthropic", GenericLiteLLMParams(rust=True))
    assert _rust_responses_websocket_enabled("openai", GenericLiteLLMParams(rust=True))


def test_explicit_false_overrides_process_enable() -> None:
    configuration.use_litellm_rust(True)

    assert not _rust_responses_websocket_enabled("openai", GenericLiteLLMParams(rust=False))


def test_process_enable_applies_without_request_override() -> None:
    configuration.use_litellm_rust(True)

    assert _rust_responses_websocket_enabled("openai", GenericLiteLLMParams())


@pytest.mark.asyncio
async def test_adapter_raises_clean_close_when_rust_connection_ends() -> None:
    adapter = responses_websocket._ConnectionAdapter(_ClosedNativeConnection())

    with pytest.raises(responses_websocket.ConnectionClosedOK):
        await adapter.recv()


@pytest.mark.asyncio
async def test_bridge_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)

    result = await responses_websocket.connect(
        url="wss://example.test/responses",
        headers={},
        timeout=None,
    )
    assert result.value is None
    assert result.source is responses_websocket.CoreEngine.PYTHON


@pytest.mark.asyncio
async def test_enabled_bridge_connects_and_adapts_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses_websocket.set_rust_responses_websocket(connection=_FakeNativeBridge)

    result = await responses_websocket.connect(
        url="wss://example.test/responses",
        headers={"Authorization": "Bearer key"},
        timeout=1.0,
    )

    assert result.source is responses_websocket.CoreEngine.RUST
    connection = result.value
    assert connection is not None
    assert connection.core_engine is responses_websocket.CoreEngine.RUST
    await connection.send("response.create")
    assert await connection.recv() == "response.completed"
    await connection.close()
