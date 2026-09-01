from __future__ import annotations

from collections.abc import Mapping

import pytest

from litellm.llms.custom_httpx.llm_http_handler import _rust_responses_websocket_enabled
from litellm.rust_bridge import configuration, responses_websocket
from litellm.types.router import GenericLiteLLMParams


class _FakeNativeConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def send_event(self, event: Mapping[str, object]) -> None:
        self.sent.append(dict(event))

    async def recv_event(self) -> dict[str, object]:
        return {"type": "response.completed"}

    async def close(self) -> None:
        self.closed = True


class _ClosedNativeConnection:
    async def send_event(self, event: Mapping[str, object]) -> None:
        return None

    async def recv_event(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _FakeNativeBridge:
    @classmethod
    async def connect(
        cls,
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
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
    monkeypatch.setattr(responses_websocket, "_STATE", responses_websocket._RustResponsesWebSocketState())
    monkeypatch.setattr(responses_websocket, "get_native_bridge", lambda: None)

    assert (
        await responses_websocket.connect(
            provider="openai",
            api_key=None,
            api_base="https://example.test",
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
        provider="openai",
        api_key="key",
        api_base="https://example.test",
        headers={"Authorization": "Bearer key"},
        timeout=1.0,
    )

    assert connection is not None
    await connection.send('{"type":"response.create","model":"gpt-5"}')
    assert await connection.recv() == '{"type":"response.completed"}'
    await connection.close()
