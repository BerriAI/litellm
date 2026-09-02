from __future__ import annotations

from collections.abc import Mapping

import pytest

from litellm.exceptions import APIError
from litellm.llms.custom_httpx.llm_http_handler import _rust_responses_websocket_enabled
from litellm.rust_bridge import bindings, configuration, responses_websocket, streaming
from litellm.types.router import GenericLiteLLMParams


class _FakeNativeConnection:
    def __init__(self) -> None:
        self.sent: list[dict[str, object]] = []
        self.closed = False

    async def send_event(self, event: Mapping[str, object]) -> None:
        self.sent.append(dict(event))

    async def recv_event(self) -> Mapping[str, object]:
        return {"type": "response.completed"}

    async def close(self) -> None:
        self.closed = True


class _ClosedNativeConnection:
    async def recv_event(self) -> None:
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


class _Declined(Exception):
    pass


class _Upstream(Exception):
    pass


class _NativeErrors:
    RustBridgeDeclined = _Declined
    RustUpstreamError = _Upstream


class _DecliningNativeBridge:
    @classmethod
    async def connect(
        cls,
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> _FakeNativeConnection:
        raise _Declined("provider unsupported")


class _FailingNativeBridge:
    @classmethod
    async def connect(
        cls,
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
    ) -> _FakeNativeConnection:
        raise _Upstream(503, "request may have executed")


@pytest.fixture(autouse=True)
def reset_responses_websocket():
    streaming.set_rust_streaming(capability=None)
    responses_websocket.set_rust_responses_websocket(connection=None)
    configuration.reset_rust_configuration()
    yield
    streaming.set_rust_streaming(capability=None)
    responses_websocket.set_rust_responses_websocket(connection=None)
    configuration.reset_rust_configuration()


def test_rust_websocket_bridge_is_disabled_without_flag() -> None:
    assert not _rust_responses_websocket_enabled("openai", GenericLiteLLMParams())
    assert not _rust_responses_websocket_enabled("anthropic", GenericLiteLLMParams(rust=True))
    assert not _rust_responses_websocket_enabled("openai", GenericLiteLLMParams(rust=True))


def test_injected_typed_capability_enables_the_gate() -> None:
    streaming.set_rust_streaming(
        capability=lambda api, provider, transport: (api, provider, transport) == ("responses", "openai", "websocket")
    )

    assert _rust_responses_websocket_enabled("openai", GenericLiteLLMParams(rust=True))


def test_explicit_false_overrides_process_enable() -> None:
    streaming.set_rust_streaming(capability=lambda api, provider, transport: True)
    configuration.use_litellm_rust(True)

    assert not _rust_responses_websocket_enabled("openai", GenericLiteLLMParams(rust=False))


def test_process_enable_applies_without_request_override() -> None:
    streaming.set_rust_streaming(capability=lambda api, provider, transport: True)
    configuration.use_litellm_rust(True)

    assert _rust_responses_websocket_enabled("openai", GenericLiteLLMParams())


@pytest.mark.asyncio
async def test_adapter_raises_clean_close_when_rust_connection_ends() -> None:
    adapter = responses_websocket._ConnectionAdapter(
        _ClosedNativeConnection(),
        responses_websocket.BridgeErrorContext(route="responses websocket", provider="openai", model=""),
    )

    with pytest.raises(responses_websocket.ConnectionClosedOK):
        await adapter.recv()


@pytest.mark.asyncio
async def test_bridge_unavailable_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: None)
    streaming.set_rust_streaming(capability=lambda api, provider, transport: True)

    result = await responses_websocket.connect(
        provider="openai",
        api_key=None,
        api_base="https://example.test",
        headers={},
        timeout=None,
    )
    assert result.value is None
    assert result.source is responses_websocket.CoreEngine.PYTHON


@pytest.mark.asyncio
async def test_enabled_bridge_connects_and_adapts_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    streaming.set_rust_streaming(capability=lambda api, provider, transport: True)
    responses_websocket.set_rust_responses_websocket(connection=_FakeNativeBridge)

    result = await responses_websocket.connect(
        provider="openai",
        api_key="key",
        api_base="https://example.test",
        headers={"Authorization": "Bearer key"},
        timeout=1.0,
    )

    assert result.source is responses_websocket.CoreEngine.RUST
    connection = result.value
    assert connection is not None
    assert connection.core_engine is responses_websocket.CoreEngine.RUST
    await connection.send('{"type":"response.create","model":"gpt-5"}')
    assert await connection.recv() == '{"type":"response.completed"}'
    await connection.close()


@pytest.mark.asyncio
async def test_declined_connect_falls_back_before_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _NativeErrors())
    streaming.set_rust_streaming(capability=lambda api, provider, transport: True)
    responses_websocket.set_rust_responses_websocket(connection=_DecliningNativeBridge)

    result = await responses_websocket.connect(
        provider="openai",
        api_key="key",
        api_base="https://example.test",
        headers={},
        timeout=None,
    )

    assert result.value is None
    assert result.source is responses_websocket.CoreEngine.PYTHON


@pytest.mark.asyncio
async def test_upstream_connect_failure_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: _NativeErrors())
    streaming.set_rust_streaming(capability=lambda api, provider, transport: True)
    responses_websocket.set_rust_responses_websocket(connection=_FailingNativeBridge)

    with pytest.raises(APIError, match="request may have executed") as caught:
        await responses_websocket.connect(
            provider="openai",
            api_key="key",
            api_base="https://example.test",
            headers={},
            timeout=None,
        )
    assert caught.value.headers["x-litellm-core"] == "rust"
