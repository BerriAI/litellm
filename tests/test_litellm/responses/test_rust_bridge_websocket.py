from __future__ import annotations

import pytest

from litellm.llms.custom_httpx.llm_http_handler import _rust_responses_websocket_enabled
from litellm.rust_bridge import configuration, responses_websocket


class _Declined(Exception):
    pass


class _FakeNativeBridge:
    RustBridgeDeclined = _Declined


@pytest.fixture(autouse=True)
def reset_responses_websocket(monkeypatch: pytest.MonkeyPatch):
    responses_websocket.set_rust_responses_websocket(route=None)
    configuration.reset_rust_configuration()
    monkeypatch.setattr(responses_websocket, "get_native_bridge", lambda: _FakeNativeBridge)
    yield
    responses_websocket.set_rust_responses_websocket(route=None)
    configuration.reset_rust_configuration()


def test_rust_websocket_bridge_uses_process_enablement() -> None:
    configuration.rust(False)
    assert not _rust_responses_websocket_enabled("openai")
    configuration.rust(True)
    assert _rust_responses_websocket_enabled("openai")
    assert not _rust_responses_websocket_enabled("anthropic")


def test_bridge_unavailable_declines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(responses_websocket, "_STATE", responses_websocket._RustResponsesWebSocketState())
    monkeypatch.setattr(responses_websocket, "get_native_bridge", lambda: None)
    assert not responses_websocket.admit()


def test_host_only_lifecycle_declines_before_provider_io() -> None:
    def decline() -> None:
        raise _Declined("host guardrails required")

    responses_websocket.set_rust_responses_websocket(route=decline)
    assert not responses_websocket.admit()


def test_unexpected_bridge_error_does_not_allow_fallback() -> None:
    def fail() -> None:
        raise RuntimeError("bridge failed")

    responses_websocket.set_rust_responses_websocket(route=fail)
    with pytest.raises(RuntimeError, match="bridge failed"):
        responses_websocket.admit()
