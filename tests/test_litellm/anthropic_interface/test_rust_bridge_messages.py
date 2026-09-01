"""Tests for the optional Rust-backed Anthropic Messages path."""

import importlib
from typing import cast

import httpx
import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
    anthropic_messages_stream_hidden_params,
)
from litellm.llms.custom_httpx.llm_http_handler import (
    BaseLLMHTTPHandler,
    _anthropic_messages_with_core_engine,
)
from litellm.rust_bridge import configuration
from litellm.rust_bridge.messages import RustAmessages, RustMessages
from litellm.rust_bridge.runtime import UNSET, CoreEngine, Unset
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams

rust_messages = importlib.import_module("litellm.rust_bridge.messages")
rust_bridge_loader = importlib.import_module("litellm.rust_bridge.loader")
rust_bridge_runtime = importlib.import_module("litellm.rust_bridge.runtime")

FAKE_MESSAGES_RESPONSE: dict[str, object] = {
    "id": "msg_123",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-5-20250929",
    "content": [{"type": "text", "text": "hello world"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 5, "output_tokens": 3},
}

REQUEST_BODY: dict[str, object] = {
    "model": "claude-sonnet-4-5",
    "max_tokens": 64,
    "messages": [{"role": "user", "content": "hi"}],
}


def _use_test_rust(
    enabled: bool = True,
    *,
    messages: RustMessages | None | Unset = UNSET,
    amessages: RustAmessages | None | Unset = UNSET,
) -> None:
    rust_messages.set_rust_messages(messages=messages, amessages=amessages)
    litellm.use_litellm_rust(enabled)


def test_python_messages_response_has_symmetric_provenance():
    response = _anthropic_messages_with_core_engine(
        cast(AnthropicMessagesResponse, dict(FAKE_MESSAGES_RESPONSE)),
        CoreEngine.PYTHON,
    )

    assert response["_hidden_params"]["core_engine"] == "python"
    assert response["_hidden_params"]["additional_headers"] == {"x-litellm-core": "python"}


def test_python_messages_stream_provenance_precedes_iteration_and_cannot_be_spoofed():
    hidden_params = anthropic_messages_stream_hidden_params(
        httpx.Headers({"x-litellm-core": "rust", "x-litellm-rust": "true", "x-provider": "kept"})
    )

    assert hidden_params["core_engine"] == "python"
    assert hidden_params["additional_headers"]["x-litellm-core"] == "python"
    assert "x-litellm-rust" not in hidden_params["additional_headers"]
    assert hidden_params["additional_headers"]["llm_provider-x-litellm-core"] == "rust"
    assert hidden_params["additional_headers"]["llm_provider-x-litellm-rust"] == "true"
    assert hidden_params["additional_headers"]["llm_provider-x-provider"] == "kept"


class RecordingMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "body": body,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "timeout_seconds": timeout_seconds,
            }
        )
        return dict(FAKE_MESSAGES_RESPONSE)


class RecordingAsyncMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "model": model,
                "body": body,
                "api_key": api_key,
                "api_base": api_base,
                "custom_llm_provider": custom_llm_provider,
                "extra_headers": extra_headers,
                "timeout_seconds": timeout_seconds,
            }
        )
        return dict(FAKE_MESSAGES_RESPONSE)


class ExplodingAsyncMessages:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        raise AssertionError("bridge must not be called")


class RaisingAsyncMessages:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        raise RuntimeError("upstream request failed with status 400: bad request")


class _FakeDeclined(Exception):
    pass


class _FakeUpstream(Exception):
    pass


class _FakeNativeErrors:
    RustBridgeDeclined = _FakeDeclined
    RustUpstreamError = _FakeUpstream


class RaisingTypedAsyncMessages:
    def __init__(self, error: Exception) -> None:
        self.error = error
        self.calls = 0

    async def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls += 1
        raise self.error


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    rust_messages.set_rust_messages(messages=None, amessages=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    yield
    rust_messages.set_rust_messages(messages=None, amessages=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL


def test_load_rust_messages_returns_injected_impl():
    bridge = RecordingMessages()
    _use_test_rust(True, messages=bridge)
    assert rust_messages.load_rust_messages() is bridge


def test_bare_use_litellm_rust_still_toggles_ocr():
    from litellm.rust_bridge.ocr import rust_ocr_enabled

    litellm.use_litellm_rust(True)
    assert rust_ocr_enabled() is True

    litellm.use_litellm_rust(False)
    assert rust_ocr_enabled() is False


def test_load_rust_amessages_returns_injected_impl():
    bridge = RecordingAsyncMessages()
    _use_test_rust(True, amessages=bridge)
    assert rust_messages.load_rust_amessages() is bridge


def test_messages_wrapper_returns_none_when_bridge_absent(monkeypatch):
    monkeypatch.setattr(rust_bridge_runtime, "get_native_bridge", lambda: None)
    litellm.use_litellm_rust(True)
    assert rust_messages.load_rust_messages() is None
    result = rust_messages.messages(
        model="claude",
        body=REQUEST_BODY,
        api_key="k",
        api_base="b",
        custom_llm_provider="azure_ai",
        extra_headers={},
        timeout=30.0,
    )
    assert result is None


def test_messages_wrapper_forwards_args_and_converts_timeout():
    bridge = RecordingMessages()
    _use_test_rust(True, messages=bridge)

    response = rust_messages.messages(
        model="claude-sonnet-4-5",
        body=REQUEST_BODY,
        api_key="sk-azure",
        api_base="https://resource.services.ai.azure.com/anthropic",
        custom_llm_provider="azure_ai",
        extra_headers={"anthropic-beta": "token-efficient-tools-2025-02-19"},
        timeout=httpx.Timeout(600.0, read=42.0),
    )

    assert response == FAKE_MESSAGES_RESPONSE
    assert bridge.calls[0] == {
        "model": "claude-sonnet-4-5",
        "body": REQUEST_BODY,
        "api_key": "sk-azure",
        "api_base": "https://resource.services.ai.azure.com/anthropic",
        "custom_llm_provider": "azure_ai",
        "extra_headers": {"anthropic-beta": "token-efficient-tools-2025-02-19"},
        "timeout_seconds": 42.0,
    }


@pytest.mark.asyncio
async def test_amessages_wrapper_forwards_args():
    bridge = RecordingAsyncMessages()
    _use_test_rust(True, amessages=bridge)

    response = await rust_messages.amessages(
        model="claude-sonnet-4-5",
        body=REQUEST_BODY,
        api_key="sk-azure",
        api_base="https://resource.services.ai.azure.com/anthropic",
        custom_llm_provider="azure_ai",
        extra_headers=None,
        timeout=12.5,
    )

    assert response == FAKE_MESSAGES_RESPONSE
    assert bridge.calls[0]["model"] == "claude-sonnet-4-5"
    assert bridge.calls[0]["timeout_seconds"] == 12.5


def _gate(**overrides):
    kwargs = {
        "custom_llm_provider": "azure_ai",
        "litellm_params": GenericLiteLLMParams(api_key="sk-azure", rust=True),
        "has_agentic_hook": False,
        "model": "claude-sonnet-4-5",
        "api_key": "sk-azure",
        "api_base": "https://resource.services.ai.azure.com/anthropic",
        "headers": {"x-api-key": "sk-azure", "anthropic-version": "2023-06-01"},
        "request_body": dict(REQUEST_BODY),
        "timeout": 30.0,
    }
    kwargs.update(overrides)
    return BaseLLMHTTPHandler._maybe_rust_anthropic_messages(**kwargs)


@pytest.mark.asyncio
async def test_gate_invokes_rust_and_marks_response_header():
    bridge = RecordingAsyncMessages()
    _use_test_rust(True, amessages=bridge)

    response = await _gate()

    assert response is not None
    assert response["id"] == "msg_123"
    assert response["_hidden_params"]["core_engine"] == "rust"
    assert response["_hidden_params"]["additional_headers"] == {
        "x-litellm-core": "rust",
        "x-litellm-rust": "true",
    }
    call = bridge.calls[0]
    assert call["model"] == "claude-sonnet-4-5"
    assert call["body"] == REQUEST_BODY
    assert call["api_key"] == "sk-azure"
    assert call["api_base"] == "https://resource.services.ai.azure.com/anthropic"
    assert call["extra_headers"] == {"x-api-key": "sk-azure", "anthropic-version": "2023-06-01"}
    assert call["timeout_seconds"] == 30.0


@pytest.mark.asyncio
async def test_gate_does_not_fall_back_for_unknown_bridge_error():
    bridge = RaisingAsyncMessages()
    _use_test_rust(True, amessages=bridge)

    with pytest.raises(RuntimeError, match="upstream request failed"):
        await _gate()
    assert bridge.calls == 1


@pytest.mark.asyncio
async def test_gate_does_not_fall_back_after_upstream_send(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rust_bridge_runtime, "get_native_bridge", lambda: _FakeNativeErrors())
    bridge = RaisingTypedAsyncMessages(_FakeUpstream(429, "rate limited"))
    _use_test_rust(True, amessages=bridge)

    with pytest.raises(litellm.APIError, match="rate limited"):
        await _gate()
    assert bridge.calls == 1


@pytest.mark.asyncio
async def test_gate_falls_back_when_rust_declines(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(rust_bridge_runtime, "get_native_bridge", lambda: _FakeNativeErrors())
    bridge = RaisingTypedAsyncMessages(_FakeDeclined("unsupported"))
    _use_test_rust(True, amessages=bridge)

    assert await _gate() is None
    assert bridge.calls == 1


@pytest.mark.asyncio
async def test_gate_skips_rust_when_flag_absent():
    bridge = ExplodingAsyncMessages()
    rust_messages.set_rust_messages(amessages=bridge)

    response = await _gate(litellm_params=GenericLiteLLMParams(api_key="sk-azure"))

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_skips_rust_when_flag_false():
    bridge = ExplodingAsyncMessages()
    _use_test_rust(True, amessages=bridge)

    response = await _gate(litellm_params=GenericLiteLLMParams(api_key="sk-azure", rust=False))

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_invokes_rust_for_native_anthropic_provider():
    bridge = RecordingAsyncMessages()
    _use_test_rust(True, amessages=bridge)

    response = await _gate(
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(api_key="sk-ant", rust=True),
        api_key="sk-ant",
        api_base="https://api.anthropic.com",
        headers={"x-api-key": "sk-ant", "anthropic-version": "2023-06-01"},
    )

    assert response is not None
    assert response["_hidden_params"]["core_engine"] == "rust"
    assert response["_hidden_params"]["additional_headers"] == {
        "x-litellm-core": "rust",
        "x-litellm-rust": "true",
    }
    assert bridge.calls[0]["custom_llm_provider"] == "anthropic"
    assert bridge.calls[0]["api_key"] == "sk-ant"


@pytest.mark.asyncio
async def test_gate_invokes_rust_when_env_var_set(monkeypatch):
    bridge = RecordingAsyncMessages()
    rust_messages.set_rust_messages(amessages=bridge)
    monkeypatch.setenv("LITELLM_RUST", "1")

    response = await _gate(
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(api_key="sk-ant"),
    )

    assert response is not None
    assert bridge.calls[0]["custom_llm_provider"] == "anthropic"


@pytest.mark.asyncio
async def test_gate_env_var_falsey_does_not_enable(monkeypatch):
    bridge = ExplodingAsyncMessages()
    rust_messages.set_rust_messages(amessages=bridge)
    monkeypatch.setenv("LITELLM_RUST", "0")

    response = await _gate(
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(api_key="sk-ant"),
    )

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_skips_rust_for_unsupported_provider():
    bridge = ExplodingAsyncMessages()
    _use_test_rust(True, amessages=bridge)

    response = await _gate(custom_llm_provider="openai")

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_skips_rust_for_agentic_hook():
    bridge = ExplodingAsyncMessages()
    _use_test_rust(True, amessages=bridge)

    response = await _gate(has_agentic_hook=True)

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_streams_through_rust_when_eligible_and_strips_stream_flag():
    bridge = RecordingAsyncMessages()
    _use_test_rust(True, amessages=bridge)

    streaming_body = {**REQUEST_BODY, "stream": True}
    response = await _gate(
        has_agentic_hook=False,
        request_body=streaming_body,
    )

    assert response is not None
    assert response["_hidden_params"]["core_engine"] == "rust"
    assert response["_hidden_params"]["additional_headers"] == {
        "x-litellm-core": "rust",
        "x-litellm-rust": "true",
    }
    assert "stream" not in bridge.calls[0]["body"]
    assert bridge.calls[0]["body"] == REQUEST_BODY


@pytest.mark.asyncio
async def test_fake_stream_wraps_rust_response_as_anthropic_sse():
    response = cast(AnthropicMessagesResponse, dict(FAKE_MESSAGES_RESPONSE))
    stream = BaseLLMHTTPHandler._rust_anthropic_messages_fake_stream(response)

    assert stream._hidden_params["core_engine"] == "rust"
    assert stream._hidden_params["additional_headers"] == {
        "x-litellm-core": "rust",
        "x-litellm-rust": "true",
    }

    chunks = [chunk async for chunk in stream]
    joined = b"".join(chunks)

    assert b"event: message_start" in joined
    assert b"event: content_block_delta" in joined
    assert b"hello world" in joined
    assert b"event: message_stop" in joined


@pytest.mark.asyncio
async def test_gate_falls_back_when_bridge_unavailable(monkeypatch):
    monkeypatch.setattr(rust_bridge_runtime, "get_native_bridge", lambda: None)
    litellm.use_litellm_rust(True)

    response = await _gate()

    assert response is None
