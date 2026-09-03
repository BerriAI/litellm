"""Tests for the optional Rust-backed Anthropic Messages path."""

import importlib
from typing import AsyncIterator

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.rust_bridge import configuration
from litellm.types.router import GenericLiteLLMParams

rust_messages = importlib.import_module("litellm.rust_bridge.messages")
rust_messages_stream = importlib.import_module("litellm.rust_bridge.messages_stream")
rust_bridge_loader = importlib.import_module("litellm.rust_bridge.loader")

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


FAKE_SSE_FRAMES: tuple[bytes, ...] = (
    b'event: message_start\ndata: {"type":"message_start"}\n\n',
    b'event: content_block_delta\ndata: {"delta":"hello world"}\n\n',
    b'event: message_stop\ndata: {"type":"message_stop"}\n\n',
)


class FakeNativeStream:
    """Stands in for the native `MessagesStream` async iterator."""

    def __init__(self, frames: tuple[bytes, ...] = FAKE_SSE_FRAMES) -> None:
        self._frames = frames
        self._index = 0

    def __aiter__(self) -> "FakeNativeStream":
        return self

    async def __anext__(self) -> bytes:
        if self._index >= len(self._frames):
            raise StopAsyncIteration
        frame = self._frames[self._index]
        self._index += 1
        return frame


class RecordingAsyncMessagesStream:
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
    ) -> object:
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
        return FakeNativeStream()


class ExplodingAsyncMessagesStream:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        raise AssertionError("stream bridge must not be called")


class RaisingAsyncMessagesStream:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kwargs: object) -> object:
        self.calls += 1
        raise RuntimeError("upstream request failed with status 429: rate limited")


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    rust_messages.set_rust_messages(messages=None, amessages=None)
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    yield
    rust_messages.set_rust_messages(messages=None, amessages=None)
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL


def test_load_rust_messages_returns_injected_impl():
    bridge = RecordingMessages()
    litellm.use_litellm_rust(True, messages=bridge)
    assert rust_messages.load_rust_messages() is bridge


def test_bare_use_litellm_rust_still_toggles_ocr():
    from litellm.rust_bridge.ocr import rust_ocr_enabled

    litellm.use_litellm_rust(True)
    assert rust_ocr_enabled() is True

    litellm.use_litellm_rust(False)
    assert rust_ocr_enabled() is False


def test_load_rust_amessages_returns_injected_impl():
    bridge = RecordingAsyncMessages()
    litellm.use_litellm_rust(True, amessages=bridge)
    assert rust_messages.load_rust_amessages() is bridge


def test_messages_wrapper_returns_none_when_bridge_absent(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: None,
    )
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
    litellm.use_litellm_rust(True, messages=bridge)

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
    litellm.use_litellm_rust(True, amessages=bridge)

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
    litellm.use_litellm_rust(True, amessages=bridge)

    response = await _gate()

    assert response is not None
    assert response["id"] == "msg_123"
    assert response["_hidden_params"]["additional_headers"] == {"x-litellm-rust": "true"}
    call = bridge.calls[0]
    assert call["model"] == "claude-sonnet-4-5"
    assert call["body"] == REQUEST_BODY
    assert call["api_key"] == "sk-azure"
    assert call["api_base"] == "https://resource.services.ai.azure.com/anthropic"
    assert call["extra_headers"] == {"x-api-key": "sk-azure", "anthropic-version": "2023-06-01"}
    assert call["timeout_seconds"] == 30.0


@pytest.mark.asyncio
async def test_gate_falls_back_to_python_when_bridge_raises():
    bridge = RaisingAsyncMessages()
    litellm.use_litellm_rust(True, amessages=bridge)

    response = await _gate()

    assert response is None
    assert bridge.calls == 1


@pytest.mark.asyncio
async def test_gate_skips_rust_when_flag_absent():
    bridge = ExplodingAsyncMessages()
    rust_messages.set_rust_messages(amessages=bridge)

    response = await _gate(litellm_params=GenericLiteLLMParams(api_key="sk-azure"))

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_uses_process_enable_without_request_override():
    bridge = RecordingAsyncMessages()
    rust_messages.set_rust_messages(amessages=bridge)
    litellm.use_litellm_rust(True)

    response = await _gate(litellm_params=GenericLiteLLMParams(api_key="sk-azure"))

    assert response is not None
    assert bridge.calls[0]["custom_llm_provider"] == "azure_ai"


@pytest.mark.asyncio
async def test_gate_skips_rust_when_flag_false():
    bridge = ExplodingAsyncMessages()
    litellm.use_litellm_rust(True, amessages=bridge)

    response = await _gate(litellm_params=GenericLiteLLMParams(api_key="sk-azure", rust=False))

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_invokes_rust_for_native_anthropic_provider():
    bridge = RecordingAsyncMessages()
    litellm.use_litellm_rust(True, amessages=bridge)

    response = await _gate(
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(api_key="sk-ant", rust=True),
        api_key="sk-ant",
        api_base="https://api.anthropic.com",
        headers={"x-api-key": "sk-ant", "anthropic-version": "2023-06-01"},
    )

    assert response is not None
    assert response["_hidden_params"]["additional_headers"] == {"x-litellm-rust": "true"}
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
    litellm.use_litellm_rust(True, amessages=bridge)

    response = await _gate(custom_llm_provider="openai")

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_skips_rust_for_agentic_hook():
    bridge = ExplodingAsyncMessages()
    litellm.use_litellm_rust(True, amessages=bridge)

    response = await _gate(has_agentic_hook=True)

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_streams_through_rust_when_eligible_and_strips_stream_flag():
    bridge = RecordingAsyncMessages()
    litellm.use_litellm_rust(True, amessages=bridge)

    streaming_body = {**REQUEST_BODY, "stream": True}
    response = await _gate(
        has_agentic_hook=False,
        request_body=streaming_body,
    )

    assert response is not None
    assert response["_hidden_params"]["additional_headers"] == {"x-litellm-rust": "true"}
    assert "stream" not in bridge.calls[0]["body"]
    assert bridge.calls[0]["body"] == REQUEST_BODY


def _stream_gate(**overrides):
    kwargs = {
        "custom_llm_provider": "anthropic",
        "litellm_params": GenericLiteLLMParams(api_key="sk-ant", rust=True),
        "has_agentic_hook": False,
        "model": "claude-sonnet-4-5",
        "api_key": "sk-ant",
        "api_base": "https://api.anthropic.com",
        "headers": {"x-api-key": "sk-ant", "anthropic-version": "2023-06-01"},
        "request_body": {**REQUEST_BODY, "stream": True},
        "timeout": 30.0,
    }
    kwargs.update(overrides)
    return BaseLLMHTTPHandler._maybe_rust_anthropic_messages_stream(**kwargs)


@pytest.mark.asyncio
async def test_stream_gate_returns_sse_frames_and_keeps_stream_flag():
    bridge = RecordingAsyncMessagesStream()
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=bridge)

    response = await _stream_gate()

    assert response is not None
    assert response._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}
    chunks = [chunk async for chunk in response]
    assert chunks == list(FAKE_SSE_FRAMES)
    call = bridge.calls[0]
    assert call["model"] == "claude-sonnet-4-5"
    assert call["body"]["stream"] is True
    assert call["timeout_seconds"] == 30.0


@pytest.mark.asyncio
async def test_stream_gate_routes_azure_ai_to_the_native_stream():
    bridge = RecordingAsyncMessagesStream()
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=bridge)

    response = await _stream_gate(
        custom_llm_provider="azure_ai",
        litellm_params=GenericLiteLLMParams(api_key="sk-azure", rust=True),
        api_key="sk-azure",
        api_base="https://resource.services.ai.azure.com/anthropic",
        headers={"x-api-key": "sk-azure", "anthropic-version": "2023-06-01"},
    )

    assert response is not None
    assert response._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}
    chunks = [chunk async for chunk in response]
    assert chunks == list(FAKE_SSE_FRAMES)
    call = bridge.calls[0]
    assert call["custom_llm_provider"] == "azure_ai"
    assert call["api_key"] == "sk-azure"
    assert call["body"]["stream"] is True


@pytest.mark.asyncio
async def test_stream_gate_falls_back_to_python_when_bridge_raises():
    bridge = RaisingAsyncMessagesStream()
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=bridge)

    response = await _stream_gate()

    assert response is None
    assert bridge.calls == 1


@pytest.mark.asyncio
async def test_stream_gate_skips_rust_when_flag_absent():
    bridge = ExplodingAsyncMessagesStream()
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=bridge)

    response = await _stream_gate(
        litellm_params=GenericLiteLLMParams(api_key="sk-ant"),
    )

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_stream_gate_skips_rust_for_agentic_hook():
    bridge = ExplodingAsyncMessagesStream()
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=bridge)

    response = await _stream_gate(has_agentic_hook=True)

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_stream_gate_skips_rust_for_unsupported_provider():
    bridge = ExplodingAsyncMessagesStream()
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=bridge)

    response = await _stream_gate(custom_llm_provider="openai")

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_stream_gate_falls_back_when_bridge_unavailable(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: None,
    )
    litellm.use_litellm_rust(True)

    response = await _stream_gate()

    assert response is None


@pytest.mark.asyncio
async def test_messages_stream_wrapper_forwards_args_and_adapts_frames():
    bridge = RecordingAsyncMessagesStream()
    rust_messages_stream.set_rust_amessages_stream(amessages_stream=bridge)

    stream = await rust_messages_stream.amessages_stream(
        model="claude-sonnet-4-5",
        body={**REQUEST_BODY, "stream": True},
        api_key="sk-ant",
        api_base="https://api.anthropic.com",
        custom_llm_provider="anthropic",
        extra_headers=None,
        timeout=httpx.Timeout(600.0, read=42.0),
    )

    assert stream is not None
    assert [chunk async for chunk in stream] == list(FAKE_SSE_FRAMES)
    assert bridge.calls[0]["timeout_seconds"] == 42.0
    assert bridge.calls[0]["body"]["stream"] is True


@pytest.mark.asyncio
async def test_messages_stream_adapter_aclose_ends_iteration():
    adapter = rust_messages_stream.RustMessagesStreamAdapter(FakeNativeStream())
    assert await adapter.__anext__() == FAKE_SSE_FRAMES[0]
    await adapter.aclose()
    with pytest.raises(StopAsyncIteration):
        await adapter.__anext__()


@pytest.mark.asyncio
async def test_messages_stream_wrapper_returns_none_when_bridge_absent(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: None,
    )
    litellm.use_litellm_rust(True)

    stream = await rust_messages_stream.amessages_stream(
        model="claude-sonnet-4-5",
        body={**REQUEST_BODY, "stream": True},
        api_key="sk-ant",
        api_base="https://api.anthropic.com",
        custom_llm_provider="anthropic",
        extra_headers=None,
        timeout=30.0,
    )

    assert stream is None


@pytest.mark.asyncio
async def test_gate_falls_back_when_bridge_unavailable(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: None,
    )
    litellm.use_litellm_rust(True)

    response = await _gate()

    assert response is None
