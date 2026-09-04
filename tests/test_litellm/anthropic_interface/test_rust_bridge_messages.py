"""Tests for the optional Rust-backed Anthropic Messages path."""

import importlib
from typing import cast

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.rust_bridge import configuration
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams

rust_messages = importlib.import_module("litellm.rust_bridge.messages")
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


@pytest.fixture(autouse=True)
def _reset_rust_flag():
    rust_messages.set_rust_messages(messages=None, amessages=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL
    yield
    rust_messages.set_rust_messages(messages=None, amessages=None)
    configuration.reset_rust_configuration()
    rust_bridge_loader._cached_bridge = rust_bridge_loader._BRIDGE_SENTINEL


def test_messages_wrapper_forwards_args_and_converts_timeout():
    bridge = RecordingMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(messages=bridge)

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
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

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
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

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
@pytest.mark.parametrize(
    ("enablement", "expected_calls"),
    (
        pytest.param("disabled", 0, id="disabled"),
        pytest.param("request-false", 0, id="request-false-over-process"),
        pytest.param("process", 1, id="process-override"),
        pytest.param("environment", 1, id="environment"),
    ),
)
async def test_gate_respects_shared_enablement(
    monkeypatch: pytest.MonkeyPatch,
    enablement: str,
    expected_calls: int,
) -> None:
    bridge = RecordingAsyncMessages()
    rust_messages.set_rust_messages(amessages=bridge)
    if enablement in {"request-false", "process"}:
        litellm.rust(True)
    if enablement == "environment":
        monkeypatch.setenv("LITELLM_RUST", "1")
    params = GenericLiteLLMParams(api_key="sk-azure", rust=False if enablement == "request-false" else None)

    response = await _gate(litellm_params=params)

    assert len(bridge.calls) == expected_calls
    assert (response["id"] if response is not None else None) == ("msg_123" if expected_calls else None)


@pytest.mark.asyncio
async def test_gate_invokes_rust_for_native_anthropic_provider():
    bridge = RecordingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

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
async def test_gate_skips_rust_for_unsupported_provider():
    bridge = ExplodingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

    response = await _gate(custom_llm_provider="openai")

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_skips_rust_for_agentic_hook():
    bridge = ExplodingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

    response = await _gate(has_agentic_hook=True)

    assert response is None
    assert bridge.calls == 0


@pytest.mark.asyncio
async def test_gate_streams_through_rust_when_eligible_and_strips_stream_flag():
    bridge = RecordingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

    streaming_body = {**REQUEST_BODY, "stream": True}
    response = await _gate(
        has_agentic_hook=False,
        request_body=streaming_body,
    )

    assert response is not None
    assert response["_hidden_params"]["additional_headers"] == {"x-litellm-rust": "true"}
    assert "stream" not in bridge.calls[0]["body"]
    assert bridge.calls[0]["body"] == REQUEST_BODY


@pytest.mark.asyncio
async def test_fake_stream_wraps_rust_response_as_anthropic_sse():
    response = cast(AnthropicMessagesResponse, dict(FAKE_MESSAGES_RESPONSE))
    stream = BaseLLMHTTPHandler._rust_anthropic_messages_fake_stream(response)

    assert stream._hidden_params["additional_headers"] == {"x-litellm-rust": "true"}

    chunks = [chunk async for chunk in stream]
    joined = b"".join(chunks)

    assert b"event: message_start" in joined
    assert b"event: content_block_delta" in joined
    assert b"hello world" in joined
    assert b"event: message_stop" in joined
