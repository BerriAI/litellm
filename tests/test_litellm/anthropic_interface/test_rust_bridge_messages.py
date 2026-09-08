"""Tests for the optional Rust-backed Anthropic Messages path."""

import asyncio
import importlib
import weakref
from datetime import datetime
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
        arguments: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append(arguments)
        return dict(FAKE_MESSAGES_RESPONSE)


class RecordingAsyncMessages:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def __call__(
        self,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append(arguments)
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


class _CommittedMessagesError(Exception):
    pass


class _DeclinedMessagesError(Exception):
    pass


class _NativeExceptions:
    RustBridgeDeclined = _DeclinedMessagesError
    RustUpstreamError = _CommittedMessagesError


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
    litellm.rust(True)
    rust_messages.set_rust_messages(messages=bridge)
    assert rust_messages.load_rust_messages() is bridge


def test_load_rust_amessages_returns_injected_impl():
    bridge = RecordingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)
    assert rust_messages.load_rust_amessages() is bridge


def test_messages_wrapper_returns_none_when_bridge_absent(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: None,
    )
    litellm.rust(True)
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
        "_rust_lifecycle_owner": "bridge",
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
        "litellm_params": GenericLiteLLMParams(api_key="sk-azure"),
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
async def test_gate_propagates_unknown_errors_without_replaying():
    bridge = RaisingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

    with pytest.raises(RuntimeError, match="upstream request failed"):
        await _gate()

    assert bridge.calls == 1


@pytest.mark.asyncio
async def test_gate_does_not_fall_back_after_provider_commit(monkeypatch):
    calls = 0

    async def committed(**kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        raise _CommittedMessagesError(429, "rate limited")

    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: _NativeExceptions())
    rust_messages.set_rust_messages(amessages=committed)
    litellm.rust(True)

    with pytest.raises(litellm.APIError) as raised:
        await _gate()

    assert raised.value.status_code == 429
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_gate_only_accepts_declines_before_callback_setup(monkeypatch, started):
    from litellm.rust_bridge._lifecycle import LIFECYCLE_STARTED_KEY

    error = _DeclinedMessagesError("declined")

    async def decline(arguments):
        if started:
            arguments[LIFECYCLE_STARTED_KEY] = True
        raise error

    monkeypatch.setattr("litellm.rust_bridge.bindings.get_native_bridge", lambda: _NativeExceptions())
    rust_messages.set_rust_messages(amessages=decline)
    litellm.rust(True)

    if not started:
        assert await _gate() is None
        return

    with pytest.raises(_DeclinedMessagesError) as raised:
        await _gate()
    assert raised.value is error


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
    litellm.rust(True)

    response = await _gate(litellm_params=GenericLiteLLMParams(api_key="sk-azure"))

    assert response is not None
    assert bridge.calls[0]["custom_llm_provider"] == "azure_ai"


@pytest.mark.asyncio
async def test_gate_invokes_rust_for_native_anthropic_provider():
    bridge = RecordingAsyncMessages()
    litellm.rust(True)
    rust_messages.set_rust_messages(amessages=bridge)

    response = await _gate(
        custom_llm_provider="anthropic",
        litellm_params=GenericLiteLLMParams(api_key="sk-ant"),
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


def test_fake_stream_completes_retained_response_after_sync_exhaustion():
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    completed: list[bool] = []
    stream = FakeAnthropicMessagesStreamIterator(
        response=cast(AnthropicMessagesResponse, dict(FAKE_MESSAGES_RESPONSE)),
        on_complete=lambda: completed.append(True),
    )

    assert list(stream)
    assert completed == [True]
    assert list(stream) == []
    assert completed == [True]


@pytest.mark.asyncio
async def test_fake_stream_async_exhaustion_and_close_settle_exactly_once():
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )

    completed: list[bool] = []
    stream = FakeAnthropicMessagesStreamIterator(
        response=cast(AnthropicMessagesResponse, dict(FAKE_MESSAGES_RESPONSE)),
        on_complete=lambda: completed.append(True),
    )

    assert [chunk async for chunk in stream]
    await stream.aclose()
    stream.close()
    assert completed == [True]


@pytest.mark.asyncio
async def test_fake_stream_context_closes_after_early_cancellation():
    from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
        FakeAnthropicMessagesStreamIterator,
    )
    from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
        AnthropicMessagesStreamHiddenParams,
        AnthropicMessagesStreamingResponse,
    )

    completed: list[bool] = []
    completion_stream = FakeAnthropicMessagesStreamIterator(
        cast(AnthropicMessagesResponse, dict(FAKE_MESSAGES_RESPONSE)),
        on_complete=lambda: completed.append(True),
    )
    stream = AnthropicMessagesStreamingResponse(
        completion_stream,
        AnthropicMessagesStreamHiddenParams(additional_headers={}),
    )

    async def cancel_inside_context() -> None:
        async with stream:
            await anext(stream)
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cancel_inside_context()

    await stream.aclose()
    assert completed == [True]


def test_retained_response_releases_roots_and_restores_when_terminal_fails(monkeypatch):
    class Root:
        pass

    class Logger:
        def __init__(self) -> None:
            self.model_call_details: dict[str, object] = {}

        def _handle_anthropic_messages_response_logging(self, result: object) -> object:
            return result

    root = Root()
    root_ref = weakref.ref(root)
    restored: list[object] = []

    terminal_calls: list[bool] = []

    def terminal(*args: object) -> None:
        terminal_calls.append(True)
        if len(terminal_calls) == 1:
            raise RuntimeError("terminal callback failed")

    monkeypatch.setattr(rust_messages, "invoke_terminal", terminal)
    monkeypatch.setattr(
        litellm.utils,
        "_restore_correlation_context_if_supported",
        lambda logger: restored.append(logger),
    )
    logger = Logger()
    response = rust_messages.retain_stream_response(dict(FAKE_MESSAGES_RESPONSE), root, logger, datetime.now())
    del root

    with pytest.raises(RuntimeError, match="terminal callback failed"):
        response.complete()
    response.complete()

    assert root_ref() is None
    assert terminal_calls == [True, True]
    assert restored == [logger]


@pytest.mark.asyncio
async def test_gate_falls_back_when_bridge_unavailable(monkeypatch):
    monkeypatch.setattr(
        importlib.import_module("litellm.rust_bridge"),
        "get_native_bridge",
        lambda: None,
    )
    litellm.rust(True)

    response = await _gate()

    assert response is None
