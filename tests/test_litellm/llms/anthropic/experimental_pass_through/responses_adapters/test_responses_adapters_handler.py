import json
import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

import litellm
from litellm.llms.anthropic.experimental_pass_through.responses_adapters.handler import (
    LiteLLMMessagesToResponsesAPIHandler,
    _build_responses_kwargs,
)

MESSAGES = [{"role": "user", "content": "hello"}]


def test_build_responses_kwargs_derives_prompt_cache_key_from_user_id():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        metadata={"user_id": "session-abc"},
        extra_kwargs={"custom_llm_provider": "openai"},
    )
    assert responses_kwargs["user"] == "session-abc"
    assert responses_kwargs["prompt_cache_key"] == "session-abc"


def test_build_responses_kwargs_prefers_explicit_prompt_cache_key_over_derived():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        metadata={"user_id": "session-abc"},
        extra_kwargs={"custom_llm_provider": "openai", "prompt_cache_key": "explicit-key"},
    )
    assert responses_kwargs["user"] == "session-abc"
    assert responses_kwargs["prompt_cache_key"] == "explicit-key"


def test_build_responses_kwargs_without_metadata_sets_no_prompt_cache_key():
    responses_kwargs = _build_responses_kwargs(
        max_tokens=1024,
        messages=MESSAGES,
        model="openai/gpt-5.6-luna",
        extra_kwargs={"custom_llm_provider": "openai"},
    )
    assert "user" not in responses_kwargs
    assert "prompt_cache_key" not in responses_kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested_model, expected_reported_model",
    [
        ("openai/gpt-5.6-luna", "gpt-5.6-luna"),
        ("perplexity/perplexity/kimi-k3", "perplexity/kimi-k3"),
    ],
)
async def test_streaming_message_start_reports_the_provider_local_model(requested_model, expected_reported_model):
    """
    BerriAI/litellm#37716 sends the caller's unresolved id down this bridge so the provider
    resolves it once. ``message_start`` is a reporting field rather than a wire value, so it
    keeps naming the model as the provider knows it, with only the leading provider segment gone.
    """

    async def empty_stream():
        return
        yield

    with patch.object(litellm, "aresponses", AsyncMock(return_value=empty_stream())):
        sse = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
            max_tokens=1024,
            messages=MESSAGES,
            model=requested_model,
            stream=True,
            custom_llm_provider=requested_model.split("/")[0],
        )
        events = [json.loads(chunk.decode().split("data: ", 1)[1]) async for chunk in sse]

    message_start = next(e for e in events if e["type"] == "message_start")
    assert message_start["message"]["model"] == expected_reported_model


class _FakeResponsesStream:
    """Minimal stand-in for BaseResponsesAPIStreamingIterator: iterable, and
    able to carry `_hidden_params` the way the real iterator does from its
    response headers."""

    def __init__(self, hidden_params=None):
        self._hidden_params = hidden_params

    def __aiter__(self):
        return self

    async def __anext__(self):
        if getattr(self, "_sent", False):
            raise StopAsyncIteration
        self._sent = True
        return {
            "type": "response.completed",
            "response": {"id": "resp_001", "model": "gpt-5.6-luna", "usage": {}},
        }


@pytest.mark.asyncio
async def test_streaming_headers_are_carried_on_the_returned_response_object():
    """
    The SSE wrapper returns a bare async generator, which cannot carry
    attributes -- the proxy would drop the upstream header mirror the inner
    Responses iterator built from its response headers. The handler must wrap
    it in AnthropicMessagesStreamingResponse (the passthrough route's
    attribute-carrying shape, introduced in #32160) so /v1/messages keeps
    llm_provider-* headers, matching /v1/chat/completions.
    """
    from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
        AnthropicMessagesStreamingResponse,
    )

    stream = _FakeResponsesStream(hidden_params={"additional_headers": {"llm_provider-x-cortecs-provider": "tensorix"}})
    with patch.object(litellm, "aresponses", AsyncMock(return_value=stream)):
        returned = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
            max_tokens=1024,
            messages=MESSAGES,
            model="openai/gpt-5.6-luna",
            stream=True,
            custom_llm_provider="openai",
        )

    assert isinstance(returned, AnthropicMessagesStreamingResponse)
    assert returned._hidden_params["additional_headers"] == {"llm_provider-x-cortecs-provider": "tensorix"}

    # the stream still flows through the wrapper as SSE bytes
    chunks = [chunk async for chunk in returned]
    assert any(b"message_stop" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_streaming_without_headers_returns_the_bare_sse_generator():
    """No mirror on the inner stream -> no wrapper; behaviour unchanged."""
    stream = _FakeResponsesStream()  # no _hidden_params at all
    with patch.object(litellm, "aresponses", AsyncMock(return_value=stream)):
        returned = await LiteLLMMessagesToResponsesAPIHandler.async_anthropic_messages_handler(
            max_tokens=1024,
            messages=MESSAGES,
            model="openai/gpt-5.6-luna",
            stream=True,
            custom_llm_provider="openai",
        )

    import inspect

    assert inspect.isasyncgen(returned)
    found_message_stop = False
    async for chunk in returned:
        if b"message_stop" in chunk:
            found_message_stop = True
            break
    assert found_message_stop
