"""
Streaming ``/v1/messages`` against a model that is neither Anthropic nor OpenAI is served by
translating the call onto ``/v1/chat/completions``, and the ``msg_`` id the caller is streamed
is minted right here. It is the only request id such a caller ever sees, so the spend row has
to be keyed on that same value rather than on the provider's own completion id.
"""

import datetime
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Final
from unittest.mock import Mock

import pytest
import respx
from pydantic import TypeAdapter

import litellm
from litellm.llms.anthropic.messages.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)
from litellm.llms.anthropic.messages.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam
from litellm.types.utils import ModelResponseStream

_EVENT: Final = TypeAdapter(Mapping[str, object])
_MESSAGES: Final = TypeAdapter(list[AllMessageValues])
_TOOLS: Final = TypeAdapter(list[ChatCompletionToolParam])
_BLOCK_PROMPTS: Final[tuple[list[AllMessageValues], ...]] = (
    [{"role": "user", "content": [{"type": "text", "text": "Count every word. " * 100}]}],
    [{"role": "tool", "content": [{"type": "text", "text": "Count every word. " * 100}], "tool_call_id": "call_1"}],
    [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://image.example/test.png", "detail": "high"}}
            ],
        }
    ],
)


def _assert_start_and_provider_final_usage(
    raw_events: Sequence[bytes], request_body: bytes, estimate_enabled: bool
) -> None:
    request: Final = _EVENT.validate_json(request_body)
    events: Final = tuple(_EVENT.validate_json(chunk.decode().split("data: ", 1)[1]) for chunk in raw_events)
    start: Final = _EVENT.validate_python(
        next(event for event in events if event["type"] == "message_start")["message"]
    )
    final: Final = next(event for event in events if event["type"] == "message_delta")
    expected: Final = litellm.token_counter(
        model="kimi-k2",
        messages=_MESSAGES.validate_python(request["messages"]),
        tools=_TOOLS.validate_python(request["tools"]),
    )
    assert expected > 0
    assert start["usage"] == {
        "input_tokens": expected if estimate_enabled else 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert final["usage"] == {"input_tokens": 3, "output_tokens": 4}


@pytest.mark.parametrize("estimate_enabled", [None, False, True, "false"])
def test_sync_message_start_opt_in_preserves_provider_final_usage(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, estimate_enabled: bool | str | None
) -> None:
    if estimate_enabled is not None:
        monkeypatch.setattr(litellm, "anthropic_messages_estimate_initial_input_tokens", estimate_enabled)
    stream: Final = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
        max_tokens=16,
        messages=MESSAGES,
        system="Reply with a short greeting.",
        tools=[{"name": "greet", "description": "Greet a person", "input_schema": {"type": "object"}}],
        model="groq/kimi-k2",
        stream=True,
        custom_llm_provider="groq",
    )
    _assert_start_and_provider_final_usage(
        list(stream), respx_mock.calls.last.request.content, estimate_enabled is True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("estimate_enabled", [None, False, True, "false"])
async def test_async_message_start_opt_in_preserves_provider_final_usage(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, estimate_enabled: bool | str | None
) -> None:
    if estimate_enabled is not None:
        monkeypatch.setattr(litellm, "anthropic_messages_estimate_initial_input_tokens", estimate_enabled)
    stream: Final = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=16,
        messages=MESSAGES,
        system="Reply with a short greeting.",
        tools=[{"name": "greet", "description": "Greet a person", "input_schema": {"type": "object"}}],
        model="groq/kimi-k2",
        stream=True,
        custom_llm_provider="groq",
    )
    raw: Final = [chunk async for chunk in stream]
    _assert_start_and_provider_final_usage(raw, respx_mock.calls.last.request.content, estimate_enabled is True)


class _FailingUpstream:
    def __iter__(self) -> Iterator[ModelResponseStream]:
        raise RuntimeError("upstream failed before its first chunk")

    def __aiter__(self) -> AsyncIterator[ModelResponseStream]:
        raise RuntimeError("upstream failed before its first chunk")


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_disabled_initial_estimate_skips_token_counting(monkeypatch: pytest.MonkeyPatch, is_async: bool) -> None:
    counter: Final = Mock(side_effect=AssertionError("disabled estimation must not count tokens"))
    monkeypatch.setattr(litellm, "anthropic_messages_estimate_initial_input_tokens", False)
    monkeypatch.setattr(litellm, "token_counter", counter)
    wrapper: Final = AnthropicStreamWrapper(
        _FailingUpstream(), model="kimi-k2", completion_request={"messages": MESSAGES}
    )
    first: Final = _EVENT.validate_python(await anext(wrapper) if is_async else next(wrapper))
    message: Final = _EVENT.validate_python(first["message"])
    assert message["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    counter.assert_not_called()


@pytest.mark.parametrize("messages", _BLOCK_PROMPTS)
def test_sync_initial_estimate_counts_content_blocks(
    messages: list[AllMessageValues], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "anthropic_messages_estimate_initial_input_tokens", True)
    image: Final = respx_mock.get("https://image.example/test.png").respond(503)
    wrapper: Final = AnthropicStreamWrapper(iter([]), model="gpt-4o", completion_request={"messages": messages})
    expected: Final = litellm.token_counter(model="gpt-4o", messages=messages, use_default_image_token_count=True)
    first: Final = _EVENT.validate_python(next(wrapper))
    message: Final = _EVENT.validate_python(first["message"])
    assert message["usage"] == {
        "input_tokens": expected,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert not image.called


@pytest.mark.asyncio
@pytest.mark.parametrize("messages", _BLOCK_PROMPTS)
async def test_async_initial_estimate_counts_content_blocks(
    messages: list[AllMessageValues], respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "anthropic_messages_estimate_initial_input_tokens", True)
    image: Final = respx_mock.get("https://image.example/test.png").respond(503)
    wrapper: Final = AnthropicStreamWrapper(iter([]), model="gpt-4o", completion_request={"messages": messages})
    expected: Final = litellm.token_counter(model="gpt-4o", messages=messages, use_default_image_token_count=True)
    first: Final = _EVENT.validate_python(await anext(wrapper))
    message: Final = _EVENT.validate_python(first["message"])
    assert message["usage"] == {
        "input_tokens": expected,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert not image.called


@pytest.mark.parametrize("prompt", [{}, {"messages": []}, {"messages": "invalid"}])
def test_sync_initial_estimate_fallback_does_not_consume_upstream(
    prompt: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "anthropic_messages_estimate_initial_input_tokens", True)
    wrapper: Final = AnthropicStreamWrapper(_FailingUpstream(), model="kimi-k2", completion_request=prompt)
    first: Final = _EVENT.validate_python(next(wrapper))
    message: Final = _EVENT.validate_python(first["message"])
    assert message["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    assert list(wrapper) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", [{}, {"messages": []}, {"messages": "invalid"}])
async def test_async_initial_estimate_fallback_does_not_consume_upstream(
    prompt: Mapping[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "anthropic_messages_estimate_initial_input_tokens", True)
    wrapper: Final = AnthropicStreamWrapper(_FailingUpstream(), model="kimi-k2", completion_request=prompt)
    first: Final = _EVENT.validate_python(await anext(wrapper))
    message: Final = _EVENT.validate_python(first["message"])
    assert message["usage"] == {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    with pytest.raises(RuntimeError, match="upstream failed before its first chunk"):
        [chunk async for chunk in wrapper]


MESSAGES = [{"role": "user", "content": "hello"}]

GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

CHAT_SSE_BODY = (
    b'data: {"id":"chatcmpl-lit6825","object":"chat.completion.chunk","created":1,'
    b'"model":"kimi-k2","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},'
    b'"finish_reason":null}]}\n\n'
    b'data: {"id":"chatcmpl-lit6825","object":"chat.completion.chunk","created":1,'
    b'"model":"kimi-k2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}}\n\n'
    b"data: [DONE]\n\n"
)


def _logging_obj(call_id: str):
    from litellm.litellm_core_utils.litellm_logging import Logging

    return Logging(
        model="kimi-k2",
        messages=MESSAGES,
        stream=True,
        call_type="anthropic_messages",
        start_time=datetime.datetime.now(datetime.timezone.utc),
        litellm_call_id=call_id,
        function_id="1234",
    )


def _streamed_message_id(raw_events: list[bytes]) -> str:
    events = [json.loads(chunk.decode().split("data: ", 1)[1]) for chunk in raw_events]
    message_start = next(e for e in events if e["type"] == "message_start")
    return message_start["message"]["id"]


@pytest.fixture(autouse=True)
def _intercept_groq(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-lit6825-test")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(GROQ_CHAT_URL).respond(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        content=CHAT_SSE_BODY,
    )


@pytest.mark.asyncio
async def test_async_streaming_hands_the_logging_object_the_message_id_the_caller_is_streamed():
    logging_obj = _logging_obj("6825beef-0000-4000-8000-000000000010")

    sse = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=1024,
        messages=MESSAGES,
        model="groq/kimi-k2",
        stream=True,
        custom_llm_provider="groq",
        litellm_logging_obj=logging_obj,
    )
    streamed_id = _streamed_message_id([chunk async for chunk in sse])

    assert streamed_id.startswith("msg_")
    assert logging_obj.streamed_anthropic_message_id == streamed_id


def test_sync_streaming_hands_the_logging_object_the_message_id_the_caller_is_streamed():
    logging_obj = _logging_obj("6825beef-0000-4000-8000-000000000011")

    sse = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
        max_tokens=1024,
        messages=MESSAGES,
        model="groq/kimi-k2",
        stream=True,
        custom_llm_provider="groq",
        litellm_logging_obj=logging_obj,
    )
    streamed_id = _streamed_message_id(list(sse))

    assert streamed_id.startswith("msg_")
    assert logging_obj.streamed_anthropic_message_id == streamed_id


def test_concurrent_streams_are_keyed_on_their_own_message_id():
    """Two callers streaming at once must not be handed, or logged under, one another's id."""
    first = AnthropicStreamWrapper(completion_stream=iter([]), model="kimi-k2")
    second = AnthropicStreamWrapper(completion_stream=iter([]), model="kimi-k2")

    assert first._message_id != second._message_id
    assert _streamed_message_id(list(first.anthropic_sse_wrapper())) == first._message_id
    assert _streamed_message_id(list(second.anthropic_sse_wrapper())) == second._message_id
