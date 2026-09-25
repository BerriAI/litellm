"""
Streaming ``/v1/messages`` against a model that is neither Anthropic nor OpenAI is served by
translating the call onto ``/v1/chat/completions``, and the ``msg_`` id the caller is streamed
is minted right here. It is the only request id such a caller ever sees, so the spend row has
to be keyed on that same value rather than on the provider's own completion id.
"""

import datetime
import json

import pytest
import respx

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)
from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)

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
