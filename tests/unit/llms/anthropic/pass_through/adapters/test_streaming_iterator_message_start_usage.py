"""
A ``/v1/messages`` stream bridged onto ``/v1/chat/completions`` sends ``message_start`` before the
provider has reported any usage, so its ``input_tokens`` is a local estimate of the prompt. The
provider's own counts still arrive on ``message_delta`` and stay the authoritative figure.
"""

import json
from collections.abc import Mapping, Sequence
from typing import Final

import pytest
import respx

import litellm
from litellm.llms.anthropic.pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)

MESSAGES: Final = [{"role": "user", "content": "Summarize the plot of Hamlet in two sentences, please."}]
SYSTEM: Final = "You are a concise literature tutor."
PROVIDER_PROMPT_TOKENS: Final = 18
PROVIDER_COMPLETION_TOKENS: Final = 4

GROQ_CHAT_URL: Final = "https://api.groq.com/openai/v1/chat/completions"

CHAT_SSE_BODY: Final = (
    b'data: {"id":"chatcmpl-48481","object":"chat.completion.chunk","created":1,'
    b'"model":"kimi-k2","choices":[{"index":0,"delta":{"role":"assistant","content":"hi"},'
    b'"finish_reason":null}]}\n\n'
    b'data: {"id":"chatcmpl-48481","object":"chat.completion.chunk","created":1,'
    b'"model":"kimi-k2","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":18,"completion_tokens":4,"total_tokens":22}}\n\n'
    b"data: [DONE]\n\n"
)


def _usage_by_event(raw_events: Sequence[bytes]) -> Mapping[str, Mapping[str, int]]:
    events: Final = [json.loads(chunk.decode().split("data: ", 1)[1]) for chunk in raw_events]
    message_start: Final = next(e for e in events if e["type"] == "message_start")
    message_delta: Final = next(e for e in events if e["type"] == "message_delta")
    return {"message_start": message_start["message"]["usage"], "message_delta": message_delta["usage"]}


@pytest.fixture(autouse=True)
def _intercept_groq(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-48481-test")
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx_mock.post(GROQ_CHAT_URL).respond(
        status_code=200,
        headers={"Content-Type": "text/event-stream"},
        content=CHAT_SSE_BODY,
    )


async def _async_stream_usage() -> Mapping[str, Mapping[str, int]]:
    sse: Final = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=1024,
        messages=MESSAGES,
        system=SYSTEM,
        model="groq/kimi-k2",
        stream=True,
        custom_llm_provider="groq",
    )
    return _usage_by_event([chunk async for chunk in sse])


def _sync_stream_usage() -> Mapping[str, Mapping[str, int]]:
    sse: Final = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
        max_tokens=1024,
        messages=MESSAGES,
        system=SYSTEM,
        model="groq/kimi-k2",
        stream=True,
        custom_llm_provider="groq",
    )
    return _usage_by_event(list(sse))


def _expected_estimate() -> int:
    return litellm.token_counter(
        model="kimi-k2",
        messages=[{"role": "system", "content": SYSTEM}, *MESSAGES],
        use_default_image_token_count=True,
    )


def _assert_start_is_estimate_and_delta_is_provider(usage: Mapping[str, Mapping[str, int]]) -> None:
    assert usage["message_start"]["input_tokens"] == _expected_estimate()
    assert usage["message_start"]["input_tokens"] > 0
    assert usage["message_start"]["output_tokens"] == 0
    assert usage["message_delta"]["input_tokens"] == PROVIDER_PROMPT_TOKENS
    assert usage["message_delta"]["output_tokens"] == PROVIDER_COMPLETION_TOKENS


@pytest.mark.asyncio
async def test_async_message_start_estimates_the_prompt_and_message_delta_keeps_provider_usage():
    _assert_start_is_estimate_and_delta_is_provider(await _async_stream_usage())


def test_sync_message_start_estimates_the_prompt_and_message_delta_keeps_provider_usage():
    _assert_start_is_estimate_and_delta_is_provider(_sync_stream_usage())


def _failing_token_counter(model: str, messages: object, tools: object, use_default_image_token_count: bool) -> int:
    raise RuntimeError("tokenizer unavailable")


@pytest.mark.asyncio
async def test_async_counting_failure_reports_zero_and_still_streams(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "token_counter", _failing_token_counter)

    usage: Final = await _async_stream_usage()

    assert usage["message_start"]["input_tokens"] == 0
    assert usage["message_delta"]["input_tokens"] == PROVIDER_PROMPT_TOKENS


def test_sync_counting_failure_reports_zero_and_still_streams(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "token_counter", _failing_token_counter)

    usage: Final = _sync_stream_usage()

    assert usage["message_start"]["input_tokens"] == 0
    assert usage["message_delta"]["input_tokens"] == PROVIDER_PROMPT_TOKENS
