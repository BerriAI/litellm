"""
Regression tests for issue #35061: Anthropic ``/v1/messages`` with a tiny
``max_tokens`` (e.g. Claude Code's ``max_tokens=1`` model-availability probe)
routed to an OpenAI-family reasoning model (GPT-5.x).

What was broken:
* The reasoning model can't finish even one output token within such a small
  budget, so the provider returns a 400 whose message contains
  "Could not finish the message because max_tokens or model output limit was
  reached". The adapter surfaced that as a ``BadRequestError``, which made
  Claude Code believe the model was unavailable.
* Anthropic's own Messages API returns a 200 with ``stop_reason="max_tokens"``
  for the same request, so the gateway must mirror that contract.

These tests drive the handler with ``litellm.(a)completion`` mocked to raise the
provider error, so they fail on the pre-fix code (which re-raised) and pass only
once the handler translates the error into a ``max_tokens`` response. An
unrelated ``BadRequestError`` must still propagate untouched.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../..")))

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)

MESSAGES = [{"role": "user", "content": "hello"}]
MODEL = "gpt-5.2"
OUTPUT_LIMIT_MESSAGE = (
    "Could not finish the message because max_tokens or model output limit "
    "was reached. Please try again with higher max_tokens."
)


def _output_limit_error() -> litellm.BadRequestError:
    return litellm.BadRequestError(message=OUTPUT_LIMIT_MESSAGE, model=MODEL, llm_provider="openai")


def _unrelated_bad_request() -> litellm.BadRequestError:
    return litellm.BadRequestError(message="Invalid 'temperature': must be <= 2", model=MODEL, llm_provider="openai")


async def _collect(stream) -> bytes:
    chunks = []
    async for chunk in stream:
        chunks.append(chunk)
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_async_non_streaming_returns_max_tokens_response(monkeypatch):
    async def _raise(**_kwargs):
        raise _output_limit_error()

    monkeypatch.setattr(litellm, "acompletion", _raise)

    response = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=1, messages=MESSAGES, model=MODEL
    )

    assert response["stop_reason"] == "max_tokens"
    assert response["type"] == "message"
    assert response["role"] == "assistant"
    # No tokens were produced (or billed) for the rejected request.
    assert response["usage"]["output_tokens"] == 0


@pytest.mark.asyncio
async def test_async_streaming_returns_max_tokens_sse(monkeypatch):
    async def _raise(**_kwargs):
        raise _output_limit_error()

    monkeypatch.setattr(litellm, "acompletion", _raise)

    stream = await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
        max_tokens=1, messages=MESSAGES, model=MODEL, stream=True
    )
    sse = await _collect(stream)

    assert b"event: message_start" in sse
    assert b"event: message_stop" in sse
    assert b'"stop_reason": "max_tokens"' in sse


@pytest.mark.asyncio
async def test_async_unrelated_bad_request_still_raises(monkeypatch):
    async def _raise(**_kwargs):
        raise _unrelated_bad_request()

    monkeypatch.setattr(litellm, "acompletion", _raise)

    with pytest.raises(litellm.BadRequestError):
        await LiteLLMMessagesToCompletionTransformationHandler.async_anthropic_messages_handler(
            max_tokens=1, messages=MESSAGES, model=MODEL
        )


def test_sync_non_streaming_returns_max_tokens_response(monkeypatch):
    def _raise(**_kwargs):
        raise _output_limit_error()

    monkeypatch.setattr(litellm, "completion", _raise)

    response = LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
        max_tokens=1, messages=MESSAGES, model=MODEL, _is_async=False
    )

    assert response["stop_reason"] == "max_tokens"
    assert response["usage"]["output_tokens"] == 0


def test_sync_unrelated_bad_request_still_raises(monkeypatch):
    def _raise(**_kwargs):
        raise _unrelated_bad_request()

    monkeypatch.setattr(litellm, "completion", _raise)

    with pytest.raises(litellm.BadRequestError):
        LiteLLMMessagesToCompletionTransformationHandler.anthropic_messages_handler(
            max_tokens=1, messages=MESSAGES, model=MODEL, _is_async=False
        )
