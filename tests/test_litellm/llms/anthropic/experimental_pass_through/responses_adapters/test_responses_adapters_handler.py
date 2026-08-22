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
