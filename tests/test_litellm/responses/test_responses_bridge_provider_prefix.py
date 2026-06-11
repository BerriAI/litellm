"""
Regression tests for https://github.com/BerriAI/litellm/issues/28505

When a chat-completions request to a GPT-5.4+ model carries both `tools` and
`reasoning_effort`, it auto-routes through the responses API bridge. The bridge
must hand the already-resolved provider to `litellm.responses()` and tell it not
to re-run get_llm_provider; otherwise a second `provider/` prefix is stripped off
the model and the wrong model name is sent upstream.
"""

import asyncio
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

import litellm
from litellm.completion_extras.litellm_responses_transformation.transformation import (
    LiteLLMResponsesTransformationHandler,
)

SYNC_HANDLER = (
    "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.response_api_handler"
)
ASYNC_HANDLER = "litellm.llms.custom_httpx.llm_http_handler.BaseLLMHTTPHandler.async_response_api_handler"


def _bridge_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


@pytest.mark.parametrize(
    "configured_model, expected_outgoing",
    [
        # one routing prefix consumed by litellm; the remaining two prefixes
        # belong to the upstream model name and must survive
        ("openai/openai/openai/gpt-5.5", "openai/openai/gpt-5.5"),
        # single prefix: the bridge sends the bare model
        ("openai/gpt-5.5", "gpt-5.5"),
        # a second prefix that is itself a known provider must not be stripped
        ("openai/azure/gpt-5.5", "azure/gpt-5.5"),
    ],
)
def test_sync_bridge_preserves_model_prefix(configured_model, expected_outgoing):
    captured = {}

    def record(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        captured["custom_llm_provider"] = kwargs.get("custom_llm_provider")
        raise RuntimeError("stop-after-capture")

    with patch(SYNC_HANDLER, side_effect=record):
        with pytest.raises(Exception):
            litellm.completion(
                model=configured_model,
                messages=[{"role": "user", "content": "hi"}],
                tools=_bridge_tools(),
                reasoning_effort="low",
                api_base="https://example.invalid",
                api_key="sk-fake",
            )

    assert captured["model"] == expected_outgoing
    assert captured["custom_llm_provider"] == "openai"


def test_async_bridge_preserves_model_prefix():
    captured = {}

    async def record(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        captured["custom_llm_provider"] = kwargs.get("custom_llm_provider")
        raise RuntimeError("stop-after-capture")

    async def go():
        with pytest.raises(Exception):
            await litellm.acompletion(
                model="openai/openai/openai/gpt-5.5",
                messages=[{"role": "user", "content": "hi"}],
                tools=_bridge_tools(),
                reasoning_effort="low",
                api_base="https://example.invalid",
                api_key="sk-fake",
            )

    with patch(ASYNC_HANDLER, side_effect=record):
        asyncio.run(go())

    assert captured["model"] == "openai/openai/gpt-5.5"
    assert captured["custom_llm_provider"] == "openai"


def test_transform_request_marks_provider_resolved():
    handler = LiteLLMResponsesTransformationHandler()

    request_data = handler.transform_request(
        model="openai/openai/gpt-5.5",
        messages=[{"role": "user", "content": "hi"}],
        optional_params={},
        litellm_params={},
        headers={},
        litellm_logging_obj=MagicMock(),
        custom_llm_provider="openai",
    )

    assert request_data["model"] == "openai/openai/gpt-5.5"
    assert request_data["custom_llm_provider"] == "openai"
    assert request_data["_provider_already_resolved"] is True
