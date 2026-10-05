from datetime import datetime
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.completion_extras.litellm_responses_transformation.handler import (
    ResponsesToCompletionBridgeHandler,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.types.utils import ModelResponse

MODEL = "openai.gpt-5.5"
REGION = "us-east-2"


def _bedrock_mantle_kwargs() -> dict:
    messages = [{"role": "user", "content": "hi"}]
    logging_obj = LiteLLMLogging(
        litellm_call_id="test-call",
        call_type="acompletion",
        model=MODEL,
        messages=messages,
        function_id="fn-id",
        stream=False,
        start_time=datetime.now(),
    )
    return {
        "model": MODEL,
        "custom_llm_provider": "bedrock_mantle",
        "messages": messages,
        "optional_params": {},
        "litellm_params": {
            "aws_region_name": REGION,
            "api_base": "https://bedrock-mantle.us-east-1.api.aws/v1",
            "custom_llm_provider": "bedrock_mantle",
        },
        "headers": {},
        "model_response": ModelResponse(),
        "logging_obj": logging_obj,
    }


def _openai_kwargs() -> dict:
    messages = [{"role": "user", "content": "hi"}]
    logging_obj = LiteLLMLogging(
        litellm_call_id="test-call",
        call_type="completion",
        model="gpt-5.5",
        messages=messages,
        function_id="fn-id",
        stream=False,
        start_time=datetime.now(),
    )
    return {
        "model": "gpt-5.5",
        "custom_llm_provider": "openai",
        "messages": messages,
        "optional_params": {},
        "litellm_params": {},
        "headers": {},
        "model_response": ModelResponse(),
        "logging_obj": logging_obj,
    }


def test_completion_forwards_custom_llm_provider_to_responses():
    bridge = ResponsesToCompletionBridgeHandler()
    cached = ModelResponse(id="chatcmpl-cached", model="gpt-5.5")

    with patch("litellm.responses", return_value=cached) as fake_responses:
        result = bridge.completion(**_openai_kwargs())

    assert result is cached
    assert fake_responses.call_args.kwargs["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_acompletion_forwards_custom_llm_provider_to_aresponses():
    bridge = ResponsesToCompletionBridgeHandler()
    cached = ModelResponse(id="chatcmpl-cached", model="gpt-5.5")

    async def _fake_aresponses(**kwargs):
        _fake_aresponses.kwargs = kwargs
        return cached

    _fake_aresponses.kwargs = {}

    with patch("litellm.aresponses", _fake_aresponses):
        result = await bridge.acompletion(**_openai_kwargs())

    assert result is cached
    assert _fake_aresponses.kwargs["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_acompletion_forwards_aws_region_name_to_aresponses():
    bridge = ResponsesToCompletionBridgeHandler()
    cached = ModelResponse(id="chatcmpl-cached", model=MODEL)

    async def _fake_aresponses(**kwargs):
        _fake_aresponses.kwargs = kwargs
        return cached

    _fake_aresponses.kwargs = {}

    with patch("litellm.aresponses", _fake_aresponses):
        result = await bridge.acompletion(**_bedrock_mantle_kwargs())

    assert result is cached
    assert _fake_aresponses.kwargs["aws_region_name"] == REGION
    assert _fake_aresponses.kwargs["custom_llm_provider"] == "bedrock_mantle"


def test_completion_forwards_cache_to_responses():
    bridge = ResponsesToCompletionBridgeHandler()
    cached = ModelResponse(id="chatcmpl-cached", model="gpt-5.5")

    with patch("litellm.responses", return_value=cached) as fake_responses:
        result = bridge.completion(**_openai_kwargs(), cache={"no-cache": True})

    assert result is cached
    assert fake_responses.call_args.kwargs["cache"] == {"no-cache": True}


def test_completion_without_cache_kwarg_sends_no_cache_key():
    bridge = ResponsesToCompletionBridgeHandler()
    cached = ModelResponse(id="chatcmpl-cached", model="gpt-5.5")

    with patch("litellm.responses", return_value=cached) as fake_responses:
        result = bridge.completion(**_openai_kwargs())

    assert result is cached
    assert "cache" not in fake_responses.call_args.kwargs


@pytest.mark.asyncio
async def test_acompletion_forwards_cache_to_aresponses():
    bridge = ResponsesToCompletionBridgeHandler()
    cached = ModelResponse(id="chatcmpl-cached", model="gpt-5.5")

    async def _fake_aresponses(**kwargs):
        _fake_aresponses.kwargs = kwargs
        return cached

    _fake_aresponses.kwargs = {}

    with patch("litellm.aresponses", _fake_aresponses):
        result = await bridge.acompletion(**_openai_kwargs(), cache={"no-cache": True})

    assert result is cached
    assert _fake_aresponses.kwargs["cache"] == {"no-cache": True}


@pytest.mark.asyncio
async def test_litellm_acompletion_forwards_cache_through_the_bridge():
    cached = ModelResponse(id="resp_cached", model="gpt-6-sol")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]

    with patch("litellm.aresponses", new=AsyncMock(return_value=cached)) as fake_aresponses:
        result = await litellm.acompletion(
            model="azure/gpt-6-sol",
            api_base="https://example.invalid",
            api_key="x",
            api_version="2025-04-01-preview",
            messages=[{"role": "user", "content": "weather?"}],
            tools=tools,
            cache={"no-cache": True},
        )

    assert result is cached
    assert fake_aresponses.call_args.kwargs["cache"] == {"no-cache": True}
