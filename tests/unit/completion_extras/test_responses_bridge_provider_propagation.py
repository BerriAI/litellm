import itertools
from datetime import datetime
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.caching.caching import Cache
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


def _responses_api_body(n: int) -> dict:
    return {
        "id": f"resp_{n}",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-6-sol",
        "output": [
            {
                "type": "message",
                "id": f"msg_{n}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": f"hi {n}", "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    }


_GPT_6_TOOLS: Final = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        },
    }
]

_GPT_6_REQUEST: Final = {
    "model": "azure/gpt-6-sol",
    "api_base": "https://example.invalid",
    "api_key": "x",
    "api_version": "2025-04-01-preview",
    "messages": [{"role": "user", "content": "weather?"}],
    "tools": _GPT_6_TOOLS,
}


@pytest.fixture
def _bridged_cache_edge(monkeypatch):
    monkeypatch.setattr(litellm, "cache", Cache(type="local"))
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    ids = itertools.count(1)
    with respx.mock(assert_all_called=True) as router:
        route = router.post(url__regex=r"https://example\.invalid/openai/responses.*").mock(
            side_effect=lambda request: httpx.Response(200, json=_responses_api_body(next(ids)))
        )
        yield route


def test_completion_no_cache_reaches_provider_each_time(_bridged_cache_edge):
    first = litellm.completion(**_GPT_6_REQUEST, cache={"no-cache": True})
    second = litellm.completion(**_GPT_6_REQUEST, cache={"no-cache": True})

    assert len(_bridged_cache_edge.calls) == 2
    assert [first.id, second.id] == ["resp_1", "resp_2"]


@pytest.mark.asyncio
async def test_acompletion_no_cache_reaches_provider_each_time(_bridged_cache_edge):
    first = await litellm.acompletion(**_GPT_6_REQUEST, cache={"no-cache": True})
    second = await litellm.acompletion(**_GPT_6_REQUEST, cache={"no-cache": True})

    assert len(_bridged_cache_edge.calls) == 2
    assert [first.id, second.id] == ["resp_1", "resp_2"]


@pytest.mark.asyncio
async def test_acompletion_without_cache_field_is_served_from_cache(_bridged_cache_edge):
    first = await litellm.acompletion(**_GPT_6_REQUEST)
    second = await litellm.acompletion(**_GPT_6_REQUEST)

    assert len(_bridged_cache_edge.calls) == 1
    assert [first.id, second.id] == ["resp_1", "resp_1"]
