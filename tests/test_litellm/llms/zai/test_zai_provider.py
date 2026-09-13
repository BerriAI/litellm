"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math

import pytest

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token


@pytest.fixture
def local_model_cost_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))


@pytest.fixture
def zai_response():
    """Mock response from Z.AI API"""
    return {
        "id": "chatcmpl-zai-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "glm-4.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def test_get_llm_provider_zai():
    """Test that get_llm_provider correctly identifies zai provider"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider("zai/glm-4.6")
    assert model == "glm-4.6"
    assert provider == "zai"
    assert api_base == "https://api.z.ai/api/paas/v4"


def test_zai_in_provider_lists():
    """Test that zai is registered in all necessary provider lists"""
    assert "zai" in litellm.openai_compatible_providers
    assert "zai" in litellm.provider_list


def test_zai_glm46_cost_calculation(local_model_cost_map):
    """Test the cost calculation for glm-4.6"""

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.6",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.6: $0.6/M input, $2.2/M output
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


def test_glm47_cost_calculation(local_model_cost_map):
    """Test cost calculation for GLM-4.7"""

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.7",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.7: $0.6/M input, $2.2/M output (same as GLM-4.6)
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


@pytest.mark.asyncio
async def test_zai_completion_call(respx_mock, zai_response, monkeypatch):
    """Test completion call with zai provider using mocked response"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    response = await litellm.acompletion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert "api.z.ai" in str(request.url)
    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"


def test_zai_sync_completion(respx_mock, zai_response, monkeypatch):
    """Test synchronous completion call"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    response = completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25


@pytest.fixture
def zai_thinking_response():
    return {
        "id": "chatcmpl-zai-thinking",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "glm-4.7",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


def _captured_body(respx_mock) -> dict:
    assert len(respx_mock.calls) == 1
    return json.loads(respx_mock.calls[0].request.content.decode("utf-8"))


def test_reasoning_params_supported_on_reasoning_models(local_model_cost_map):
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    params = ZAIChatConfig().get_supported_openai_params(model="glm-4.7")
    assert "thinking" in params
    assert "reasoning_effort" in params


def test_reasoning_params_not_supported_without_reasoning_flag(monkeypatch):
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    monkeypatch.setattr(litellm, "supports_reasoning", lambda **kwargs: False)
    params = ZAIChatConfig().get_supported_openai_params(model="glm-4-32b-0414-128k")
    assert "thinking" not in params
    assert "reasoning_effort" not in params


def test_thinking_and_reasoning_effort_move_into_extra_body(local_model_cost_map):
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    result = ZAIChatConfig()._map_openai_params(
        non_default_params={"max_tokens": 100, "thinking": {"type": "disabled"}, "reasoning_effort": "low"},
        optional_params={},
        model="glm-4.7",
        drop_params=False,
    )
    assert result["max_tokens"] == 100
    assert "thinking" not in result
    assert "reasoning_effort" not in result
    assert result["extra_body"] == {"thinking": {"type": "disabled"}, "reasoning_effort": "low"}


def test_caller_optional_params_and_extra_body_are_not_mutated(local_model_cost_map):
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    caller_extra_body = {"already_here": True}
    caller_optional_params = {"stream": False, "extra_body": caller_extra_body}
    result = ZAIChatConfig()._map_openai_params(
        non_default_params={"max_tokens": 100, "thinking": {"type": "disabled"}},
        optional_params=caller_optional_params,
        model="glm-4.7",
        drop_params=False,
    )
    assert result == {
        "stream": False,
        "max_tokens": 100,
        "extra_body": {"already_here": True, "thinking": {"type": "disabled"}},
    }
    assert result is not caller_optional_params
    assert caller_optional_params == {"stream": False, "extra_body": {"already_here": True}}
    assert caller_optional_params["extra_body"] is caller_extra_body
    assert caller_extra_body == {"already_here": True}


def test_reasoning_params_dropped_for_non_reasoning_model(monkeypatch):
    from litellm.utils import get_optional_params

    monkeypatch.setattr(litellm, "supports_reasoning", lambda **kwargs: False)
    result = get_optional_params(
        model="glm-4-32b-0414-128k",
        custom_llm_provider="zai",
        thinking={"type": "disabled"},
        reasoning_effort="low",
        drop_params=True,
    )
    assert "thinking" not in result
    assert "reasoning_effort" not in result
    assert "thinking" not in result.get("extra_body", {})
    assert "reasoning_effort" not in result.get("extra_body", {})


@pytest.mark.asyncio
async def test_thinking_and_reasoning_effort_reach_http_body(
    respx_mock, zai_thinking_response, monkeypatch, local_model_cost_map
):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_thinking_response)

    await litellm.acompletion(
        model="zai/glm-4.7",
        messages=[{"role": "user", "content": "hi"}],
        thinking={"type": "disabled"},
        reasoning_effort="low",
    )

    body = _captured_body(respx_mock)
    assert body["thinking"] == {"type": "disabled"}
    assert body["reasoning_effort"] == "low"
    assert "extra_body" not in body
