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


@pytest.mark.asyncio
async def test_zai_completion_call(respx_mock, zai_response, monkeypatch):
    """Test completion call with zai provider using mocked response"""
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

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

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(
        json=zai_response
    )

    response = completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25


def test_zai_sends_response_format_and_maps_max_completion_tokens(respx_mock, zai_response, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model="zai/glm-5.3",
        messages=[{"role": "user", "content": "Reply in JSON"}],
        response_format={"type": "json_object"},
        max_completion_tokens=64,
    )

    body = json.loads(respx_mock.calls[0].request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_tokens"] == 64
    assert "max_completion_tokens" not in body


@pytest.mark.parametrize("model", ["glm-5v-turbo", "GLM-5V-Turbo", "GLM-5.3", "GLM-5.3-Flash"])
def test_zai_cost_resolves_case_insensitively(local_model_cost_map: None, model: str) -> None:
    prompt_cost, completion_cost = cost_per_token(
        model=model, custom_llm_provider="zai", prompt_tokens=1000, completion_tokens=1000
    )

    entry = litellm.model_cost[f"zai/{model.lower()}"]
    assert math.isclose(prompt_cost, 1000 * entry["input_cost_per_token"])
    assert math.isclose(completion_cost, 1000 * entry["output_cost_per_token"])


def test_zai_glm_5v_turbo_accepts_thinking(local_model_cost_map: None, respx_mock, zai_response, monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model="zai/glm-5v-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        thinking={"type": "disabled"},
    )

    body = json.loads(respx_mock.calls[0].request.content)
    assert body["thinking"] == {"type": "disabled"}
