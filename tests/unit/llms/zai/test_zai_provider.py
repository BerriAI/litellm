"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math
from collections.abc import Mapping
from typing import Final

import pytest
import respx

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


@pytest.mark.asyncio
async def test_zai_messages_thinking_reaches_glm_as_reasoning_effort(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    response: Final = await litellm.anthropic_messages(
        model="zai/glm-5.3",
        max_tokens=2048,
        thinking={"type": "enabled", "budget_tokens": 1024},
        messages=[{"role": "user", "content": "Is 391 prime?"}],
        api_key="test-api-key",
    )

    body: Final = json.loads(respx_mock.calls[0].request.content)
    assert body["reasoning_effort"] in litellm.model_cost["zai/glm-5.3"]["reasoning_effort_levels"]
    assert response["content"][-1]["text"] == "Hello! How can I help you today?"


@pytest.mark.parametrize(
    ("requested", "sent"),
    [("minimal", "low"), ("low", "low"), ("medium", "high"), ("high", "high"), ("xhigh", "max"), ("max", "max")],
)
def test_zai_rounds_reasoning_effort_up_to_a_level_the_model_declares(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    sent: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(model="zai/glm-5.3", messages=[{"role": "user", "content": "Hello"}], reasoning_effort=requested)

    assert json.loads(respx_mock.calls[0].request.content)["reasoning_effort"] == sent


@pytest.mark.parametrize("requested", ["none", "minimal", "medium", "xhigh"])
def test_zai_forwards_reasoning_effort_as_sent_to_models_declaring_no_levels(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(model="zai/glm-5.2", messages=[{"role": "user", "content": "Hello"}], reasoning_effort=requested)

    assert json.loads(respx_mock.calls[0].request.content)["reasoning_effort"] == requested


@pytest.mark.parametrize(("drop_params", "sent"), [(True, None), (False, "none")])
def test_zai_drops_a_reasoning_effort_the_model_cannot_take_only_under_drop_params(
    local_model_cost_map: None,
    respx_mock: respx.MockRouter,
    zai_response: Mapping[str, object],
    monkeypatch: pytest.MonkeyPatch,
    drop_params: bool,
    sent: str | None,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", "test-api-key")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    completion(
        model="zai/glm-5.3",
        messages=[{"role": "user", "content": "Hello"}],
        reasoning_effort="none",
        drop_params=drop_params,
    )

    assert json.loads(respx_mock.calls[0].request.content).get("reasoning_effort") == sent
