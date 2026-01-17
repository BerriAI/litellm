"""
Tests for Z.AI (Zhipu AI) provider - GLM models
"""

import json
import math

import pytest
import respx

import litellm
from litellm import completion
from litellm.cost_calculator import cost_per_token


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
                "message": {"role": "assistant", "content": "Hello! How can I help you today?"},
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


def test_zai_models_in_model_cost():
    """Test that ZAI models are in the model cost map"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    zai_models = [
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5",
        "zai/glm-4.5v",
        "zai/glm-4.5-x",
        "zai/glm-4.5-air",
        "zai/glm-4.5-airx",
        "zai/glm-4-32b-0414-128k",
        "zai/glm-4.5-flash",
    ]

    for model in zai_models:
        assert model in litellm.model_cost, f"Model {model} not found in model_cost"
        assert litellm.model_cost[model]["litellm_provider"] == "zai"


def test_zai_glm46_cost_calculation():
    """Test the cost calculation for glm-4.6"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.6"
    info = litellm.model_cost[key]

    prompt_cost, completion_cost = cost_per_token(
        model="zai/glm-4.6",
        prompt_tokens=1000000,  # 1M tokens
        completion_tokens=1000000,
    )

    # GLM-4.6: $0.6/M input, $2.2/M output
    assert math.isclose(prompt_cost, 0.6, rel_tol=1e-6)
    assert math.isclose(completion_cost, 2.2, rel_tol=1e-6)


def test_zai_flash_model_is_free():
    """Test that glm-4.5-flash has zero cost"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.5-flash"
    info = litellm.model_cost[key]

    assert info["input_cost_per_token"] == 0
    assert info["output_cost_per_token"] == 0


def test_glm47_supports_reasoning():
    """Test that GLM-4.7 supports reasoning"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    key = "zai/glm-4.7"
    assert key in litellm.model_cost, f"Model {key} not found in model_cost"

    info = litellm.model_cost[key]
    assert info["supports_reasoning"] is True


def test_glm47_cost_calculation():
    """Test cost calculation for GLM-4.7"""
    import os

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

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
    litellm.disable_aiohttp_transport = True

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
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://api.z.ai/api/paas/v4/chat/completions").respond(json=zai_response)

    response = completion(
        model="zai/glm-4.6",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! How can I help you today?"
    assert response.usage.total_tokens == 25


# ========================================
# Web Search Tests
# ========================================


def test_zai_models_support_web_search(monkeypatch):
    """Test that GLM models have supports_web_search: true in model_cost"""
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    # Force reload of model cost map from local JSON
    litellm.model_cost = litellm.get_model_cost_map(url="")

    web_search_models = [
        "zai/glm-4.7",
        "zai/glm-4.6",
        "zai/glm-4.5",
        "zai/glm-4.5v",
    ]

    for model in web_search_models:
        assert model in litellm.model_cost, f"Model {model} not found in model_cost"
        info = litellm.model_cost[model]
        assert info.get("supports_web_search") is True, f"Model {model} should support web search"


def test_zai_web_search_options_in_supported_params(monkeypatch):
    """Test that web_search_options is in supported params for GLM models"""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    # Ensure model_cost is loaded from local JSON
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    litellm.model_cost = litellm.get_model_cost_map(url="")

    config = ZAIChatConfig()

    # Test for a model that supports web search
    params = config.get_supported_openai_params("zai/glm-4.7")
    assert "web_search_options" in params, "web_search_options should be in supported params for glm-4.7"


def test_zai_map_web_search_options_default():
    """Test _map_web_search_options with default (medium) search_context_size"""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    config = ZAIChatConfig()

    result = config._map_web_search_options({})

    assert result["type"] == "web_search"
    assert result["web_search"]["search_engine"] == "search_pro_jina"
    assert result["web_search"]["enable"] is True
    assert result["web_search"]["count"] == 10  # medium = 10
    assert result["web_search"]["content_size"] == "medium"
    assert result["web_search"]["search_recency_filter"] == "noLimit"


def test_zai_map_web_search_options_low():
    """Test _map_web_search_options with low search_context_size"""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    config = ZAIChatConfig()

    result = config._map_web_search_options({"search_context_size": "low"})

    assert result["web_search"]["count"] == 5  # low = 5
    assert result["web_search"]["content_size"] == "medium"


def test_zai_map_web_search_options_high():
    """Test _map_web_search_options with high search_context_size"""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    config = ZAIChatConfig()

    result = config._map_web_search_options({"search_context_size": "high"})

    assert result["web_search"]["count"] == 20  # high = 20
    assert result["web_search"]["content_size"] == "high"


def test_zai_map_openai_params_adds_web_search_tool():
    """Test that map_openai_params adds web_search tool to tools array"""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    config = ZAIChatConfig()

    non_default_params = {
        "web_search_options": {"search_context_size": "medium"}
    }
    optional_params = {}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="zai/glm-4.7",
        drop_params=False,
    )

    assert "tools" in result
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "web_search"
    assert result["tools"][0]["web_search"]["enable"] is True
    assert result["tools"][0]["web_search"]["count"] == 10


def test_zai_map_openai_params_preserves_existing_tools():
    """Test that map_openai_params preserves existing tools when adding web search"""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    config = ZAIChatConfig()

    existing_tool = {
        "type": "function",
        "function": {"name": "get_weather", "parameters": {}}
    }

    non_default_params = {
        "web_search_options": {"search_context_size": "medium"},
        "tools": [existing_tool],
    }
    optional_params = {"tools": [existing_tool]}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="zai/glm-4.7",
        drop_params=False,
    )

    assert "tools" in result
    assert len(result["tools"]) == 2
    # Existing tool preserved
    assert any(t.get("type") == "function" for t in result["tools"])
    # Web search tool added
    assert any(t.get("type") == "web_search" for t in result["tools"])


def test_zai_map_openai_params_no_duplicate_web_search():
    """Test that map_openai_params doesn't add duplicate web_search tool"""
    from litellm.llms.zai.chat.transformation import ZAIChatConfig

    config = ZAIChatConfig()

    existing_web_search = {
        "type": "web_search",
        "web_search": {"enable": True, "count": 5}
    }

    non_default_params = {
        "web_search_options": {"search_context_size": "high"},
        "tools": [existing_web_search],
    }
    optional_params = {"tools": [existing_web_search]}

    result = config.map_openai_params(
        non_default_params=non_default_params,
        optional_params=optional_params,
        model="zai/glm-4.7",
        drop_params=False,
    )

    # Should not add duplicate
    web_search_count = sum(1 for t in result["tools"] if t.get("type") == "web_search")
    assert web_search_count == 1
