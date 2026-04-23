"""
Mock tests for vercel_ai_gateway provider
"""
import json
from unittest.mock import MagicMock, patch

import pytest
import respx

import litellm
from litellm import completion
from litellm.llms.vercel_ai_gateway.chat.transformation import VercelAIGatewayConfig
from litellm.cost_calculator import cost_per_token
import math

@pytest.fixture
def vercel_ai_gateway_response():
    """Mock response from Vercel AI Gateway API"""
    return {
        "id": "chatcmpl-vercel-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "openai/gpt-3.5-turbo",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello! This is a test response from Vercel AI Gateway."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 15, "total_tokens": 25},
    }


def test_vercel_ai_gateway_config_initialization():
    """Test VercelAIGatewayConfig initializes correctly"""
    config = VercelAIGatewayConfig()
    assert config.custom_llm_provider == "vercel_ai_gateway"


def test_get_llm_provider_vercel_ai_gateway():
    """Test that get_llm_provider correctly identifies vercel_ai_gateway"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with vercel_ai_gateway/provider/model-name format
    model, provider, api_key, api_base = get_llm_provider("vercel_ai_gateway/openai/gpt-4o")
    assert model == "openai/gpt-4o"
    assert provider == "vercel_ai_gateway"

    # Test with api_base containing vercel ai gateway endpoint
    model, provider, api_key, api_base = get_llm_provider("gpt-4o", api_base="https://ai-gateway.vercel.sh/v1")
    assert model == "gpt-4o"
    assert provider == "vercel_ai_gateway"
    assert api_base == "https://ai-gateway.vercel.sh/v1"


def test_vercel_ai_gateway_in_provider_lists():
    """Test that vercel_ai_gateway is registered in all necessary provider lists"""
    assert "vercel_ai_gateway" in litellm.openai_compatible_providers
    assert "vercel_ai_gateway" in litellm.provider_list
    assert "https://ai-gateway.vercel.sh/v1" in litellm.openai_compatible_endpoints


@pytest.mark.asyncio
async def test_vercel_ai_gateway_completion_call(respx_mock, vercel_ai_gateway_response, monkeypatch):
    """Test completion call with vercel_ai_gateway provider using mocked response"""
    monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://ai-gateway.vercel.sh/v1/chat/completions").respond(json=vercel_ai_gateway_response)

    response = await litellm.acompletion(
        model="vercel_ai_gateway/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, this is a test"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! This is a test response from Vercel AI Gateway."
    assert response.model == "vercel_ai_gateway/openai/gpt-3.5-turbo"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    assert request.method == "POST"
    assert "ai-gateway.vercel.sh" in str(request.url)

    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-api-key"


@pytest.mark.asyncio
async def test_vercel_ai_gateway_with_oidc_token(respx_mock, vercel_ai_gateway_response, monkeypatch):
    """Test completion call with vercel_ai_gateway provider using VERCEL_OIDC_TOKEN"""
    monkeypatch.setenv("VERCEL_OIDC_TOKEN", "test-oidc-token")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://ai-gateway.vercel.sh/v1/chat/completions").respond(json=vercel_ai_gateway_response)

    response = await litellm.acompletion(
        model="vercel_ai_gateway/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello, this is a test"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! This is a test response from Vercel AI Gateway."
    assert response.model == "vercel_ai_gateway/openai/gpt-3.5-turbo"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request

    assert "Authorization" in request.headers
    assert request.headers["Authorization"] == "Bearer test-oidc-token"


def test_vercel_ai_gateway_supported_params():
    """Test that vercel_ai_gateway returns the supported parameters"""
    config = VercelAIGatewayConfig()
    supported_params = config.get_supported_openai_params("vercel_ai_gateway/openai/gpt-3.5-turbo")

    # vercel_ai_gateway should include all base OpenAI params plus extra_body
    expected_base_params = [
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "top_logprobs",
        "max_tokens",
        "max_completion_tokens",
        "modalities",
        "prediction",
        "n",
        "presence_penalty",
        "seed",
        "stop",
        "stream",
        "stream_options",
        "temperature",
        "top_p",
        "tools",
        "tool_choice",
        "function_call",
        "functions",
        "max_retries",
        "extra_headers",
        "parallel_tool_calls",
        "audio",
        "web_search_options",
        "extra_body",
    ]

    for param in expected_base_params:
        assert param in supported_params, f"Expected parameter '{param}' not found in supported params"

    assert "extra_body" in supported_params


def test_vercel_ai_gateway_sync_completion(respx_mock, vercel_ai_gateway_response, monkeypatch):
    """Test synchronous completion call"""
    monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://ai-gateway.vercel.sh/v1/chat/completions").respond(json=vercel_ai_gateway_response)

    response = completion(
        model="vercel_ai_gateway/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! This is a test response from Vercel AI Gateway."
    assert response.model == "vercel_ai_gateway/openai/gpt-3.5-turbo"
    assert response.usage.total_tokens == 25


def test_vercel_ai_gateway_with_provider_options(respx_mock, vercel_ai_gateway_response, monkeypatch):
    """Test vercel_ai_gateway with providerOptions parameter"""
    monkeypatch.setenv("VERCEL_AI_GATEWAY_API_KEY", "test-api-key")
    litellm.disable_aiohttp_transport = True

    respx_mock.post("https://ai-gateway.vercel.sh/v1/chat/completions").respond(json=vercel_ai_gateway_response)

    response = completion(
        model="vercel_ai_gateway/openai/gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hello"}],
        providerOptions={"gateway": {"order": ["azure", "openai"]}},
        max_tokens=20,
    )

    assert response.choices[0].message.content == "Hello! This is a test response from Vercel AI Gateway."
    assert response.model == "vercel_ai_gateway/openai/gpt-3.5-turbo"
    assert response.usage.total_tokens == 25

    assert len(respx_mock.calls) == 1
    request = respx_mock.calls[0].request
    request_data = json.loads(request.content.decode("utf-8"))
    assert "providerOptions" in request_data
    assert request_data["providerOptions"]["gateway"]["order"] == ["azure", "openai"]


def test_vercel_ai_gateway_models_endpoint():
    """Test the get_models functionality"""
    config = VercelAIGatewayConfig()

    with patch("litellm.module_level_client.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [{"id": "openai/gpt-4o"}, {"id": "openai/gpt-3.5-turbo"}, {"id": "anthropic/claude-3-sonnet"}]
        }
        mock_get.return_value = mock_response

        models = config.get_models()

        assert models == ["openai/gpt-4o", "openai/gpt-3.5-turbo", "anthropic/claude-3-sonnet"]
        mock_get.assert_called_once_with(url="https://ai-gateway.vercel.sh/v1/models")


def test_vercel_ai_gateway_models_endpoint_failure():
    """Test the get_models functionality with failure"""
    config = VercelAIGatewayConfig()

    with patch("litellm.module_level_client.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_get.return_value = mock_response

        with pytest.raises(Exception, match="Failed to get models: Not found"):
            config.get_models()

def test_vercel_ai_gateway_glm46_cost_math():
    """Test the cost math for glm-4.6"""

    with open("model_prices_and_context_window.json", "r") as f:
        litellm.model_cost = json.load(f)

    key = "vercel_ai_gateway/zai/glm-4.6"
    info = litellm.model_cost[key]

    prompt_cost, completion_cost = cost_per_token(
        model="vercel_ai_gateway/zai/glm-4.6",
        prompt_tokens=1000,
        completion_tokens=500,
    )

    assert math.isclose(prompt_cost, 1000 * info["input_cost_per_token"], rel_tol=1e-12)
    assert math.isclose(completion_cost, 500 * info["output_cost_per_token"], rel_tol=1e-12)
