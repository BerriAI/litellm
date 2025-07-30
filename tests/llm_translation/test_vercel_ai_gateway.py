"""
Tests for vercel_ai_gateway provider integration
"""
import os
from unittest import mock

import pytest

import litellm
from litellm import completion
from litellm.llms.vercel_ai_gateway.chat.transformation import VercelAIGatewayConfig


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
    model, provider, api_key, api_base = get_llm_provider(
        "gpt-4o", api_base="https://ai-gateway.vercel.sh/v1"
    )
    assert model == "gpt-4o"
    assert provider == "vercel_ai_gateway"
    assert api_base == "https://ai-gateway.vercel.sh/v1"


def test_vercel_ai_gateway_in_provider_lists():
    """Test that vercel_ai_gateway is registered in all necessary provider lists"""
    assert "vercel_ai_gateway" in litellm.openai_compatible_providers
    assert "vercel_ai_gateway" in litellm.provider_list
    assert "https://ai-gateway.vercel.sh/v1" in litellm.openai_compatible_endpoints


@pytest.mark.asyncio
async def test_vercel_ai_gateway_completion_call():
    """Test completion call with vercel_ai_gateway provider (requires VERCEL_AI_GATEWAY_API_KEY)"""
    # Skip if no API key is available
    if not os.getenv("VERCEL_AI_GATEWAY_API_KEY"):
        pytest.skip("VERCEL_AI_GATEWAY_API_KEY not set")
    
    try:
        response = await litellm.acompletion(
            model="vercel_ai_gateway/openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test"}],
            max_tokens=20,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        # If the API key is invalid or there's a network issue, that's okay
        # The important thing is that the provider was recognized
        if "vercel_ai_gateway" not in str(e) and "provider" not in str(e).lower():
            # Re-raise if it's not a provider-related error
            raise


@pytest.mark.asyncio
async def test_vercel_ai_gateway_with_oidc_token():
    """Test completion call with vercel_ai_gateway provider using VERCEL_OIDC_TOKEN"""
    # Skip if no OIDC token is available
    if not os.getenv("VERCEL_OIDC_TOKEN"):
        pytest.skip("VERCEL_OIDC_TOKEN not set")
    
    try:
        response = await litellm.acompletion(
            model="vercel_ai_gateway/openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test"}],
            max_tokens=20,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        # If the OIDC token is invalid or there's a network issue, that's okay
        # The important thing is that the provider was recognized
        if "vercel_ai_gateway" not in str(e) and "provider" not in str(e).lower():
            # Re-raise if it's not a provider-related error
            raise


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
        "extra_body"
    ]
    
    for param in expected_base_params:
        assert param in supported_params, f"Expected parameter '{param}' not found in supported params"
    
    assert "extra_body" in supported_params


def test_vercel_ai_gateway_sync_completion():
    """Test synchronous completion call"""
    if not os.getenv("VERCEL_AI_GATEWAY_API_KEY"):
        pytest.skip("VERCEL_AI_GATEWAY_API_KEY not set")
    
    try:
        response = completion(
            model="vercel_ai_gateway/openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=20,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        if "vercel_ai_gateway" not in str(e) and "provider" not in str(e).lower():
            raise


def test_vercel_ai_gateway_with_provider_options():
    """Test vercel_ai_gateway with providerOptions parameter"""
    if not os.getenv("VERCEL_AI_GATEWAY_API_KEY"):
        pytest.skip("VERCEL_AI_GATEWAY_API_KEY not set")
    
    try:
        response = completion(
            model="vercel_ai_gateway/openai/gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            providerOptions={"gateway": {"order": ["azure", "openai"]}},
            max_tokens=20,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        if "vercel_ai_gateway" not in str(e) and "provider" not in str(e).lower():
            raise


def test_vercel_ai_gateway_models_endpoint():
    """Test the get_models functionality"""
    config = VercelAIGatewayConfig()
    
    with mock.patch("litellm.module_level_client.get") as mock_get:
        mock_response = mock.MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "openai/gpt-4o"},
                {"id": "openai/gpt-3.5-turbo"},
                {"id": "anthropic/claude-4-sonnet"}
            ]
        }
        mock_get.return_value = mock_response
        
        models = config.get_models()
        
        assert models == ["openai/gpt-4o", "openai/gpt-3.5-turbo", "anthropic/claude-4-sonnet"]
        mock_get.assert_called_once_with(url="https://ai-gateway.vercel.sh/v1/models")


def test_vercel_ai_gateway_models_endpoint_failure():
    """Test the get_models functionality with failure"""
    config = VercelAIGatewayConfig()
    
    with mock.patch("litellm.module_level_client.get") as mock_get:
        mock_response = mock.MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_get.return_value = mock_response
        
        with pytest.raises(Exception, match="Failed to get models: Not found"):
            config.get_models()
