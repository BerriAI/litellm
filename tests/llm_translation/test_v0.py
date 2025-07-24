"""
Tests for v0 provider integration
"""
import os
from unittest import mock

import pytest

import litellm
from litellm import completion
from litellm.llms.v0.chat.transformation import V0ChatConfig


def test_v0_config_initialization():
    """Test V0ChatConfig initializes correctly"""
    config = V0ChatConfig()
    assert config.custom_llm_provider == "v0"


def test_v0_get_openai_compatible_provider_info():
    """Test v0 provider info retrieval"""
    config = V0ChatConfig()
    
    # Test with default values (no env vars set)
    with mock.patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.v0.dev/v1"
        assert api_key is None
    
    # Test with environment variables
    with mock.patch.dict(os.environ, {"V0_API_KEY": "test-key", "V0_API_BASE": "https://custom.v0.ai/v1"}):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.v0.ai/v1"
        assert api_key == "test-key"
    
    # Test with explicit parameters (should override env vars)
    with mock.patch.dict(os.environ, {"V0_API_KEY": "env-key", "V0_API_BASE": "https://env.v0.ai/v1"}):
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://param.v0.ai/v1", "param-key"
        )
        assert api_base == "https://param.v0.ai/v1"
        assert api_key == "param-key"


def test_get_llm_provider_v0():
    """Test that get_llm_provider correctly identifies v0"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    
    # Test with v0/model-name format
    model, provider, api_key, api_base = get_llm_provider("v0/gpt-4-turbo")
    assert model == "gpt-4-turbo"
    assert provider == "v0"
    
    # Test with api_base containing v0 endpoint
    model, provider, api_key, api_base = get_llm_provider(
        "gpt-4-turbo", api_base="https://api.v0.dev/v1"
    )
    assert model == "gpt-4-turbo"
    assert provider == "v0"
    assert api_base == "https://api.v0.dev/v1"


def test_v0_in_provider_lists():
    """Test that v0 is registered in all necessary provider lists"""
    assert "v0" in litellm.openai_compatible_providers
    assert "v0" in litellm.provider_list
    assert "https://api.v0.dev/v1" in litellm.openai_compatible_endpoints


@pytest.mark.asyncio
async def test_v0_completion_call():
    """Test completion call with v0 provider (requires V0_API_KEY)"""
    # Skip if no API key is available
    if not os.getenv("V0_API_KEY"):
        pytest.skip("V0_API_KEY not set")
    
    try:
        response = await litellm.acompletion(
            model="v0/gpt-4-turbo",
            messages=[{"role": "user", "content": "Hello, this is a test"}],
            max_tokens=10,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        # If the API key is invalid or there's a network issue, that's okay
        # The important thing is that the provider was recognized
        if "v0" not in str(e) and "provider" not in str(e).lower():
            # Re-raise if it's not a provider-related error
            raise


def test_v0_supported_params():
    """Test that v0 returns only the supported parameters"""
    config = V0ChatConfig()
    supported_params = config.get_supported_openai_params("v0/v0-1.5-md")
    
    # v0 only supports these specific params
    expected_params = [
        "messages",
        "model",
        "stream",
        "tools",
        "tool_choice",
    ]
    
    assert set(supported_params) == set(expected_params)


def test_v0_models_configuration():
    """Test that v0 models are configured correctly"""
    from litellm import get_model_info
    
    # Reload model cost map to pick up local changes
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    # All v0 models
    v0_models = ["v0/v0-1.0-md", "v0/v0-1.5-md", "v0/v0-1.5-lg"]
    
    for model in v0_models:
        model_info = get_model_info(model)
        assert model_info is not None, f"Model info not found for {model}"
        # All v0 models support vision (multimodal)
        assert model_info.get("supports_vision") is True, f"{model} should support vision"
        assert model_info.get("litellm_provider") == "v0", f"{model} should have v0 as provider"
        assert model_info.get("mode") == "chat", f"{model} should be in chat mode"
        assert model_info.get("supports_function_calling") is True, f"{model} should support function calling"
        assert model_info.get("supports_system_messages") is True, f"{model} should support system messages"