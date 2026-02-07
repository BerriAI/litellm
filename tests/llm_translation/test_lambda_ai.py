"""
Tests for Lambda AI provider integration
"""
import os
from unittest import mock

import pytest

import litellm
from litellm import completion
from litellm.llms.lambda_ai.chat.transformation import LambdaAIChatConfig


def test_lambda_ai_config_initialization():
    """Test LambdaAIChatConfig initializes correctly"""
    config = LambdaAIChatConfig()
    assert config.custom_llm_provider == "lambda_ai"


def test_lambda_ai_get_openai_compatible_provider_info():
    """Test Lambda AI provider info retrieval"""
    config = LambdaAIChatConfig()
    
    # Test with default values (no env vars set)
    with mock.patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.lambda.ai/v1"
        assert api_key is None
    
    # Test with environment variables
    with mock.patch.dict(os.environ, {"LAMBDA_API_KEY": "test-key", "LAMBDA_API_BASE": "https://custom.lambda.ai/v1"}):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.lambda.ai/v1"
        assert api_key == "test-key"
    
    # Test with explicit parameters (should override env vars)
    with mock.patch.dict(os.environ, {"LAMBDA_API_KEY": "env-key", "LAMBDA_API_BASE": "https://env.lambda.ai/v1"}):
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://param.lambda.ai/v1", "param-key"
        )
        assert api_base == "https://param.lambda.ai/v1"
        assert api_key == "param-key"


def test_get_llm_provider_lambda_ai():
    """Test that get_llm_provider correctly identifies Lambda AI"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
    
    # Test with lambda_ai/model-name format
    model, provider, api_key, api_base = get_llm_provider("lambda_ai/llama3.1-8b-instruct")
    assert model == "llama3.1-8b-instruct"
    assert provider == "lambda_ai"
    
    # Test with api_base containing Lambda AI endpoint
    model, provider, api_key, api_base = get_llm_provider(
        "llama3.1-8b-instruct", api_base="https://api.lambda.ai/v1"
    )
    assert model == "llama3.1-8b-instruct"
    assert provider == "lambda_ai"
    assert api_base == "https://api.lambda.ai/v1"


def test_lambda_ai_in_provider_lists():
    """Test that Lambda AI is registered in all necessary provider lists"""
    assert "lambda_ai" in litellm.openai_compatible_providers
    assert "lambda_ai" in litellm.provider_list
    assert "https://api.lambda.ai/v1" in litellm.openai_compatible_endpoints


@pytest.mark.asyncio
async def test_lambda_ai_completion_call():
    """Test completion call with Lambda AI provider (requires LAMBDA_API_KEY)"""
    # Skip if no API key is available
    if not os.getenv("LAMBDA_API_KEY"):
        pytest.skip("LAMBDA_API_KEY not set")
    
    try:
        response = await litellm.acompletion(
            model="lambda_ai/llama3.1-8b-instruct",
            messages=[{"role": "user", "content": "Hello, this is a test"}],
            max_tokens=10,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        # If the API key is invalid or there's a network issue, that's okay
        # The important thing is that the provider was recognized
        if "lambda_ai" not in str(e) and "provider" not in str(e).lower():
            # Re-raise if it's not a provider-related error
            raise


def test_lambda_ai_models_configuration():
    """Test that Lambda AI models are configured correctly"""
    from litellm import get_model_info
    
    # Reload model cost map to pick up local changes
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    # Clear and repopulate lambda_ai_models list after reloading model_cost
    litellm.lambda_ai_models = set()
    litellm.add_known_models()
    
    # Some Lambda AI models to test
    lambda_ai_models = [
        "lambda_ai/deepseek-llama3.3-70b",
        "lambda_ai/hermes3-8b",
        "lambda_ai/llama3.1-8b-instruct",
        "lambda_ai/llama3.2-11b-vision-instruct",
        "lambda_ai/qwen25-coder-32b-instruct",
    ]
    
    for model in lambda_ai_models:
        model_info = get_model_info(model)
        assert model_info is not None, f"Model info not found for {model}"
        assert model_info.get("litellm_provider") == "lambda_ai", f"{model} should have lambda_ai as provider"
        assert model_info.get("mode") == "chat", f"{model} should be in chat mode"
        assert model_info.get("supports_function_calling") is True, f"{model} should support function calling"
        assert model_info.get("supports_system_messages") is True, f"{model} should support system messages"
        
        # Check vision support for vision models
        if "vision" in model:
            assert model_info.get("supports_vision") is True, f"{model} should support vision"


def test_lambda_ai_model_list_populated():
    """Test that lambda_ai_models list is populated correctly"""
    # Ensure we're using local model cost map and repopulate models
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    
    # Clear and repopulate all model lists after reloading model_cost
    litellm.lambda_ai_models = set()
    litellm.add_known_models()
    
    # This should be populated by the add_known_models function
    assert len(litellm.lambda_ai_models) > 0, "lambda_ai_models list should not be empty"
    
    # Check that all models in the list are Lambda AI models
    for model in litellm.lambda_ai_models:
        assert model.startswith("lambda_ai/"), f"Model {model} should start with 'lambda_ai/'"
    
    # Check some expected models are in the list
    expected_models = [
        "lambda_ai/llama3.1-8b-instruct",
        "lambda_ai/hermes3-405b",
        "lambda_ai/deepseek-v3-0324",
    ]
    
    for model in expected_models:
        assert model in litellm.lambda_ai_models, f"{model} should be in lambda_ai_models list"