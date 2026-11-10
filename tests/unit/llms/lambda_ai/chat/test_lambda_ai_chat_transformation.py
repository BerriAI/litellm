import os
from unittest import mock

import litellm
import pytest

from litellm.llms.lambda_ai.chat.transformation import LambdaAIChatConfig


def test_get_llm_provider_lambda_ai():
    """Test that get_llm_provider correctly identifies Lambda AI"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with lambda_ai/model-name format
    model, provider, api_key, api_base = get_llm_provider(
        "lambda_ai/llama3.1-8b-instruct"
    )
    assert model == "llama3.1-8b-instruct"
    assert provider == "lambda_ai"

    # Test with api_base containing Lambda AI endpoint
    model, provider, api_key, api_base = get_llm_provider(
        "llama3.1-8b-instruct", api_base="https://api.lambda.ai/v1"
    )
    assert model == "llama3.1-8b-instruct"
    assert provider == "lambda_ai"
    assert api_base == "https://api.lambda.ai/v1"


def test_lambda_ai_config_initialization() -> None:
    """Test LambdaAIChatConfig initializes correctly"""
    config = LambdaAIChatConfig()
    assert config.custom_llm_provider == "lambda_ai"


def test_lambda_ai_get_openai_compatible_provider_info() -> None:
    """Test Lambda AI provider info retrieval"""
    config = LambdaAIChatConfig()

    with mock.patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://api.lambda.ai/v1"
        assert api_key is None

    with mock.patch.dict(
        os.environ,
        {
            "LAMBDA_API_KEY": "test-key",
            "LAMBDA_API_BASE": "https://custom.lambda.ai/v1",
        },
    ):
        api_base, api_key = config.get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.lambda.ai/v1"
        assert api_key == "test-key"

    with mock.patch.dict(
        os.environ,
        {"LAMBDA_API_KEY": "env-key", "LAMBDA_API_BASE": "https://env.lambda.ai/v1"},
    ):
        api_base, api_key = config.get_openai_compatible_provider_info("https://param.lambda.ai/v1", "param-key")
        assert api_base == "https://param.lambda.ai/v1"
        assert api_key == "param-key"


def test_lambda_ai_in_provider_lists() -> None:
    """Test that Lambda AI is registered in all necessary provider lists"""
    assert "lambda_ai" in litellm.openai_compatible_providers
    assert "lambda_ai" in litellm.provider_list
    assert "https://api.lambda.ai/v1" in litellm.openai_compatible_endpoints
