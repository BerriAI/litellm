"""
Tests for Crusoe provider integration
"""
import os
from unittest import mock

import litellm

CRUSOE_API_BASE = "https://managed-inference-api-proxy.crusoecloud.com/v1/"


def test_crusoe_json_registry():
    """Test CrusoeChatConfig is loaded from JSON provider registry"""
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.exists("crusoe")
    config = JSONProviderRegistry.get("crusoe")
    assert config is not None
    assert config.base_url == CRUSOE_API_BASE
    assert config.api_key_env == "CRUSOE_API_KEY"
    assert config.api_base_env == "CRUSOE_API_BASE"


def test_crusoe_get_openai_compatible_provider_info():
    """Test Crusoe provider info retrieval"""
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("crusoe"))()

    # Test with default values (no env vars set)
    with mock.patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == CRUSOE_API_BASE
        assert api_key is None

    # Test with environment variables
    with mock.patch.dict(
        os.environ,
        {
            "CRUSOE_API_KEY": "test-key",
            "CRUSOE_API_BASE": "https://custom.crusoecloud.com/v1/",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)
        assert api_base == "https://custom.crusoecloud.com/v1/"
        assert api_key == "test-key"

    # Test with explicit parameters (should override env vars)
    with mock.patch.dict(
        os.environ,
        {
            "CRUSOE_API_KEY": "env-key",
            "CRUSOE_API_BASE": "https://env.crusoecloud.com/v1/",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://param.crusoecloud.com/v1/", "param-key"
        )
        assert api_base == "https://param.crusoecloud.com/v1/"
        assert api_key == "param-key"


def test_get_llm_provider_crusoe():
    """Test that get_llm_provider correctly identifies Crusoe"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with crusoe/model-name format
    model, provider, api_key, api_base = get_llm_provider(
        "crusoe/meta-llama/Llama-3.3-70B-Instruct"
    )
    assert model == "meta-llama/Llama-3.3-70B-Instruct"
    assert provider == "crusoe"


def test_crusoe_models_configuration():
    """Test that Crusoe models are configured correctly"""
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    crusoe_models = [
        "crusoe/meta-llama/Llama-3.3-70B-Instruct",
        "crusoe/deepseek-ai/DeepSeek-R1-0528",
        "crusoe/deepseek-ai/DeepSeek-V3-0324",
        "crusoe/Qwen/Qwen3-235B-A22B-Instruct-2507",
        "crusoe/moonshotai/Kimi-K2-Thinking",
        "crusoe/openai/gpt-oss-120b",
        "crusoe/google/gemma-3-12b-it",
    ]

    for model in crusoe_models:
        model_info = get_model_info(model)
        assert model_info is not None, f"Model info not found for {model}"
        assert model_info.get("litellm_provider") == "crusoe", (
            f"{model} should have crusoe as provider"
        )
        assert model_info.get("mode") == "chat", f"{model} should be in chat mode"
