"""
Tests for Crusoe provider integration
"""
import os
from unittest import mock

import litellm

CRUSOE_API_BASE = "https://api.inference.crusoecloud.com/v1"


def test_crusoe_native_config():
    """Test CrusoeConfig is registered as the native provider config"""
    from litellm.llms.crusoe.chat.transformation import CrusoeConfig
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="meta-llama/Llama-3.3-70B-Instruct",
        provider=LlmProviders.CRUSOE,
    )
    assert isinstance(config, CrusoeConfig)


def test_crusoe_provider_info_resolution():
    """Test Crusoe provider info retrieval via get_llm_provider"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    # Test with default values (no env vars set)
    with mock.patch.dict(os.environ, {}, clear=True):
        model, provider, api_key, api_base = get_llm_provider(
            "crusoe/meta-llama/Llama-3.3-70B-Instruct"
        )
        assert api_base == CRUSOE_API_BASE
        assert api_key is None

    # Test with environment variables
    with mock.patch.dict(
        os.environ,
        {
            "CRUSOE_API_KEY": "test-key",
            "CRUSOE_API_BASE": "https://custom.crusoecloud.com/v1",
        },
    ):
        model, provider, api_key, api_base = get_llm_provider(
            "crusoe/meta-llama/Llama-3.3-70B-Instruct"
        )
        assert api_base == "https://custom.crusoecloud.com/v1"
        assert api_key == "test-key"

    # Test with explicit parameters (should override env vars)
    with mock.patch.dict(
        os.environ,
        {
            "CRUSOE_API_KEY": "env-key",
            "CRUSOE_API_BASE": "https://env.crusoecloud.com/v1",
        },
    ):
        model, provider, api_key, api_base = get_llm_provider(
            "crusoe/meta-llama/Llama-3.3-70B-Instruct",
            api_base="https://param.crusoecloud.com/v1",
            api_key="param-key",
        )
        assert api_base == "https://param.crusoecloud.com/v1"
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

    original_model_cost = litellm.model_cost
    original_env = os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP")
    try:
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
    finally:
        litellm.model_cost = original_model_cost
        if original_env is None:
            os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
        else:
            os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = original_env
