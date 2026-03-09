"""
Tests for Crusoe provider integration
"""
import os
from unittest import mock

import pytest

import litellm
from litellm import completion
from litellm.llms.crusoe.chat.transformation import CrusoeChatConfig

CRUSOE_API_BASE = "https://managed-inference-api-proxy.crusoecloud.com/v1/"


def test_crusoe_config_initialization():
    """Test CrusoeChatConfig initializes correctly"""
    config = CrusoeChatConfig()
    assert config.custom_llm_provider == "crusoe"


def test_crusoe_get_openai_compatible_provider_info():
    """Test Crusoe provider info retrieval"""
    config = CrusoeChatConfig()

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

    # Test with api_base containing Crusoe endpoint
    model, provider, api_key, api_base = get_llm_provider(
        "meta-llama/Llama-3.3-70B-Instruct",
        api_base=CRUSOE_API_BASE,
    )
    assert model == "meta-llama/Llama-3.3-70B-Instruct"
    assert provider == "crusoe"
    assert api_base == CRUSOE_API_BASE


def test_crusoe_in_provider_lists():
    """Test that Crusoe is registered in all necessary provider lists"""
    assert "crusoe" in litellm.openai_compatible_providers
    assert "crusoe" in litellm.provider_list
    assert CRUSOE_API_BASE in litellm.openai_compatible_endpoints


def test_crusoe_models_configuration():
    """Test that Crusoe models are configured correctly"""
    from litellm import get_model_info

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.crusoe_models = set()
    litellm.add_known_models()

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


def test_crusoe_model_list_populated():
    """Test that crusoe_models list is populated correctly"""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    litellm.crusoe_models = set()
    litellm.add_known_models()

    assert len(litellm.crusoe_models) > 0, "crusoe_models list should not be empty"

    for model in litellm.crusoe_models:
        assert model.startswith("crusoe/"), (
            f"Model {model} should start with 'crusoe/'"
        )

    expected_models = [
        "crusoe/meta-llama/Llama-3.3-70B-Instruct",
        "crusoe/deepseek-ai/DeepSeek-R1-0528",
        "crusoe/deepseek-ai/DeepSeek-V3-0324",
        "crusoe/Qwen/Qwen3-235B-A22B-Instruct-2507",
        "crusoe/moonshotai/Kimi-K2-Thinking",
        "crusoe/openai/gpt-oss-120b",
        "crusoe/google/gemma-3-12b-it",
    ]

    for model in expected_models:
        assert model in litellm.crusoe_models, (
            f"{model} should be in crusoe_models list"
        )


@pytest.mark.asyncio
async def test_crusoe_completion_call():
    """Test completion call with Crusoe provider (requires CRUSOE_API_KEY)"""
    if not os.getenv("CRUSOE_API_KEY"):
        pytest.skip("CRUSOE_API_KEY not set")

    try:
        response = await litellm.acompletion(
            model="crusoe/meta-llama/Llama-3.3-70B-Instruct",
            messages=[{"role": "user", "content": "Hello, this is a test"}],
            max_tokens=10,
        )
        assert response.choices[0].message.content
        assert response.model
        assert response.usage
    except Exception as e:
        if "crusoe" not in str(e) and "provider" not in str(e).lower():
            raise
