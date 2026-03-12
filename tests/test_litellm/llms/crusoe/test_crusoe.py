import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))
from unittest.mock import patch

CRUSOE_API_BASE = "https://managed-inference-api-proxy.crusoecloud.com/v1/"


def test_crusoe_json_registry():
    """Test Crusoe is registered in the JSON provider registry"""
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.exists("crusoe")
    config = JSONProviderRegistry.get("crusoe")
    assert config is not None
    assert config.base_url == CRUSOE_API_BASE
    assert config.api_key_env == "CRUSOE_API_KEY"
    assert config.api_base_env == "CRUSOE_API_BASE"


def test_crusoe_dynamic_config_defaults():
    """Test dynamic config returns correct default API base"""
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("crusoe"))()

    with patch.dict(os.environ, {}, clear=True):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == CRUSOE_API_BASE
    assert api_key is None


def test_crusoe_dynamic_config_env_vars():
    """Test dynamic config reads CRUSOE_API_KEY and CRUSOE_API_BASE from env"""
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("crusoe"))()

    with patch.dict(
        os.environ,
        {"CRUSOE_API_KEY": "test-key", "CRUSOE_API_BASE": "https://custom.crusoe.com/v1/"},
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://custom.crusoe.com/v1/"
    assert api_key == "test-key"


def test_crusoe_dynamic_config_explicit_params():
    """Test explicit params override env vars"""
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("crusoe"))()

    with patch.dict(os.environ, {"CRUSOE_API_KEY": "env-key"}):
        api_base, api_key = config._get_openai_compatible_provider_info(
            "https://override.crusoe.com/v1/", "override-key"
        )

    assert api_base == "https://override.crusoe.com/v1/"
    assert api_key == "override-key"


def test_crusoe_supported_params():
    """Test dynamic config returns standard OpenAI params"""
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("crusoe"))()
    params = config.get_supported_openai_params(model="meta-llama/Llama-3.3-70B-Instruct")

    assert isinstance(params, list)
    assert len(params) > 0
    assert "temperature" in params
    assert "max_tokens" in params
    assert "stream" in params


def test_crusoe_provider_detection_by_prefix():
    """Test crusoe/model prefix is correctly routed"""
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, _ = get_llm_provider("crusoe/meta-llama/Llama-3.3-70B-Instruct")
    assert provider == "crusoe"
    assert model == "meta-llama/Llama-3.3-70B-Instruct"


def test_crusoe_model_list_populated():
    """Test Crusoe models are present in model_prices_and_context_window.json"""
    import litellm

    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    expected = [
        "crusoe/meta-llama/Llama-3.3-70B-Instruct",
        "crusoe/deepseek-ai/DeepSeek-R1-0528",
        "crusoe/deepseek-ai/DeepSeek-V3-0324",
        "crusoe/Qwen/Qwen3-235B-A22B-Instruct-2507",
        "crusoe/moonshotai/Kimi-K2-Thinking",
        "crusoe/openai/gpt-oss-120b",
        "crusoe/google/gemma-3-12b-it",
    ]
    for model in expected:
        assert model in litellm.model_cost, f"{model} not found in model_cost"
        assert litellm.model_cost[model].get("litellm_provider") == "crusoe"
