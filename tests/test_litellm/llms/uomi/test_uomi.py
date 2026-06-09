import json
import os
from pathlib import Path
from unittest.mock import patch

UOMI_API_BASE = "https://gateway.uomi.ai/v1"
UOMI_MODEL = "deepseek/deepseek-v4-flash"


def test_uomi_json_registry():
    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.UOMI.value == "uomi"
    assert litellm.LlmProviders("uomi") == litellm.LlmProviders.UOMI
    assert JSONProviderRegistry.exists("uomi")

    config = JSONProviderRegistry.get("uomi")
    assert config is not None
    assert config.base_url == UOMI_API_BASE
    assert config.api_key_env == "UOMI_API_KEY"
    assert config.api_base_env == "UOMI_API_BASE"
    assert config.param_mappings.get("max_completion_tokens") == "max_tokens"


def test_uomi_listed_in_openai_compatible_providers():
    from litellm.constants import (
        openai_compatible_providers,
        openai_text_completion_compatible_providers,
    )

    assert "uomi" in openai_compatible_providers
    assert "uomi" in openai_text_completion_compatible_providers


def test_uomi_dynamic_config_env_vars():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("uomi"))()

    with patch.dict(
        os.environ,
        {
            "UOMI_API_KEY": "test-key",
            "UOMI_API_BASE": "https://custom.uomi.test/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://custom.uomi.test/v1"
    assert api_key == "test-key"


def test_uomi_provider_detection_by_prefix():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider(f"uomi/{UOMI_MODEL}")

    assert model == UOMI_MODEL
    assert provider == "uomi"
    assert api_base == UOMI_API_BASE


def test_uomi_provider_detection_by_api_base():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider(
        model="custom-uomi-model",
        api_base=UOMI_API_BASE,
    )

    assert model == "custom-uomi-model"
    assert provider == "uomi"
    assert api_base == UOMI_API_BASE


def test_uomi_chat_complete_url():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("uomi"))()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model=UOMI_MODEL,
            optional_params={},
            litellm_params={},
        )
        == f"{UOMI_API_BASE}/chat/completions"
    )


def test_uomi_model_info_contains_catalog_pricing():
    repo_root = Path(__file__).parents[4]
    model_prices_path = repo_root / "model_prices_and_context_window.json"
    model_prices = json.loads(model_prices_path.read_text())
    model_info = model_prices[f"uomi/{UOMI_MODEL}"]

    assert model_info["litellm_provider"] == "uomi"
    assert model_info["max_input_tokens"] == 1048576
    assert model_info["max_output_tokens"] == 8192
    assert model_info["input_cost_per_token"] == 7.864e-8
    assert model_info["output_cost_per_token"] == 1.5728e-7
