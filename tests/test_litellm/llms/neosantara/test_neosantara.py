import os
from unittest.mock import patch

NEOSANTARA_API_BASE = "https://api.neosantara.xyz/v1"


def test_neosantara_json_registry():
    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.NEOSANTARA.value == "neosantara"
    assert litellm.LlmProviders("neosantara") == litellm.LlmProviders.NEOSANTARA
    assert JSONProviderRegistry.exists("neosantara")
    config = JSONProviderRegistry.get("neosantara")
    assert config is not None
    assert config.base_url == NEOSANTARA_API_BASE
    assert config.api_key_env == "NEOSANTARA_API_KEY"
    assert config.api_base_env == "NEOSANTARA_API_BASE"
    assert config.param_mappings["max_completion_tokens"] == "max_tokens"
    assert "/v1/chat/completions" in config.supported_endpoints
    assert "/v1/responses" in config.supported_endpoints


def test_neosantara_dynamic_config_env_vars():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("neosantara"))()

    with patch.dict(
        os.environ,
        {
            "NEOSANTARA_API_KEY": "test-key",
            "NEOSANTARA_API_BASE": "https://custom.neosantara.example/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://custom.neosantara.example/v1"
    assert api_key == "test-key"


def test_neosantara_provider_detection_by_prefix():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider("neosantara/gemini-3-flash")

    assert model == "gemini-3-flash"
    assert provider == "neosantara"
    assert api_base == NEOSANTARA_API_BASE


def test_neosantara_chat_complete_url():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("neosantara"))()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-3-flash",
            optional_params={},
            litellm_params={},
        )
        == "https://api.neosantara.xyz/v1/chat/completions"
    )


def test_neosantara_maps_max_completion_tokens_to_max_tokens():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("neosantara"))()
    optional_params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 7},
        optional_params={},
        model="gemini-3-flash",
        drop_params=False,
    )

    assert optional_params == {"max_tokens": 7}


def test_neosantara_responses_api_config():
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="neosantara",
        model="claude-opus-4-6",
    )

    assert isinstance(config, OpenAIResponsesAPIConfig)
    assert config.custom_llm_provider == "neosantara"
    assert (
        config.get_complete_url(api_base=None, litellm_params={})
        == "https://api.neosantara.xyz/v1/responses"
    )
