import os
from unittest.mock import patch

KENARI_API_BASE = "https://kenari.id/v1"


def test_kenari_json_registry():
    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.KENARI.value == "kenari"
    assert litellm.LlmProviders("kenari") == litellm.LlmProviders.KENARI
    assert JSONProviderRegistry.exists("kenari")
    config = JSONProviderRegistry.get("kenari")
    assert config is not None
    assert config.base_url == KENARI_API_BASE
    assert config.api_key_env == "KENARI_API_KEY"
    assert config.api_base_env == "KENARI_API_BASE"
    assert config.supported_endpoints == ["/v1/chat/completions"]


def test_kenari_dynamic_config_env_vars():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("kenari"))()

    with patch.dict(
        os.environ,
        {
            "KENARI_API_KEY": "test-key",
            "KENARI_API_BASE": "https://custom.kenari.example/v1",
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://custom.kenari.example/v1"
    assert api_key == "test-key"


def test_kenari_provider_detection_by_prefix():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider("kenari/deepseek-v4-pro")

    assert model == "deepseek-v4-pro"
    assert provider == "kenari"
    assert api_base == KENARI_API_BASE


def test_kenari_chat_complete_url():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("kenari"))()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://kenari.id/v1/chat/completions"
    )
