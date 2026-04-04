"""
Unit tests for the AIHubMix provider registration and configuration.
"""

import litellm
from litellm.llms.aihubmix.chat.transformation import AIHubMixChatConfig
from litellm.types.utils import LlmProviders


def test_aihubmix_in_llm_providers_enum():
    assert LlmProviders.AIHUBMIX == "aihubmix"


def test_aihubmix_in_provider_list():
    assert "aihubmix" in [p.value for p in LlmProviders]


def test_aihubmix_config_registered_in_provider_config_manager():
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="gpt-4o", provider=LlmProviders.AIHUBMIX
    )
    assert config is not None
    assert isinstance(config, AIHubMixChatConfig)


def test_aihubmix_get_llm_provider_slash_notation():
    model, provider, _, _ = litellm.get_llm_provider("aihubmix/gpt-4o")
    assert provider == "aihubmix"
    assert model == "gpt-4o"


def test_aihubmix_get_llm_provider_explicit():
    model, provider, _, _ = litellm.get_llm_provider(
        "claude-3-5-sonnet-20241022", custom_llm_provider="aihubmix"
    )
    assert provider == "aihubmix"
    assert model == "claude-3-5-sonnet-20241022"


def test_aihubmix_default_base_url():
    config = AIHubMixChatConfig()
    api_base, _ = config._get_openai_compatible_provider_info(
        api_base=None, api_key=None
    )
    assert api_base == "https://aihubmix.com/v1"
