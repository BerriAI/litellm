import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.zai.messages.transformation import ZAIAnthropicMessagesConfig
from litellm.utils import ProviderConfigManager


def test_zai_provider_uses_anthropic_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="glm-5.3",
        provider=litellm.LlmProviders.ZAI,
    )

    assert isinstance(config, ZAIAnthropicMessagesConfig)
    assert config.custom_llm_provider == "zai"


def test_anthropic_provider_keeps_default_config_for_zai_named_model():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="glm-5.3",
        provider=litellm.LlmProviders.ANTHROPIC,
    )

    assert isinstance(config, AnthropicMessagesConfig)
    assert not isinstance(config, ZAIAnthropicMessagesConfig)


def test_zai_anthropic_messages_config_defaults():
    config = ZAIAnthropicMessagesConfig()

    assert config.custom_llm_provider == "zai"
    assert config.get_api_base() == "https://api.z.ai/api/anthropic"


def test_zai_anthropic_messages_url_defaults_to_anthropic_endpoint():
    config = ZAIAnthropicMessagesConfig()

    url_cases = {
        None: "https://api.z.ai/api/anthropic/v1/messages",
        "https://api.z.ai/api/anthropic": "https://api.z.ai/api/anthropic/v1/messages",
        "https://api.z.ai/api/anthropic/v1": "https://api.z.ai/api/anthropic/v1/messages",
        "https://api.z.ai/api/anthropic/v1/messages": "https://api.z.ai/api/anthropic/v1/messages",
        "https://api.z.ai/api": "https://api.z.ai/api/anthropic/v1/messages",
    }

    for api_base, expected_url in url_cases.items():
        assert (
            config.get_complete_url(
                api_base=api_base,
                api_key=None,
                model="glm-5.3",
                optional_params={},
                litellm_params={},
            )
            == expected_url
        )


def test_zai_anthropic_messages_headers_use_zai_key():
    config = ZAIAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="glm-5.3",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-zai",
        api_base="https://example.test/anthropic",
    )

    assert api_base == "https://example.test/anthropic"
    assert headers["x-api-key"] == "sk-zai"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"


def test_zai_anthropic_messages_respects_existing_case_insensitive_auth_headers():
    config = ZAIAnthropicMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={"Authorization": "Bearer caller-token"},
        model="glm-5.3",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-zai",
        api_base="https://api.z.ai/api/anthropic",
    )

    assert headers == {
        "Authorization": "Bearer caller-token",
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
