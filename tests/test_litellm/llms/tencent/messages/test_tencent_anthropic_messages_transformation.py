import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.tencent.messages.transformation import (
    TencentAnthropicMessagesConfig,
)
from litellm.utils import ProviderConfigManager


def test_tencent_provider_uses_anthropic_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.TENCENT,
    )

    assert isinstance(config, TencentAnthropicMessagesConfig)
    assert config.custom_llm_provider == "tencent"


def test_anthropic_provider_keeps_default_config_for_tencent_named_model():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="deepseek-v4-pro",
        provider=litellm.LlmProviders.ANTHROPIC,
    )

    assert isinstance(config, AnthropicMessagesConfig)
    assert not isinstance(config, TencentAnthropicMessagesConfig)


def test_strips_billing_metadata():
    config = TencentAnthropicMessagesConfig()

    assert config.should_strip_billing_metadata() is True


def test_get_api_base_default():
    config = TencentAnthropicMessagesConfig()

    assert config.get_api_base() == "https://tokenhub-intl.tencentcloudmaas.com"


def test_get_api_base_from_arg():
    config = TencentAnthropicMessagesConfig()

    assert config.get_api_base(api_base="https://custom.example.com") == "https://custom.example.com"


def test_messages_url_default():
    config = TencentAnthropicMessagesConfig()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://tokenhub-intl.tencentcloudmaas.com/v1/messages"
    )


def test_messages_url_with_base_ending_in_v1():
    config = TencentAnthropicMessagesConfig()

    assert (
        config.get_complete_url(
            api_base="https://tokenhub-intl.tencentcloudmaas.com/v1",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://tokenhub-intl.tencentcloudmaas.com/v1/messages"
    )


def test_messages_url_with_base_ending_in_v1_messages():
    config = TencentAnthropicMessagesConfig()

    url = config.get_complete_url(
        api_base="https://tokenhub-intl.tencentcloudmaas.com/v1/messages",
        api_key=None,
        model="deepseek-v4-pro",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://tokenhub-intl.tencentcloudmaas.com/v1/messages"


def test_messages_url_with_base_ending_in_v1_chat_completions():
    config = TencentAnthropicMessagesConfig()

    assert (
        config.get_complete_url(
            api_base="https://tokenhub-intl.tencentcloudmaas.com/v1/chat/completions",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://tokenhub-intl.tencentcloudmaas.com/v1/messages"
    )


def test_messages_url_with_custom_base_no_v1():
    config = TencentAnthropicMessagesConfig()

    assert (
        config.get_complete_url(
            api_base="https://tokenhub.tencentcloudmaas.com",
            api_key=None,
            model="deepseek-v4-pro",
            optional_params={},
            litellm_params={},
        )
        == "https://tokenhub.tencentcloudmaas.com/v1/messages"
    )


def test_validate_environment_sets_headers():
    config = TencentAnthropicMessagesConfig()

    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="deepseek-v4-pro",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-tencent-key",
        api_base="https://custom.test",
    )

    assert headers["x-api-key"] == "sk-tencent-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert api_base == "https://custom.test"


def test_validate_environment_injects_anthropic_beta_headers():
    config = TencentAnthropicMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="deepseek-v4-pro",
        messages=[],
        optional_params={"speed": "fast"},
        litellm_params={},
        api_key="sk-tencent-key",
        api_base=None,
    )

    assert "anthropic-beta" in headers


def test_validate_environment_preserves_existing_headers():
    config = TencentAnthropicMessagesConfig()

    headers, _ = config.validate_anthropic_messages_environment(
        headers={"authorization": "Bearer existing", "anthropic-version": "2024-01-01"},
        model="deepseek-v4-pro",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-tencent-key",
        api_base=None,
    )

    assert headers["authorization"] == "Bearer existing"
    assert headers["anthropic-version"] == "2024-01-01"
    assert "x-api-key" not in headers
