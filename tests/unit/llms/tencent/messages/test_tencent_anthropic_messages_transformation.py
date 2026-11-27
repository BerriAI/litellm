from typing import Final

import pytest

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.anthropic.messages.native.transformation import (
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


@pytest.mark.parametrize("tencent_api_key", [None, "   "])
@pytest.mark.parametrize(
    ("anthropic_env", "anthropic_secret"),
    [
        ("ANTHROPIC_API_KEY", "sk-ant-secret"),
        ("ANTHROPIC_AUTH_TOKEN", "oauth-ant-secret"),
    ],
)
def test_missing_tencent_key_does_not_fall_back_to_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    tencent_api_key: str | None,
    anthropic_env: str,
    anthropic_secret: str,
) -> None:
    monkeypatch.delenv("TENCENT_API_KEY", raising=False)
    if tencent_api_key is not None:
        monkeypatch.setenv("TENCENT_API_KEY", tencent_api_key)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv(anthropic_env, anthropic_secret)
    monkeypatch.setattr(litellm, "api_key", None)
    headers: Final[dict[str, object]] = {}

    with pytest.raises(AuthenticationError) as error:
        TencentAnthropicMessagesConfig().validate_anthropic_messages_environment(
            headers=headers,
            model="deepseek-v4-pro",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base="https://tokenhub-intl.tencentcloudmaas.com",
        )

    assert "Tencent" in str(error.value)
    assert anthropic_secret not in str(error.value)
    assert {name.lower() for name in headers}.isdisjoint({"x-api-key", "authorization"})


@pytest.mark.parametrize(
    ("header_name", "header_value"),
    [
        ("x-api-key", "caller-key"),
        ("Authorization", "Bearer caller-token"),
    ],
)
def test_caller_auth_header_is_honored_without_tencent_key(
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
    header_value: str,
) -> None:
    monkeypatch.delenv("TENCENT_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
    monkeypatch.setattr(litellm, "api_key", None)
    headers: Final[dict[str, object]] = {header_name: header_value}

    resolved_headers, _ = TencentAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers=headers,
        model="deepseek-v4-pro",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base="https://tokenhub-intl.tencentcloudmaas.com",
    )

    assert resolved_headers[header_name] == header_value
    assert "sk-ant-secret" not in resolved_headers.values()
