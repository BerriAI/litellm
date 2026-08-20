import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.melious.messages.transformation import (
    MeliousAnthropicMessagesConfig,
)
from litellm.utils import ProviderConfigManager

MESSAGES_URL = "https://api.melious.ai/v1/messages"


def complete_url(config, api_base):
    return config.get_complete_url(
        api_base=api_base,
        api_key=None,
        model="glm-5.1",
        optional_params={},
        litellm_params={},
    )


def test_provider_config_manager_returns_melious_messages_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="glm-5.1",
        provider=litellm.LlmProviders.MELIOUS,
    )

    assert isinstance(config, MeliousAnthropicMessagesConfig)
    assert config.custom_llm_provider == "melious"


def test_anthropic_provider_keeps_the_default_config():
    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="glm-5.1",
        provider=litellm.LlmProviders.ANTHROPIC,
    )

    assert isinstance(config, AnthropicMessagesConfig)
    assert not isinstance(config, MeliousAnthropicMessagesConfig)


def test_billing_metadata_is_stripped():
    """Melious rejects Anthropic's x-anthropic-billing-header system blocks."""
    assert MeliousAnthropicMessagesConfig().should_strip_billing_metadata() is True


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "https://api.melious.ai",
        "https://api.melious.ai/",
        "https://api.melious.ai/v1",
        "https://api.melious.ai/v1/",
        "https://api.melious.ai/v1/messages",
        "https://api.melious.ai/v1/chat/completions",
    ],
)
def test_complete_url_normalizes_every_base_shape(api_base, monkeypatch):
    monkeypatch.delenv("MELIOUS_API_BASE", raising=False)

    assert complete_url(MeliousAnthropicMessagesConfig(), api_base) == MESSAGES_URL


def test_complete_url_uses_a_self_hosted_base(monkeypatch):
    monkeypatch.delenv("MELIOUS_API_BASE", raising=False)

    url = complete_url(MeliousAnthropicMessagesConfig(), "https://melious.internal.example")

    assert url == "https://melious.internal.example/v1/messages"


def test_complete_url_reads_the_api_base_env(monkeypatch):
    monkeypatch.setenv("MELIOUS_API_BASE", "https://env.melious.test/v1")

    assert complete_url(MeliousAnthropicMessagesConfig(), None) == "https://env.melious.test/v1/messages"


def test_headers_carry_the_melious_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("MELIOUS_API_KEY", "sk-mel-env")

    headers, api_base = MeliousAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model="glm-5.1",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert headers["x-api-key"] == "sk-mel-env"
    assert headers["anthropic-version"] == "2023-06-01"
    assert headers["content-type"] == "application/json"
    assert api_base is None


def test_headers_prefer_the_passed_key_over_the_environment(monkeypatch):
    monkeypatch.setenv("MELIOUS_API_KEY", "sk-mel-env")

    headers, _ = MeliousAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model="glm-5.1",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="sk-mel-arg",
        api_base=None,
    )

    assert headers["x-api-key"] == "sk-mel-arg"


def test_existing_auth_headers_are_left_alone(monkeypatch):
    monkeypatch.setenv("MELIOUS_API_KEY", "sk-mel-env")

    headers, _ = MeliousAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={"authorization": "Bearer sk-mel-forwarded"},
        model="glm-5.1",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "x-api-key" not in headers
    assert headers["authorization"] == "Bearer sk-mel-forwarded"


def test_anthropic_beta_headers_are_still_injected(monkeypatch):
    """The base config owns beta-header handling; the override must not drop it."""
    monkeypatch.setenv("MELIOUS_API_KEY", "sk-mel-env")

    headers, _ = MeliousAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model="glm-5.1",
        messages=[],
        optional_params={"output_format": {"type": "json_schema"}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "structured-outputs-2025-11-13" in headers["anthropic-beta"]
