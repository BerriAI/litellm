from unittest.mock import patch

import pytest

import litellm
from litellm.llms.melious.chat.transformation import MeliousChatConfig
from litellm.utils import ProviderConfigManager

MODEL = "melious/glm-5.1"


def test_provider_config_manager_returns_melious_chat_config():
    config = ProviderConfigManager.get_provider_chat_config(
        model="glm-5.1",
        provider=litellm.LlmProviders.MELIOUS,
    )

    assert isinstance(config, MeliousChatConfig)
    assert config.custom_llm_provider == "melious"


@pytest.mark.parametrize(
    "api_base",
    [
        None,
        "https://api.melious.ai",
        "https://api.melious.ai/",
        "https://api.melious.ai/v1",
        "https://api.melious.ai/v1/",
        "https://api.melious.ai/v1/chat/completions",
    ],
)
def test_provider_info_normalizes_every_base_shape_to_v1(api_base):
    """The OpenAI SDK path uses this value as its base_url, so it must be the /v1 root."""
    with patch("litellm.llms.melious.chat.transformation.get_secret_str", return_value=None):
        resolved_api_base, _ = MeliousChatConfig()._get_openai_compatible_provider_info(api_base, "sk-mel-arg")

    assert resolved_api_base == "https://api.melious.ai/v1"


def test_provider_info_keeps_a_self_hosted_base_and_adds_v1():
    with patch("litellm.llms.melious.chat.transformation.get_secret_str", return_value=None):
        resolved_api_base, _ = MeliousChatConfig()._get_openai_compatible_provider_info(
            "https://melious.internal.example", "sk-mel-arg"
        )

    assert resolved_api_base == "https://melious.internal.example/v1"


def test_provider_info_prefers_the_argument_over_the_environment():
    secrets = {"MELIOUS_API_BASE": "https://env.melious.test/v1", "MELIOUS_API_KEY": "sk-mel-env"}
    with patch(
        "litellm.llms.melious.chat.transformation.get_secret_str",
        side_effect=secrets.get,
    ):
        api_base, api_key = MeliousChatConfig()._get_openai_compatible_provider_info(
            "https://arg.melious.test/v1", "sk-mel-arg"
        )

    assert api_base == "https://arg.melious.test/v1"
    assert api_key == "sk-mel-arg"


def test_provider_info_reads_env_secrets():
    secrets = {"MELIOUS_API_BASE": "https://env.melious.test", "MELIOUS_API_KEY": "sk-mel-env"}
    with patch(
        "litellm.llms.melious.chat.transformation.get_secret_str",
        side_effect=secrets.get,
    ):
        api_base, api_key = MeliousChatConfig()._get_openai_compatible_provider_info(None, None)

    assert api_base == "https://env.melious.test/v1"
    assert api_key == "sk-mel-env"


def test_get_llm_provider_routes_the_melious_prefix(monkeypatch):
    monkeypatch.setenv("MELIOUS_API_KEY", "sk-mel-env")
    monkeypatch.delenv("MELIOUS_API_BASE", raising=False)

    model, custom_llm_provider, dynamic_api_key, api_base = litellm.get_llm_provider(model=MODEL)

    assert (model, custom_llm_provider) == ("glm-5.1", "melious")
    assert dynamic_api_key == "sk-mel-env"
    assert api_base == "https://api.melious.ai/v1"


def test_get_llm_provider_keeps_the_routing_flavor_suffix(monkeypatch):
    """Melious selects a routing flavor via a `:<flavor>` suffix, which must survive the split."""
    monkeypatch.setenv("MELIOUS_API_KEY", "sk-mel-env")

    model, custom_llm_provider, _, _ = litellm.get_llm_provider(model=f"{MODEL}:eco")

    assert (model, custom_llm_provider) == ("glm-5.1:eco", "melious")


def test_melious_is_registered_as_an_openai_compatible_chat_provider():
    from litellm.constants import LITELLM_CHAT_PROVIDERS, openai_compatible_providers

    assert "melious" in openai_compatible_providers
    assert "melious" in LITELLM_CHAT_PROVIDERS


def test_supported_openai_params_inherit_the_openai_surface():
    params = litellm.get_supported_openai_params(model="glm-5.1", custom_llm_provider="melious")

    assert params is not None
    for expected in ("stream", "temperature", "tools", "tool_choice", "response_format", "max_completion_tokens"):
        assert expected in params


def test_get_optional_params_maps_through_the_melious_config():
    optional_params = litellm.utils.get_optional_params(
        model="glm-5.1",
        custom_llm_provider="melious",
        max_completion_tokens=256,
        temperature=0.2,
        stream=True,
    )

    assert optional_params["max_completion_tokens"] == 256
    assert optional_params["temperature"] == 0.2
    assert optional_params["stream"] is True


def test_validate_environment_reports_the_melious_key(monkeypatch):
    monkeypatch.delenv("MELIOUS_API_KEY", raising=False)
    assert litellm.validate_environment(model=MODEL)["missing_keys"] == ["MELIOUS_API_KEY"]

    monkeypatch.setenv("MELIOUS_API_KEY", "sk-mel-env")
    assert litellm.validate_environment(model=MODEL)["keys_in_environment"] is True
