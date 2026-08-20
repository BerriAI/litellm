import litellm


def test_modelsell_registered_as_openai_compatible_provider():
    from litellm import LlmProviders
    from litellm.constants import openai_compatible_providers

    assert LlmProviders.MODELSELL.value == "modelsell"
    assert "modelsell" in litellm.provider_list
    assert "modelsell" in openai_compatible_providers


def test_modelsell_json_config_supports_only_chat_completions():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("modelsell")

    assert provider is not None
    assert provider.base_url == "https://modelsell.com/v1"
    assert provider.api_key_env == "MODELSELL_API_KEY"
    assert provider.api_base_env == "MODELSELL_API_BASE"
    assert provider.param_mappings == {"max_completion_tokens": "max_tokens"}
    assert provider.supported_endpoints == ["/v1/chat/completions"]
    assert JSONProviderRegistry.supports_responses_api("modelsell") is False


def test_modelsell_prefixed_model_uses_default_configuration():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="modelsell/test-model",
        custom_llm_provider=None,
        api_base=None,
        api_key="sk-test",
    )

    assert model == "test-model"
    assert provider == "modelsell"
    assert api_key == "sk-test"
    assert api_base == "https://modelsell.com/v1"


def test_modelsell_configuration_can_be_overridden_from_env(monkeypatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MODELSELL_API_KEY", "sk-env-test")
    monkeypatch.setenv("MODELSELL_API_BASE", "https://proxy.example.com/v1")

    _, provider, api_key, api_base = get_llm_provider(
        model="modelsell/test-model",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert provider == "modelsell"
    assert api_key == "sk-env-test"
    assert api_base == "https://proxy.example.com/v1"


def test_modelsell_url_autodetection_uses_environment_api_key(monkeypatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv("MODELSELL_API_KEY", "sk-env-test")

    model, provider, api_key, api_base = get_llm_provider(
        model="test-model",
        custom_llm_provider=None,
        api_base="https://modelsell.com/v1",
        api_key=None,
    )

    assert model == "test-model"
    assert provider == "modelsell"
    assert api_key == "sk-env-test"
    assert api_base == "https://modelsell.com/v1"


def test_modelsell_chat_completions_url():
    config = litellm.ProviderConfigManager.get_provider_chat_config(
        model="test-model",
        provider=litellm.LlmProviders.MODELSELL,
    )

    assert config is not None
    assert (
        config.get_complete_url(
            api_base=None,
            api_key="sk-test",
            model="test-model",
            optional_params={},
            litellm_params={},
        )
        == "https://modelsell.com/v1/chat/completions"
    )
