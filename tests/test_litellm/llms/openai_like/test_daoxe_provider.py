import litellm


def test_daoxe_in_provider_list():
    from litellm import LlmProviders

    assert LlmProviders.DAOXE.value == "daoxe"
    assert "daoxe" in litellm.provider_list


def test_daoxe_json_config():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.exists("daoxe")
    provider = JSONProviderRegistry.get("daoxe")
    assert provider is not None
    assert provider.base_url == "https://daoxe.com/v1"
    assert provider.api_key_env == "DAOXE_API_KEY"
    assert provider.api_base_env == "DAOXE_API_BASE"


def test_daoxe_provider_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="daoxe/account-model",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "account-model"
    assert provider == "daoxe"
    assert api_key is None
    assert api_base == "https://daoxe.com/v1"


def test_daoxe_responses_api_config():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry
    from litellm.utils import ProviderConfigManager

    assert JSONProviderRegistry.supports_responses_api("daoxe") is True
    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="daoxe",
        model="daoxe/account-model",
    )
    assert config is not None
    assert config.custom_llm_provider == "daoxe"
