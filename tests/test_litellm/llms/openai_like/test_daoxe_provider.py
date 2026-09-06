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


def test_daoxe_max_completion_tokens_mapped():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("daoxe")
    assert provider is not None
    config = create_config_class(provider)()

    optional_params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 512},
        optional_params={},
        model="account-model",
        drop_params=False,
    )
    assert optional_params["max_tokens"] == 512
    assert "max_completion_tokens" not in optional_params


def test_daoxe_passthrough_params_kept():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("daoxe")
    assert provider is not None
    config = create_config_class(provider)()

    optional_params = config.map_openai_params(
        non_default_params={"temperature": 0.4, "top_p": 0.9},
        optional_params={},
        model="account-model",
        drop_params=False,
    )
    assert optional_params["temperature"] == 0.4
    assert optional_params["top_p"] == 0.9


def test_daoxe_chat_completions_url():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("daoxe")
    assert provider is not None
    config = create_config_class(provider)()

    url = config.get_complete_url(
        api_base=None,
        api_key="test-key",
        model="account-model",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://daoxe.com/v1/chat/completions"


def test_daoxe_chat_completions_url_custom_base():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("daoxe")
    assert provider is not None
    config = create_config_class(provider)()

    url = config.get_complete_url(
        api_base="https://api.daoxe.com/v1",
        api_key="test-key",
        model="account-model",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.daoxe.com/v1/chat/completions"


def test_daoxe_responses_url():
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="daoxe",
        model="daoxe/account-model",
    )
    assert config is not None
    url = config.get_complete_url(api_base=None, litellm_params={})
    assert url == "https://daoxe.com/v1/responses"


def test_daoxe_env_base_overrides_default(monkeypatch):
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    monkeypatch.setenv("DAOXE_API_BASE", "https://api.daoxe.com/v1")
    provider = JSONProviderRegistry.get("daoxe")
    assert provider is not None
    config = create_config_class(provider)()

    resolved_base, _resolved_key = config._get_openai_compatible_provider_info(
        None, None
    )
    assert resolved_base == "https://api.daoxe.com/v1"


def test_daoxe_responses_config_url_construction():
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="daoxe",
        model="daoxe/account-model",
    )
    assert config is not None
    url = config.get_complete_url(api_base=None, litellm_params={})
    assert url == "https://daoxe.com/v1/responses"
