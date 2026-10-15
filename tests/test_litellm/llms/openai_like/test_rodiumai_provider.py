import litellm


def test_rodiumai_in_provider_list():
    from litellm import LlmProviders

    assert LlmProviders.RODIUMAI.value == "rodiumai"
    assert "rodiumai" in litellm.provider_list


def test_rodiumai_json_config():
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert JSONProviderRegistry.exists("rodiumai")
    provider = JSONProviderRegistry.get("rodiumai")
    assert provider is not None
    assert provider.base_url == "https://api.rodiumai.io/v1"
    assert provider.api_key_env == "RODIUMAI_API_KEY"
    assert provider.api_base_env == "RODIUMAI_API_BASE"


def test_rodiumai_provider_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="rodiumai/openai/gpt-4o",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "openai/gpt-4o"
    assert provider == "rodiumai"
    assert api_key is None
    assert api_base == "https://api.rodiumai.io/v1"


def test_rodiumai_router_model_resolution():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider(
        model="rodiumai/rodiumai/smart",
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
    )

    assert model == "rodiumai/smart"
    assert provider == "rodiumai"
    assert api_base == "https://api.rodiumai.io/v1"


def test_rodiumai_max_completion_tokens_mapped():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("rodiumai")
    assert provider is not None
    config = create_config_class(provider)()

    optional_params = config.map_openai_params(
        non_default_params={"max_completion_tokens": 512},
        optional_params={},
        model="openai/gpt-4o",
        drop_params=False,
    )
    assert optional_params["max_tokens"] == 512
    assert "max_completion_tokens" not in optional_params


def test_rodiumai_chat_completions_url():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    provider = JSONProviderRegistry.get("rodiumai")
    assert provider is not None
    config = create_config_class(provider)()

    url = config.get_complete_url(
        api_base=None,
        api_key="rd_sk_test",
        model="openai/gpt-4o",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.rodiumai.io/v1/chat/completions"
