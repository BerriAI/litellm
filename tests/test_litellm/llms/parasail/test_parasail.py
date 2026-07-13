import os
from unittest.mock import patch

PARASAIL_API_BASE = "https://api.parasail.io/v1"
PARASAIL_RESPONSES_GATEWAY = "https://api-webflux.saas.parasail.io/v1"


def test_parasail_json_registry():
    import litellm
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    assert litellm.LlmProviders.PARASAIL.value == "parasail"
    assert litellm.LlmProviders("parasail") == litellm.LlmProviders.PARASAIL
    assert JSONProviderRegistry.exists("parasail")
    config = JSONProviderRegistry.get("parasail")
    assert config is not None
    assert config.base_url == PARASAIL_API_BASE
    assert config.api_key_env == "PARASAIL_API_KEY"
    assert config.api_base_env == "PARASAIL_API_BASE"
    assert "/v1/chat/completions" in config.supported_endpoints
    assert "/v1/responses" in config.supported_endpoints
    assert config.special_handling.get("force_store_false") is True


def test_parasail_listed_in_openai_compatible_providers():
    from litellm.constants import openai_compatible_providers

    assert "parasail" in openai_compatible_providers


def test_parasail_dynamic_config_env_vars():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("parasail"))()

    with patch.dict(
        os.environ,
        {
            "PARASAIL_API_KEY": "test-key",
            "PARASAIL_API_BASE": PARASAIL_RESPONSES_GATEWAY,
        },
    ):
        api_base, api_key = config._get_openai_compatible_provider_info(None, None)

    assert api_base == PARASAIL_RESPONSES_GATEWAY
    assert api_key == "test-key"


def test_parasail_provider_detection_by_prefix():
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, _, api_base = get_llm_provider(
        "parasail/parasail-llama-33-70b-fp8"
    )

    assert model == "parasail-llama-33-70b-fp8"
    assert provider == "parasail"
    assert api_base == PARASAIL_API_BASE


def test_parasail_chat_complete_url():
    from litellm.llms.openai_like.dynamic_config import create_config_class
    from litellm.llms.openai_like.json_loader import JSONProviderRegistry

    config = create_config_class(JSONProviderRegistry.get("parasail"))()

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="parasail-llama-33-70b-fp8",
            optional_params={},
            litellm_params={},
        )
        == f"{PARASAIL_API_BASE}/chat/completions"
    )


def test_parasail_responses_api_config():
    from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="parasail",
        model="parasail-kimi-k25-elicit",
    )

    assert isinstance(config, OpenAIResponsesAPIConfig)
    assert config.custom_llm_provider == "parasail"
    assert (
        config.get_complete_url(api_base=None, litellm_params={})
        == f"{PARASAIL_API_BASE}/responses"
    )


def test_parasail_responses_api_honors_api_base_override():
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="parasail",
        model="parasail-kimi-k25-elicit",
    )

    with patch.dict(
        os.environ,
        {"PARASAIL_API_BASE": PARASAIL_RESPONSES_GATEWAY},
    ):
        url = config.get_complete_url(api_base=None, litellm_params={})

    assert url == f"{PARASAIL_RESPONSES_GATEWAY}/responses"


def test_parasail_responses_api_forces_store_false_when_caller_sets_true():
    from litellm.types.router import GenericLiteLLMParams
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="parasail",
        model="parasail-kimi-k25-elicit",
    )

    request_params: dict = {"store": True, "temperature": 0.2}
    transformed = config.transform_responses_api_request(
        model="parasail-kimi-k25-elicit",
        input="hello",
        response_api_optional_request_params=request_params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert transformed["store"] is False
    assert transformed["temperature"] == 0.2


def test_parasail_responses_api_forces_store_false_when_caller_omits_store():
    from litellm.types.router import GenericLiteLLMParams
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="parasail",
        model="parasail-kimi-k25-elicit",
    )

    transformed = config.transform_responses_api_request(
        model="parasail-kimi-k25-elicit",
        input="hello",
        response_api_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert transformed["store"] is False


def test_parasail_responses_api_validate_environment_sets_bearer_token():
    from litellm.types.router import GenericLiteLLMParams
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="parasail",
        model="parasail-kimi-k25-elicit",
    )

    with patch.dict(os.environ, {"PARASAIL_API_KEY": "secret-from-env"}):
        headers = config.validate_environment(
            headers={},
            model="parasail-kimi-k25-elicit",
            litellm_params=GenericLiteLLMParams(),
        )

    assert headers["Authorization"] == "Bearer secret-from-env"
