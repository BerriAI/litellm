import litellm
from litellm.llms.zai.responses.transformation import ZAIResponsesAPIConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_zai_provider_uses_responses_api_config():
    config = ProviderConfigManager.get_provider_responses_api_config(
        model="glm-5.3",
        provider=litellm.LlmProviders.ZAI,
    )

    assert isinstance(config, ZAIResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.ZAI


def test_zai_responses_url_defaults_to_responses_endpoint():
    config = ZAIResponsesAPIConfig()

    url_cases = {
        None: "https://api.z.ai/api/v1/responses",
        "https://api.z.ai/api/v1": "https://api.z.ai/api/v1/responses",
        "https://api.z.ai/api/v1/": "https://api.z.ai/api/v1/responses",
        "https://api.z.ai/api/v1/responses": "https://api.z.ai/api/v1/responses",
    }

    for api_base, expected_url in url_cases.items():
        assert config.get_complete_url(api_base=api_base, litellm_params={}) == expected_url


def test_zai_responses_url_ignores_chat_completions_api_base():
    config = ZAIResponsesAPIConfig()

    chat_bases = (
        "https://api.z.ai/api/paas/v4",
        "https://api.z.ai/api/paas/v4/",
        "https://api.z.ai/api/coding/paas/v4",
    )

    for chat_base in chat_bases:
        assert config.get_complete_url(api_base=chat_base, litellm_params={}) == "https://api.z.ai/api/v1/responses"


def test_zai_responses_url_keeps_custom_api_base():
    config = ZAIResponsesAPIConfig()

    assert (
        config.get_complete_url(
            api_base="https://gateway.example.com/openai/v1",
            litellm_params={},
        )
        == "https://gateway.example.com/openai/v1/responses"
    )


def test_zai_responses_headers_use_bearer_token():
    config = ZAIResponsesAPIConfig()
    litellm_params = GenericLiteLLMParams(api_key="sk-zai")

    headers = config.validate_environment(
        headers={},
        model="glm-5.3",
        litellm_params=litellm_params,
    )

    assert headers["Authorization"] == "Bearer sk-zai"
    assert headers["Content-Type"] == "application/json"


def test_zai_responses_headers_fall_back_to_environment_key(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-zai-env")
    config = ZAIResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="glm-5.3",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers["Authorization"] == "Bearer sk-zai-env"


def test_zai_responses_headers_prefer_zai_key_over_global_key(monkeypatch):
    monkeypatch.setenv("ZAI_API_KEY", "sk-zai-env")
    monkeypatch.setattr(litellm, "api_key", "sk-global-other-provider", raising=False)

    config = ZAIResponsesAPIConfig()

    headers = config.validate_environment(
        headers={},
        model="glm-5.3",
        litellm_params=GenericLiteLLMParams(),
    )

    assert headers["Authorization"] == "Bearer sk-zai-env"
