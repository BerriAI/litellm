import json

import httpx
import pytest
import respx

import litellm
from litellm.llms.azure_ai.responses.transformation import AzureAIResponsesAPIConfig
from litellm.responses.main import _will_bridge_to_chat_completions
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

FOUNDRY_PROJECT_BASE = "https://res.services.ai.azure.com/api/projects/proj"
FOUNDRY_RESPONSES_URL = f"{FOUNDRY_PROJECT_BASE}/openai/v1/responses"
SERVERLESS_BASE = "https://endpoint.eastus.models.ai.azure.com"
WEATHER_TOOL = {
    "type": "function",
    "name": "get_weather",
    "description": "Get weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


@pytest.fixture(autouse=True)
def clear_azure_ai_env(monkeypatch):
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "enable_azure_ad_token_refresh", False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    for env_var in (
        "AZURE_AI_API_BASE",
        "AZURE_AI_API_KEY",
        "AZURE_AD_TOKEN",
        "AZURE_TENANT_ID",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
    ):
        monkeypatch.delenv(env_var, raising=False)


def _responses_payload(model: str) -> dict:
    return {
        "id": "resp_123",
        "object": "response",
        "created_at": 1741369938,
        "status": "completed",
        "model": model,
        "output": [],
        "parallel_tool_calls": False,
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "error": None,
        "tool_choice": "auto",
        "tools": [],
        "metadata": None,
        "temperature": None,
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "truncation": None,
        "instructions": None,
        "incomplete_details": None,
        "user": None,
    }


def _chat_completion_payload(model: str) -> dict:
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1741369938,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.mark.parametrize("model", ["gpt-5.6-luna-20260710154139", "gpt-5.5-20260504143601", "DeepSeek-R1-0528", None])
@pytest.mark.parametrize(
    "api_base", [FOUNDRY_PROJECT_BASE, "https://res.services.ai.azure.com", "https://res.openai.azure.com"]
)
def test_azure_openai_v1_hosts_resolve_native_config(model, api_base):
    config = ProviderConfigManager.get_provider_responses_api_config(provider="azure_ai", model=model, api_base=api_base)
    assert isinstance(config, AzureAIResponsesAPIConfig)


def test_api_base_from_env_resolves_native_config(monkeypatch):
    monkeypatch.setenv("AZURE_AI_API_BASE", FOUNDRY_PROJECT_BASE)
    config = ProviderConfigManager.get_provider_responses_api_config(provider="azure_ai", model="gpt-5.6-luna", api_base=None)
    assert isinstance(config, AzureAIResponsesAPIConfig)


@pytest.mark.parametrize("model", ["gpt-5.6-luna", None])
@pytest.mark.parametrize(
    "api_base",
    [SERVERLESS_BASE, "https://endpoint.eastus.inference.ml.azure.com/score", "https://res.cognitiveservices.azure.com"],
)
def test_other_hosts_keep_chat_bridge(model, api_base):
    config = ProviderConfigManager.get_provider_responses_api_config(provider="azure_ai", model=model, api_base=api_base)
    assert config is None


@pytest.mark.parametrize("model", ["claude-3-5-sonnet", "model_router/gpt-5", "agents/my-agent"])
def test_non_openai_surfaces_keep_chat_bridge(model):
    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="azure_ai", model=model, api_base=FOUNDRY_PROJECT_BASE
    )
    assert config is None


@pytest.mark.parametrize("api_base,bridged", [(FOUNDRY_PROJECT_BASE, False), (SERVERLESS_BASE, True)])
def test_will_bridge_to_chat_completions_follows_host(api_base, bridged):
    assert _will_bridge_to_chat_completions("gpt-5.6-luna", "azure_ai", False, None, api_base) is bridged


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (FOUNDRY_PROJECT_BASE, FOUNDRY_RESPONSES_URL),
        (f"{FOUNDRY_PROJECT_BASE}/", FOUNDRY_RESPONSES_URL),
        (f"{FOUNDRY_PROJECT_BASE}/openai/v1", FOUNDRY_RESPONSES_URL),
        (FOUNDRY_RESPONSES_URL, FOUNDRY_RESPONSES_URL),
        ("https://res.services.ai.azure.com", "https://res.services.ai.azure.com/openai/v1/responses"),
        ("https://res.services.ai.azure.com/models", "https://res.services.ai.azure.com/openai/v1/responses"),
        (
            "https://res.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview",
            "https://res.services.ai.azure.com/openai/v1/responses",
        ),
        ("https://res.openai.azure.com", "https://res.openai.azure.com/openai/v1/responses"),
        (
            "https://res.openai.azure.com/openai/deployments/gpt-5?api-version=2025-04-01-preview",
            "https://res.openai.azure.com/openai/v1/responses",
        ),
    ],
)
def test_get_complete_url(api_base, expected):
    assert AzureAIResponsesAPIConfig().get_complete_url(api_base=api_base, litellm_params={}) == expected


def test_get_complete_url_ignores_api_version():
    url = AzureAIResponsesAPIConfig().get_complete_url(
        api_base=FOUNDRY_PROJECT_BASE, litellm_params={"api_version": "2025-04-01-preview"}
    )
    assert url == FOUNDRY_RESPONSES_URL


def test_get_complete_url_uses_env_api_base(monkeypatch):
    monkeypatch.setenv("AZURE_AI_API_BASE", FOUNDRY_PROJECT_BASE)
    assert AzureAIResponsesAPIConfig().get_complete_url(api_base=None, litellm_params={}) == FOUNDRY_RESPONSES_URL


def test_get_complete_url_raises_without_api_base():
    with pytest.raises(ValueError, match="AZURE_AI_API_BASE"):
        AzureAIResponsesAPIConfig().get_complete_url(api_base=None, litellm_params={})


def test_native_websocket_stays_off():
    assert AzureAIResponsesAPIConfig().supports_native_websocket() is False


def test_validate_environment_sends_api_key_header():
    headers = AzureAIResponsesAPIConfig().validate_environment(
        headers={"x-custom": "1"},
        model="gpt-5.6-luna",
        litellm_params=GenericLiteLLMParams(api_key="secret", api_base=FOUNDRY_PROJECT_BASE),
    )
    assert headers == {"x-custom": "1", "api-key": "secret", "Content-Type": "application/json"}


def test_validate_environment_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_AI_API_KEY", "env-secret")
    headers = AzureAIResponsesAPIConfig().validate_environment(
        headers={}, model="gpt-5.6-luna", litellm_params=GenericLiteLLMParams(api_base=FOUNDRY_PROJECT_BASE)
    )
    assert headers["api-key"] == "env-secret"


def test_validate_environment_uses_entra_token_without_api_key():
    headers = AzureAIResponsesAPIConfig().validate_environment(
        headers={},
        model="gpt-5.6-luna",
        litellm_params=GenericLiteLLMParams(azure_ad_token="entra-token", api_base=FOUNDRY_PROJECT_BASE),
    )
    assert headers["Authorization"] == "Bearer entra-token"
    assert "api-key" not in headers


def test_validate_environment_raises_without_credentials():
    with pytest.raises(ValueError, match="AZURE_AI_API_KEY"):
        AzureAIResponsesAPIConfig().validate_environment(
            headers={}, model="gpt-5.6-luna", litellm_params=GenericLiteLLMParams(api_base=FOUNDRY_PROJECT_BASE)
        )


NATIVE_RESPONSES_CASES = [
    ("azure_ai/gpt-5.6-luna-20260710154139", FOUNDRY_PROJECT_BASE, FOUNDRY_RESPONSES_URL, "gpt-5.6-luna-20260710154139"),
    (
        "azure_ai/gpt-5.6-luna",
        "https://res.services.ai.azure.com/models",
        "https://res.services.ai.azure.com/openai/v1/responses",
        "gpt-5.6-luna",
    ),
    (
        "azure_ai/gpt-5.6-sol",
        "https://res.services.ai.azure.com",
        "https://res.services.ai.azure.com/openai/v1/responses",
        "gpt-5.6-sol",
    ),
    (
        "azure_ai/gpt-5.6-luna-20260710154139",
        "https://res.openai.azure.com",
        "https://res.openai.azure.com/openai/v1/responses",
        "gpt-5.6-luna-20260710154139",
    ),
    (
        "azure_ai/gpt-5.6-sol",
        "https://res.openai.azure.com",
        "https://res.openai.azure.com/openai/v1/responses",
        "gpt-5.6-sol",
    ),
]


def _assert_native_responses_request(route, expected_url, expected_model):
    request = route.calls.last.request
    body = json.loads(request.content)
    assert f"{request.url.scheme}://{request.url.host}{request.url.path}" == expected_url
    assert request.headers["api-key"] == "fake-key"
    assert body["model"] == expected_model
    assert body["input"] == "What is the weather in SF?"
    assert "messages" not in body
    assert body["reasoning"] == {"effort": "high"}
    assert body["tools"] == [WEATHER_TOOL]


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("model,api_base,expected_url,expected_model", NATIVE_RESPONSES_CASES)
async def test_aresponses_sends_reasoning_and_tools_to_native_endpoint(model, api_base, expected_url, expected_model):
    route = respx.post(url__regex=r".*/openai/v1/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload(expected_model))
    )

    await litellm.aresponses(
        model=model,
        input="What is the weather in SF?",
        reasoning_effort="high",
        tools=[WEATHER_TOOL],
        api_base=api_base,
        api_key="fake-key",
    )

    _assert_native_responses_request(route, expected_url, expected_model)


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("model,api_base,expected_url,expected_model", NATIVE_RESPONSES_CASES)
async def test_router_aresponses_sends_bare_deployment_name(model, api_base, expected_url, expected_model):
    route = respx.post(url__regex=r".*/openai/v1/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload(expected_model))
    )
    router = litellm.Router(
        model_list=[{"model_name": "gpt-5.6", "litellm_params": {"model": model, "api_base": api_base, "api_key": "fake-key"}}],
        num_retries=0,
    )

    await router.aresponses(
        model="gpt-5.6", input="What is the weather in SF?", reasoning={"effort": "high"}, tools=[WEATHER_TOOL]
    )

    _assert_native_responses_request(route, expected_url, expected_model)


@pytest.mark.asyncio
@respx.mock
async def test_aresponses_serverless_host_stays_on_chat_bridge():
    chat_route = respx.post(url__regex=r".*/chat/completions$").mock(
        return_value=httpx.Response(200, json=_chat_completion_payload("gpt-5.6-luna"))
    )
    responses_route = respx.post(url__regex=r".*/responses$")

    await litellm.aresponses(
        model="azure_ai/gpt-5.6-luna-20260710154139",
        input="What is the weather in SF?",
        tools=[WEATHER_TOOL],
        api_base=SERVERLESS_BASE,
        api_key="fake-key",
    )

    assert chat_route.called
    assert not responses_route.called
    assert chat_route.calls.last.request.headers["Authorization"] == "Bearer fake-key"
