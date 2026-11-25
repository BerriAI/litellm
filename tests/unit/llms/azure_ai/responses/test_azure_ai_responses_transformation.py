import asyncio
import json
import queue
from collections.abc import Awaitable, Callable
from types import NoneType, SimpleNamespace
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx
from pydantic import BaseModel

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.azure_ai.common_utils import AZURE_AI_AGENTS_SCOPE, AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY
from litellm.llms.azure_ai.responses.transformation import AzureAIResponsesAPIConfig
from litellm.responses.main import _will_bridge_to_chat_completions, provider_stores_the_conversation
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.utils import ProviderConfigManager

FOUNDRY_PROJECT_BASE = "https://res.services.ai.azure.com/api/projects/proj"
FOUNDRY_RESPONSES_URL = f"{FOUNDRY_PROJECT_BASE}/openai/v1/responses"
GATEWAY_PROJECT_BASE = "https://apim.example.com/foundry/api/projects/proj"
GATEWAY_RESPONSES_URL = f"{GATEWAY_PROJECT_BASE}/openai/v1/responses"
SERVERLESS_BASE = "https://endpoint.eastus.models.ai.azure.com"
RESOURCE_V1_BASE = "https://res.services.ai.azure.com/openai/v1"
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
    "api_base",
    [FOUNDRY_PROJECT_BASE, GATEWAY_PROJECT_BASE, "https://res.services.ai.azure.com", "https://res.openai.azure.com"],
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


@pytest.mark.parametrize("model", ["claude-3-5-sonnet", "agents/my-agent"])
def test_non_openai_surfaces_keep_chat_bridge(model):
    config = ProviderConfigManager.get_provider_responses_api_config(
        provider="azure_ai", model=model, api_base=FOUNDRY_PROJECT_BASE
    )
    assert config is None


@pytest.mark.parametrize(
    "api_base,expected_config",
    [
        (FOUNDRY_PROJECT_BASE, AzureAIResponsesAPIConfig),
        (f"{FOUNDRY_PROJECT_BASE}/openai/v1", AzureAIResponsesAPIConfig),
        (RESOURCE_V1_BASE, AzureAIResponsesAPIConfig),
        ("https://res.openai.azure.com/openai/v1", NoneType),
        ("https://res.openai.azure.com/openai/v1/responses", NoneType),
        (f"{RESOURCE_V1_BASE}/responses/compact", AzureAIResponsesAPIConfig),
        (f"{RESOURCE_V1_BASE}/chat/completions", AzureAIResponsesAPIConfig),
        ("https://res.cognitiveservices.azure.com/openai/v1", NoneType),
        ("https://res.cognitiveservices.azure.com/api/projects/proj/openai/v1", NoneType),
        ("https://res.openai.azure.com/api/projects/proj/openai/v1", NoneType),
        ("https://res.services.ai.azure.com/models", NoneType),
        ("https://res.services.ai.azure.com", NoneType),
        ("https://res.openai.azure.com/openai/deployments/model-router", NoneType),
        (SERVERLESS_BASE, NoneType),
    ],
)
@pytest.mark.parametrize("model", ["model_router/model-router", "model-router"])
def test_model_router_calls_responses_natively_on_a_foundry_project_or_resource(model, api_base, expected_config):
    config: Final = ProviderConfigManager.get_provider_responses_api_config(
        provider="azure_ai", model=model, api_base=api_base
    )
    assert type(config) is expected_config


@pytest.mark.parametrize(
    "api_base,bridged", [(FOUNDRY_PROJECT_BASE, False), (GATEWAY_PROJECT_BASE, False), (SERVERLESS_BASE, True)]
)
def test_will_bridge_to_chat_completions_follows_host(api_base, bridged):
    assert _will_bridge_to_chat_completions("gpt-5.6-luna", "azure_ai", False, None, api_base) is bridged


@pytest.mark.parametrize(
    "api_base,expected",
    [
        (FOUNDRY_PROJECT_BASE, FOUNDRY_RESPONSES_URL),
        (f"{FOUNDRY_PROJECT_BASE}/", FOUNDRY_RESPONSES_URL),
        (f"{FOUNDRY_PROJECT_BASE}/openai/v1", FOUNDRY_RESPONSES_URL),
        (f"{FOUNDRY_PROJECT_BASE}/models", FOUNDRY_RESPONSES_URL),
        (f"{FOUNDRY_PROJECT_BASE}/v1", FOUNDRY_RESPONSES_URL),
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


@pytest.mark.parametrize(
    "api_base,expected_auth",
    [
        ("https://res.services.ai.azure.com", {"api-key": "secret"}),
        (FOUNDRY_PROJECT_BASE, {"Authorization": "Bearer secret"}),
    ],
)
def test_validate_environment_sends_the_api_key_the_way_the_endpoint_takes_it(api_base, expected_auth):
    headers = AzureAIResponsesAPIConfig().validate_environment(
        headers={"x-custom": "1"},
        model="gpt-5.6-luna",
        litellm_params=GenericLiteLLMParams(api_key="secret", api_base=api_base),
    )
    assert headers == {"x-custom": "1", **expected_auth, "Content-Type": "application/json"}


def test_validate_environment_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("AZURE_AI_API_KEY", "env-secret")
    headers = AzureAIResponsesAPIConfig().validate_environment(
        headers={}, model="gpt-5.6-luna", litellm_params=GenericLiteLLMParams(api_base="https://res.services.ai.azure.com")
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


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "configured_scope,env_scope,expected_scope",
    [
        (None, None, AZURE_AI_AGENTS_SCOPE),
        ("api://custom-app/.default", None, "api://custom-app/.default"),
        (None, "https://ai.azure.us/.default", "https://ai.azure.us/.default"),
        ("api://custom-app/.default", "https://ai.azure.us/.default", "api://custom-app/.default"),
    ],
)
async def test_aresponses_mints_the_service_principal_token_for_the_project_scope(
    monkeypatch, configured_scope, env_scope, expected_scope
):
    if env_scope is None:
        monkeypatch.delenv("AZURE_SCOPE", raising=False)
    else:
        monkeypatch.setenv("AZURE_SCOPE", env_scope)
    route = respx.post(url__regex=r".*/api/projects/proj/openai/v1/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("gpt-4.1"))
    )

    with patch("litellm.llms.azure.common_utils.get_azure_ad_token_from_entra_id") as mock_entra_id:  # test-quality-ok: stubs the Entra token fetch to assert the scope a project entry requests and the bearer it sends
        mock_entra_id.return_value = lambda: "project-sp-token"
        await litellm.aresponses(
            model="azure_ai/gpt-4.1",
            input="What is 2+2?",
            api_base=FOUNDRY_PROJECT_BASE,
            tenant_id="tenant",
            client_id="client",
            client_secret="secret",
            azure_scope=configured_scope,
        )

    mock_entra_id.assert_called_once_with(
        tenant_id="tenant", client_id="client", client_secret="secret", scope=expected_scope
    )
    assert route.calls.last.request.headers["Authorization"] == "Bearer project-sp-token"


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
    ("azure_ai/gpt-5.6-sol", GATEWAY_PROJECT_BASE, GATEWAY_RESPONSES_URL, "gpt-5.6-sol"),
    ("azure_ai/gpt-5.6-sol", f"{FOUNDRY_PROJECT_BASE}/models", FOUNDRY_RESPONSES_URL, "gpt-5.6-sol"),
]


def _assert_native_responses_request(route, expected_url, expected_model):
    request = route.calls.last.request
    body = json.loads(request.content)
    assert f"{request.url.scheme}://{request.url.host}{request.url.path}" == expected_url
    on_project: Final = expected_url == FOUNDRY_RESPONSES_URL
    assert request.headers.get("authorization" if on_project else "api-key") == (
        "Bearer fake-key" if on_project else "fake-key"
    )
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

    response: Final = await litellm.aresponses(
        model=model,
        input="What is the weather in SF?",
        reasoning_effort="high",
        tools=[WEATHER_TOOL],
        api_base=api_base,
        api_key="fake-key",
    )

    _assert_native_responses_request(route, expected_url, expected_model)
    assert AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY not in response._hidden_params


@pytest.mark.asyncio
@respx.mock
async def test_aresponses_catalog_name_remapped_to_azure_sends_bare_deployment_name(monkeypatch):
    monkeypatch.setenv("AZURE_AI_API_BASE", "https://res.openai.azure.com")
    route = respx.post(url__regex=r".*/openai/v1/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("gpt-5.4-nano"))
    )

    await litellm.aresponses(
        model="azure_ai/gpt-5.4-nano",
        input="What is the weather in SF?",
        api_base="https://res.openai.azure.com",
        api_key="fake-key",
    )

    assert json.loads(route.calls.last.request.content)["model"] == "gpt-5.4-nano"


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
@pytest.mark.parametrize(
    "api_base,api_version,expected_url",
    [
        (GATEWAY_PROJECT_BASE, None, GATEWAY_RESPONSES_URL),
        (
            "https://res.cognitiveservices.azure.com/openai/v1",
            "preview",
            "https://res.cognitiveservices.azure.com/openai/v1/responses",
        ),
    ],
)
async def test_aresponses_keeps_previous_response_id_on_gateway_and_cognitiveservices_bases(api_base, api_version, expected_url):
    route = respx.post(url__regex=r".*/openai/v1/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("gpt-4.1"))
    )

    await litellm.aresponses(
        model="azure_ai/gpt-4.1",
        input="What was the codeword?",
        previous_response_id="resp_earlier",
        api_base=api_base,
        api_version=api_version,
        api_key="fake-key",
    )

    request = route.calls.last.request
    assert f"{request.url.scheme}://{request.url.host}{request.url.path}" == expected_url
    assert json.loads(request.content)["previous_response_id"] == "resp_earlier"


@pytest.mark.asyncio
@respx.mock
async def test_aresponses_on_gateway_project_base_keeps_its_query_without_api_version():
    route = respx.post(url__regex=r".*/openai/v1/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("gpt-4.1"))
    )

    await litellm.aresponses(
        model="azure_ai/gpt-4.1",
        input="What is 2+2?",
        api_base=f"{GATEWAY_PROJECT_BASE}?subscription-key=gateway-key&api-version=preview",
        api_key="fake-key",
    )

    assert str(route.calls.last.request.url) == f"{GATEWAY_RESPONSES_URL}?subscription-key=gateway-key"


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "api_base,expected_url,auth_header,auth_value",
    [
        (f"{FOUNDRY_PROJECT_BASE}/openai/v1", FOUNDRY_RESPONSES_URL, "authorization", "Bearer fake-key"),
        (RESOURCE_V1_BASE, f"{RESOURCE_V1_BASE}/responses", "api-key", "fake-key"),
    ],
)
async def test_aresponses_sends_a_model_router_to_the_foundry_responses_api_by_deployment_name(
    api_base, expected_url, auth_header, auth_value
):
    route: Final = respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("FW-GLM-5.3-Flash"))
    )

    response: Final = await litellm.aresponses(
        model="azure_ai/model_router/model-router",
        input="What is the weather in SF?",
        api_base=api_base,
        api_key="fake-key",
    )

    request: Final = route.calls.last.request
    assert str(request.url) == expected_url
    assert request.headers[auth_header] == auth_value
    assert json.loads(request.content) == {"model": "model-router", "input": "What is the weather in SF?"}
    assert response.model == "FW-GLM-5.3-Flash"
    assert response._hidden_params[AZURE_MODEL_ROUTER_SELECTED_MODEL_KEY] == "azure_ai/FW-GLM-5.3-Flash"


@pytest.mark.asyncio
@respx.mock
async def test_acompact_responses_sends_a_model_router_on_a_project_by_deployment_name():
    route: Final = respx.post(url__regex=r".*/responses/compact(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("FW-GLM-5.3-Flash"))
    )

    await litellm.acompact_responses(
        model="azure_ai/model_router/model-router",
        input="What is the weather in SF?",
        api_base=f"{FOUNDRY_PROJECT_BASE}/openai/v1",
        api_key="fake-key",
    )

    request: Final = route.calls.last.request
    assert str(request.url) == f"{FOUNDRY_RESPONSES_URL}/compact"
    assert json.loads(request.content)["model"] == "model-router"


CUSTOM_TOOL: Final = {"type": "custom", "name": "apply_patch", "description": "Apply a patch"}
RESPONSES_CALLS: Final = (
    pytest.param(r".*/responses(\?.*)?$", litellm.aresponses, id="responses"),
    pytest.param(r".*/responses/compact(\?.*)?$", litellm.acompact_responses, id="compact"),
)


PATCH_GRAMMAR: Final = {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"}
ROUTER_PATCH_TOOL: Final = {
    "type": "function",
    "name": "apply_patch",
    "description": "Apply a patch\n\nFormat:\n```lark\nstart: /.+/\n```",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The apply_patch content following the specified format"}
        },
        "required": ["content"],
    },
}


async def _sent_router_request(
    call: Callable[..., Awaitable[object]], url_regex: str, **params: object
) -> dict[str, object]:
    route: Final = respx.post(url__regex=url_regex).mock(
        return_value=httpx.Response(200, json=_responses_payload("FW-GLM-5.3-Flash"))
    )
    await call(
        model="azure_ai/model_router/model-router",
        input="Fix the typo in README.md",
        api_base=FOUNDRY_PROJECT_BASE,
        api_key="fake-key",
        **params,
    )
    return json.loads(route.calls.last.request.content)


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("url_regex,call", RESPONSES_CALLS)
async def test_a_model_router_sends_custom_tools_as_function_tools(url_regex, call):
    sent: Final = await _sent_router_request(
        call, url_regex, tools=[WEATHER_TOOL, {**CUSTOM_TOOL, "format": PATCH_GRAMMAR}]
    )

    assert sent["tools"] == [WEATHER_TOOL, ROUTER_PATCH_TOOL]


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "tool_choice,sent_tool_choice",
    [
        ({"type": "custom", "name": "apply_patch"}, {"type": "function", "name": "apply_patch"}),
        (
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "custom", "name": "apply_patch"}, WEATHER_TOOL],
            },
            {
                "type": "allowed_tools",
                "mode": "required",
                "tools": [{"type": "function", "name": "apply_patch"}, WEATHER_TOOL],
            },
        ),
        ("required", "required"),
    ],
)
async def test_a_model_router_points_a_custom_tool_choice_at_its_function_tool(tool_choice, sent_tool_choice):
    sent: Final = await _sent_router_request(
        litellm.aresponses, r".*/responses(\?.*)?$", tools=[WEATHER_TOOL, CUSTOM_TOOL], tool_choice=tool_choice
    )

    assert sent["tool_choice"] == sent_tool_choice


@pytest.mark.asyncio
@respx.mock
async def test_a_model_router_sends_custom_tools_inside_a_namespace_as_function_tools():
    namespace: Final = {"type": "namespace", "name": "editor", "description": "Edit files"}
    sent: Final = await _sent_router_request(
        litellm.aresponses,
        r".*/responses(\?.*)?$",
        tools=[{**namespace, "tools": [WEATHER_TOOL, {**CUSTOM_TOOL, "format": PATCH_GRAMMAR}]}],
    )

    assert sent["tools"] == [{**namespace, "tools": [WEATHER_TOOL, ROUTER_PATCH_TOOL]}]


def _function_call(name: str, arguments: str) -> dict[str, str]:
    return {
        "type": "function_call",
        "id": f"fc_{name}",
        "call_id": f"call_{name}",
        "name": name,
        "arguments": arguments,
        "status": "completed",
    }


PATCH_TEXT: Final = "*** Begin Patch\n*** End Patch"
ROUTER_PATCH_CALL: Final = _function_call("apply_patch", json.dumps({"content": PATCH_TEXT}))
ROUTER_WEATHER_CALL: Final = _function_call("get_weather", json.dumps({"city": "SF"}))
CUSTOM_PATCH_CALL: Final = {
    "type": "custom_tool_call",
    "id": "fc_apply_patch",
    "call_id": "call_apply_patch",
    "name": "apply_patch",
    "input": PATCH_TEXT,
    "status": "completed",
}


def _dumped_item(item: BaseModel | dict[str, object]) -> dict[str, object]:
    return item if isinstance(item, dict) else item.model_dump(exclude_none=True)


def _dumped_output(response: ResponsesAPIResponse) -> list[dict[str, object]]:
    return [_dumped_item(item) for item in response.output]


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "model,tools,expected_output",
    [
        (
            "azure_ai/model_router/model-router",
            [WEATHER_TOOL, CUSTOM_TOOL],
            [CUSTOM_PATCH_CALL, ROUTER_WEATHER_CALL],
        ),
        (
            "azure_ai/model_router/model-router",
            [WEATHER_TOOL, {**WEATHER_TOOL, "name": "apply_patch"}],
            [ROUTER_PATCH_CALL, ROUTER_WEATHER_CALL],
        ),
        ("azure_ai/gpt-4.1", [WEATHER_TOOL, CUSTOM_TOOL], [ROUTER_PATCH_CALL, ROUTER_WEATHER_CALL]),
    ],
    ids=["router_custom_tool", "router_function_tool_with_that_name", "default_model"],
)
async def test_a_model_router_answers_a_custom_tool_call_as_one(model, tools, expected_output):
    respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(
            200,
            json={**_responses_payload("gpt-5-nano"), "output": [ROUTER_PATCH_CALL, ROUTER_WEATHER_CALL]},
        )
    )

    response: Final = await litellm.aresponses(
        model=model, input="Fix the typo in README.md", tools=tools, api_base=FOUNDRY_PROJECT_BASE, api_key="fake-key"
    )

    assert _dumped_output(response) == expected_output


@pytest.mark.asyncio
@respx.mock
async def test_a_model_router_answers_only_the_namespaced_call_as_custom_beside_a_function_of_that_name():
    namespaced_call: Final = {**ROUTER_PATCH_CALL, "namespace": "editor"}
    respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(
            200, json={**_responses_payload("gpt-5-nano"), "output": [namespaced_call, ROUTER_PATCH_CALL]}
        )
    )

    response: Final = await litellm.aresponses(
        model="azure_ai/model_router/model-router",
        input="Fix the typo in README.md",
        tools=[
            {**WEATHER_TOOL, "name": "apply_patch"},
            {"type": "namespace", "name": "editor", "description": "Edit files", "tools": [CUSTOM_TOOL]},
        ],
        api_base=FOUNDRY_PROJECT_BASE,
        api_key="fake-key",
    )

    assert _dumped_output(response) == [{**CUSTOM_PATCH_CALL, "namespace": "editor"}, ROUTER_PATCH_CALL]


@pytest.mark.asyncio
@respx.mock
async def test_a_streamed_model_router_finishes_a_custom_tool_call_as_one():
    in_progress: Final = {**ROUTER_PATCH_CALL, "arguments": "", "status": "in_progress"}
    completed: Final = {**_responses_payload("gpt-5-nano"), "output": [ROUTER_PATCH_CALL]}
    stream_body: Final = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in (
            {"type": "response.created", "response": {**completed, "status": "in_progress", "output": []}},
            {"type": "response.output_item.added", "output_index": 0, "item": in_progress},
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_apply_patch",
                "output_index": 0,
                "delta": ROUTER_PATCH_CALL["arguments"],
            },
            {
                "type": "response.function_call_arguments.done",
                "item_id": "fc_apply_patch",
                "output_index": 0,
                "arguments": ROUTER_PATCH_CALL["arguments"],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": ROUTER_PATCH_CALL},
            {"type": "response.completed", "response": completed},
        )
    )
    respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})
    )

    stream: Final = await litellm.aresponses(
        model="azure_ai/model_router/model-router",
        input="Fix the typo in README.md",
        tools=[CUSTOM_TOOL],
        stream=True,
        api_base=FOUNDRY_PROJECT_BASE,
        api_key="fake-key",
    )
    events: Final = {event.type: event async for event in stream}

    assert _dumped_item(events["response.output_item.added"].item)["type"] == "function_call"
    assert _dumped_item(events["response.output_item.done"].item) == CUSTOM_PATCH_CALL
    assert _dumped_output(events["response.completed"].response) == [CUSTOM_PATCH_CALL]


CUSTOM_TOOL_HISTORY: Final = [
    {"type": "message", "role": "user", "content": "Fix the typo in README.md"},
    {
        "type": "custom_tool_call",
        "id": "ctc_1",
        "call_id": "call_1",
        "name": "apply_patch",
        "input": "*** Begin Patch\n*** End Patch",
        "status": "completed",
    },
    {"type": "custom_tool_call_output", "call_id": "call_1", "output": "Done"},
]


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize("url_regex,call", RESPONSES_CALLS)
async def test_a_model_router_sends_custom_tool_history_as_function_calls(url_regex, call):
    route: Final = respx.post(url__regex=url_regex).mock(
        return_value=httpx.Response(200, json=_responses_payload("FW-GLM-5.3-Flash"))
    )

    await call(
        model="azure_ai/model_router/model-router",
        input=CUSTOM_TOOL_HISTORY,
        api_base=FOUNDRY_PROJECT_BASE,
        api_key="fake-key",
    )

    sent_call, sent_output = json.loads(route.calls.last.request.content)["input"][1:]
    assert sent_call == {
        "type": "function_call",
        "call_id": "call_1",
        "name": "apply_patch",
        "arguments": json.dumps({"content": "*** Begin Patch\n*** End Patch"}),
    }
    assert sent_output == {"type": "function_call_output", "call_id": "call_1", "output": "Done"}


@pytest.mark.asyncio
@respx.mock
async def test_a_model_router_keeps_the_namespace_of_custom_tool_history():
    route: Final = respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("FW-GLM-5.3-Flash"))
    )

    await litellm.aresponses(
        model="azure_ai/model_router/model-router",
        input=[{**item, "namespace": "editor"} if item["type"] == "custom_tool_call" else item for item in CUSTOM_TOOL_HISTORY],
        api_base=FOUNDRY_PROJECT_BASE,
        api_key="fake-key",
    )

    sent_call: Final = json.loads(route.calls.last.request.content)["input"][1]
    assert (sent_call["type"], sent_call["namespace"], sent_call["name"]) == ("function_call", "editor", "apply_patch")


class _EarlierBridgedTurn:
    async def query_raw(self, query: str, request_id: str) -> list[dict[str, object]]:
        return [
            {
                "request_id": request_id,
                "call_type": "aresponses",
                "session_id": "earlier-session",
                "proxy_server_request": {"input": "What is the first step?"},
                "response": _chat_completion_payload("FW-GLM-5.3-Flash"),
            }
        ]


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "model,earlier_response_id,bridged",
    [
        pytest.param("azure_ai/model_router/model-router", "chatcmpl-earlier", True, id="router-after-bridged-turn"),
        pytest.param("azure_ai/model_router/model-router", "resp_earlier", False, id="router-after-native-turn"),
        pytest.param("azure_ai/gpt-5.6-terra", "chatcmpl-earlier", False, id="deployment-after-bridged-turn"),
    ],
)
async def test_only_a_model_router_follow_up_to_a_bridged_turn_stays_on_the_bridge(model, earlier_response_id, bridged):
    respx.post(url__regex=r".*/chat/completions(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_chat_completion_payload("FW-GLM-5.3-Flash"))
    )
    respx.post(url__regex=r".*/openai/v1/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("FW-GLM-5.3-Flash"))
    )

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_EarlierBridgedTurn())):
        await litellm.aresponses(
            model=model,
            input="And the next step?",
            previous_response_id=ResponsesAPIRequestUtils._build_responses_api_response_id(
                custom_llm_provider="azure_ai", model_id=None, response_id=earlier_response_id
            ),
            api_base=FOUNDRY_PROJECT_BASE,
            api_key="fake-key",
        )

    assert [call.request.url.path.endswith("/chat/completions") for call in respx.calls] == [bridged]


@pytest.mark.parametrize(
    "model,earlier_response_id,stored",
    [
        pytest.param("azure_ai/model_router/model-router", "chatcmpl-earlier", False, id="router-after-bridged-turn"),
        pytest.param("azure_ai/model_router/model-router", "resp_earlier", True, id="router-after-native-turn"),
        pytest.param("azure_ai/gpt-5.6-terra", "chatcmpl-earlier", True, id="deployment-after-bridged-turn"),
        pytest.param("azure/model-router", "chatcmpl-earlier", True, id="azure-openai-router-after-bridged-turn"),
    ],
)
def test_a_follow_up_is_stored_by_the_provider_unless_it_stays_on_the_model_router_bridge(
    model, earlier_response_id, stored
):
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id=None, response_id=earlier_response_id
    )

    assert (
        provider_stores_the_conversation(
            model,
            None,
            {"api_base": FOUNDRY_PROJECT_BASE, "previous_response_id": previous_response_id},
        )
        is stored
    )


@pytest.mark.asyncio
@respx.mock
async def test_a_default_foundry_model_keeps_custom_tool_history():
    route: Final = respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("gpt-4.1"))
    )

    await litellm.aresponses(
        model="azure_ai/gpt-4.1",
        input=CUSTOM_TOOL_HISTORY,
        api_base=FOUNDRY_PROJECT_BASE,
        api_key="fake-key",
    )

    sent_types: Final = [item["type"] for item in json.loads(route.calls.last.request.content)["input"]]
    assert sent_types == ["message", "custom_tool_call", "custom_tool_call_output"]


@pytest.mark.asyncio
@respx.mock
async def test_a_default_foundry_model_keeps_custom_tools():
    route: Final = respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, json=_responses_payload("gpt-4.1"))
    )

    await litellm.aresponses(
        model="azure_ai/gpt-4.1",
        input="What is the weather in SF?",
        tools=[CUSTOM_TOOL],
        api_base=FOUNDRY_PROJECT_BASE,
        api_key="fake-key",
    )

    assert json.loads(route.calls.last.request.content)["tools"] == [CUSTOM_TOOL]


@pytest.mark.asyncio
@respx.mock
async def test_aresponses_streams_a_model_router_on_a_project_as_it_answers():
    completed: Final = _responses_payload("FW-GLM-5.3-Flash")
    deltas: Final = tuple(
        {"type": "response.output_text.delta", "item_id": "msg_1", "output_index": 0, "content_index": 0, "delta": d}
        for d in ("1", "2")
    )
    stream_body: Final = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in (
            {"type": "response.created", "response": {**completed, "status": "in_progress"}},
            *deltas,
            {"type": "response.completed", "response": completed},
        )
    )
    route: Final = respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})
    )

    stream: Final = await litellm.aresponses(
        model="azure_ai/model_router/model-router",
        input="Count to 2",
        stream=True,
        api_base=f"{FOUNDRY_PROJECT_BASE}/openai/v1",
        api_key="fake-key",
    )
    events: Final = [event async for event in stream]
    sent: Final = json.loads(route.calls.last.request.content)

    assert sent == {"model": "model-router", "input": "Count to 2", "stream": True}
    assert [event.type for event in events] == [
        "response.created",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.completed",
    ]
    assert [events[1].delta, events[2].delta] == ["1", "2"]


class _LoggedModels(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.models: queue.Queue[object] = queue.Queue()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.models.put(kwargs["standard_logging_object"]["model"])


@pytest.mark.asyncio
@respx.mock
async def test_a_streamed_model_router_answer_logs_the_routed_model(monkeypatch):
    logged: Final = _LoggedModels()
    monkeypatch.setattr(litellm, "callbacks", [logged])
    completed: Final = _responses_payload("FW-GLM-5.3-Flash")
    stream_body: Final = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
        for event in (
            {"type": "response.created", "response": {**completed, "status": "in_progress"}},
            {"type": "response.completed", "response": completed},
        )
    )
    respx.post(url__regex=r".*/responses(\?.*)?$").mock(
        return_value=httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})
    )

    stream: Final = await litellm.aresponses(
        model="azure_ai/model_router/model-router",
        input="Count to 2",
        stream=True,
        api_base=FOUNDRY_PROJECT_BASE,
        api_key="fake-key",
    )
    _ = [event async for event in stream]

    assert await asyncio.to_thread(logged.models.get, timeout=10) == "azure_ai/FW-GLM-5.3-Flash"


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


def _responses_params(**fields: object) -> ResponsesAPIOptionalRequestParams:
    return ResponsesAPIOptionalRequestParams(**fields)  # pyright: ignore[reportArgumentType]  # TypedDict kwargs from a test helper


def test_foundry_responses_effort_gate_reads_the_azure_ai_row(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
):
    """A Foundry deployment and its Azure OpenAI namesake are different products, so the Foundry
    Responses route answers from the azure_ai/ row the Foundry chat route already reads, never from
    the azure/ twin. The rows are made to disagree on low, a level neither real row flags, so a
    leak from the wrong row cannot hide behind rows that happen to agree."""
    monkeypatch.setitem(
        litellm.model_cost,
        "azure_ai/gpt-6-sol",
        {**litellm.model_cost["azure_ai/gpt-6-sol"], "supports_low_reasoning_effort": False},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "azure/gpt-6-sol",
        {**litellm.model_cost["azure/gpt-6-sol"], "supports_low_reasoning_effort": True},
    )

    dropped: Final = AzureAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=_responses_params(reasoning={"effort": "low"}),
        model="gpt-6-sol",
        drop_params=True,
    )
    assert "reasoning" not in dropped

    with pytest.raises(litellm.UnsupportedParamsError):
        AzureAIResponsesAPIConfig().map_openai_params(
            response_api_optional_params=_responses_params(reasoning={"effort": "low"}),
            model="gpt-6-sol",
            drop_params=False,
        )

    kept_by_azure_openai: Final = AzureOpenAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=_responses_params(reasoning={"effort": "low"}),
        model="gpt-6-sol",
        drop_params=True,
    )
    assert kept_by_azure_openai["reasoning"] == {"effort": "low"}


def test_foundry_responses_none_unlocks_temperature_off_the_azure_ai_row(
    local_model_cost_map: None, monkeypatch: pytest.MonkeyPatch
):
    """The sampling-param unlock that rides on none reads the same azure_ai/ row as the effort gate:
    with the azure/ twin turned off for none, the Foundry row still lets none through and, only
    then, a non-default temperature."""
    monkeypatch.setitem(
        litellm.model_cost,
        "azure_ai/gpt-6-sol",
        {**litellm.model_cost["azure_ai/gpt-6-sol"], "supports_none_reasoning_effort": True},
    )
    monkeypatch.setitem(
        litellm.model_cost,
        "azure/gpt-6-sol",
        {**litellm.model_cost["azure/gpt-6-sol"], "supports_none_reasoning_effort": False},
    )

    params: Final = AzureAIResponsesAPIConfig().map_openai_params(
        response_api_optional_params=_responses_params(temperature=0.2, reasoning={"effort": "none"}),
        model="gpt-6-sol",
        drop_params=False,
    )

    assert params["temperature"] == 0.2
    assert params["reasoning"] == {"effort": "none"}
