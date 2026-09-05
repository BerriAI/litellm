import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, TypedDict
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_input_param import FunctionCallOutput
from openai.types.responses.tool_param import Mcp
from typing_extensions import ReadOnly

import litellm
from litellm.llms.fireworks_ai.responses.transformation import FireworksAIResponsesAPIConfig
from litellm.types.llms.openai import InputTokensDetails, ResponseAPIUsage, ResponsesAPIResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

FIREWORKS_RESPONSES_URL: Final = "https://api.fireworks.ai/inference/v1/responses"
HTTPX_CLIENT_FACTORY: Final = "litellm.llms.custom_httpx.llm_http_handler._get_httpx_client"
NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})
NO_PARAMS: Final[Mapping[str, object]] = MappingProxyType({})


class _GeneratedSessionMetadata(TypedDict):
    litellm_session_id_generated: ReadOnly[bool]


def _fireworks_response(model: str) -> Mapping[str, object]:
    return ResponsesAPIResponse(
        id="resp_0e946f2d46bf4b49bf8b29ff78083583",
        object="response",
        created_at=1788550000,
        model=model,
        status="completed",
        output=(
            ResponseReasoningItem(id="rs_1", summary=(), type="reasoning"),
            ResponseOutputMessage(
                id="msg_1",
                status="completed",
                role="assistant",
                type="message",
                content=(ResponseOutputText(type="output_text", text="Paris is clear and 21C.", annotations=()),),
            ),
            ResponseFunctionToolCall(
                id="fc_1",
                call_id="call_abc123",
                name="get_weather",
                arguments='{"city": "Paris"}',
                status="completed",
                type="function_call",
            ),
        ),
        usage=ResponseAPIUsage(
            input_tokens=179,
            output_tokens=100,
            total_tokens=279,
            input_tokens_details=InputTokensDetails(cached_tokens=0),
        ),
    ).model_dump(mode="json", exclude_none=True)


def _mock_http_client(response_body: Mapping[str, object]) -> MagicMock:
    client: Final = MagicMock()
    response: Final = MagicMock()
    response.status_code = 200
    response.headers = httpx.Headers((("content-type", "application/json"),))
    response.json.return_value = response_body
    response.text = json.dumps(response_body)
    client.post.return_value = response
    return client


def _sent_request(client: MagicMock) -> tuple[str, Mapping[str, str], Mapping[str, object]]:
    kwargs: Final = client.post.call_args.kwargs
    body: Final = kwargs["json"] if "json" in kwargs else json.loads(kwargs["data"])
    return kwargs["url"], kwargs["headers"], body


@pytest.fixture(autouse=True)
def fireworks_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "FIREWORKS_API_KEY",
        "FIREWORKS_AI_API_KEY",
        "FIREWORKSAI_API_KEY",
        "FIREWORKS_AI_TOKEN",
        "FIREWORKS_API_BASE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_fireworks_ai_provider_config_registration() -> None:
    config: Final = ProviderConfigManager.get_provider_responses_api_config(
        model="accounts/fireworks/models/kimi-k3", provider=LlmProviders.FIREWORKS_AI
    )
    assert isinstance(config, FireworksAIResponsesAPIConfig)
    assert config.custom_llm_provider == LlmProviders.FIREWORKS_AI


def test_responses_call_hits_native_endpoint_with_mcp_tool_untouched() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    mcp_tool: Final[Mcp] = {
        "type": "mcp",
        "server_label": "deepwiki",
        "server_url": "https://mcp.deepwiki.com/mcp",
        "require_approval": "never",
    }
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        response: Final = litellm.responses(
            model="fireworks_ai/accounts/fireworks/models/kimi-k3",
            input="What is litellm?",
            tools=[mcp_tool],  # mutable-ok: the Responses API takes tools as a JSON list
            api_key="fw-test-key",
        )
    url, headers, body = _sent_request(client)
    assert url == FIREWORKS_RESPONSES_URL
    assert headers["Authorization"] == "Bearer fw-test-key"
    assert body["model"] == "accounts/fireworks/models/kimi-k3"
    assert tuple(body["tools"]) == (mcp_tool,)
    assert "messages" not in body
    assert isinstance(response, ResponsesAPIResponse)
    function_calls: Final = tuple(item for item in response.output if getattr(item, "type", None) == "function_call")
    assert getattr(function_calls[0], "call_id", None) == "call_abc123"


def test_responses_call_expands_bare_model_name_to_fireworks_resource() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/glm-5p3"))
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(model="fireworks_ai/glm-5p3", input="hi", api_key="fw-test-key")
    _, _, body = _sent_request(client)
    assert body["model"] == "accounts/fireworks/models/glm-5p3"


def test_responses_call_forwards_previous_response_id_and_store() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    tool_output: Final[FunctionCallOutput] = {
        "type": "function_call_output",
        "call_id": "call_abc123",
        "output": "{}",
    }
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(
            model="fireworks_ai/kimi-k3",
            input=[tool_output],  # mutable-ok: the Responses API takes input items as a JSON list
            previous_response_id="resp_0e946f2d46bf4b49bf8b29ff78083583",
            store=True,
            api_key="fw-test-key",
        )
    _, _, body = _sent_request(client)
    assert body["previous_response_id"] == "resp_0e946f2d46bf4b49bf8b29ff78083583"
    assert body["store"] is True
    assert body["input"][0]["call_id"] == "call_abc123"


def test_responses_call_sends_session_affinity_for_caller_session_id() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(model="fireworks_ai/kimi-k3", input="hi", api_key="fw-test-key", litellm_session_id="sess-42")
    _, headers, _ = _sent_request(client)
    assert headers["x-session-affinity"] == "sess-42"


def test_responses_call_keeps_caller_supplied_session_affinity_header() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    pinned: Final[Mapping[str, str]] = MappingProxyType({"x-session-affinity": "explicit-node"})
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(
            model="fireworks_ai/kimi-k3",
            input="hi",
            api_key="fw-test-key",
            litellm_session_id="sess-42",
            extra_headers=pinned,
        )
    _, headers, _ = _sent_request(client)
    assert headers["x-session-affinity"] == "explicit-node"


def test_responses_call_maps_provider_errors_to_fireworks_ai() -> None:
    client: Final = MagicMock()
    request: Final = httpx.Request("POST", FIREWORKS_RESPONSES_URL)
    client.post.side_effect = httpx.HTTPStatusError(
        "unauthorized",
        request=request,
        response=httpx.Response(401, text='{"error": {"message": "invalid api key"}}', request=request),
    )
    with patch(HTTPX_CLIENT_FACTORY, return_value=client), pytest.raises(litellm.AuthenticationError) as raised:
        litellm.responses(model="fireworks_ai/kimi-k3", input="hi", api_key="fw-bad-key")
    assert raised.value.llm_provider == "fireworks_ai"
    assert raised.value.status_code == 401
    assert "invalid api key" in str(raised.value)


def test_responses_call_skips_session_affinity_for_proxy_generated_session_id() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    generated: Final[_GeneratedSessionMetadata] = {"litellm_session_id_generated": True}
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(
            model="fireworks_ai/kimi-k3",
            input="hi",
            api_key="fw-test-key",
            litellm_session_id="generated-1",
            litellm_metadata=generated,
        )
    _, headers, _ = _sent_request(client)
    assert "x-session-affinity" not in headers


@pytest.mark.parametrize(
    "api_base, expected",
    (
        (None, FIREWORKS_RESPONSES_URL),
        ("https://api.fireworks.ai/inference/v1", FIREWORKS_RESPONSES_URL),
        ("https://api.fireworks.ai/inference/v1/", FIREWORKS_RESPONSES_URL),
        ("https://gateway.example.com/fireworks", "https://gateway.example.com/fireworks/responses"),
    ),
)
def test_get_complete_url(api_base: str | None, expected: str) -> None:
    assert FireworksAIResponsesAPIConfig().get_complete_url(api_base=api_base, litellm_params=NO_PARAMS) == expected


def test_responses_call_reads_fireworks_api_base_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_BASE", "https://self-hosted.example.com/v1")
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(model="fireworks_ai/kimi-k3", input="hi", api_key="fw-test-key")
    url, _, _ = _sent_request(client)
    assert url == "https://self-hosted.example.com/v1/responses"


@pytest.mark.parametrize(
    "env_name", ("FIREWORKS_API_KEY", "FIREWORKS_AI_API_KEY", "FIREWORKSAI_API_KEY", "FIREWORKS_AI_TOKEN")
)
def test_validate_environment_reads_every_fireworks_key_name(monkeypatch: pytest.MonkeyPatch, env_name: str) -> None:
    monkeypatch.setenv(env_name, "env-key")
    headers: Final = FireworksAIResponsesAPIConfig().validate_environment(
        headers=NO_HEADERS, model="accounts/fireworks/models/kimi-k3", litellm_params=GenericLiteLLMParams()
    )
    assert headers["Authorization"] == "Bearer env-key"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_prefers_explicit_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "env-key")
    headers: Final = FireworksAIResponsesAPIConfig().validate_environment(
        headers=NO_HEADERS,
        model="accounts/fireworks/models/kimi-k3",
        litellm_params=GenericLiteLLMParams(api_key="explicit"),
    )
    assert headers["Authorization"] == "Bearer explicit"


def test_validate_environment_without_any_key_raises() -> None:
    with pytest.raises(ValueError, match="FIREWORKS_API_KEY"):
        FireworksAIResponsesAPIConfig().validate_environment(
            headers=NO_HEADERS, model="accounts/fireworks/models/kimi-k3", litellm_params=None
        )
