import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final, TypedDict, cast
from unittest.mock import MagicMock, patch
from urllib.parse import quote

import httpx
import pytest
from openai.types.responses import (
    EasyInputMessage,
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
from litellm.responses.file_search.emulated_handler import should_use_emulated_file_search
from litellm.types.llms.openai import InputTokensDetails, ResponseAPIUsage, ResponseInputParam, ResponsesAPIResponse
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


def test_responses_call_hoists_developer_items_into_one_leading_system_message() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(
            model="fireworks_ai/accounts/fireworks/models/kimi-k3",
            input=[  # mutable-ok: the Responses API takes input as a JSON list
                {"role": "user", "content": "Hi there"},
                {"role": "developer", "content": "Answer with exactly one word."},
                {"role": "user", "content": [{"type": "input_text", "text": "What is the capital of France?"}]},
            ],
            api_key="fw-test-key",
        )
    _, _, body = _sent_request(client)
    assert tuple(body["input"]) == (
        {"role": "system", "content": "Answer with exactly one word.", "type": "message"},
        {"role": "user", "content": "Hi there"},
        {"role": "user", "content": [{"type": "input_text", "text": "What is the capital of France?"}]},
    )


def test_responses_call_folds_instructions_and_developer_item_into_one_leading_system_message() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/qwen3p8-2p4t-a95b"))
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(
            model="fireworks_ai/accounts/fireworks/models/qwen3p8-2p4t-a95b",
            instructions="You are a coding agent running in the Codex CLI.",
            input=[  # mutable-ok: the Responses API takes input as a JSON list
                {
                    "role": "developer",
                    "content": [{"type": "input_text", "text": "<permissions instructions>read-only</permissions instructions>"}],
                },
                {"role": "user", "content": [{"type": "input_text", "text": "What is the capital of France?"}]},
                {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "A trivial question."}]},
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Paris is the capital of France.", "annotations": []}],
                },
                {"role": "user", "content": [{"type": "input_text", "text": "And of Spain?"}]},
            ],
            store=False,
            api_key="fw-test-key",
        )
    _, _, body = _sent_request(client)
    assert "instructions" not in body
    assert tuple(body["input"]) == (
        {
            "role": "system",
            "content": (
                "You are a coding agent running in the Codex CLI.\n\n"
                "<permissions instructions>read-only</permissions instructions>"
            ),
            "type": "message",
        },
        {"role": "user", "content": [{"type": "input_text", "text": "What is the capital of France?"}]},
        {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "A trivial question."}]},
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "Paris is the capital of France.", "annotations": []}],
        },
        {"role": "user", "content": [{"type": "input_text", "text": "And of Spain?"}]},
    )


def test_responses_call_turns_string_input_with_instructions_into_system_then_user_messages() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(
            model="fireworks_ai/accounts/fireworks/models/kimi-k3",
            instructions="Answer with exactly one word.",
            input="What is the capital of France?",
            api_key="fw-test-key",
        )
    _, _, body = _sent_request(client)
    assert "instructions" not in body
    assert tuple(body["input"]) == (
        {"role": "system", "content": "Answer with exactly one word.", "type": "message"},
        {"role": "user", "content": "What is the capital of France?"},
    )


def test_responses_call_maps_pydantic_developer_items_and_replays_pydantic_output_items() -> None:
    client: Final = _mock_http_client(_fireworks_response("accounts/fireworks/models/kimi-k3"))
    pydantic_input: Final = cast(
        ResponseInputParam,
        [  # mutable-ok: the Responses API takes input as a JSON list
            EasyInputMessage(role="developer", content="Answer with exactly one word.", type="message"),
            ResponseReasoningItem(id="rs_1", summary=(), type="reasoning"),
            ResponseFunctionToolCall(
                id="fc_1",
                call_id="call_abc123",
                name="get_weather",
                arguments='{"city": "Paris"}',
                status="completed",
                type="function_call",
            ),
            FunctionCallOutput(type="function_call_output", call_id="call_abc123", output="21C"),
        ],
    )
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        litellm.responses(
            model="fireworks_ai/accounts/fireworks/models/kimi-k3", input=pydantic_input, api_key="fw-test-key"
        )
    _, _, body = _sent_request(client)
    assert tuple(body["input"]) == (
        {"role": "system", "content": "Answer with exactly one word.", "type": "message"},
        {"id": "rs_1", "summary": [], "type": "reasoning"},
        {
            "id": "fc_1",
            "call_id": "call_abc123",
            "name": "get_weather",
            "arguments": '{"city": "Paris"}',
            "status": "completed",
            "type": "function_call",
        },
        {"type": "function_call_output", "call_id": "call_abc123", "output": "21C"},
    )


def test_file_search_tools_take_litellm_emulated_search_not_fireworks() -> None:
    config: Final = FireworksAIResponsesAPIConfig()
    file_search: Final = ({"type": "file_search", "vector_store_ids": ("vs_kb",)},)
    function_tool: Final = ({"type": "function", "name": "get_weather", "parameters": {"type": "object"}},)
    assert should_use_emulated_file_search(tools=file_search, provider_config=config)
    assert not should_use_emulated_file_search(tools=function_tool, provider_config=config)


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


def test_delete_responses_maps_fireworks_message_only_body_to_deleted_result() -> None:
    response_id: Final = (
        "resp_xFaIJR9Nc_OXmqKRqL78UuAGj2Te5GY5BT_knpZiMrYoNOVmu5oc2mQW1HI7hCtEYB4mcx2lEYS0DYP1U5yEQskHunuB4=="
    )
    request: Final = httpx.Request("DELETE", f"{FIREWORKS_RESPONSES_URL}/{quote(response_id, safe='')}")
    client: Final = MagicMock()
    client.delete.return_value = httpx.Response(200, json={"message": "Response deleted successfully"}, request=request)
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        result: Final = litellm.delete_responses(
            response_id=response_id, custom_llm_provider="fireworks_ai", api_key="fw-test-key"
        )
    assert client.delete.call_args.kwargs["url"] == str(request.url)
    assert (result.id, result.object, result.deleted) == (response_id, "response", True)


def _fireworks_stream_response(status: str, output: tuple[Mapping[str, object], ...]) -> Mapping[str, object]:
    return {
        "id": "resp_htnkJ8piNKeOHkn9LfAusC38O2OgcDQs4S8trSOJ6anLeqjUDGqu2PkWmg5N",
        "object": "response",
        "created_at": 1788567245,
        "model": "accounts/fireworks/models/kimi-k3",
        "status": status,
        "output": output,
        "usage": None
        if status == "in_progress"
        else {
            "input_tokens": 95,
            "output_tokens": 89,
            "total_tokens": 184,
            "input_tokens_details": {"cached_tokens": 94},
        },
    }


FIREWORKS_SSE_EVENTS: Final[tuple[Mapping[str, object], ...]] = (
    {"type": "response.created", "sequence_number": 0, "response": _fireworks_stream_response("in_progress", ())},
    {
        "type": "response.output_item.added",
        "sequence_number": 1,
        "output_index": 0,
        "item": {"id": "rs_1", "type": "reasoning", "summary": []},
    },
    {
        "type": "response.reasoning_summary_text.delta",
        "sequence_number": 2,
        "item_id": "rs_1",
        "output_index": 0,
        "summary_index": 0,
        "delta": "pong",
    },
    {
        "type": "response.output_item.added",
        "sequence_number": 3,
        "output_index": 1,
        "item": {"id": "msg_1", "type": "message", "role": "assistant", "status": "in_progress", "content": []},
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 4,
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "delta": "po",
    },
    {
        "type": "response.output_text.delta",
        "sequence_number": 5,
        "item_id": "msg_1",
        "output_index": 1,
        "content_index": 0,
        "delta": "ng",
    },
    {
        "type": "response.completed",
        "sequence_number": 6,
        "response": _fireworks_stream_response(
            "completed",
            (
                {"id": "rs_1", "type": "reasoning", "summary": [{"type": "summary_text", "text": "pong"}]},
                {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "pong", "annotations": []}],
                },
            ),
        ),
    },
)


def _sse_body(events: tuple[Mapping[str, object], ...]) -> bytes:
    return b"".join(f"data: {json.dumps(dict(event))}\n\n".encode() for event in events) + b"data: [DONE]\n\n"


def test_streaming_responses_call_hits_native_endpoint_and_yields_every_fireworks_event() -> None:
    request: Final = httpx.Request("POST", FIREWORKS_RESPONSES_URL)
    client: Final = MagicMock()
    client.post.return_value = httpx.Response(
        200, content=_sse_body(FIREWORKS_SSE_EVENTS), headers={"content-type": "text/event-stream"}, request=request
    )
    with patch(HTTPX_CLIENT_FACTORY, return_value=client):
        received: Final = tuple(
            litellm.responses(
                model="fireworks_ai/kimi-k3",
                input="Reply with the single word pong.",
                stream=True,
                api_key="fw-test-key",
            )
        )
    url, _, body = _sent_request(client)
    assert (url, body["model"], body["stream"], client.post.call_args.kwargs["stream"]) == (
        FIREWORKS_RESPONSES_URL,
        "accounts/fireworks/models/kimi-k3",
        True,
        True,
    )
    assert tuple(event.type for event in received) == tuple(event["type"] for event in FIREWORKS_SSE_EVENTS)
    assert "".join(event.delta for event in received if event.type == "response.output_text.delta") == "pong"
    assert received[-1].response.usage.output_tokens == 89
