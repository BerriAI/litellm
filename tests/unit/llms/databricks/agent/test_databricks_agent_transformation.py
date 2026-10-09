import json
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.llms.databricks.agent.responses_output import (
    AgentResponse,
    DatabricksAgentResponsesIterator,
    output_text,
)
from litellm.llms.databricks.agent.transformation import (
    DatabricksAgentConfig,
    resolve_endpoint_url,
)
from litellm.llms.databricks.common_utils import DatabricksException
from litellm.types.utils import ModelResponse

WORKSPACE: Final = "https://adb-1.azuredatabricks.net"
INVOCATIONS_URL: Final = f"{WORKSPACE}/serving-endpoints/my-agent/invocations"
UNIFIED_URL: Final = f"{WORKSPACE}/serving-endpoints/responses"
APP_URL: Final = "https://my-app-1.azure.databricksapps.com/responses"
MESSAGE_ITEM: Final = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "pong", "annotations": []}],
}


@pytest.fixture
def httpx_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def _sse(events: list[dict[str, object]]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


@pytest.mark.parametrize(
    ("api_base", "model", "expected"),
    [
        (WORKSPACE, "my-agent", INVOCATIONS_URL),
        (f"{WORKSPACE}/", "databricks_agent/my-agent", INVOCATIONS_URL),
        (f"{WORKSPACE}/serving-endpoints", "my-agent", INVOCATIONS_URL),
        (INVOCATIONS_URL, "anything", INVOCATIONS_URL),
        (f"{APP_URL}/", "agent", APP_URL),
        (f"{UNIFIED_URL}/", "databricks_agent/my-agent", UNIFIED_URL),
        (WORKSPACE, "team/agent v2", f"{WORKSPACE}/serving-endpoints/team%2Fagent%20v2/invocations"),
    ],
)
def test_resolve_endpoint_url(api_base: str, model: str, expected: str) -> None:
    assert resolve_endpoint_url(api_base, model) == expected


def test_resolve_endpoint_url_needs_an_endpoint_name_for_a_workspace_url() -> None:
    with pytest.raises(DatabricksException, match="model must name the Databricks serving endpoint") as info:
        resolve_endpoint_url(WORKSPACE, "databricks_agent/")
    assert info.value.status_code == 400


def test_unified_responses_url_needs_an_endpoint_name_for_the_body_model() -> None:
    with pytest.raises(DatabricksException, match="model must name the Databricks serving endpoint") as info:
        resolve_endpoint_url(UNIFIED_URL, "databricks_agent/")
    assert info.value.status_code == 400


def test_get_complete_url_requires_api_base() -> None:
    with pytest.raises(DatabricksException, match="api_base is required for databricks_agent"):
        DatabricksAgentConfig().get_complete_url(
            api_base=None, api_key=None, model="my-agent", optional_params={}, litellm_params={}
        )


def test_transform_request_maps_messages_to_responses_input_with_passthrough_fields() -> None:
    body = DatabricksAgentConfig().transform_request(
        model="my-agent",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": [{"type": "text", "text": "ping"}, {"type": "text", "text": "pong?"}]},
            {"role": "assistant", "content": "pong"},
        ],
        optional_params={"custom_inputs": {"tenant": "t-1"}, "context": {"conversation_id": "c-1"}, "ignored": 1},
        litellm_params={},
        headers={},
    )
    assert body == {
        "input": [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "pingpong?"},
            {"role": "assistant", "content": "pong"},
        ],
        "custom_inputs": {"tenant": "t-1"},
        "context": {"conversation_id": "c-1"},
    }


def test_transform_request_sets_stream_only_when_streaming() -> None:
    config = DatabricksAgentConfig()
    messages = [{"role": "user", "content": "ping"}]
    assert "stream" not in config.transform_request("m", messages, {"stream": False}, {}, {})
    assert config.transform_request("m", messages, {"stream": True}, {}, {})["stream"] is True


def test_transform_request_rejects_roles_a_responses_agent_cannot_take() -> None:
    with pytest.raises(DatabricksException, match="role 'tool' is not supported") as info:
        DatabricksAgentConfig().transform_request(
            model="my-agent",
            messages=[{"role": "tool", "content": "42", "tool_call_id": "call_1"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
    assert info.value.status_code == 400


def test_validate_environment_keeps_a_minted_authorization_header_over_the_api_key() -> None:
    headers = DatabricksAgentConfig().validate_environment(
        headers={"authorization": "Bearer minted-oauth", "X-LiteLLM-User-Id": "u1"},
        model="my-agent",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="pat-1",
    )
    assert headers["authorization"] == "Bearer minted-oauth"
    assert "Authorization" not in headers
    assert headers["X-LiteLLM-User-Id"] == "u1"
    assert headers["Content-Type"] == "application/json"


def test_validate_environment_uses_the_pat_as_bearer() -> None:
    headers = DatabricksAgentConfig().validate_environment(
        headers={}, model="my-agent", messages=[], optional_params={}, litellm_params={}, api_key="pat-1"
    )
    assert headers == {"Content-Type": "application/json", "Authorization": "Bearer pat-1"}


def test_validate_environment_without_any_credential_is_a_400() -> None:
    with pytest.raises(DatabricksException, match="Missing Databricks credentials") as info:
        DatabricksAgentConfig().validate_environment(
            headers={}, model="my-agent", messages=[], optional_params={}, litellm_params={}, api_key=None
        )
    assert info.value.status_code == 400


def test_output_text_joins_message_items_and_skips_tool_trace_items() -> None:
    output = [
        {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "secret trace"},
        {"type": "message", "content": [{"type": "output_text", "text": "Hello"}, {"type": "refusal", "refusal": "no"}]},
        {"type": "message", "content": [{"type": "output_text", "text": ", world"}]},
    ]
    assert output_text(AgentResponse.model_validate({"output": output}).output) == "Hello, world"
    assert output_text(None) == ""


def test_transform_response_reads_usage_and_custom_outputs() -> None:
    raw = httpx.Response(
        200,
        json={
            "object": "response",
            "output": [MESSAGE_ITEM],
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "custom_outputs": {"trace_id": "tr-1"},
        },
        request=httpx.Request("POST", INVOCATIONS_URL),
    )
    logging_obj = MagicMock()
    response = DatabricksAgentConfig().transform_response(
        model="my-agent",
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=logging_obj,
        request_data={},
        messages=[{"role": "user", "content": "ping"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert logging_obj.post_call.call_args.kwargs["original_response"] == raw.text
    assert response.choices[0].message.content == "pong"
    assert response.choices[0].message.provider_specific_fields == {"custom_outputs": {"trace_id": "tr-1"}}
    assert (response.usage.prompt_tokens, response.usage.completion_tokens, response.usage.total_tokens) == (7, 3, 10)


def test_transform_response_estimates_usage_when_the_agent_reports_none() -> None:
    raw = httpx.Response(
        200, json={"object": "response", "output": [MESSAGE_ITEM]}, request=httpx.Request("POST", INVOCATIONS_URL)
    )
    response = DatabricksAgentConfig().transform_response(
        model="my-agent",
        raw_response=raw,
        model_response=ModelResponse(),
        logging_obj=MagicMock(),
        request_data={},
        messages=[{"role": "user", "content": "ping"}],
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.choices[0].message.provider_specific_fields is None
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


@pytest.mark.parametrize("body", ["<html>login</html>", "[]", '{"output": "pong"}'])
def test_transform_response_rejects_a_body_that_is_not_a_responses_agent_response(body: str) -> None:
    raw = httpx.Response(200, text=body, request=httpx.Request("POST", APP_URL))
    with pytest.raises(DatabricksException, match="not a ResponsesAgent response"):
        DatabricksAgentConfig().transform_response(
            model="my-agent",
            raw_response=raw,
            model_response=ModelResponse(),
            logging_obj=MagicMock(),
            request_data={},
            messages=[],
            optional_params={},
            litellm_params={},
            encoding=None,
        )


def test_stream_iterator_emits_deltas_once_and_whole_items_only_when_nothing_streamed() -> None:
    iterator = DatabricksAgentResponsesIterator(streaming_response=iter(()), sync_stream=True)
    chunks = [
        iterator.chunk_parser({"type": "response.output_text.delta", "item_id": "msg_1", "delta": "po"}),
        iterator.chunk_parser({"type": "response.output_text.delta", "item_id": "msg_1", "delta": "ng"}),
        iterator.chunk_parser({"type": "response.output_item.done", "item": MESSAGE_ITEM}),
        iterator.chunk_parser({"type": "response.output_item.done", "item": {**MESSAGE_ITEM, "id": "msg_2"}}),
        iterator.chunk_parser({"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc_1"}}),
    ]
    assert [chunk["text"] for chunk in chunks] == ["po", "ng", "", "pong", ""]
    assert all(chunk["is_finished"] is False for chunk in chunks)


def test_stream_iterator_surfaces_agent_errors() -> None:
    iterator = DatabricksAgentResponsesIterator(streaming_response=iter(()), sync_stream=True)
    with pytest.raises(DatabricksException, match="tool failed"):
        iterator.chunk_parser({"type": "error", "message": "tool failed"})


@respx.mock
async def test_acompletion_posts_responses_input_to_model_serving(httpx_transport: None) -> None:
    route = respx.post(INVOCATIONS_URL).mock(
        return_value=httpx.Response(200, json={"object": "response", "output": [MESSAGE_ITEM]})
    )
    response = await litellm.acompletion(
        model="databricks_agent/my-agent",
        messages=[{"role": "user", "content": "ping"}],
        api_base=WORKSPACE,
        api_key="pat-1",
        custom_inputs={"tenant": "t-1"},
    )
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer pat-1"
    assert json.loads(sent.content) == {"input": [{"role": "user", "content": "ping"}], "custom_inputs": {"tenant": "t-1"}}
    assert response.choices[0].message.content == "pong"
    assert response.model == "my-agent"


@respx.mock
async def test_acompletion_names_the_endpoint_in_the_body_for_the_unified_responses_url(
    httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABRICKS_API_BASE", UNIFIED_URL)
    monkeypatch.setenv("DATABRICKS_API_KEY", "pat-env")
    route = respx.post(UNIFIED_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    response = await litellm.acompletion(
        model="databricks_agent/my-agent",
        messages=[{"role": "user", "content": "ping"}],
    )
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer pat-env"
    assert json.loads(sent.content) == {"model": "my-agent", "input": [{"role": "user", "content": "ping"}]}
    assert response.choices[0].message.content == "pong"


@respx.mock
async def test_acompletion_forwards_extra_headers_and_keeps_the_minted_token(httpx_transport: None) -> None:
    route = respx.post(APP_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    await litellm.acompletion(
        model="databricks_agent/agent",
        messages=[{"role": "user", "content": "ping"}],
        api_base=APP_URL,
        extra_headers={"Authorization": "Bearer minted-oauth", "X-LiteLLM-Trace-Id": "tr-1"},
    )
    sent = route.calls.last.request
    assert sent.headers["Authorization"] == "Bearer minted-oauth"
    assert sent.headers["X-LiteLLM-Trace-Id"] == "tr-1"


@respx.mock
async def test_acompletion_streams_output_text_deltas(httpx_transport: None) -> None:
    events: list[dict[str, object]] = [
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "po"},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "ng"},
        {"type": "response.output_item.done", "item": MESSAGE_ITEM},
    ]
    route = respx.post(INVOCATIONS_URL).mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, text=_sse(events))
    )
    stream = await litellm.acompletion(
        model="databricks_agent/my-agent",
        messages=[{"role": "user", "content": "ping"}],
        api_base=WORKSPACE,
        api_key="pat-1",
        stream=True,
    )
    deltas = [chunk.choices[0].delta.content async for chunk in stream if chunk.choices[0].delta.content]
    assert json.loads(route.calls.last.request.content)["stream"] is True
    assert "".join(deltas) == "pong"


async def test_unsupported_openai_params_follow_drop_params(httpx_transport: None) -> None:
    messages = [{"role": "user", "content": "ping"}]
    with pytest.raises(litellm.UnsupportedParamsError):
        await litellm.acompletion(
            model="databricks_agent/my-agent", messages=messages, api_base=WORKSPACE, api_key="pat-1", temperature=0.2
        )
    with respx.mock:
        route = respx.post(INVOCATIONS_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
        await litellm.acompletion(
            model="databricks_agent/my-agent",
            messages=messages,
            api_base=WORKSPACE,
            api_key="pat-1",
            temperature=0.2,
            drop_params=True,
        )
    assert "temperature" not in json.loads(route.calls.last.request.content)
