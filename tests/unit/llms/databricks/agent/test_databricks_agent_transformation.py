import json
from typing import Final
from uuid import uuid4
from unittest.mock import MagicMock

import httpx
import pytest
import responses
import respx

import litellm
from litellm.llms.databricks.agent.responses_output import (
    AgentResponse,
    DatabricksAgentResponsesIterator,
    output_text,
)
from litellm.llms.databricks.agent.transformation import (
    DatabricksAgentConfig,
    is_databricks_app_url,
    resolve_endpoint_url,
    workspace_url_from_api_base,
)
from litellm.llms.databricks.common_utils import DatabricksException
from litellm.types.utils import ModelResponse

WORKSPACE: Final = "https://adb-1.azuredatabricks.net"
TOKEN_URL: Final = f"{WORKSPACE}/oidc/v1/token"
INVOCATIONS_URL: Final = f"{WORKSPACE}/serving-endpoints/my-agent/invocations"
UNIFIED_URL: Final = f"{WORKSPACE}/serving-endpoints/responses"
APP_HOST: Final = "https://my-app-1.azure.databricksapps.com"
APP_URL: Final = f"{APP_HOST}/responses"
MESSAGE_ITEM: Final = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "pong", "annotations": []}],
}
_TOOL_CALL: Final = {
    "id": "call_1",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "{}"},
}


@pytest.fixture
def env_oauth_m2m(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_HOST", raising=False)
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "sp-id")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", f"sp-secret-{uuid4().hex}")


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
        (f"{WORKSPACE}/serving-endpoints/my-agent", "", INVOCATIONS_URL),
        (f"{APP_URL}/", "agent", APP_URL),
        (APP_HOST, "", APP_URL),
        (f"{APP_HOST}/", "agent", APP_URL),
        (f"{APP_HOST}/api/agent", "", f"{APP_HOST}/api/agent"),
        (f"{UNIFIED_URL}/", "databricks_agent/my-agent", UNIFIED_URL),
        (WORKSPACE, "team/agent v2", f"{WORKSPACE}/serving-endpoints/team%2Fagent%20v2/invocations"),
    ],
)
def test_resolve_endpoint_url(api_base: str, model: str, expected: str) -> None:
    assert resolve_endpoint_url(api_base, model) == expected


@pytest.mark.parametrize("app_base", [APP_HOST, f"{APP_HOST}/", APP_URL, "https://other.example/responses"])
def test_an_app_url_is_never_a_workspace(app_base: str) -> None:
    assert is_databricks_app_url(app_base)
    assert workspace_url_from_api_base(app_base) is None


@pytest.mark.parametrize("api_base", [WORKSPACE, f"{WORKSPACE}/serving-endpoints", INVOCATIONS_URL, UNIFIED_URL])
def test_model_serving_urls_resolve_to_the_workspace(api_base: str) -> None:
    assert not is_databricks_app_url(api_base)
    assert workspace_url_from_api_base(api_base) == WORKSPACE


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


def test_transform_request_rejects_roles_a_responses_agent_cannot_take(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(DatabricksException, match="role 'tool' is not supported") as info:
        DatabricksAgentConfig().transform_request(
            model="my-agent",
            messages=[{"role": "tool", "content": "42", "tool_call_id": "call_1"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
    assert info.value.status_code == 400


def test_transform_request_rejects_assistant_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(DatabricksException, match="remove tool_calls from the message") as info:
        DatabricksAgentConfig().transform_request(
            model="my-agent",
            messages=[
                {"role": "user", "content": "weather?"},
                {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL]},
            ],
            optional_params={},
            litellm_params={},
            headers={},
        )
    assert info.value.status_code == 400


def test_transform_request_drops_tool_turns_and_empty_messages_under_drop_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    body = DatabricksAgentConfig().transform_request(
        model="my-agent",
        messages=[
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": None, "tool_calls": [_TOOL_CALL]},
            {"role": "tool", "content": "sunny", "tool_call_id": "call_1"},
            {"role": "assistant", "content": "Checking the weather", "tool_calls": [_TOOL_CALL]},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]},
            {"role": "user", "content": "and tomorrow?"},
        ],
        optional_params={},
        litellm_params={"drop_params": True},
        headers={},
    )
    assert body == {
        "input": [
            {"role": "user", "content": "weather?"},
            {"role": "assistant", "content": "Checking the weather"},
            {"role": "user", "content": "and tomorrow?"},
        ]
    }


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
        {"type": "file", "file": {"file_id": "file-1"}},
    ],
)
def test_transform_request_rejects_content_a_responses_agent_cannot_take(
    part: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", False)
    with pytest.raises(DatabricksException, match=str(part["type"])) as info:
        DatabricksAgentConfig().transform_request(
            model="my-agent",
            messages=[{"role": "user", "content": [{"type": "text", "text": "describe this"}, part]}],
            optional_params={},
            litellm_params={},
            headers={},
        )
    assert info.value.status_code == 400


@pytest.mark.parametrize(
    ("global_drop_params", "litellm_params"),
    [(True, {}), (False, {"drop_params": True})],
)
def test_transform_request_drops_non_text_content_under_drop_params(
    global_drop_params: bool, litellm_params: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "drop_params", global_drop_params)
    body = DatabricksAgentConfig().transform_request(
        model="my-agent",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe this"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
                ],
            }
        ],
        optional_params={},
        litellm_params=litellm_params,
        headers={},
    )
    assert body == {"input": [{"role": "user", "content": "describe this"}]}


def test_transform_request_joins_text_content_parts() -> None:
    body = DatabricksAgentConfig().transform_request(
        model="my-agent",
        messages=[{"role": "user", "content": [{"type": "text", "text": "pi"}, {"type": "text", "text": "ng"}]}],
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert body == {"input": [{"role": "user", "content": "ping"}]}


def test_validate_environment_lets_the_api_key_replace_a_header_authorization() -> None:
    headers = DatabricksAgentConfig().validate_environment(
        headers={"authorization": "Bearer client-token", "X-LiteLLM-User-Id": "u1"},
        model="my-agent",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="pat-1",
    )
    assert headers == {"X-LiteLLM-User-Id": "u1", "Content-Type": "application/json", "Authorization": "Bearer pat-1"}


def test_validate_environment_uses_a_header_authorization_when_no_api_key_is_set() -> None:
    headers = DatabricksAgentConfig().validate_environment(
        headers={"authorization": "Bearer client-token"},
        model="my-agent",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=None,
    )
    assert headers == {"authorization": "Bearer client-token", "Content-Type": "application/json"}


def test_validate_environment_uses_the_pat_as_bearer() -> None:
    headers = DatabricksAgentConfig().validate_environment(
        headers={}, model="my-agent", messages=[], optional_params={}, litellm_params={}, api_key="pat-1"
    )
    assert headers == {"Content-Type": "application/json", "Authorization": "Bearer pat-1"}


def test_validate_environment_without_any_credential_is_a_400(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABRICKS_CLIENT_ID", raising=False)
    monkeypatch.delenv("DATABRICKS_CLIENT_SECRET", raising=False)
    with pytest.raises(DatabricksException, match="Missing Databricks credentials") as info:
        DatabricksAgentConfig().validate_environment(
            headers={}, model="my-agent", messages=[], optional_params={}, litellm_params={}, api_key=None
        )
    assert info.value.status_code == 400


def _validate_environment(api_key: str | None, api_base: str | None) -> dict[str, object]:
    return DatabricksAgentConfig().validate_environment(
        headers={"authorization": "Bearer client-token"},
        model="my-agent",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key=api_key,
        api_base=api_base,
    )


@responses.activate
@pytest.mark.parametrize("api_base", [WORKSPACE, f"{WORKSPACE}/serving-endpoints", INVOCATIONS_URL])
def test_validate_environment_mints_an_oauth_m2m_token_from_the_env_service_principal(
    env_oauth_m2m: None, api_base: str
) -> None:
    token = responses.post(TOKEN_URL, json={"access_token": "m2m-token", "expires_in": 3600})
    headers = _validate_environment(api_key=None, api_base=api_base)
    assert headers == {"Content-Type": "application/json", "Authorization": "Bearer m2m-token"}
    assert token.calls[0].request.headers["Authorization"].startswith("Basic ")
    assert "grant_type=client_credentials" in str(token.calls[0].request.body)


@responses.activate
def test_validate_environment_mints_the_env_token_from_databricks_host_for_an_app(
    env_oauth_m2m: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DATABRICKS_HOST", f"{WORKSPACE}/")
    responses.post(TOKEN_URL, json={"access_token": "m2m-token"})
    assert _validate_environment(api_key=None, api_base=APP_HOST)["Authorization"] == "Bearer m2m-token"


@responses.activate
def test_validate_environment_never_posts_the_env_secret_to_an_app_host(env_oauth_m2m: None) -> None:
    with pytest.raises(DatabricksException, match="set DATABRICKS_HOST") as info:
        _validate_environment(api_key=None, api_base=APP_URL)
    assert info.value.status_code == 400
    assert not responses.calls


@responses.activate
def test_validate_environment_prefers_the_api_key_over_the_env_service_principal(env_oauth_m2m: None) -> None:
    assert _validate_environment(api_key="pat-1", api_base=WORKSPACE)["Authorization"] == "Bearer pat-1"
    assert not responses.calls


def test_output_text_joins_message_items_and_skips_tool_trace_items() -> None:
    output = [
        {"type": "function_call", "name": "lookup", "arguments": "{}", "call_id": "c1"},
        {"type": "function_call_output", "call_id": "c1", "output": "secret trace"},
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "Let me check."}, {"type": "refusal", "refusal": "no"}],
        },
        {"type": "message", "content": [{"type": "output_text", "text": "It is sunny."}]},
    ]
    assert output_text(AgentResponse.model_validate({"output": output}).output) == "Let me check.\n\nIt is sunny."
    assert output_text(AgentResponse.model_validate({"output": output[:2]}).output) == ""
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


@pytest.mark.parametrize("body", ["<html>login</html>", "[]", "{}", '{"output": "pong"}'])
def test_transform_response_rejects_a_body_that_is_not_a_responses_agent_response(body: str) -> None:
    raw = httpx.Response(200, text=body, request=httpx.Request("POST", APP_URL))
    with pytest.raises(DatabricksException, match="not a ResponsesAgent response") as info:
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
    assert info.value.status_code == 502


def test_stream_iterator_emits_deltas_once_and_whole_items_only_when_nothing_streamed() -> None:
    iterator = DatabricksAgentResponsesIterator(streaming_response=iter(()), sync_stream=True)
    chunks = [
        iterator.chunk_parser({"type": "response.output_text.delta", "item_id": "msg_1", "delta": "po"}),
        iterator.chunk_parser({"type": "response.output_text.delta", "item_id": "msg_1", "delta": "ng"}),
        iterator.chunk_parser({"type": "response.output_item.done", "item": MESSAGE_ITEM}),
        iterator.chunk_parser({"type": "response.output_item.done", "item": {**MESSAGE_ITEM, "id": "msg_2"}}),
        iterator.chunk_parser({"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc_1"}}),
    ]
    assert [chunk["text"] for chunk in chunks] == ["po", "ng", "", "\n\npong", ""]
    assert all(chunk["is_finished"] is False for chunk in chunks)


def test_stream_iterator_separates_consecutive_message_items_like_the_non_streaming_reply() -> None:
    iterator = DatabricksAgentResponsesIterator(streaming_response=iter(()), sync_stream=True)
    events: list[dict[str, object]] = [
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": ""},
        {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Let me check."},
        {"type": "response.output_item.done", "item": {"type": "function_call", "id": "fc_1"}},
        {"type": "response.output_text.delta", "item_id": "msg_2", "delta": "It is"},
        {"type": "response.output_text.delta", "item_id": "msg_2", "delta": " sunny."},
    ]
    streamed = "".join(iterator.chunk_parser(event)["text"] for event in events)
    whole = output_text(
        AgentResponse.model_validate(
            {
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": "Let me check."}]},
                    {"type": "function_call", "id": "fc_1"},
                    {"type": "message", "content": [{"type": "output_text", "text": "It is sunny."}]},
                ]
            }
        ).output
    )
    assert streamed == whole == "Let me check.\n\nIt is sunny."


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
    assert json.loads(sent.content) == {
        "input": [{"role": "user", "content": "ping"}],
        "custom_inputs": {"tenant": "t-1"},
    }
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


@responses.activate
@respx.mock
async def test_acompletion_authenticates_with_the_env_service_principal(
    httpx_transport: None, env_oauth_m2m: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    responses.post(TOKEN_URL, json={"access_token": "m2m-token"})
    route = respx.post(INVOCATIONS_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    response = await litellm.acompletion(
        model="databricks_agent/my-agent",
        messages=[{"role": "user", "content": "ping"}],
        api_base=WORKSPACE,
    )
    assert route.calls.last.request.headers["Authorization"] == "Bearer m2m-token"
    assert response.choices[0].message.content == "pong"


async def test_unsupported_openai_params_follow_drop_params(httpx_transport: None) -> None:
    messages = [{"role": "user", "content": "ping"}]
    with pytest.raises(litellm.UnsupportedParamsError):
        await litellm.acompletion(
            model="databricks_agent/my-agent",
            messages=messages,
            api_base=WORKSPACE,
            api_key="pat-1",
            frequency_penalty=0.2,
        )
    with respx.mock:
        route = respx.post(INVOCATIONS_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
        await litellm.acompletion(
            model="databricks_agent/my-agent",
            messages=messages,
            api_base=WORKSPACE,
            api_key="pat-1",
            frequency_penalty=0.2,
            drop_params=True,
        )
    assert "frequency_penalty" not in json.loads(route.calls.last.request.content)


@respx.mock
@pytest.mark.parametrize("token_limit", [{"max_tokens": 64}, {"max_completion_tokens": 64}])
async def test_acompletion_sends_sampling_params_in_the_responses_agent_shape(
    httpx_transport: None, token_limit: dict[str, int]
) -> None:
    route = respx.post(INVOCATIONS_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    await litellm.acompletion(
        model="databricks_agent/my-agent",
        messages=[{"role": "user", "content": "ping"}],
        api_base=WORKSPACE,
        api_key="pat-1",
        temperature=0.2,
        top_p=0.9,
        user="user-1",
        **token_limit,
    )
    assert json.loads(route.calls.last.request.content) == {
        "input": [{"role": "user", "content": "ping"}],
        "max_output_tokens": 64,
        "temperature": 0.2,
        "top_p": 0.9,
        "user": "user-1",
    }


@respx.mock
async def test_acompletion_sends_string_metadata_and_refuses_other_values_unless_dropped(
    httpx_transport: None,
) -> None:
    route = respx.post(INVOCATIONS_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    request: Final = {
        "model": "databricks_agent/my-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "api_base": WORKSPACE,
        "api_key": "pat-1",
    }
    await litellm.acompletion(**request, extra_body={"metadata": {"session": "s-1"}})
    assert json.loads(route.calls.last.request.content)["metadata"] == {"session": "s-1"}

    with pytest.raises(litellm.BadRequestError, match="string values") as info:
        await litellm.acompletion(**request, extra_body={"metadata": {"session": "s-1", "turn": 1}})
    assert info.value.status_code == 400
    assert route.call_count == 1

    await litellm.acompletion(**request, extra_body={"metadata": {"session": "s-1", "turn": 1}}, drop_params=True)
    assert json.loads(route.calls.last.request.content)["metadata"] == {"session": "s-1"}
    await litellm.acompletion(**request, extra_body={"metadata": {"turn": 1}}, drop_params=True)
    assert "metadata" not in json.loads(route.calls.last.request.content)


@responses.activate
@respx.mock
async def test_acompletion_reuses_the_env_service_principal_token_until_the_secret_rotates(
    httpx_transport: None, env_oauth_m2m: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    token = responses.post(TOKEN_URL, json={"access_token": "m2m-token", "expires_in": 3600})
    route = respx.post(INVOCATIONS_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    request: Final = {
        "model": "databricks_agent/my-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "api_base": WORKSPACE,
    }
    await litellm.acompletion(**request)
    await litellm.acompletion(**request)
    assert token.call_count == 1
    assert route.call_count == 2

    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", f"rotated-{uuid4().hex}")
    await litellm.acompletion(**request)
    assert token.call_count == 2


@responses.activate
@respx.mock
async def test_acompletion_mints_a_fresh_env_token_when_it_expires_within_the_refresh_buffer(
    httpx_transport: None, env_oauth_m2m: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DATABRICKS_API_KEY", raising=False)
    token = responses.post(TOKEN_URL, json={"access_token": "m2m-token", "expires_in": 30})
    respx.post(INVOCATIONS_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    request: Final = {
        "model": "databricks_agent/my-agent",
        "messages": [{"role": "user", "content": "ping"}],
        "api_base": WORKSPACE,
    }
    await litellm.acompletion(**request)
    await litellm.acompletion(**request)
    assert token.call_count == 2


@respx.mock
async def test_acompletion_resolves_os_environ_api_base_and_api_key(
    httpx_transport: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_DBX_HOST", WORKSPACE)
    monkeypatch.setenv("MY_DBX_PAT", "pat-from-env")
    route = respx.post(INVOCATIONS_URL).mock(return_value=httpx.Response(200, json={"output": [MESSAGE_ITEM]}))
    response = await litellm.acompletion(
        model="databricks_agent/my-agent",
        messages=[{"role": "user", "content": "ping"}],
        api_base="os.environ/MY_DBX_HOST",
        api_key="os.environ/MY_DBX_PAT",
    )
    assert route.calls.last.request.headers["Authorization"] == "Bearer pat-from-env"
    assert response.choices[0].message.content == "pong"
