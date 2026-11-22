"""Regression tests for the responses -> completion fallback bridge guard.

When the Responses API falls back to chat completions (no native responses
config), it must tag the forwarded ``litellm.completion`` / ``litellm.acompletion``
call with ``_skip_responses_api_bridge=True`` so ``completion()`` does not bridge
the request straight back to the Responses API and mutually recurse forever.

Both fallback paths are covered: the sync ``response_api_handler`` (``_is_async``
False) and the async ``async_response_api_handler`` (``_is_async`` True). The
module-level ``litellm.completion`` / ``litellm.acompletion`` are patched to
capture the forwarded kwargs; if the flag-setting line is removed the captured
kwargs lack the flag and these tests fail.
"""

import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.utils import ADDRESSED_RESPONSE_ID_FIELD


_UPSTREAM_PARAM_CASES: Final = [
    pytest.param("max_tokens", "max_output_tokens", id="renamed_by_the_bridge"),
    pytest.param("max_completion_tokens", "max_output_tokens", id="renamed_twice"),
    pytest.param("reasoning_effort", "reasoning.effort", id="moved_under_reasoning"),
    pytest.param("response_format.json_schema.schema", "text.format", id="moved_under_text"),
    pytest.param("tools[0].function.name", "tools", id="reshaped_tool"),
    pytest.param("messages[0].content[1].image_url", "input", id="message_path_under_input"),
    pytest.param("stream_options", None, id="sent_only_by_the_bridge"),
    pytest.param("frequency_penalty", None, id="not_a_responses_field"),
    pytest.param("user", "user", id="shared_field"),
    pytest.param(["body", "max_tokens"], None, id="non_string_param"),
]


def _request_sending_every_renamed_field() -> ResponsesAPIOptionalRequestParams:
    return {
        "max_output_tokens": 16,
        "reasoning": {"effort": "low"},
        "text": {"format": {"type": "text"}},
        "tools": [{"type": "function", "name": "lookup", "parameters": {"type": "object", "properties": {}}}],
        "user": "user-1",
    }


def _upstream_rejects(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, param: object) -> respx.Route:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    return respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "rejected", "type": "invalid_request_error", "param": param, "code": None}},
        )
    )


@pytest.mark.parametrize("upstream_param, relayed_param", _UPSTREAM_PARAM_CASES)
def test_sync_bridged_turn_names_an_upstream_param_as_the_responses_field_the_caller_sent(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, upstream_param: object, relayed_param: str | None
):
    upstream: Final = _upstream_rejects(respx_mock, monkeypatch, upstream_param)

    with pytest.raises(litellm.BadRequestError) as raised:
        LiteLLMCompletionTransformationHandler().response_api_handler(
            model="gpt-5.6",
            input="hi",
            responses_api_request=_request_sending_every_renamed_field(),
            custom_llm_provider="openai",
            api_key="sk-bridge-test",
        )

    assert upstream.call_count == 1
    assert raised.value.param == relayed_param


@pytest.mark.asyncio
@pytest.mark.parametrize("upstream_param, relayed_param", _UPSTREAM_PARAM_CASES)
async def test_async_bridged_turn_names_an_upstream_param_as_the_responses_field_the_caller_sent(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, upstream_param: object, relayed_param: str | None
):
    upstream: Final = _upstream_rejects(respx_mock, monkeypatch, upstream_param)
    bridged_turn: Final = LiteLLMCompletionTransformationHandler().response_api_handler(
        model="gpt-5.6",
        input="hi",
        responses_api_request=_request_sending_every_renamed_field(),
        custom_llm_provider="openai",
        _is_async=True,
        api_key="sk-bridge-test",
    )

    with pytest.raises(litellm.BadRequestError) as raised:
        await bridged_turn

    assert upstream.call_count == 1
    assert raised.value.param == relayed_param


@pytest.mark.parametrize(
    "upstream_param",
    ["max_tokens", "max_completion_tokens", "reasoning_effort", "response_format", "tools[0].function.name"],
)
def test_bridged_turn_names_no_responses_field_the_caller_did_not_send(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, upstream_param: str
):
    upstream: Final = _upstream_rejects(respx_mock, monkeypatch, upstream_param)

    with pytest.raises(litellm.BadRequestError) as raised:
        LiteLLMCompletionTransformationHandler().response_api_handler(
            model="gpt-5.6",
            input="hi",
            responses_api_request={"user": "user-1"},
            custom_llm_provider="openai",
            api_key="sk-bridge-test",
            max_tokens=100000,
        )

    assert upstream.call_count == 1
    assert raised.value.param is None


@pytest.mark.parametrize(
    "responses_api_request, upstream_param, relayed_param",
    [
        pytest.param({"instructions": "Answer in one word"}, "messages.[0].content", "instructions", id="instructions"),
        pytest.param({"instructions": "Answer in one word"}, "messages.[1].role", "input", id="input_after_instructions"),
        pytest.param({"previous_response_id": "resp_stored"}, "messages.[1].role", None, id="stored_turns_first"),
    ],
)
def test_bridged_turn_names_a_message_path_by_the_request_field_the_message_came_from(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    responses_api_request: dict[str, str],  # mutable-ok: response_api_handler takes the request as a dict
    upstream_param: str,
    relayed_param: str | None,
):
    upstream: Final = _upstream_rejects(respx_mock, monkeypatch, upstream_param)

    with pytest.raises(litellm.BadRequestError) as raised:
        LiteLLMCompletionTransformationHandler().response_api_handler(
            model="gpt-5.6",
            input="hi",
            responses_api_request=responses_api_request,
            custom_llm_provider="openai",
            api_key="sk-bridge-test",
        )

    assert upstream.call_count == 1
    assert raised.value.param == relayed_param


class _StopForwarding(Exception):
    """Raised by the mocked (a)completion once the forwarded kwargs are captured."""


def test_sync_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="gpt-4o",
                input="hello",
                responses_api_request={},
                custom_llm_provider="openai",
                _is_async=False,
            )

    assert captured.get("_skip_responses_api_bridge") is True


@pytest.mark.asyncio
async def test_async_fallback_tags_skip_responses_api_bridge():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.acompletion", fake_acompletion):
        coro = handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
        )
        with pytest.raises(_StopForwarding):
            await coro

    assert captured.get("_skip_responses_api_bridge") is True


_CODEX_ADDITIONAL_TOOLS_ITEM = {
    "type": "additional_tools",
    "id": "at_codex",
    "role": "developer",
    "tools": [
        {
            "type": "namespace",
            "name": "functions",
            "description": "",
            "tools": [
                {
                    "type": "custom",
                    "name": "exec",
                    "description": "Runs a shell command.",
                    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/"},
                },
                {
                    "type": "function",
                    "name": "wait",
                    "description": "Waits for a background command.",
                    "parameters": {"type": "object", "properties": {"id": {"type": "string"}}},
                },
            ],
        }
    ],
}
_CODEX_INPUT = [_CODEX_ADDITIONAL_TOOLS_ITEM, {"type": "message", "role": "user", "content": "Run ls"}]


def test_sync_fallback_hoists_additional_tools_input_items_into_chat_tools():
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.completion", fake_completion):  # test-quality-ok: no DI seam; the file stubs this same boundary
        with pytest.raises(_StopForwarding):
            handler.response_api_handler(
                model="bedrock/us.openai.gpt-5.6",
                input=_CODEX_INPUT,
                responses_api_request={},
                custom_llm_provider="bedrock",
                _is_async=False,
            )

    assert [message["role"] for message in captured["messages"]] == ["user"]
    functions_by_name = {tool["function"]["name"]: tool["function"] for tool in captured["tools"]}
    assert set(functions_by_name) == {"exec", "functions__wait"}
    assert set(functions_by_name["exec"]["parameters"]["properties"]) == {"content"}


@pytest.mark.asyncio
async def test_async_fallback_returns_hoisted_nested_custom_tool_call_as_custom_tool_call():
    from litellm.responses.litellm_completion_transformation.transformation import TOOL_CALLS_CACHE
    from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Function, Message, ModelResponse

    handler = LiteLLMCompletionTransformationHandler()
    tool_call_id = "call_exec_hoisted"

    async def fake_acompletion(**kwargs):
        return ModelResponse(
            id="chatcmpl-exec",
            created=1,
            model="us.openai.gpt-5.6",
            object="chat.completion",
            choices=[
                Choices(
                    finish_reason="tool_calls",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=[
                            ChatCompletionMessageToolCall(
                                id=tool_call_id,
                                type="function",
                                function=Function(name="exec", arguments='{"content": "ls"}'),
                            )
                        ],
                    ),
                )
            ],
        )

    try:
        with patch("litellm.acompletion", fake_acompletion):  # test-quality-ok: no DI seam; file stubs this boundary
            response = await handler.response_api_handler(
                model="bedrock/us.openai.gpt-5.6",
                input=_CODEX_INPUT,
                responses_api_request={},
                custom_llm_provider="bedrock",
                _is_async=True,
            )
    finally:
        TOOL_CALLS_CACHE.delete_cache(key=tool_call_id)

    tool_calls = [(item.type, item.name, item.input) for item in response.output if item.type == "custom_tool_call"]
    assert tool_calls == [("custom_tool_call", "exec", "ls")]


class _RecordingAnthropicHandler:
    def __init__(self, reply: Mapping[str, object]) -> None:
        self.reply: Final = reply
        self.request_body: Mapping[str, object] | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request_body = json.loads(request.content)
        return httpx.Response(200, json=dict(self.reply), request=request)


_ANTHROPIC_MESSAGE_PAYLOAD: Final = {
    "id": "msg_turn_two",
    "type": "message",
    "role": "assistant",
    "model": "claude-sonnet-4-6",
    "content": [{"type": "text", "text": "14"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 12, "output_tokens": 1},
}


@pytest.mark.asyncio
async def test_bridged_follow_up_turn_keeps_the_addressed_response_id_off_the_provider_body():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        previous_response_id="resp_turn_one",
        client=client,
        **{ADDRESSED_RESPONSE_ID_FIELD: "resp_turn_one"},
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert ADDRESSED_RESPONSE_ID_FIELD not in provider.request_body, (
        f"the addressed response id reached the provider body: {sorted(provider.request_body)}"
    )
    assert isinstance(response, ResponsesAPIResponse)
    assert [item.type for item in response.output] == ["message"]


@pytest.mark.asyncio
async def test_bridged_turn_after_a_client_tool_search_reports_auto_for_an_allowed_tools_choice_it_dropped():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        tools=[
            {"type": "tool_search", "execution": "client"},
            {"type": "function", "name": "get_time", "parameters": {"type": "object", "properties": {}}},
        ],
        tool_choice={"type": "allowed_tools", "mode": "auto", "tools": [{"type": "function", "name": "get_time"}]},
        drop_params=True,
        client=client,
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert "tool_choice" not in provider.request_body
    assert isinstance(response, ResponsesAPIResponse)
    assert response.tool_choice == "auto"
