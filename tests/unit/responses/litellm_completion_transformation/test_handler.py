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

import asyncio
import inspect
import json
import sys
import threading
from collections.abc import Mapping, Sequence
from itertools import chain
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import patch

import httpx
import pytest
import respx

import litellm
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import _PENDING_CACHE_WRITES
from litellm.constants import PREVIOUS_RESPONSE_IDS_METADATA_KEY, RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.tool_fields import (
    BRIDGED_TOOL_NAMES_KWARG,
    BridgedToolNames,
    NamespaceMember,
    full_names_by_sent_name,
    qualified_member_name,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.responses.litellm_completion_transformation.compaction_trigger import SUMMARY_PROMPT
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_api import RoutingContext
from litellm.router_utils.cooldown_handlers import _should_cooldown_deployment
from litellm.types.caching import LiteLLMCacheType
from litellm.types.llms.openai import ResponsesAPIOptionalRequestParams, ResponsesAPIResponse
from litellm.types.router import RetryPolicy
from litellm.types.utils import ADDRESSED_RESPONSE_ID_FIELD, ChatCompletionDeltaToolCall, ModelResponseStream


_UPSTREAM_PARAM_CASES: Final = [
    pytest.param("max_tokens", "max_output_tokens", id="renamed_by_the_bridge"),
    pytest.param("max_completion_tokens", "max_output_tokens", id="renamed_twice"),
    pytest.param("reasoning_effort", "reasoning.effort", id="moved_under_reasoning"),
    pytest.param("response_format.json_schema.schema", "text.format", id="moved_under_text"),
    pytest.param("verbosity", "text.verbosity", id="verbosity_moved_under_text"),
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
        "text": {"format": {"type": "text"}, "verbosity": "low"},
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


@pytest.mark.asyncio
async def test_bridged_follow_up_names_no_request_field_for_a_message_path_inside_the_stored_turns(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    upstream: Final = _upstream_rejects(respx_mock, monkeypatch, "messages.[1].role")
    bridged_turn: Final = LiteLLMCompletionTransformationHandler().response_api_handler(
        model="gpt-5.6",
        input="hi",
        responses_api_request={"previous_response_id": "resp_stored"},
        custom_llm_provider="openai",
        _is_async=True,
        api_key="sk-bridge-test",
    )

    stored_turns: Final = _StoredTurns((_stored_mcp_turn("resp_stored"),))

    with (
        patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored_turns)),
        pytest.raises(litellm.BadRequestError) as raised,
    ):
        await bridged_turn

    assert upstream.call_count == 1
    assert raised.value.param is None


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
async def test_a_first_turn_stores_an_empty_path_to_the_root_so_a_fork_off_it_replays_only_it():
    litellm_metadata: Final[dict[str, object]] = {"user_api_key": "hashed"}
    handler = LiteLLMCompletionTransformationHandler()
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        raise _StopForwarding()

    with patch("litellm.acompletion", fake_acompletion), pytest.raises(_StopForwarding):
        await handler.response_api_handler(
            model="gpt-4o",
            input="hello",
            responses_api_request={},
            custom_llm_provider="openai",
            _is_async=True,
            litellm_metadata=litellm_metadata,
        )

    assert captured["litellm_metadata"] == {"user_api_key": "hashed", PREVIOUS_RESPONSE_IDS_METADATA_KEY: []}
    assert captured["litellm_metadata"] is litellm_metadata


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

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        SimpleNamespace(db=_StoredTurns((_stored_mcp_turn("resp_turn_one"),))),
    ):
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
async def test_bridged_turn_with_a_client_field_named_previous_response_client_ids_reaches_the_provider():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        client=client,
        previous_response_client_ids=["resp_from_the_client_body"],
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert isinstance(response, ResponsesAPIResponse)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_field", ["caller_input", "names_non_string_fields"])
async def test_bridged_turn_with_a_client_field_named_like_a_bridge_keyword_reaches_the_provider(client_field: str):
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    response = await litellm.aresponses(
        model="azure_ai/claude-sonnet-4-6",
        api_base="https://fake-foundry-resource.services.ai.azure.com",
        api_key="fake-api-key",
        input="Double it",
        client=client,
        **{client_field: "from the client body"},
    )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert isinstance(response, ResponsesAPIResponse)


class _StoredTurns:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows: Final = rows

    async def query_raw(self, query: str, request_ids: Sequence[str]) -> list[dict[str, object]]:
        sessions: Final = {row["session_id"] for row in self.rows if row["request_id"] in request_ids}
        return [dict(row) for row in self.rows if row["session_id"] in sessions]


def _stored_mcp_turn(request_id: str) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "turn-one-session",
        "proxy_server_request": {"input": "My favorite number is 7.", "model": "claude-sonnet-4-6"},
        "response": {
            "id": request_id,
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Noted.", "annotations": []}],
                },
                {"type": "mcp_tools_fetched", "tools": [{"name": "demo-echo"}]},
            ],
        },
    }


_TURN_ONE_ID: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
    custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
)


def _turn_stored_without_prompts(request_id: str) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "turn-one-session",
        "proxy_server_request": "{}",
        "response": "{}",
    }


async def _bridged_follow_up(
    asynchronous: bool,
    provider: _RecordingAnthropicHandler,
    proxy_fields: Mapping[str, object] = MappingProxyType({}),
    follow_up_input: str | Sequence[Mapping[str, object]] = "Double it",
    previous_response_id: object = _TURN_ONE_ID,
) -> object:
    request: Final = {
        **proxy_fields,
        "model": "azure_ai/claude-sonnet-4-6",
        "api_base": "https://fake-foundry-resource.services.ai.azure.com",
        "api_key": "fake-api-key",
        "input": follow_up_input,
        "previous_response_id": previous_response_id,
    }
    if not asynchronous:
        sync_client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(provider)))
        return litellm.responses(**request, client=sync_client)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    return await litellm.aresponses(**request, client=client)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("asynchronous", "prisma_client"),
    [
        pytest.param(False, None, id="sync-no_database"),
        pytest.param(True, None, id="async-no_database"),
        pytest.param(True, SimpleNamespace(db=_StoredTurns(())), id="async-unknown_id"),
        pytest.param(
            True,
            SimpleNamespace(db=_StoredTurns((_turn_stored_without_prompts(_TURN_ONE_ID),))),
            id="async-prompts_not_stored",
        ),
    ],
)
@pytest.mark.parametrize(
    "follow_up_input",
    [
        pytest.param("Double it", id="user_turn"),
        pytest.param([{"type": "function_call_output", "call_id": "call_1", "output": "7"}], id="tool_output_only"),
    ],
)
async def test_bridged_follow_up_without_a_replayable_previous_turn_is_refused_before_the_provider_call(
    asynchronous: bool, prisma_client: SimpleNamespace | None, follow_up_input: str | Sequence[Mapping[str, object]]
):
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)

    with patch("litellm.proxy.proxy_server.prisma_client", prisma_client):
        with pytest.raises(litellm.BadRequestError) as refused:
            await _bridged_follow_up(asynchronous, provider, follow_up_input=follow_up_input)

    assert provider.request_body is None, "the follow-up reached the provider with no conversation history"
    assert (refused.value.status_code, refused.value.param, refused.value.code) == (
        400,
        "previous_response_id",
        "previous_response_not_found",
    )
    assert f"Previous response with id '{_TURN_ONE_ID}' not found." in refused.value.message
    assert "`store_prompts_in_spend_logs: true`" in refused.value.message



@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("previous_response_id", [123, [_TURN_ONE_ID]], ids=["int", "list"])
async def test_a_non_string_previous_response_id_is_a_400_before_the_provider_call(
    asynchronous: bool, previous_response_id: object
):
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)

    with pytest.raises(litellm.BadRequestError) as refused:
        await _bridged_follow_up(asynchronous, provider, previous_response_id=previous_response_id)

    assert provider.request_body is None, "the follow-up reached the provider with an id it cannot look up"
    assert (refused.value.status_code, refused.value.param) == (400, "previous_response_id")

@pytest.mark.asyncio
async def test_refused_follow_ups_do_not_count_toward_the_deployments_failure_rate():
    litellm_params: Final = {
        "model": "azure_ai/claude-sonnet-4-6",
        "api_base": "https://fake-foundry-resource.services.ai.azure.com",
        "api_key": "fake-api-key",
    }
    router: Final = litellm.Router(
        model_list=[
            {"model_name": "bridged", "litellm_params": litellm_params, "model_info": {"id": "d1"}},
            {"model_name": "bridged", "litellm_params": litellm_params, "model_info": {"id": "d2"}},
        ]
    )

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        for _ in range(6):
            with pytest.raises(litellm.BadRequestError, match="not found"):
                await router.aresponses(model="d1", input="Double it", previous_response_id=_TURN_ONE_ID)

    assert _should_cooldown_deployment(RoutingContext(router), "d1", 500, Exception("Test")) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fallback_settings",
    [
        pytest.param({"fallbacks": [{"bridged": ["native"]}]}, id="fallbacks"),
        pytest.param({"default_fallbacks": ["native"]}, id="default_fallbacks"),
    ],
)
@pytest.mark.parametrize(
    ("previous_response_id", "refusal"),
    [
        pytest.param(_TURN_ONE_ID, "not found", id="unresolvable_id"),
        pytest.param(123, "expected a string, got int", id="non_string_id"),
    ],
)
async def test_a_refused_follow_up_is_neither_retried_nor_sent_to_a_fallback(
    fallback_settings: Mapping[str, object], previous_response_id: object, refusal: str
):
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "bridged",
                "litellm_params": {
                    "model": "azure_ai/claude-sonnet-4-6",
                    "api_base": "https://fake-foundry-resource.services.ai.azure.com",
                    "api_key": "fake-api-key",
                },
                "model_info": {"id": "d1"},
            },
            {
                "model_name": "native",
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "fake-api-key",
                    "mock_response": "answered by the fallback without the history",
                },
                "model_info": {"id": "n1"},
            },
        ],
        num_retries=3,
        retry_policy=RetryPolicy(BadRequestErrorRetries=2),
        **fallback_settings,
    )

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        with pytest.raises(litellm.BadRequestError, match=refusal) as refused:
            await router.aresponses(model="bridged", input="Double it", previous_response_id=previous_response_id)

    assert not getattr(refused.value, "num_retries", None)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "client_id_fields",
    [
        pytest.param({}, id="previous_response_id_only"),
        pytest.param({ADDRESSED_RESPONSE_ID_FIELD: "resp_the_client_held"}, id="with_the_id_the_client_sent"),
    ],
)
async def test_a_follow_up_to_a_native_turn_the_bridged_group_cannot_rebuild_goes_to_its_fallback(
    client_id_fields: Mapping[str, str],
):
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "bridged",
                "litellm_params": {
                    "model": "azure_ai/claude-sonnet-4-6",
                    "api_base": "https://fake-foundry-resource.services.ai.azure.com",
                    "api_key": "fake-api-key",
                },
                "model_info": {"id": "d1"},
            },
            {
                "model_name": "native",
                "litellm_params": {
                    "model": "openai/gpt-5.4-mini",
                    "api_key": "fake-api-key",
                    "mock_response": "answered by the native fallback",
                },
                "model_info": {"id": "n1"},
            },
        ],
        fallbacks=[{"bridged": ["native"]}],
    )
    native_turn_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="openai", model_id="native-elsewhere", response_id="resp_turn_one"
    )

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        response: Final = await router.aresponses(
            model="bridged", input="Double it", previous_response_id=native_turn_id, **client_id_fields
        )

    assert response._hidden_params["model_id"] == "n1"


class _CountedLookups:
    def __init__(self) -> None:
        self.lookups = 0

    async def query_raw(self, query: str, request_ids: Sequence[str]) -> list[dict[str, object]]:
        self.lookups += 1
        return []


@pytest.mark.asyncio
async def test_a_refused_sdk_follow_up_is_not_rerun_for_num_retries():
    spend_logs: Final = _CountedLookups()

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=spend_logs)):
        with pytest.raises(litellm.BadRequestError, match="not found"):
            await _bridged_follow_up(
                True, _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD), proxy_fields={"num_retries": 2}
            )

    assert spend_logs.lookups == RESPONSES_SESSION_LOOKUP_MAX_ATTEMPTS


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_an_sdk_install_without_the_proxy_extras_gets_the_400_for_a_follow_up(
    asynchronous: bool, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delitem(sys.modules, "litellm.proxy.proxy_server", raising=False)
    monkeypatch.setitem(sys.modules, "fastapi", None)
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)

    with pytest.raises(litellm.BadRequestError) as refused:
        await _bridged_follow_up(asynchronous, provider)

    assert provider.request_body is None, "the follow-up reached the provider with no conversation history"
    assert (refused.value.status_code, refused.value.param) == (400, "previous_response_id")
    assert "send the earlier turns in `input`" in refused.value.message
    assert "general_settings" not in refused.value.message


@pytest.mark.asyncio
async def test_a_refused_proxy_follow_up_names_the_id_the_client_sent():
    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_StoredTurns(()))):
        with pytest.raises(litellm.BadRequestError) as refused:
            await _bridged_follow_up(
                True,
                _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD),
                {ADDRESSED_RESPONSE_ID_FIELD: "resp_the_client_held"},
            )

    assert "Previous response with id 'resp_the_client_held' not found." in refused.value.message


class _DatabaseOnTheProxyLoop:
    async def query_raw(self, query: str, request_ids: Sequence[str]) -> list[dict[str, object]]:
        raise AssertionError("a synchronous call queried the proxy's database off the proxy's event loop")


@pytest.mark.asyncio
async def test_a_sync_follow_up_beside_the_proxy_database_asks_for_aresponses_without_querying_it():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_DatabaseOnTheProxyLoop())):
        with pytest.raises(litellm.BadRequestError) as refused:
            await _bridged_follow_up(False, provider, {ADDRESSED_RESPONSE_ID_FIELD: "resp_the_client_held"})

    assert provider.request_body is None, "the follow-up reached the provider with no conversation history"
    assert (refused.value.status_code, refused.value.param) == (400, "previous_response_id")
    assert "'resp_the_client_held'" in refused.value.message
    assert "litellm.aresponses()" in refused.value.message


@pytest.mark.asyncio
async def test_a_sync_follow_up_carrying_a_compaction_summary_beside_the_proxy_database_reaches_the_provider():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    summarized_input: Final = [
        {
            "type": "compaction",
            "id": "cmp_1",
            "encrypted_content": json.dumps({"type": "compaction", "content": "The number is 7."}),
        },
        {"role": "user", "content": "Double it"},
    ]

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_DatabaseOnTheProxyLoop())):
        response = await _bridged_follow_up(False, provider, follow_up_input=summarized_input)

    assert isinstance(response, ResponsesAPIResponse)
    assert provider.request_body is not None, "the summarized follow-up never reached the provider"
    sent_messages: Final = json.dumps(provider.request_body["messages"])
    assert "The number is 7." in sent_messages and "Double it" in sent_messages


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("follow_up_input", "earlier_reply"),
    [
        pytest.param(
            [
                {"role": "user", "content": "My favorite number is 7."},
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": "Double it"},
            ],
            "Noted.",
            id="assistant_message",
        ),
        pytest.param(
            [
                {"role": "user", "content": "Look up my favorite number."},
                {"type": "function_call", "call_id": "call_1", "name": "lookup_number", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call_1", "output": "7"},
            ],
            "lookup_number",
            id="function_call",
        ),
    ],
)
@pytest.mark.parametrize(
    ("asynchronous", "prisma_client"),
    [
        pytest.param(False, None, id="sync-no_database"),
        pytest.param(False, SimpleNamespace(db=_DatabaseOnTheProxyLoop()), id="sync-proxy_database"),
        pytest.param(True, None, id="async-no_database"),
        pytest.param(True, SimpleNamespace(db=_StoredTurns(())), id="async-unknown_id"),
        pytest.param(
            True,
            SimpleNamespace(db=_StoredTurns((_turn_stored_without_prompts(_TURN_ONE_ID),))),
            id="async-prompts_not_stored",
        ),
    ],
)
async def test_a_follow_up_carrying_its_earlier_reply_in_input_reaches_the_provider_when_the_id_does_not_resolve(
    asynchronous: bool,
    prisma_client: SimpleNamespace | None,
    follow_up_input: Sequence[Mapping[str, object]],
    earlier_reply: str,
):
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)

    with patch("litellm.proxy.proxy_server.prisma_client", prisma_client):
        response = await _bridged_follow_up(asynchronous, provider, follow_up_input=follow_up_input)

    assert isinstance(response, ResponsesAPIResponse)
    assert provider.request_body is not None, "the follow-up that carried its history never reached the provider"
    assert earlier_reply in json.dumps(provider.request_body["messages"])


@pytest.mark.asyncio
async def test_bridged_follow_up_replays_the_stored_previous_turn():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)

    with patch(
        "litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_StoredTurns((_stored_mcp_turn(_TURN_ONE_ID),)))
    ):
        await _bridged_follow_up(True, provider)

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert [(message["role"], json.dumps(message["content"])) for message in provider.request_body["messages"]] == [
        ("user", json.dumps([{"type": "text", "text": "My favorite number is 7."}])),
        ("assistant", json.dumps([{"type": "text", "text": "Noted."}])),
        ("user", json.dumps([{"type": "text", "text": "Double it"}])),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("stored_under", ["litellm_encoded_id", "proxy_issued_id"])
async def test_bridged_next_turn_replays_a_turn_stored_under_the_id_the_client_holds(stored_under: str):
    litellm_encoded_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    proxy_issued_id: Final = "resp_proxy-issued-turn-one"
    stored_turns: Final = _StoredTurns(
        (_stored_mcp_turn(litellm_encoded_id if stored_under == "litellm_encoded_id" else proxy_issued_id),)
    )
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored_turns)):
        await litellm.aresponses(
            model="azure_ai/claude-sonnet-4-6",
            api_base="https://fake-foundry-resource.services.ai.azure.com",
            api_key="fake-api-key",
            input="Double it",
            previous_response_id=litellm_encoded_id,
            client=client,
            **{ADDRESSED_RESPONSE_ID_FIELD: proxy_issued_id},
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert [(message["role"], json.dumps(message["content"])) for message in provider.request_body["messages"]] == [
        ("user", json.dumps([{"type": "text", "text": "My favorite number is 7."}])),
        ("assistant", json.dumps([{"type": "text", "text": "Noted."}])),
        ("user", json.dumps([{"type": "text", "text": "Double it"}])),
    ]


@pytest.mark.asyncio
async def test_bridged_follow_up_after_a_client_tool_search_reports_auto_for_an_allowed_tools_choice_it_dropped():
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        SimpleNamespace(db=_StoredTurns((_stored_mcp_turn(previous_response_id),))),
    ):
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
            previous_response_id=previous_response_id,
            client=client,
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert "tool_choice" not in provider.request_body
    assert isinstance(response, ResponsesAPIResponse)
    assert response.tool_choice == "auto"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_field", ["caller_input", "names_non_string_fields"])
async def test_bridged_follow_up_after_a_client_tool_search_with_a_client_field_named_like_a_bridge_keyword_reaches_the_provider(
    client_field: str,
):
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    stored_turns: Final = _StoredTurns((_stored_mcp_turn(previous_response_id),))
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored_turns)):
        response = await litellm.aresponses(
            model="azure_ai/claude-sonnet-4-6",
            api_base="https://fake-foundry-resource.services.ai.azure.com",
            api_key="fake-api-key",
            input="Double it",
            tools=[{"type": "tool_search", "execution": "client"}],
            previous_response_id=previous_response_id,
            client=client,
            **{client_field: "from the client body"},
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert isinstance(response, ResponsesAPIResponse)


_LONG_NAMESPACE: Final = "mcp__codex_apps__codex_document_control_handler_test"
_LONG_MEMBER: Final = "get_document_tool_schemas"


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
async def test_bridged_turn_logs_the_full_name_of_a_namespace_member_sent_under_a_shortened_name(is_async: bool):
    logging_obj: Final = LiteLLMLoggingObj(
        model="gemini/gemini-flash-latest",
        messages=[],
        stream=False,
        call_type="responses",
        start_time=1.0,
        litellm_call_id="full-tool-names",
        function_id="test",
    )
    captured: Final[dict[str, object]] = {}

    def fake_completion(**kwargs: object) -> None:
        captured.update(kwargs)
        raise _StopForwarding()

    async def fake_acompletion(**kwargs: object) -> None:
        fake_completion(**kwargs)

    namespace: Final = {
        "type": "namespace",
        "name": _LONG_NAMESPACE,
        "description": "",
        "tools": [
            {
                "type": "function",
                "name": _LONG_MEMBER,
                "description": "Reads the document tool schemas.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }

    async def forward() -> None:
        outcome: Final = LiteLLMCompletionTransformationHandler().response_api_handler(
            model="gemini/gemini-flash-latest",
            input="Read the document tool schemas",
            responses_api_request={"tools": [namespace]},
            custom_llm_provider="gemini",
            _is_async=is_async,
            litellm_logging_obj=logging_obj,
        )
        if inspect.iscoroutine(outcome):
            await outcome

    with (
        patch("litellm.completion", fake_completion),  # test-quality-ok: no DI seam; the file stubs this same boundary
        patch("litellm.acompletion", fake_acompletion),  # test-quality-ok: no DI seam; the file stubs this same boundary
        pytest.raises(_StopForwarding),
    ):
        await forward()

    full_name: Final = qualified_member_name(_LONG_NAMESPACE, _LONG_MEMBER)
    sent_names: Final = tuple(tool["function"]["name"] for tool in captured["tools"])
    logged: Final = logging_obj.model_call_details.get(BRIDGED_TOOL_NAMES_KWARG)
    assert full_name not in sent_names
    assert isinstance(logged, BridgedToolNames)
    assert full_names_by_sent_name(logged.tools()) == {
        sent_name: full_name for sent_name in (*sent_names, _LONG_MEMBER)
    }


@pytest.mark.asyncio
async def test_bridged_turn_logs_a_shortened_name_known_only_from_an_earlier_call_in_history():
    """Codex can resume before it lists its MCP tools again, so the earlier call is the only trace of the tool."""
    logging_obj: Final = LiteLLMLoggingObj(
        model="gemini/gemini-flash-latest",
        messages=[],
        stream=False,
        call_type="responses",
        start_time=1.0,
        litellm_call_id="history-tool-names",
        function_id="test",
    )

    def fake_completion(**kwargs: object) -> None:
        raise _StopForwarding()

    history: Final = [
        {"type": "message", "role": "user", "content": "Read the document tool schemas"},
        {
            "type": "function_call",
            "call_id": "call_earlier",
            "name": _LONG_MEMBER,
            "namespace": _LONG_NAMESPACE,
            "arguments": "{}",
        },
        {"type": "function_call_output", "call_id": "call_earlier", "output": "schemas"},
    ]

    with (
        patch("litellm.completion", fake_completion),  # test-quality-ok: no DI seam; the file stubs this same boundary
        pytest.raises(_StopForwarding),
    ):
        LiteLLMCompletionTransformationHandler().response_api_handler(
            model="gemini/gemini-flash-latest",
            input=history,
            responses_api_request={"tools": []},
            custom_llm_provider="gemini",
            _is_async=False,
            litellm_logging_obj=logging_obj,
        )

    logged: Final = logging_obj.model_call_details.get(BRIDGED_TOOL_NAMES_KWARG)
    assert isinstance(logged, BridgedToolNames)
    assert list(logged.tools().values()) == [NamespaceMember(_LONG_NAMESPACE, _LONG_MEMBER)]
    assert all(len(sent_name) <= 64 for sent_name in logged.tools())


@pytest.mark.parametrize(
    ("returned_name", "logged_name"),
    [("run", "exec__run"), ("exec__run", "exec__run"), ("read", "read"), ("shell", "shell")],
    ids=["bare namespace member", "joined namespace member", "deferred top-level tool", "plain function"],
)
def test_spend_logs_name_a_bridged_call_by_the_tool_the_caller_gets_back(returned_name: str, logged_name: str):
    from litellm.proxy.db.spend_log_tool_index import logged_full_tool_names, response_tool_call_names

    logging_obj: Final = LiteLLMLoggingObj(
        model="anthropic/claude-sonnet-5-5",
        messages=[],
        stream=False,
        call_type="responses",
        start_time=1.0,
        litellm_call_id="spend-log-tool-names",
        function_id="test",
    )

    def fake_completion(**kwargs: object) -> None:
        raise _StopForwarding()

    tools: Final = [
        {"type": "namespace", "name": "exec", "description": "", "tools": [{"type": "function", "name": "run"}]},
        {"type": "function", "name": "read", "defer_loading": True, "parameters": {"type": "object"}},
        {"type": "function", "name": "shell", "parameters": {"type": "object"}},
    ]
    with (
        patch("litellm.completion", fake_completion),  # test-quality-ok: no DI seam; the file stubs this same boundary
        pytest.raises(_StopForwarding),
    ):
        LiteLLMCompletionTransformationHandler().response_api_handler(
            model="anthropic/claude-sonnet-5-5",
            input="go",
            responses_api_request={"tools": tools},
            custom_llm_provider="anthropic",
            _is_async=False,
            litellm_logging_obj=logging_obj,
        )

    response: Final = litellm.ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": returned_name, "arguments": "{}"}}
                    ],
                }
            }
        ]
    )
    logged_names: Final = response_tool_call_names(response, logged_full_tool_names(logging_obj.model_call_details))
    assert logged_names == (logged_name,)


def _anthropic_streams_a_call_to_the_first_tool(request: httpx.Request) -> httpx.Response:
    sent_name: Final = json.loads(request.content)["tools"][0]["name"]
    events: Final = (
        {
            "type": "message_start",
            "message": {
                "id": "msg_stream",
                "type": "message",
                "role": "assistant",
                "model": "claude-haiku-5-5",
                "content": [],
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": sent_name, "input": {}},
        },
        {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{}"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )
    body: Final = b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body, request=request)


def _chunk_tool_names(chunk: object) -> tuple[str, ...]:
    if not isinstance(chunk, ModelResponseStream):
        return ()
    tool_calls: Final = chain.from_iterable(choice.delta.tool_calls or () for choice in chunk.choices)
    return tuple(
        tool_call.function.name
        for tool_call in tool_calls
        if isinstance(tool_call, ChatCompletionDeltaToolCall) and tool_call.function.name
    )


class _StreamChunkToolNames(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.names: tuple[str, ...] = ()
        self.saw_tool_call: Final = threading.Event()

    def _record(self, chunk: object) -> None:
        names: Final = _chunk_tool_names(chunk)
        self.names = (*self.names, *names)
        if names:
            self.saw_tool_call.set()

    def log_stream_event(self, kwargs, response_obj, start_time, end_time):
        self._record(response_obj)

    async def async_log_stream_event(self, kwargs, response_obj, start_time, end_time):
        self._record(response_obj)


def test_per_chunk_hooks_of_a_sync_bridged_stream_see_the_full_name_of_a_namespace_member(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder: Final = _StreamChunkToolNames()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    namespace: Final = {
        "type": "namespace",
        "name": _LONG_NAMESPACE,
        "description": "",
        "tools": [
            {
                "type": "function",
                "name": _LONG_MEMBER,
                "description": "Reads the document tool schemas.",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
    }

    stream: Final = litellm.responses(
        model="anthropic/claude-haiku-5-5",
        input="Read the document tool schemas",
        tools=[namespace],
        stream=True,
        api_key="sk-ant-fake",
        client=HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(_anthropic_streams_a_call_to_the_first_tool))),
    )
    events: Final = tuple(stream)

    assert events
    assert recorder.saw_tool_call.wait(timeout=60), "no stream chunk with a tool call reached the per-chunk hooks"
    assert set(recorder.names) == {qualified_member_name(_LONG_NAMESPACE, _LONG_MEMBER)}


@pytest.mark.asyncio
async def test_bridged_follow_up_with_a_lone_user_item_without_content_ends_on_a_user_turn():
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    stored_turns: Final = _StoredTurns((_stored_mcp_turn(previous_response_id),))
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored_turns)):
        await litellm.aresponses(
            model="azure_ai/claude-opus-5-5",
            api_base="https://fake-foundry-resource.services.ai.azure.com",
            api_key="fake-api-key",
            input=[{"role": "user", "content": None}],
            previous_response_id=previous_response_id,
            client=client,
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert [message["role"] for message in provider.request_body["messages"]] == ["user", "assistant", "user"]


_CONVERSE_REPLY: Final = {
    "output": {"message": {"role": "assistant", "content": [{"text": "Eight."}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 12, "outputTokens": 1, "totalTokens": 13},
}


def _stored_converse_turn(request_id: str, input_items: object, reply: str) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "converse-session",
        "model": "us.anthropic.claude-opus-5-5",
        "custom_llm_provider": "bedrock",
        "proxy_server_request": {"input": input_items, "model": "bedrock/us.anthropic.claude-opus-5-5"},
        "response": {
            "id": request_id,
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": reply, "annotations": []}],
                }
            ],
        },
    }


async def _sent_converse_messages(
    stored: tuple[Mapping[str, object], ...],
    input_items: object,
    previous_response_id: str,
    continue_settings: Mapping[str, object],
) -> object:
    provider: Final = _RecordingAnthropicHandler(_CONVERSE_REPLY)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_StoredTurns(stored))):
        await litellm.aresponses(
            model="bedrock/us.anthropic.claude-opus-5-5",
            aws_access_key_id="fake-access-key",
            aws_secret_access_key="fake-secret-key",
            aws_region_name="us-east-1",
            input=input_items,
            previous_response_id=previous_response_id,
            client=client,
            **continue_settings,
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    return provider.request_body["messages"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("modify_params", "continue_settings"),
    [(False, {}), (True, {}), (False, {"user_continue_message": {"role": "user", "content": "Go on."}})],
    ids=["no_continue_setting", "modify_params", "user_continue_message"],
)
@pytest.mark.parametrize(
    "blank_input", [[{"role": "user", "content": None}], "   "], ids=["null_user_item", "whitespace_string"]
)
async def test_converse_turn_after_a_blank_user_turn_resends_the_messages_that_follow_up_sent(
    monkeypatch: pytest.MonkeyPatch,
    modify_params: bool,
    continue_settings: Mapping[str, object],
    blank_input: object,
):
    monkeypatch.setattr(litellm, "modify_params", modify_params)
    first_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="bedrock", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    follow_up_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="bedrock", model_id="deployment-a", response_id="chatcmpl-turn-two"
    )
    first_turn: Final = _stored_converse_turn(first_id, "What is 2+2?", "Four.")

    follow_up_messages: Final = await _sent_converse_messages((first_turn,), blank_input, first_id, continue_settings)
    next_turn_messages: Final = await _sent_converse_messages(
        (first_turn, _stored_converse_turn(follow_up_id, blank_input, "Eight.")),
        [{"role": "user", "content": "Double it."}],
        follow_up_id,
        continue_settings,
    )

    assert isinstance(follow_up_messages, list) and isinstance(next_turn_messages, list)
    assert [message["role"] for message in next_turn_messages] == ["user", "assistant", "user", "assistant", "user"]
    assert next_turn_messages[: len(follow_up_messages)] == follow_up_messages


@pytest.mark.asyncio
async def test_a_compaction_trigger_after_a_stored_turn_asks_for_the_summary_after_that_turn():
    previous_response_id: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
        custom_llm_provider="azure_ai", model_id="deployment-a", response_id="chatcmpl-turn-one"
    )
    stored_turns: Final = _StoredTurns((_stored_mcp_turn(previous_response_id),))
    provider: Final = _RecordingAnthropicHandler(
        {**_ANTHROPIC_MESSAGE_PAYLOAD, "content": [{"type": "text", "text": "<summary>The number is 7.</summary>"}]}
    )
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=stored_turns)):
        response = await litellm.aresponses(
            model="azure_ai/claude-sonnet-4-6",
            api_base="https://fake-foundry-resource.services.ai.azure.com",
            api_key="fake-api-key",
            input=[{"type": "compaction_trigger"}],
            previous_response_id=previous_response_id,
            client=client,
        )

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert [
        (message["role"], json.dumps(message["content"])) for message in provider.request_body["messages"]
    ] == [
        ("user", json.dumps([{"type": "text", "text": "My favorite number is 7."}])),
        ("assistant", json.dumps([{"type": "text", "text": "Noted."}])),
        ("user", json.dumps([{"type": "text", "text": SUMMARY_PROMPT}])),
    ]
    assert isinstance(response, ResponsesAPIResponse)
    assert [(item.type, item.encrypted_content) for item in response.output] == [
        ("compaction", '{"type":"compaction","content":"The number is 7."}')
    ]


_USER_ASKS_FOR_TWO_CALLS: Final = MappingProxyType({"role": "user", "content": "weather in a and b"})


def _function_call(call_id: str) -> Mapping[str, str]:
    return MappingProxyType({"type": "function_call", "call_id": call_id, "name": "get_weather", "arguments": "{}"})


def _function_call_output(call_id: str) -> Mapping[str, str]:
    return MappingProxyType({"type": "function_call_output", "call_id": call_id, "output": "sunny"})


_GROUPED_TOOL_ITEMS: Final = (
    _USER_ASKS_FOR_TWO_CALLS,
    _function_call("call_1"),
    _function_call("call_2"),
    _function_call_output("call_1"),
    _function_call_output("call_2"),
)
_INTERLEAVED_TOOL_ITEMS: Final = (
    _USER_ASKS_FOR_TWO_CALLS,
    _function_call("call_1"),
    _function_call_output("call_1"),
    _function_call("call_2"),
    _function_call_output("call_2"),
)
_FIVE_ITEM_INPUTS: Final = (
    pytest.param(_GROUPED_TOOL_ITEMS, id="grouped-calls-merge-into-one-chat-message"),
    pytest.param(_INTERLEAVED_TOOL_ITEMS, id="interleaved-calls-stay-separate-chat-messages"),
)


@pytest.fixture
def anthropic_messages_route(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, fake_provider_credentials: None
) -> respx.Route:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    litellm.in_memory_llm_clients_cache.flush_cache()
    return respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=_ANTHROPIC_MESSAGE_PAYLOAD)
    )


def _five_items(items: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [dict(item) for item in items]


@pytest.mark.parametrize("items", _FIVE_ITEM_INPUTS)
def test_sync_bridged_responses_past_max_messages_neither_read_nor_write_the_cache(
    monkeypatch: pytest.MonkeyPatch, anthropic_messages_route: respx.Route, items: Sequence[Mapping[str, str]]
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", cache)

    litellm.responses(model="anthropic/claude-sonnet-5-5", input=_five_items(items))
    repeat: Final = litellm.responses(model="anthropic/claude-sonnet-5-5", input=_five_items(items))
    cache.max_messages = None
    litellm.responses(model="anthropic/claude-sonnet-5-5", input=_five_items(items))

    assert repeat._hidden_params.get("cache_hit") is not True
    assert anthropic_messages_route.call_count == 3, "a request past max_messages used the cache"


async def _drain_cache_writes() -> None:
    while _PENDING_CACHE_WRITES:
        await asyncio.gather(*tuple(_PENDING_CACHE_WRITES))


@pytest.mark.asyncio
@pytest.mark.parametrize("items", _FIVE_ITEM_INPUTS)
async def test_async_bridged_responses_past_max_messages_neither_read_nor_write_the_cache(
    monkeypatch: pytest.MonkeyPatch, anthropic_messages_route: respx.Route, items: Sequence[Mapping[str, str]]
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    monkeypatch.setattr(litellm, "cache", cache)

    await litellm.aresponses(model="anthropic/claude-sonnet-5-5", input=_five_items(items))
    await _drain_cache_writes()
    repeat: Final = await litellm.aresponses(model="anthropic/claude-sonnet-5-5", input=_five_items(items))
    await _drain_cache_writes()
    cache.max_messages = None
    await litellm.aresponses(model="anthropic/claude-sonnet-5-5", input=_five_items(items))

    assert repeat._hidden_params.get("cache_hit") is not True
    assert anthropic_messages_route.call_count == 3, "a request past max_messages used the cache"
