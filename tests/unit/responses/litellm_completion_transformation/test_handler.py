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
import json
import sys
from collections.abc import Mapping
from types import MappingProxyType, SimpleNamespace
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.router_utils.router_callbacks.track_deployment_metrics import get_deployment_failures_for_current_minute
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ADDRESSED_RESPONSE_ID_FIELD


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

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        SimpleNamespace(db=_StoredTurns((_stored_turn("resp_turn_one"),))),
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


class _StoredTurns:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows: Final = rows

    async def query_raw(self, query: str, request_ids: list[str]) -> list[dict[str, object]]:
        sessions: Final = {row["session_id"] for row in self.rows if row["request_id"] in request_ids}
        return [dict(row) for row in self.rows if row["session_id"] in sessions]


def _stored_turn(request_id: str) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "turn-one-session",
        "proxy_server_request": {"input": "My favorite number is 7.", "model": "claude-sonnet-4-6"},
        "response": {
            "id": request_id,
            "object": "chat.completion",
            "model": "claude-sonnet-4-6",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Noted."}}],
        },
    }


_TURN_ONE_RESPONSE_ID: Final = "chatcmpl-turn-one"
_TURN_ONE_ID: Final = ResponsesAPIRequestUtils._build_responses_api_response_id(
    custom_llm_provider="azure_ai", model_id="deployment-a", response_id=_TURN_ONE_RESPONSE_ID
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
    asynchronous: bool, provider: _RecordingAnthropicHandler, proxy_fields: Mapping[str, object] = MappingProxyType({})
) -> object:
    request: Final = {
        **proxy_fields,
        "model": "azure_ai/claude-sonnet-4-6",
        "api_base": "https://fake-foundry-resource.services.ai.azure.com",
        "api_key": "fake-api-key",
        "input": "Double it",
        "previous_response_id": _TURN_ONE_ID,
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
            SimpleNamespace(db=_StoredTurns((_turn_stored_without_prompts(_TURN_ONE_RESPONSE_ID),))),
            id="async-prompts_not_stored",
        ),
    ],
)
async def test_bridged_follow_up_without_a_replayable_previous_turn_is_refused_before_the_provider_call(
    asynchronous: bool, prisma_client: SimpleNamespace | None
):
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)

    with patch("litellm.proxy.proxy_server.prisma_client", prisma_client):
        with pytest.raises(litellm.BadRequestError) as refused:
            await _bridged_follow_up(asynchronous, provider)

    assert provider.request_body is None, "the follow-up reached the provider with no conversation history"
    assert (refused.value.status_code, refused.value.param, refused.value.code) == (
        400,
        "previous_response_id",
        "previous_response_not_found",
    )
    assert f"Previous response with id '{_TURN_ONE_ID}' not found." in refused.value.message
    assert "`store_prompts_in_spend_logs: true`" in refused.value.message


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
            {"model_name": "bridged-fallback", "litellm_params": litellm_params, "model_info": {"id": "d2"}},
        ],
        fallbacks=[{"bridged": ["bridged-fallback"]}],
    )

    with patch("litellm.proxy.proxy_server.prisma_client", None):
        for _ in range(6):
            with pytest.raises(litellm.BadRequestError, match="not found"):
                await router.aresponses(model="bridged", input="Double it", previous_response_id=_TURN_ONE_ID)
    await asyncio.sleep(1)

    assert get_deployment_failures_for_current_minute(router, "d1") == 0
    assert get_deployment_failures_for_current_minute(router, "d2") == 0


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
    assert "spend logs" not in refused.value.message


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
    async def query_raw(self, query: str, *request_ids: str) -> list[dict[str, object]]:
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
async def test_bridged_follow_up_replays_the_stored_previous_turn():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        SimpleNamespace(db=_StoredTurns((_stored_turn(_TURN_ONE_RESPONSE_ID),))),
    ):
        await _bridged_follow_up(True, provider)

    assert provider.request_body is not None, "the bridged turn never reached the provider"
    assert [(message["role"], json.dumps(message["content"])) for message in provider.request_body["messages"]] == [
        ("user", json.dumps([{"type": "text", "text": "My favorite number is 7."}])),
        ("assistant", json.dumps([{"type": "text", "text": "Noted."}])),
        ("user", json.dumps([{"type": "text", "text": "Double it"}])),
    ]


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
