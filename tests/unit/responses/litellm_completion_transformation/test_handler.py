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
from types import SimpleNamespace
from typing import Final
from unittest.mock import patch

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
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


class _StoredSession:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows: Final = rows

    async def query_raw(self, query: str, *args: object) -> list[dict[str, object]]:
        return [dict(row) for row in self.rows]


_STORED_FIRST_TURN: Final = {
    "request_id": "chatcmpl-turn-one",
    "call_type": "aresponses",
    "session_id": "turn-one-session",
    "proxy_server_request": {"input": "My favorite number is 7.", "model": "claude-opus-5-5"},
    "response": {
        "id": "chatcmpl-turn-one",
        "object": "chat.completion",
        "created": 1748575031,
        "model": "claude-opus-5-5",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Noted."}, "finish_reason": "stop"}],
    },
}


@pytest.mark.asyncio
async def test_bridged_follow_up_with_a_lone_user_item_without_content_ends_on_a_user_turn():
    provider: Final = _RecordingAnthropicHandler(_ANTHROPIC_MESSAGE_PAYLOAD)
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(provider))

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_StoredSession((_STORED_FIRST_TURN,)))):
        await litellm.aresponses(
            model="azure_ai/claude-opus-5-5",
            api_base="https://fake-foundry-resource.services.ai.azure.com",
            api_key="fake-api-key",
            input=[{"role": "user", "content": None}],
            previous_response_id="chatcmpl-turn-one",
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
            "object": "chat.completion",
            "created": 1748575031,
            "model": "claude-opus-5-5",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}],
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

    with patch("litellm.proxy.proxy_server.prisma_client", SimpleNamespace(db=_StoredSession(stored))):
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
    first_turn: Final = _stored_converse_turn("chatcmpl-turn-one", "What is 2+2?", "Four.")

    follow_up_messages: Final = await _sent_converse_messages(
        (first_turn,), blank_input, "chatcmpl-turn-one", continue_settings
    )
    next_turn_messages: Final = await _sent_converse_messages(
        (first_turn, _stored_converse_turn("chatcmpl-turn-two", blank_input, "Eight.")),
        [{"role": "user", "content": "Double it."}],
        "chatcmpl-turn-two",
        continue_settings,
    )

    assert isinstance(follow_up_messages, list) and isinstance(next_turn_messages, list)
    assert [message["role"] for message in next_turn_messages] == ["user", "assistant", "user", "assistant", "user"]
    assert next_turn_messages[: len(follow_up_messages)] == follow_up_messages
