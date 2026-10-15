import json
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from pydantic import BaseModel

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import Message, ModelResponse


def test_response_api_handler_merges_metadata_and_service_tier_without_error():
    """Sync path must merge kwargs like async; double-splat raises TypeError."""
    handler = LiteLLMCompletionTransformationHandler()

    with patch("litellm.completion", new_callable=MagicMock) as mock_completion:
        mock_completion.return_value = ModelResponse(
            id="id", created=0, model="test", object="chat.completion", choices=[]
        )
        handler.response_api_handler(
            model="test",
            input="hi",
            responses_api_request={},
            metadata={"trace": "abc"},
            service_tier="auto",
        )
        assert mock_completion.call_count == 1
        assert mock_completion.call_args.kwargs["metadata"] == {"trace": "abc"}
        assert mock_completion.call_args.kwargs["service_tier"] == "auto"


@pytest.mark.asyncio
async def test_async_response_api_handler_merges_trace_id_without_error():
    handler = LiteLLMCompletionTransformationHandler()

    async def fake_session_handler(
        previous_response_id: str,
        litellm_completion_request: dict[str, object],
        instructions: str | None = None,
        request_kwargs: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        litellm_completion_request["litellm_trace_id"] = "session-trace"
        return litellm_completion_request

    with patch.object(
        LiteLLMCompletionResponsesConfig,
        "async_responses_api_session_handler",
        side_effect=fake_session_handler,
    ):
        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = ModelResponse(
                id="id", created=0, model="test", object="chat.completion", choices=[]
            )
            await handler.async_response_api_handler(
                litellm_completion_request={"model": "test"},
                request_input="hi",
                responses_api_request={"previous_response_id": "123"},
                litellm_trace_id="original-trace",
            )

            assert mock_acompletion.call_count == 1
            assert mock_acompletion.call_args.kwargs["litellm_trace_id"] == "session-trace"


@pytest.mark.asyncio
async def test_aresponses_forwards_timeout_to_acompletion():
    """Regression test: timeout passed to aresponses() must reach acompletion()
    on the completion transformation path (Anthropic, Bedrock, Vertex etc.).

    Previously, `timeout` was a named param of `responses()` but was NOT
    forwarded to `litellm_completion_transformation_handler.response_api_handler`,
    so it was silently dropped — `Router(timeout=N)` was a no-op for Anthropic
    and similar providers, with calls falling back to the provider SDK default
    (~600s for Anthropic).
    """
    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = ModelResponse(
            id="id",
            created=0,
            model="anthropic/claude-sonnet-4-5",
            object="chat.completion",
            choices=[],
        )

        await litellm.aresponses(
            model="anthropic/claude-sonnet-4-5",
            input="hello",
            timeout=42,
            api_key="sk-ant-fake",
        )

    assert mock_acompletion.call_count == 1
    forwarded_timeout = mock_acompletion.call_args.kwargs.get("timeout")
    assert forwarded_timeout == 42, (
        f"timeout was not forwarded to acompletion (got {forwarded_timeout!r}); "
        "this means Router(timeout=N) silently fails for providers on the "
        "completion transformation path."
    )


class _FakeSpendLogsDB:
    def __init__(self, spend_logs: Sequence[Mapping[str, object]]) -> None:
        self._spend_logs = spend_logs

    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        return self._spend_logs


class _FakePrismaClient:
    def __init__(self, spend_logs: Sequence[Mapping[str, object]]) -> None:
        self.db = _FakeSpendLogsDB(spend_logs)


class _AnthropicBlock(BaseModel, frozen=True):
    type: str
    id: str | None = None
    tool_use_id: str | None = None
    text: str | None = None


class _AnthropicMessage(BaseModel, frozen=True):
    role: str
    content: tuple[_AnthropicBlock, ...]


class _AnthropicRequest(BaseModel, frozen=True):
    messages: tuple[_AnthropicMessage, ...]
    system: tuple[_AnthropicBlock, ...]


class _RecordingAnthropicMessages:
    def __init__(self) -> None:
        self.request: _AnthropicRequest | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.request = _AnthropicRequest.model_validate_json(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg_second_turn",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-5-5",
                "content": [{"type": "text", "text": "Il fait 47C."}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            request=request,
        )


_MODEL: Final = "anthropic/claude-sonnet-5-5"
_FIRST_TURN_INSTRUCTIONS: Final = "Be terse."
_FIRST_TURN: Final = {
    "request_id": "chatcmpl-first-turn",
    "call_type": "aresponses",
    "session_id": "session-1",
    "proxy_server_request": {
        "model": _MODEL,
        "input": "What is the weather in Tokyo?",
        "instructions": _FIRST_TURN_INSTRUCTIONS,
    },
    "response": {
        "id": "chatcmpl-first-turn",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "toolu_weather",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                        }
                    ],
                },
            }
        ],
    },
}
_EXPECTED_MESSAGES: Final = [
    ("user", [("text", "What is the weather in Tokyo?")]),
    ("assistant", [("tool_use", "toolu_weather")]),
    ("user", [("tool_result", "toolu_weather")]),
]


async def _continue_first_turn_with_tool_output(instructions: str | None) -> _AnthropicRequest:
    anthropic: Final = _RecordingAnthropicMessages()
    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([_FIRST_TURN])):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="chatcmpl-first-turn",
            input=[{"type": "function_call_output", "call_id": "toolu_weather", "output": "47C"}],
            instructions=instructions,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                }
            ],
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )
    assert anthropic.request is not None
    return anthropic.request


def _message_shapes(request: _AnthropicRequest) -> list[tuple[str, list[tuple[str, str | None]]]]:
    return [
        (message.role, [(block.type, block.id or block.tool_use_id or block.text) for block in message.content])
        for message in request.messages
    ]


@pytest.mark.asyncio
async def test_previous_response_id_tool_output_with_new_instructions_builds_valid_anthropic_request() -> None:
    """
    A continuation that resends `instructions` must not land a system message between the replayed
    tool_use and its tool_result, and the previous turn's instructions do not carry over (OpenAI semantics)
    """
    request: Final = await _continue_first_turn_with_tool_output(instructions="Answer in French.")

    assert _message_shapes(request) == _EXPECTED_MESSAGES
    assert [block.text for block in request.system] == ["Answer in French."]


@pytest.mark.asyncio
async def test_previous_response_id_tool_output_without_instructions_keeps_the_previous_turns() -> None:
    """
    A continuation that sends no `instructions` keeps the previous turn's instructions as the system prompt
    """
    request: Final = await _continue_first_turn_with_tool_output(instructions=None)

    assert _message_shapes(request) == _EXPECTED_MESSAGES
    assert [block.text for block in request.system] == [_FIRST_TURN_INSTRUCTIONS]


def _second_turn(instructions: str | None) -> Mapping[str, object]:
    return {
        "request_id": "chatcmpl-second-turn",
        "call_type": "aresponses",
        "session_id": "session-1",
        "proxy_server_request": {
            "model": _MODEL,
            "input": [{"type": "function_call_output", "call_id": "toolu_weather", "output": "47C"}],
            **({"instructions": instructions} if instructions else {}),
        },
        "response": {
            "id": "chatcmpl-second-turn",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "47C in Tokyo."}}
            ],
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("second_turn_instructions", "expected_system"),
    [("Answer in French.", "Answer in French."), (None, _FIRST_TURN_INSTRUCTIONS)],
)
async def test_previous_response_id_without_instructions_carries_the_latest_ones_ahead_of_a_tool_roundtrip(
    second_turn_instructions: str | None, expected_system: str
) -> None:
    anthropic: Final = _RecordingAnthropicMessages()
    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        _FakePrismaClient([_FIRST_TURN, _second_turn(second_turn_instructions)]),
    ):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="chatcmpl-second-turn",
            input="What about Osaka?",
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    assert anthropic.request is not None
    assert _message_shapes(anthropic.request) == [
        *_EXPECTED_MESSAGES,
        ("assistant", [("text", "47C in Tokyo.")]),
        ("user", [("text", "What about Osaka?")]),
    ]
    assert [block.text for block in anthropic.request.system] == [expected_system]


_SEARCH_CALL: Final = {
    "id": "srvtoolu_search",
    "type": "function",
    "function": {"name": "tool_search_tool_regex", "arguments": '{"pattern": "weather"}'},
}
_WEATHER_CALL: Final = {
    "id": "toolu_weather",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
}


def _search_turn(tool_calls: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    return {
        "request_id": "chatcmpl-search-turn",
        "call_type": "aresponses",
        "session_id": "session-search",
        "proxy_server_request": {"model": _MODEL, "input": "What is the weather in Tokyo?"},
        "response": {
            "id": "chatcmpl-search-turn",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {"role": "assistant", "content": None, "tool_calls": list(tool_calls)},
                }
            ],
        },
    }


async def _follow_up_messages(
    tool_calls: Sequence[Mapping[str, object]], model: str, custom_llm_provider: str
) -> Sequence[object]:
    return await _follow_up_messages_after(_search_turn(tool_calls), model, custom_llm_provider)


async def _follow_up_messages_after(
    spend_log: Mapping[str, object], model: str, custom_llm_provider: str
) -> Sequence[object]:
    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([spend_log])):
        request: Final = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id="chatcmpl-search-turn",
            litellm_completion_request={
                "model": model,
                "custom_llm_provider": custom_llm_provider,
                "messages": [{"role": "user", "content": "And in Osaka?"}],
            },
            instructions=None,
        )
    return request["messages"]


def _replayed_tool_call_ids(messages: Sequence[object]) -> list[str]:
    assistant_turns: Final = [message for message in messages if isinstance(message, Message)]
    return [tool_call.id for turn in assistant_turns for tool_call in turn.tool_calls or ()]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expected_tool_call_ids"),
    [
        ("mistral/mistral-small-latest", "mistral", ["toolu_weather"]),
        ("gemini/gemini-2.5-flash", "gemini", ["toolu_weather"]),
        ("databricks/databricks-claude-sonnet-4-5", "databricks", ["toolu_weather"]),
        (_MODEL, "anthropic", ["srvtoolu_search", "toolu_weather"]),
    ],
)
async def test_previous_response_id_keeps_a_stored_claude_search_call_only_on_an_anthropic_route(
    model: str, custom_llm_provider: str, expected_tool_call_ids: list[str]
) -> None:
    """
    A follow-up that routes a Claude turn's history elsewhere must not replay Anthropic's server-side
    search call as a client tool call, which Mistral answers 400 "Unexpected tool call id"
    """
    messages: Final = await _follow_up_messages([_SEARCH_CALL, _WEATHER_CALL], model, custom_llm_provider)

    assert _replayed_tool_call_ids(messages) == expected_tool_call_ids


@pytest.mark.asyncio
async def test_previous_response_id_drops_a_claude_turn_that_held_only_a_search_call_off_anthropic() -> None:
    messages: Final = await _follow_up_messages([_SEARCH_CALL], "mistral/mistral-small-latest", "mistral")

    assert [message["role"] for message in messages] == ["user", "user"]


def _replayed_search_turn(search_answered: bool) -> Mapping[str, object]:
    search_result: Final = (
        [{"type": "function_call_output", "call_id": "srvtoolu_search", "output": "get_weather"}]
        if search_answered
        else []
    )
    return {
        **_search_turn([]),
        "proxy_server_request": {
            "model": _MODEL,
            "input": [
                {"role": "user", "content": "What is the weather in Tokyo?"},
                {
                    "type": "function_call",
                    "call_id": "srvtoolu_search",
                    "name": "tool_search_tool_regex",
                    "arguments": '{"pattern": "weather"}',
                },
                *search_result,
                {
                    "type": "function_call",
                    "call_id": "toolu_weather",
                    "name": "get_weather",
                    "arguments": '{"city": "Tokyo"}',
                },
                {"type": "function_call_output", "call_id": "toolu_weather", "output": '{"temp_c": 18}'},
            ],
        },
        "response": {
            "id": "chatcmpl-search-turn",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "It is 18C in Tokyo."},
                }
            ],
        },
    }


def _tool_call_ids_of(message: object) -> Sequence[str]:
    if isinstance(message, Message):
        return [tool_call.id or "" for tool_call in message.tool_calls or ()]
    if isinstance(message, dict):
        return [tool_call["id"] for tool_call in message.get("tool_calls") or ()]
    return []


def _history_tool_ids(messages: Sequence[object]) -> tuple[Sequence[str], Sequence[str]]:
    called: Final = list(chain.from_iterable(map(_tool_call_ids_of, messages)))
    answered: Final = [
        message["tool_call_id"] for message in messages if isinstance(message, dict) and message["role"] == "tool"
    ]
    return called, answered


@pytest.mark.asyncio
@pytest.mark.parametrize("search_answered", [False, True])
@pytest.mark.parametrize(
    ("model", "custom_llm_provider", "expect_search"),
    [("mistral/mistral-small-latest", "mistral", False), (_MODEL, "anthropic", True)],
)
async def test_previous_response_id_drops_a_client_replayed_search_call_only_off_anthropic(
    model: str, custom_llm_provider: str, expect_search: bool, search_answered: bool
) -> None:
    """
    A client can replay Claude's search call as a ``function_call`` input item with its ``srvtoolu_`` id. A later
    turn sent elsewhere must not carry it, which OpenAI-compatible providers answer 400 "did not have response messages"
    """
    messages: Final = await _follow_up_messages_after(
        _replayed_search_turn(search_answered), model, custom_llm_provider
    )

    called, answered = _history_tool_ids(messages)
    search_ids: Final = ["srvtoolu_search"] if expect_search else []
    assert called == [*search_ids, "toolu_weather"]
    assert answered == [*(search_ids if search_answered else []), "toolu_weather"]


def _anthropic_sse(*events: Mapping[str, object]) -> bytes:
    return b"".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def test_streamed_responses_over_anthropic_emit_the_responses_event_sequence(respx_mock: respx.MockRouter) -> None:
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_anthropic_sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_responses_stream",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-4-5",
                        "content": [],
                        "usage": {"input_tokens": 9, "output_tokens": 1},
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Argentina "}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "won."}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
                {"type": "message_stop"},
            ),
        )
    )

    stream: Final = litellm.responses(
        model="anthropic/claude-sonnet-4-5",
        input="Who won the World Cup in 2022?",
        max_output_tokens=100,
        stream=True,
        api_key="mock_api_key",
    )
    events: Final = tuple(stream)
    event_types: Final = tuple(
        event.type for index, event in enumerate(events) if index == 0 or event.type != events[index - 1].type
    )
    deltas: Final = "".join(event.delta for event in events if event.type == ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA)
    completed: Final = events[-1]

    assert event_types == (
        ResponsesAPIStreamEvents.RESPONSE_CREATED,
        ResponsesAPIStreamEvents.RESPONSE_IN_PROGRESS,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED,
        ResponsesAPIStreamEvents.CONTENT_PART_ADDED,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DELTA,
        ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE,
        ResponsesAPIStreamEvents.CONTENT_PART_DONE,
        ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE,
        ResponsesAPIStreamEvents.RESPONSE_COMPLETED,
    )
    assert deltas == "Argentina won."
    assert completed.response.status == "completed"
    assert completed.response.output[0].content[0].text == "Argentina won."


_ANTHROPIC_MESSAGES_URL: Final = "https://api.anthropic.com/v1/messages"
_COMPACTION_BLOCK: Final = {"type": "compaction", "content": "The user pasted a long document and asked about it."}
_FIRST_INPUT: Final = ({"role": "user", "content": "What is the first word of the document?"},)
_NEXT_INPUT: Final = ({"role": "user", "content": "And the last word?"},)
_COMPACT_AT_50K: Final = ({"type": "compaction", "compact_threshold": 50000},)


def _anthropic_message(*content: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_compacted",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": list(content),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def _replay(output: Sequence[BaseModel]) -> list[object]:
    return [*_FIRST_INPUT, *(item.model_dump(exclude_none=True) for item in output), *_NEXT_INPUT]


def _sent_assistant_content(route: respx.Route) -> list[object]:
    sent: Final = json.loads(route.calls.last.request.content)
    return next(message["content"] for message in sent["messages"] if message["role"] == "assistant")


def test_responses_over_anthropic_return_the_compaction_block_and_send_it_back(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        side_effect=[
            _anthropic_message(_COMPACTION_BLOCK, {"type": "text", "text": "The."}),
            _anthropic_message({"type": "text", "text": "dog."}),
        ]
    )

    first: Final = litellm.responses(
        model=_MODEL, input=list(_FIRST_INPUT), context_management=list(_COMPACT_AT_50K), api_key="mock_api_key"
    )
    litellm.responses(
        model=_MODEL, input=_replay(first.output), context_management=list(_COMPACT_AT_50K), api_key="mock_api_key"
    )

    assert [item.type for item in first.output] == ["compaction", "message"]
    assert _sent_assistant_content(route) == [_COMPACTION_BLOCK, {"type": "text", "text": "The."}]


def test_streamed_responses_over_anthropic_emit_the_compaction_item_and_send_it_back(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        side_effect=[
            httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_anthropic_sse(
                    {
                        "type": "message_start",
                        "message": {
                            "id": "msg_compacted_stream",
                            "type": "message",
                            "role": "assistant",
                            "model": "claude-sonnet-5-5",
                            "content": [],
                            "usage": {"input_tokens": 9, "output_tokens": 1},
                        },
                    },
                    {"type": "content_block_start", "index": 0, "content_block": {"type": "compaction", "content": None}},
                    {
                        "type": "content_block_delta",
                        "index": 0,
                        "delta": {"type": "compaction_delta", "content": _COMPACTION_BLOCK["content"]},
                    },
                    {"type": "content_block_stop", "index": 0},
                    {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
                    {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The."}},
                    {"type": "content_block_stop", "index": 1},
                    {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
                    {"type": "message_stop"},
                ),
            ),
            _anthropic_message({"type": "text", "text": "dog."}),
        ]
    )

    events: Final = tuple(
        litellm.responses(
            model=_MODEL,
            input=list(_FIRST_INPUT),
            context_management=list(_COMPACT_AT_50K),
            stream=True,
            api_key="mock_api_key",
        )
    )
    item_events: Final = tuple(
        event
        for event in events
        if event.type in (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE)
    )
    completed: Final = events[-1].response
    litellm.responses(
        model=_MODEL, input=_replay(completed.output), context_management=list(_COMPACT_AT_50K), api_key="mock_api_key"
    )

    assert [(event.type, event.output_index, event.item.type) for event in item_events] == [
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, 0, "compaction"),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, 0, "compaction"),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, 1, "message"),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, 1, "message"),
    ]
    assert [(item.type, getattr(item, "encrypted_content", None)) for item in completed.output[:1]] == [
        ("compaction", item_events[1].item.encrypted_content)
    ]
    assert completed.output[0].id == item_events[1].item.id
    assert _sent_assistant_content(route) == [_COMPACTION_BLOCK, {"type": "text", "text": "The."}]


def test_responses_over_anthropic_send_a_compaction_item_as_summary_text_when_the_request_does_not_compact(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message({"type": "text", "text": "dog."})
    )
    compaction_item: Final = {
        "type": "compaction",
        "id": "cmp_earlier",
        "encrypted_content": json.dumps(_COMPACTION_BLOCK),
    }

    litellm.responses(model=_MODEL, input=[compaction_item, *_NEXT_INPUT], api_key="mock_api_key")

    assert _sent_assistant_content(route) == [{"type": "text", "text": _COMPACTION_BLOCK["content"]}]
