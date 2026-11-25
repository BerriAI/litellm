import json
from collections.abc import Mapping, Sequence
from itertools import chain
from typing import Final, cast
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from pydantic import BaseModel

import litellm
from litellm.litellm_core_utils.redact_messages import perform_redaction
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.openai.responses.guardrail_translation.handler import OpenAIResponsesHandler
from litellm.proxy.guardrails.guardrail_hooks.litellm_content_filter.content_filter import ContentFilterGuardrail
from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)
from litellm.responses.litellm_completion_transformation.tool_search import anthropic_tool_search_chat_tool
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.types.guardrails import BlockedWord, ContentFilterAction
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES
from litellm.types.llms.openai import ResponsesAPIStreamEvents
from litellm.types.utils import Message, ModelResponse
from tests.unit.llms.anthropic.anthropic_deferred_mcp_result import (
    CLIENT_TOOL_CALL_ID,
    CLIENT_TOOL_USE,
    DEFERRED_MCP_CALL_ID,
    DEFERRED_MCP_FOLLOW_UP,
    DEFERRED_MCP_PROMPT,
    DEFERRED_MCP_TOOL_RESULT,
    LOCAL_TIME_OUTPUT,
    LOCAL_TIME_TOOL,
    MCP_TOOL_ARGUMENTS,
    MCP_TOOL_USE,
    TURN_ONE_MESSAGE,
    TURN_ONE_STREAM_EVENTS,
    TURN_THREE_MESSAGE,
    TURN_TWO_MESSAGE,
    TURN_TWO_STREAM_EVENTS,
    TURN_TWO_TEXT,
    WIKI_PAGES,
    anthropic_sse,
)
from tests.unit.proxy.conftest import httpx_transport


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
        previous_response_client_ids: tuple[str, ...] = (),
        request_input: str | None = None,
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
                previous_response_id="123",
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


_ANTHROPIC_SEARCH_TOOL: Final = anthropic_tool_search_chat_tool()


async def _follow_up_messages(
    tool_calls: Sequence[Mapping[str, object]], model: str, custom_llm_provider: str
) -> Sequence[object]:
    return await _follow_up_messages_after(_search_turn(tool_calls), model, custom_llm_provider)


async def _follow_up_messages_after(
    spend_log: Mapping[str, object],
    model: str,
    custom_llm_provider: str,
    tools: Sequence[Mapping[str, object]] = (_ANTHROPIC_SEARCH_TOOL,),
) -> Sequence[object]:
    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([spend_log])):
        request: Final = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
            previous_response_id="chatcmpl-search-turn",
            litellm_completion_request={
                "model": model,
                "custom_llm_provider": custom_llm_provider,
                "messages": [{"role": "user", "content": "And in Osaka?"}],
                **({"tools": [dict(tool) for tool in tools]} if tools else {}),
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
    [
        ("mistral/mistral-small-latest", "mistral", False),
        ("anthropic/claude-3-7-sonnet-20250219", "anthropic", True),
        ("azure_ai/claude-deployment", "azure_ai", True),
        (_MODEL, "anthropic", True),
    ],
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


@pytest.mark.parametrize(
    "non_object_frame",
    [
        b"data: 5\n\n",
        b"event: ping\ndata: []\n\n",
        b'data: "x"\n\n',
        b"data: " + b'{"a": ' * 100_000 + b"1" + b"}" * 100_000 + b"\n\n",
    ],
    ids=["number", "array", "string", "deep-object"],
)
def test_streamed_responses_over_anthropic_skip_a_data_line_that_is_not_an_object(
    respx_mock: respx.MockRouter, non_object_frame: bytes
) -> None:
    respx_mock.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_anthropic_sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_responses_non_object",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-sonnet-5-5",
                        "content": [],
                        "usage": {"input_tokens": 9, "output_tokens": 1},
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            )
            + non_object_frame
            + _anthropic_sse(
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Argentina won."}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
                {"type": "message_stop"},
            ),
        )
    )

    events: Final = tuple(
        litellm.responses(
            model="anthropic/claude-sonnet-5-5",
            input="Who won the World Cup in 2022?",
            max_output_tokens=100,
            stream=True,
            api_key="mock_api_key",
        )
    )

    assert events[-1].type == ResponsesAPIStreamEvents.RESPONSE_COMPLETED
    assert events[-1].response.output[0].content[0].text == "Argentina won."


_ANTHROPIC_MESSAGES_URL: Final = "https://api.anthropic.com/v1/messages"


def _anthropic_message(*content: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_tool_search",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": list(content),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


def _sent_assistant_content(route: respx.Route) -> list[object]:
    sent: Final = json.loads(route.calls.last.request.content)
    return next(message["content"] for message in sent["messages"] if message["role"] == "assistant")


class _RecordingRawAnthropicMessages:
    def __init__(self) -> None:
        self.body: str = ""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.body = request.content.decode()
        return _anthropic_message({"type": "text", "text": "dog."})


_WEATHER_TOOL: Final = {
    "type": "function",
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    "defer_loading": True,
}
_SEARCHABLE_TOOLS: Final = ({"type": "tool_search"}, _WEATHER_TOOL)
_WEATHER_QUESTION: Final = ({"role": "user", "content": "What is the weather in Paris?"},)
_SEARCH_USE: Final = {
    "type": "server_tool_use",
    "id": "srvtoolu_01Search",
    "name": "tool_search_tool_regex",
    "input": {"pattern": "weather"},
}
_SEARCH_RESULT: Final = {
    "type": "tool_search_tool_result",
    "tool_use_id": "srvtoolu_01Search",
    "content": {
        "type": "tool_search_tool_search_result",
        "tool_references": [{"type": "tool_reference", "tool_name": "get_weather"}],
    },
}
_WEATHER_USE: Final = {"type": "tool_use", "id": "toolu_01Weather", "name": "get_weather", "input": {"city": "Paris"}}
_WEATHER_ANSWER: Final = {"type": "function_call_output", "call_id": "toolu_01Weather", "output": "18C and sunny"}


_TOOL_NAMES_ON_EACH_SIDE: Final = pytest.mark.parametrize(
    ("caller_name", "claude_name"),
    [("get_weather", "get_weather"), ("weather.get", "weather_get")],
    ids=["valid_name", "name_anthropic_rewrites"],
)


def _search_result_finding(tool_name: str) -> Mapping[str, object]:
    return {
        **_SEARCH_RESULT,
        "content": {
            "type": "tool_search_tool_search_result",
            "tool_references": [{"type": "tool_reference", "tool_name": tool_name}],
        },
    }


@_TOOL_NAMES_ON_EACH_SIDE
def test_responses_over_anthropic_return_the_tool_search_output_and_send_the_search_back_without_a_stored_turn(
    respx_mock: respx.MockRouter, caller_name: str, claude_name: str
) -> None:
    tools: Final = [{"type": "tool_search"}, {**_WEATHER_TOOL, "name": caller_name}]
    search_result: Final = _search_result_finding(claude_name)
    weather_use: Final = {**_WEATHER_USE, "name": claude_name}
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message(_SEARCH_USE, search_result, weather_use)
    )

    first: Final = litellm.responses(model=_MODEL, input=list(_WEATHER_QUESTION), tools=tools, api_key="mock_api_key")
    litellm.responses(
        model=_MODEL,
        input=[*_WEATHER_QUESTION, *(item.model_dump(exclude_none=True) for item in first.output), _WEATHER_ANSWER],
        tools=tools,
        api_key="mock_api_key",
    )

    assert [item.type for item in first.output] == ["tool_search_call", "tool_search_output", "function_call"]
    search_output: Final = first.output[1]
    assert (search_output.id, search_output.call_id) == ("tso_srvtoolu_01Search", "srvtoolu_01Search")
    assert (search_output.execution, search_output.status) == ("server", "completed")
    assert [dict(tool) for tool in search_output.tools] == tools[1:]
    assert first.output[2].name == caller_name
    assert _sent_assistant_content(route) == [_SEARCH_USE, search_result, weather_use]



@pytest.mark.parametrize(
    ("search_tools", "search_use"),
    [
        pytest.param((), _SEARCH_USE, id="regex"),
        pytest.param(
            ({"type": "tool_search_tool_bm25_20251119", "name": "tool_search_tool_bm25"},),
            {**_SEARCH_USE, "name": "tool_search_tool_bm25", "input": {"query": "weather"}},
            id="bm25",
        ),
    ],
)
def test_responses_over_anthropic_send_a_search_back_under_the_anthropic_search_tool_that_ran_it(
    respx_mock: respx.MockRouter, search_tools: tuple[Mapping[str, object], ...], search_use: Mapping[str, object]
) -> None:
    tools: Final = [{"type": "tool_search"}, *search_tools, _WEATHER_TOOL]
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message(search_use, _SEARCH_RESULT, _WEATHER_USE)
    )

    first: Final = litellm.responses(model=_MODEL, input=list(_WEATHER_QUESTION), tools=tools, api_key="mock_api_key")
    litellm.responses(
        model=_MODEL,
        input=[*_WEATHER_QUESTION, *(item.model_dump(exclude_none=True) for item in first.output), _WEATHER_ANSWER],
        tools=tools,
        api_key="mock_api_key",
    )

    assert _sent_assistant_content(route) == [search_use, _SEARCH_RESULT, _WEATHER_USE]

def _tool_search_stream(*events: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_anthropic_sse(
            {
                "type": "message_start",
                "message": {
                    "id": "msg_tool_search_stream",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5-5",
                    "content": [],
                    "usage": {"input_tokens": 9, "output_tokens": 1},
                },
            },
            *events,
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 4}},
            {"type": "message_stop"},
        ),
    )


def _streamed_tool_use(index: int, block: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return (
        {"type": "content_block_start", "index": index, "content_block": {**block, "input": {}}},
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
        },
        {"type": "content_block_stop", "index": index},
    )


@_TOOL_NAMES_ON_EACH_SIDE
def test_streamed_responses_over_anthropic_emit_the_tool_search_output_in_order_at_its_completed_position(
    respx_mock: respx.MockRouter, caller_name: str, claude_name: str
) -> None:
    tools: Final = [{"type": "tool_search"}, {**_WEATHER_TOOL, "name": caller_name}]
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_tool_search_stream(
            *_streamed_tool_use(0, _SEARCH_USE),
            {"type": "content_block_start", "index": 1, "content_block": _search_result_finding(claude_name)},
            {"type": "content_block_stop", "index": 1},
            *_streamed_tool_use(2, {**_WEATHER_USE, "name": claude_name}),
        )
    )

    events: Final = tuple(
        litellm.responses(model=_MODEL, input=list(_WEATHER_QUESTION), tools=tools, stream=True, api_key="mock")
    )

    completed: Final = events[-1].response.output
    added: Final = {
        event.item.id: event.output_index
        for event in events
        if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    }
    done: Final = [event for event in events if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE]
    assert [item.type for item in completed] == ["tool_search_call", "tool_search_output", "function_call"]
    assert [event.output_index for event in events if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED] == [
        0,
        1,
        2,
    ]
    assert [(event.output_index, event.item.id) for event in done] == [
        (index, item.id) for index, item in enumerate(completed)
    ]
    assert {event.item.id: event.output_index for event in done} == added
    search_output: Final = next(event.item for event in done if event.item.type == "tool_search_output")
    assert search_output.model_dump() == completed[1].model_dump()
    assert (search_output.call_id, search_output.status) == ("srvtoolu_01Search", "completed")
    assert [dict(tool) for tool in search_output.tools] == tools[1:]
    assert completed[2].name == caller_name


_STATELESS_SEARCH_TURN: Final = {
    **_search_turn([]),
    "proxy_server_request": {
        "model": _MODEL,
        "input": [
            *_WEATHER_QUESTION,
            {
                "type": "tool_search_call",
                "id": "tsc_srvtoolu_01Search",
                "call_id": "srvtoolu_01Search",
                "arguments": {"pattern": "weather"},
                "execution": "server",
                "status": "completed",
            },
            {
                "type": "tool_search_output",
                "id": "tso_srvtoolu_01Search",
                "call_id": "srvtoolu_01Search",
                "execution": "server",
                "status": "completed",
                "tools": [_WEATHER_TOOL],
            },
            {
                "type": "function_call",
                "id": "fc_toolu_01Weather",
                "call_id": "toolu_01Weather",
                "name": "get_weather",
                "arguments": '{"city": "Paris"}',
            },
            _WEATHER_ANSWER,
        ],
    },
    "response": {
        "id": "chatcmpl-search-turn",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [
            {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "It is 18C in Paris."}}
        ],
    },
}


@pytest.mark.asyncio
async def test_previous_response_id_sends_back_a_search_the_stored_turn_got_as_input_items() -> None:
    anthropic: Final = _RecordingRawAnthropicMessages()

    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([_STATELESS_SEARCH_TURN])):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="chatcmpl-search-turn",
            input="And in Osaka?",
            tools=list(_SEARCHABLE_TOOLS),
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    messages: Final = json.loads(anthropic.body)["messages"]
    assert next(message["content"] for message in messages if message["role"] == "assistant") == [
        _SEARCH_USE,
        _SEARCH_RESULT,
        _WEATHER_USE,
    ]


@pytest.mark.asyncio
async def test_previous_response_id_keeps_a_stored_search_when_the_follow_up_sends_no_search_tool() -> None:
    anthropic: Final = _RecordingRawAnthropicMessages()
    loaded_weather_tool: Final = {key: value for key, value in _WEATHER_TOOL.items() if key != "defer_loading"}

    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([_STATELESS_SEARCH_TURN])):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="chatcmpl-search-turn",
            input="And in Osaka?",
            tools=[loaded_weather_tool],
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    messages: Final = json.loads(anthropic.body)["messages"]
    assert next(message["content"] for message in messages if message["role"] == "assistant") == [
        _SEARCH_USE,
        _SEARCH_RESULT,
        _WEATHER_USE,
    ]


_SERVER_HELD_SEARCH_TURN: Final = {
    **_search_turn([]),
    "proxy_server_request": {"model": _MODEL, "input": list(_WEATHER_QUESTION)},
    "response": {
        "id": "chatcmpl-search-turn",
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
                            "id": "srvtoolu_01Search",
                            "type": "function",
                            "function": {"name": "tool_search_tool_regex", "arguments": '{"pattern": "weather"}'},
                        },
                        {
                            "id": "toolu_01Weather",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        },
                    ],
                    "provider_specific_fields": {"tool_results": [_SEARCH_RESULT]},
                },
            }
        ],
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tools",
    [
        list(_SEARCHABLE_TOOLS),
        [{key: value for key, value in _WEATHER_TOOL.items() if key != "defer_loading"}],
    ],
    ids=["follow_up_sends_the_search_tool", "follow_up_sends_no_search_tool"],
)
async def test_previous_response_id_replays_a_search_claude_ran_whatever_search_tool_the_follow_up_sends(
    tools: list[Mapping[str, object]],
) -> None:
    anthropic: Final = _RecordingRawAnthropicMessages()

    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([_SERVER_HELD_SEARCH_TURN])):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="chatcmpl-search-turn",
            input=[_WEATHER_ANSWER, {"role": "user", "content": "And in Osaka?"}],
            tools=tools,
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    messages: Final = json.loads(anthropic.body)["messages"]
    assert next(message["content"] for message in messages if message["role"] == "assistant") == [
        _SEARCH_USE,
        _SEARCH_RESULT,
        _WEATHER_USE,
    ]


@pytest.mark.asyncio
async def test_previous_response_id_drops_a_search_the_stored_turn_got_as_input_items_off_anthropic() -> None:
    messages: Final = await _follow_up_messages_after(_STATELESS_SEARCH_TURN, "mistral/mistral-small-latest", "mistral")

    assert _history_tool_ids(messages) == (["toolu_01Weather"], ["toolu_01Weather"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "custom_llm_provider"),
    [
        ("anthropic/claude-3-7-sonnet-20250219", "anthropic"),
        ("azure_ai/claude-deployment", "azure_ai"),
        ("bedrock/invoke/us.anthropic.claude-sonnet-5-5", "bedrock"),
    ],
)
async def test_previous_response_id_keeps_a_stored_search_on_claude_routes_the_cost_map_does_not_flag(
    model: str, custom_llm_provider: str
) -> None:
    messages: Final = await _follow_up_messages_after(_STATELESS_SEARCH_TURN, model, custom_llm_provider)

    assert _history_tool_ids(messages)[0] == ["srvtoolu_01Search", "toolu_01Weather"]


_FLAT_ALLOWED_WEATHER: Final = {
    "type": "allowed_tools",
    "mode": "required",
    "tools": [{"type": "function", "name": "get_weather"}],
}
_RESPONSES_TOOLS: Final = (
    {"type": "function", "name": "get_weather", "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "get_time", "parameters": {"type": "object", "properties": {}}},
)


def _chat_answer() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "chatcmpl-echo",
            "object": "chat.completion",
            "created": 1739462947,
            "model": "the-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _anthropic_allowed_tools_message(*content: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_allowed_tools",
            "type": "message",
            "role": "assistant",
            "model": "claude-haiku-4-5",
            "content": list(content),
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )


@pytest.mark.usefixtures(httpx_transport.__name__)
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.asyncio
async def test_responses_reports_auto_for_an_allowed_tools_choice_it_dropped(
    respx_mock: respx.MockRouter, use_async: bool
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_allowed_tools_message({"type": "text", "text": "done"})
    )
    call: Final = {
        "model": "anthropic/claude-haiku-4-5",
        "input": "What is the weather?",
        "tools": list(_RESPONSES_TOOLS),
        "tool_choice": _FLAT_ALLOWED_WEATHER,
        "drop_params": True,
        "api_key": "sk-ant-fake",
    }

    response: Final = await litellm.aresponses(**call) if use_async else litellm.responses(**call)

    assert "tool_choice" not in json.loads(route.calls[0].request.content)
    assert response.tool_choice == "auto"


@pytest.mark.usefixtures(httpx_transport.__name__)
@pytest.mark.asyncio
async def test_responses_drop_the_choice_beside_a_tool_the_provider_runs(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_allowed_tools_message({"type": "text", "text": "done"})
    )

    response: Final = await litellm.aresponses(
        model="anthropic/claude-haiku-4-5",
        input="What is the weather?",
        tools=[{"type": "mcp", "server_label": "docs", "server_url": "https://mcp.example.com/sse"}],
        tool_choice=_FLAT_ALLOWED_WEATHER,
        drop_params=True,
        api_key="sk-ant-fake",
    )

    assert "tool_choice" not in json.loads(route.calls[0].request.content)
    assert response.tool_choice == "auto"


@pytest.mark.usefixtures(httpx_transport.__name__)
def test_responses_report_auto_when_an_old_azure_api_version_dropped_the_choice(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post(url__regex=r"https://azure-bridge\.example\.com/.*chat/completions.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-old-azure",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    response: Final = litellm.responses(
        model="azure/gpt-4o",
        input="What is the weather?",
        tools=list(_RESPONSES_TOOLS),
        tool_choice=_FLAT_ALLOWED_WEATHER,
        drop_params=True,
        use_chat_completions_api=True,
        api_key="fake",
        api_base="https://azure-bridge.example.com",
        api_version="2023-07-01-preview",
    )

    assert "tool_choice" not in json.loads(route.calls[0].request.content)
    assert response.tool_choice == "auto"


def test_streamed_responses_report_auto_for_an_allowed_tools_choice_it_dropped(respx_mock: respx.MockRouter) -> None:
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_anthropic_sse(
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_dropped_choice",
                        "type": "message",
                        "role": "assistant",
                        "model": "claude-haiku-4-5",
                        "content": [],
                        "usage": {"input_tokens": 9, "output_tokens": 1},
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 4}},
                {"type": "message_stop"},
            ),
        )
    )

    events: Final = tuple(
        litellm.responses(
            model="anthropic/claude-haiku-4-5",
            input="What is the weather?",
            tools=list(_RESPONSES_TOOLS),
            tool_choice=_FLAT_ALLOWED_WEATHER,
            drop_params=True,
            stream=True,
            api_key="sk-ant-fake",
        )
    )

    reported: Final = tuple(
        event.response.tool_choice
        for event in events
        if event.type in (ResponsesAPIStreamEvents.RESPONSE_CREATED, ResponsesAPIStreamEvents.RESPONSE_COMPLETED)
    )
    assert reported == ("auto", "auto")


def test_responses_through_chat_keeps_reporting_a_choice_the_provider_received(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("http://localhost:12346/v1/chat/completions").mock(return_value=_chat_answer())

    response: Final = litellm.responses(
        model="openai/gpt-5.5",
        input="What is the weather?",
        tools=list(_RESPONSES_TOOLS),
        tool_choice=_FLAT_ALLOWED_WEATHER,
        use_chat_completions_api=True,
        api_base="http://localhost:12346/v1",
        api_key="sk-fake",
    )

    assert json.loads(route.calls[0].request.content)["tool_choice"]["type"] == "allowed_tools"
    assert response.tool_choice["type"] == "allowed_tools"
    assert list(response.tool_choice["tools"]) == _FLAT_ALLOWED_WEATHER["tools"]


def _sent_assistant_content(route: respx.Route) -> list[object]:
    sent: Final = json.loads(route.calls.last.request.content)
    return next(message["content"] for message in sent["messages"] if message["role"] == "assistant")


class _RecordingRawAnthropicMessages:
    def __init__(self, reply: Mapping[str, object] | None = None) -> None:
        self.body: str = ""
        self._reply = reply

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.body = request.content.decode()
        if self._reply is None:
            return _anthropic_message({"type": "text", "text": "dog."})
        return httpx.Response(200, json=self._reply)


_MCP_CALL_ID: Final = "mcptoolu_01WDx6ixKDtjRVLLUhL652Xa"
_MCP_ARGUMENTS: Final = '{"repoName": "facebook/react"}'
_DEEPWIKI_SERVER: Final = {
    "type": "mcp",
    "server_label": "deepwiki",
    "server_url": "https://mcp.deepwiki.com/mcp",
    "require_approval": "never",
}
_MCP_TOOL_USE: Final = {
    "type": "mcp_tool_use",
    "id": _MCP_CALL_ID,
    "name": "read_wiki_structure",
    "input": {"repoName": "facebook/react"},
    "server_name": "deepwiki",
}


def _mcp_tool_result(is_error: bool, *texts: str) -> Mapping[str, object]:
    return {
        "type": "mcp_tool_result",
        "tool_use_id": _MCP_CALL_ID,
        "is_error": is_error,
        "content": [{"type": "text", "text": text} for text in texts],
    }


@pytest.mark.parametrize(
    ("result_blocks", "expected_result"),
    [
        pytest.param(
            (_mcp_tool_result(False, "1 React Repository Overview", "2 Core Reconciler Architecture"),),
            {"status": "completed", "output": "1 React Repository Overview\n2 Core Reconciler Architecture"},
            id="completed",
        ),
        pytest.param(
            (_mcp_tool_result(True, "Repository not found"),),
            {"status": "failed", "error": "Repository not found"},
            id="is_error",
        ),
        pytest.param((), {"status": "in_progress"}, id="no_result"),
    ],
)
def test_responses_over_anthropic_return_an_mcp_connector_call_as_an_mcp_call_item(
    respx_mock: respx.MockRouter,
    result_blocks: tuple[Mapping[str, object], ...],
    expected_result: Mapping[str, str],
) -> None:
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message(_MCP_TOOL_USE, *result_blocks, {"type": "text", "text": "Done."})
    )

    response: Final = litellm.responses(
        model=_MODEL, input="List the wiki pages of facebook/react.", tools=[_DEEPWIKI_SERVER], api_key="sk-ant-fake"
    )

    assert [item.type for item in response.output] == ["mcp_call", "message"]
    assert response.output[0].model_dump(exclude_none=True) == {
        "id": f"mcp_{_MCP_CALL_ID}",
        "type": "mcp_call",
        "name": "read_wiki_structure",
        "arguments": _MCP_ARGUMENTS,
        "server_label": "deepwiki",
        **expected_result,
    }


_WEB_SEARCH_CALL_ID: Final = "srvtoolu_01WebSearchReactFiber"
_WEB_SEARCH_TOOL_USE: Final = {
    "type": "server_tool_use",
    "id": _WEB_SEARCH_CALL_ID,
    "name": "web_search",
    "input": {"query": "react fiber reconciler"},
}


def _streamed_block_events(index: int, block: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    if block["type"] == "text":
        return (
            {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": block["text"]}},
            {"type": "content_block_stop", "index": index},
        )
    if "input" not in block:
        return (
            {"type": "content_block_start", "index": index, "content_block": block},
            {"type": "content_block_stop", "index": index},
        )
    return (
        {"type": "content_block_start", "index": index, "content_block": {**block, "input": {}}},
        {
            "type": "content_block_delta",
            "index": index,
            "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
        },
        {"type": "content_block_stop", "index": index},
    )


def _anthropic_reply(stream: bool, *content: Mapping[str, object]) -> httpx.Response:
    if not stream:
        return _anthropic_message(*content)
    message_start: Final = {
        "type": "message_start",
        "message": {
            "id": "msg_mcp_and_search",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    }
    block_events: Final = chain.from_iterable(_streamed_block_events(index, block) for index, block in enumerate(content))
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        content=_anthropic_sse(
            message_start,
            *block_events,
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
            {"type": "message_stop"},
        ),
    )


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_responses_over_anthropic_list_an_mcp_call_only_as_its_item_and_keep_other_provider_fields(
    respx_mock: respx.MockRouter, stream: bool
) -> None:
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_reply(
            stream,
            _MCP_TOOL_USE,
            _mcp_tool_result(False, "1 React Repository Overview"),
            _WEB_SEARCH_TOOL_USE,
            {"type": "text", "text": "Done."},
        )
    )

    response: Final = litellm.responses(
        model=_MODEL,
        input="List the wiki pages of facebook/react.",
        tools=[_DEEPWIKI_SERVER],
        stream=stream,
        api_key="sk-ant-fake",
    )
    completed: Final = tuple(response)[-1].response if stream else response

    provider_fields: Final = completed.provider_specific_fields or {}
    assert [item.id for item in completed.output if item.type == "mcp_call"] == [f"mcp_{_MCP_CALL_ID}"]
    assert "server_tool_calls" not in provider_fields
    assert "tool_results" in provider_fields


def _mcp_call_item(item_id: str, **result: str) -> Mapping[str, object]:
    return {
        "type": "mcp_call",
        "id": item_id,
        "name": "read_wiki_structure",
        "arguments": _MCP_ARGUMENTS,
        "server_label": "deepwiki",
        **result,
    }


def _replayed_mcp_input(item: Mapping[str, object]) -> list[object]:
    return [
        {"role": "user", "content": "List the wiki pages of facebook/react."},
        item,
        {"role": "user", "content": "Which page covers the reconciler?"},
    ]


@pytest.mark.parametrize(
    ("result", "expected_result_block"),
    [
        pytest.param(
            {"status": "completed", "output": "1 React Repository Overview"},
            _mcp_tool_result(False, "1 React Repository Overview"),
            id="completed",
        ),
        pytest.param(
            {"status": "failed", "error": "Repository not found"},
            _mcp_tool_result(True, "Repository not found"),
            id="is_error",
        ),
    ],
)
def test_responses_over_anthropic_send_a_replayed_mcp_call_back_as_the_mcp_block_pair(
    respx_mock: respx.MockRouter, result: Mapping[str, str], expected_result_block: Mapping[str, object]
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message({"type": "text", "text": "Page 2."})
    )

    litellm.responses(
        model=_MODEL,
        input=_replayed_mcp_input(_mcp_call_item(f"mcp_{_MCP_CALL_ID}", **result)),
        tools=[_DEEPWIKI_SERVER],
        api_key="sk-ant-fake",
    )

    assert _sent_assistant_content(route) == [_MCP_TOOL_USE, expected_result_block]


def test_responses_over_anthropic_send_a_replayed_mcp_call_with_the_mcp_beta_when_the_turn_has_no_mcp_tool(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message({"type": "text", "text": "Page 2."})
    )

    litellm.responses(
        model=_MODEL,
        input=_replayed_mcp_input(
            _mcp_call_item(f"mcp_{_MCP_CALL_ID}", status="completed", output="1 React Repository Overview")
        ),
        api_key="sk-ant-fake",
    )

    assert _sent_assistant_content(route) == [_MCP_TOOL_USE, _mcp_tool_result(False, "1 React Repository Overview")]
    assert ANTHROPIC_BETA_HEADER_VALUES.MCP_CLIENT_2025_04_04.value in route.calls.last.request.headers[
        "anthropic-beta"
    ].split(",")


def test_responses_over_anthropic_drop_a_replayed_mcp_call_that_holds_no_claude_call_id(
    respx_mock: respx.MockRouter,
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        return_value=_anthropic_message({"type": "text", "text": "Page 2."})
    )

    litellm.responses(
        model=_MODEL,
        input=_replayed_mcp_input(_mcp_call_item("mcp_68a1b2c3d4e5f6a7b8c9d0e1", status="completed", output="1")),
        tools=[_DEEPWIKI_SERVER],
        api_key="sk-ant-fake",
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert [message["role"] for message in sent["messages"]] == ["user"]


def test_responses_through_openai_chat_drop_a_replayed_claude_mcp_call(respx_mock: respx.MockRouter) -> None:
    route: Final = respx_mock.post("http://localhost:12346/v1/chat/completions").mock(return_value=_chat_answer())

    litellm.responses(
        model="openai/gpt-5.5",
        input=_replayed_mcp_input(_mcp_call_item(f"mcp_{_MCP_CALL_ID}", status="completed", output="1")),
        use_chat_completions_api=True,
        api_base="http://localhost:12346/v1",
        api_key="sk-fake",
    )

    sent: Final = json.loads(route.calls.last.request.content)
    assert [message["role"] for message in sent["messages"]] == ["user", "user"]


def _stored_mcp_turn(response: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "request_id": "chatcmpl-mcp-turn",
        "call_type": "aresponses",
        "session_id": "session-mcp",
        "proxy_server_request": {
            "model": _MODEL,
            "input": "List the wiki pages of facebook/react.",
            "tools": [_DEEPWIKI_SERVER],
        },
        "response": response,
    }


_STORED_CHAT_COMPLETION_MCP_TURN: Final = _stored_mcp_turn(
    {
        "id": "chatcmpl-mcp-turn",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Done.",
                    "provider_specific_fields": {
                        "server_tool_calls": [
                            {
                                "id": _MCP_CALL_ID,
                                "type": "function",
                                "function": {
                                    "name": "read_wiki_structure",
                                    "arguments": _MCP_ARGUMENTS,
                                    "provider_specific_fields": {"server_name": "deepwiki"},
                                },
                            }
                        ],
                        "tool_results": [_mcp_tool_result(False, "1 React Repository Overview")],
                    },
                },
            }
        ],
    }
)
_STORED_MCP_CALL_INPUT_TURN: Final = {
    **_stored_mcp_turn(
        {
            "id": "chatcmpl-mcp-turn",
            "object": "chat.completion",
            "created": 0,
            "model": _MODEL,
            "choices": [
                {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Done."}}
            ],
        }
    ),
    "proxy_server_request": {
        "model": _MODEL,
        "input": [
            {"role": "user", "content": "List the wiki pages of facebook/react."},
            _mcp_call_item(f"mcp_{_MCP_CALL_ID}", status="completed", output="1 React Repository Overview"),
        ],
        "tools": [_DEEPWIKI_SERVER],
    },
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_turn",
    [
        pytest.param(_STORED_CHAT_COMPLETION_MCP_TURN, id="chat_completion_message"),
        pytest.param(_STORED_MCP_CALL_INPUT_TURN, id="stored_request_input"),
    ],
)
async def test_previous_response_id_sends_a_stored_mcp_call_and_its_result_back_to_claude(
    stored_turn: Mapping[str, object],
) -> None:
    anthropic: Final = _RecordingRawAnthropicMessages()

    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([stored_turn])):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="chatcmpl-mcp-turn",
            input="Which page covers the reconciler?",
            tools=[_DEEPWIKI_SERVER],
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    sent_blocks: Final = chain.from_iterable(
        message["content"] for message in json.loads(anthropic.body)["messages"] if message["role"] == "assistant"
    )
    assert [block for block in sent_blocks if block["type"] != "text"] == [
        _MCP_TOOL_USE,
        _mcp_tool_result(False, "1 React Repository Overview"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored_turn",
    [
        pytest.param(_STORED_CHAT_COMPLETION_MCP_TURN, id="chat_completion_message"),
    ],
)
async def test_previous_response_id_leaves_out_a_stored_mcp_call_that_message_logging_redacted(
    stored_turn: Mapping[str, object],
) -> None:
    anthropic: Final = _RecordingRawAnthropicMessages()
    redacted_turn: Final = {
        **stored_turn,
        "response": perform_redaction(model_call_details={}, result=json.loads(json.dumps(stored_turn["response"]))),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([redacted_turn])):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="chatcmpl-mcp-turn",
            input="Which page covers the reconciler?",
            tools=[_DEEPWIKI_SERVER],
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    messages: Final = json.loads(anthropic.body)["messages"]
    assert [message["role"] for message in messages] == ["user", "assistant", "user"]
    assert "mcp_tool_use" not in json.dumps(messages)
    assert "mcp_tool_result" not in json.dumps(messages)


_CODE_EXECUTION_CALL_ID: Final = "srvtoolu_01Bash"
_STORED_CODE_EXECUTION_TURN: Final = _stored_mcp_turn(
    {
        "id": "chatcmpl-mcp-turn",
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "2**40 is 1099511627776.",
                    "tool_calls": [
                        {
                            "id": _CODE_EXECUTION_CALL_ID,
                            "type": "function",
                            "function": {
                                "name": "bash_code_execution",
                                "arguments": json.dumps({"command": "python3 -c 'print(2**40)'"}),
                            },
                        }
                    ],
                    "provider_specific_fields": {
                        "tool_results": [
                            {
                                "type": "bash_code_execution_tool_result",
                                "tool_use_id": _CODE_EXECUTION_CALL_ID,
                                "content": {
                                    "type": "bash_code_execution_result",
                                    "stdout": "1099511627776\n",
                                    "stderr": "",
                                    "return_code": 0,
                                    "content": [],
                                },
                            }
                        ],
                    },
                },
            }
        ],
    }
)


@pytest.mark.asyncio
async def test_previous_response_id_pairs_a_code_execution_call_with_its_result_after_message_logging_redacted_it() -> (
    None
):
    anthropic: Final = _RecordingRawAnthropicMessages()
    redacted_turn: Final = {
        **_STORED_CODE_EXECUTION_TURN,
        "response": perform_redaction(
            model_call_details={}, result=json.loads(json.dumps(_STORED_CODE_EXECUTION_TURN["response"]))
        ),
    }

    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([redacted_turn])):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="chatcmpl-mcp-turn",
            input="Now divide it by 2.",
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    sent_blocks: Final = chain.from_iterable(
        message["content"] for message in json.loads(anthropic.body)["messages"] if message["role"] == "assistant"
    )
    assert [
        (block["type"], block.get("id", block.get("tool_use_id"))) for block in sent_blocks if block["type"] != "text"
    ] == [
        ("server_tool_use", _CODE_EXECUTION_CALL_ID),
        ("bash_code_execution_tool_result", _CODE_EXECUTION_CALL_ID),
    ]


_LOCAL_TIME_FUNCTION: Final = {
    "type": "function",
    "name": LOCAL_TIME_TOOL["name"],
    "description": LOCAL_TIME_TOOL["description"],
    "parameters": LOCAL_TIME_TOOL["input_schema"],
}
_DEFERRED_MCP_TOOLS: Final = (_DEEPWIKI_SERVER, _LOCAL_TIME_FUNCTION)
_PENDING_MCP_CALL_ITEM: Final = {
    "id": f"mcp_{DEFERRED_MCP_CALL_ID}",
    "type": "mcp_call",
    "name": "read_wiki_structure",
    "arguments": json.dumps(MCP_TOOL_ARGUMENTS),
    "server_label": "deepwiki",
    "status": "in_progress",
}
_ANSWERED_MCP_CALL_ITEM: Final = {
    **_PENDING_MCP_CALL_ITEM,
    "id": f"mcp_{DEFERRED_MCP_CALL_ID}_result",
    "status": "completed",
    "output": WIKI_PAGES,
}
_LOCAL_TIME_CALL_ITEM: Final = {
    "type": "function_call",
    "id": f"fc_{CLIENT_TOOL_CALL_ID}",
    "call_id": CLIENT_TOOL_CALL_ID,
    "name": "get_local_time",
    "arguments": "{}",
    "status": "completed",
}
_LOCAL_TIME_OUTPUT_ITEM: Final = {
    "type": "function_call_output",
    "call_id": CLIENT_TOOL_CALL_ID,
    "output": LOCAL_TIME_OUTPUT,
}
_TURN_TWO_TEXT_BLOCK: Final = {"type": "text", "text": TURN_TWO_TEXT}
_DEFERRED_MCP_PROMPT_ITEM: Final = {"role": "user", "content": DEFERRED_MCP_PROMPT}


def _deferred_mcp_reply(
    message: Mapping[str, object], stream_events: tuple[Mapping[str, object], ...], stream: bool
) -> httpx.Response:
    if stream:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=anthropic_sse(stream_events))
    return httpx.Response(200, json=message)


def _dumped(item: object) -> Mapping[str, object]:
    return item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else cast(Mapping[str, object], item)


def _deferred_mcp_turn(
    input_items: Sequence[object], stream: bool
) -> tuple[list[Mapping[str, object]], tuple[object, ...]]:
    response: Final = litellm.responses(
        model=_MODEL, input=list(input_items), tools=list(_DEFERRED_MCP_TOOLS), stream=stream, api_key="sk-ant-fake"
    )
    if not stream:
        return [_dumped(item) for item in response.output], ()
    events: Final = tuple(response)
    return [_dumped(item) for item in events[-1].response.output], events


def _assistant_turns(body: bytes | str) -> list[object]:
    return [message["content"] for message in json.loads(body)["messages"] if message["role"] == "assistant"]


def _first_item_of_each_id(items: Sequence[Mapping[str, object]]) -> list[Mapping[str, object]]:
    ids: Final = [item.get("id") for item in items]
    return [item for index, item in enumerate(items) if ids[index] is None or ids[index] not in ids[:index]]


@pytest.mark.parametrize("stream", [False, True], ids=["non_stream", "stream"])
def test_responses_over_anthropic_return_a_deferred_mcp_result_as_its_own_item_and_send_it_back_with_that_turn(
    respx_mock: respx.MockRouter, stream: bool
) -> None:
    route: Final = respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        side_effect=[
            _deferred_mcp_reply(TURN_ONE_MESSAGE, TURN_ONE_STREAM_EVENTS, stream),
            _deferred_mcp_reply(TURN_TWO_MESSAGE, TURN_TWO_STREAM_EVENTS, stream),
            httpx.Response(200, json=TURN_THREE_MESSAGE),
        ]
    )

    turn_one, _ = _deferred_mcp_turn([_DEFERRED_MCP_PROMPT_ITEM], stream)
    turn_two_input: Final = [_DEFERRED_MCP_PROMPT_ITEM, *turn_one, _LOCAL_TIME_OUTPUT_ITEM]
    turn_two, _ = _deferred_mcp_turn(turn_two_input, stream)
    history_of_a_client_that_drops_repeated_ids: Final = _first_item_of_each_id([*turn_two_input, *turn_two])
    _deferred_mcp_turn(
        [*history_of_a_client_that_drops_repeated_ids, {"role": "user", "content": DEFERRED_MCP_FOLLOW_UP}], False
    )

    assert [item["type"] for item in turn_one] == ["mcp_call", "function_call"]
    assert turn_one[0] == _PENDING_MCP_CALL_ITEM
    assert _assistant_turns(route.calls[1].request.content) == [[MCP_TOOL_USE, CLIENT_TOOL_USE]]
    assert [item["type"] for item in turn_two] == ["mcp_call", "message"]
    assert turn_two[0] == _ANSWERED_MCP_CALL_ITEM
    assert _assistant_turns(route.calls[2].request.content) == [
        [MCP_TOOL_USE, CLIENT_TOOL_USE],
        [DEFERRED_MCP_TOOL_RESULT, _TURN_TWO_TEXT_BLOCK],
    ]


def _item_events(events: Sequence[object], item_id: str) -> list[tuple[object, object]]:
    return [
        (getattr(event, "type"), getattr(event, "output_index", None))
        for event in events
        if (getattr(event, "item_id", None) or getattr(getattr(event, "item", None), "id", None)) == item_id
    ]


def test_streamed_responses_over_anthropic_emit_the_deferred_mcp_call_pending_then_answered(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        side_effect=[
            _deferred_mcp_reply(TURN_ONE_MESSAGE, TURN_ONE_STREAM_EVENTS, True),
            _deferred_mcp_reply(TURN_TWO_MESSAGE, TURN_TWO_STREAM_EVENTS, True),
        ]
    )

    turn_one, turn_one_events = _deferred_mcp_turn([_DEFERRED_MCP_PROMPT_ITEM], True)
    _, turn_two_events = _deferred_mcp_turn([_DEFERRED_MCP_PROMPT_ITEM, *turn_one, _LOCAL_TIME_OUTPUT_ITEM], True)

    assert _item_events(turn_one_events, _PENDING_MCP_CALL_ITEM["id"]) == [
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, 0),
        (ResponsesAPIStreamEvents.MCP_CALL_IN_PROGRESS, 0),
        (ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DELTA, 0),
        (ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DONE, 0),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, 0),
    ]
    assert _item_events(turn_two_events, _ANSWERED_MCP_CALL_ITEM["id"]) == [
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED, 0),
        (ResponsesAPIStreamEvents.MCP_CALL_IN_PROGRESS, 0),
        (ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DELTA, 0),
        (ResponsesAPIStreamEvents.MCP_CALL_ARGUMENTS_DONE, 0),
        (ResponsesAPIStreamEvents.MCP_CALL_COMPLETED, 0),
        (ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE, 0),
    ]
    added_items: Final = [
        (event.output_index, _dumped(event.item))
        for event in turn_two_events
        if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED
    ]
    assert added_items[0] == (0, {**_PENDING_MCP_CALL_ITEM, "id": _ANSWERED_MCP_CALL_ITEM["id"], "arguments": ""})
    assert [(index, item["type"]) for index, item in added_items[1:]] == [(1, "message")]
    done_mcp_items: Final = [
        _dumped(event.item)
        for event in turn_two_events
        if event.type == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE and _dumped(event.item)["type"] == "mcp_call"
    ]
    assert done_mcp_items == [_ANSWERED_MCP_CALL_ITEM]


def _streamed_mcp_call_arguments(event: object, item_id: str) -> str | None:
    item: Final = getattr(event, "item", None)
    if item is not None:
        return getattr(item, "arguments", None) if getattr(item, "id", None) == item_id else None
    if getattr(event, "item_id", None) != item_id:
        return None
    return getattr(event, "delta", None) or getattr(event, "arguments", None)


@pytest.mark.asyncio
async def test_a_masking_guardrail_leaves_no_raw_arguments_in_any_streamed_event_of_a_deferred_mcp_result(
    respx_mock: respx.MockRouter,
) -> None:
    respx_mock.post(_ANTHROPIC_MESSAGES_URL).mock(
        side_effect=[
            _deferred_mcp_reply(TURN_ONE_MESSAGE, TURN_ONE_STREAM_EVENTS, True),
            _deferred_mcp_reply(TURN_TWO_MESSAGE, TURN_TWO_STREAM_EVENTS, True),
        ]
    )
    turn_one, _ = _deferred_mcp_turn([_DEFERRED_MCP_PROMPT_ITEM], True)
    _, turn_two_events = _deferred_mcp_turn([_DEFERRED_MCP_PROMPT_ITEM, *turn_one, _LOCAL_TIME_OUTPUT_ITEM], True)

    await OpenAIResponsesHandler().process_output_streaming_response(
        responses_so_far=list(turn_two_events),
        guardrail_to_apply=ContentFilterGuardrail(
            guardrail_name="mask-repo-owner",
            blocked_words=[BlockedWord(keyword="facebook", action=ContentFilterAction.MASK)],
        ),
        request_data={"metadata": {}},
        deliver_ended_stream_rewrites=True,
    )

    streamed_arguments: Final = [
        arguments
        for event in turn_two_events
        if (arguments := _streamed_mcp_call_arguments(event, _ANSWERED_MCP_CALL_ITEM["id"]))
    ]
    assert any("repoName" in arguments for arguments in streamed_arguments)
    assert [arguments for arguments in streamed_arguments if "facebook" in arguments] == []


def _stored_deferred_mcp_turn(
    request_id: str, request_input: object, response: Mapping[str, object]
) -> Mapping[str, object]:
    return {
        "request_id": request_id,
        "call_type": "aresponses",
        "session_id": "session-deferred-mcp",
        "proxy_server_request": {"model": _MODEL, "input": request_input, "tools": list(_DEFERRED_MCP_TOOLS)},
        "response": {"id": request_id, **response},
    }


def _stored_chat_completion(finish_reason: str, message: Mapping[str, object]) -> Mapping[str, object]:
    return {
        "object": "chat.completion",
        "created": 0,
        "model": _MODEL,
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    }


_STORED_DEFERRED_MCP_TURN_ONE: Final = _stored_deferred_mcp_turn(
    "resp-deferred-mcp-1",
    DEFERRED_MCP_PROMPT,
    _stored_chat_completion(
        "tool_calls",
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": CLIENT_TOOL_CALL_ID,
                    "type": "function",
                    "function": {"name": "get_local_time", "arguments": "{}"},
                }
            ],
            "provider_specific_fields": {
                "server_tool_calls": [
                    {
                        "id": DEFERRED_MCP_CALL_ID,
                        "type": "function",
                        "function": {
                            "name": "read_wiki_structure",
                            "arguments": json.dumps(MCP_TOOL_ARGUMENTS),
                            "provider_specific_fields": {"server_name": "deepwiki"},
                        },
                    }
                ]
            },
        },
    ),
)


@pytest.mark.asyncio
async def test_a_paused_mcp_call_replayed_next_to_its_result_sends_the_result_once() -> None:
    anthropic: Final = _RecordingRawAnthropicMessages()

    await litellm.aresponses(
        model=_MODEL,
        input=[
            _DEFERRED_MCP_PROMPT_ITEM,
            _PENDING_MCP_CALL_ITEM,
            _ANSWERED_MCP_CALL_ITEM,
            {"role": "user", "content": "Thanks, which page covers the reconciler?"},
        ],
        tools=list(_DEFERRED_MCP_TOOLS),
        api_key="sk-ant-fake",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
    )

    assert _assistant_turns(anthropic.body) == [[MCP_TOOL_USE, DEFERRED_MCP_TOOL_RESULT]]


@pytest.mark.asyncio
async def test_previous_response_id_returns_a_deferred_mcp_result_with_the_fields_of_the_stored_call() -> None:
    anthropic: Final = _RecordingRawAnthropicMessages(TURN_TWO_MESSAGE)

    with patch("litellm.proxy.proxy_server.prisma_client", _FakePrismaClient([_STORED_DEFERRED_MCP_TURN_ONE])):
        response: Final = await litellm.aresponses(
            model=_MODEL,
            previous_response_id="resp-deferred-mcp-1",
            input=[_LOCAL_TIME_OUTPUT_ITEM],
            tools=list(_DEFERRED_MCP_TOOLS),
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    assert _assistant_turns(anthropic.body) == [[MCP_TOOL_USE, CLIENT_TOOL_USE]]
    assert [item.type for item in response.output] == ["mcp_call", "message"]
    assert _dumped(response.output[0]) == _ANSWERED_MCP_CALL_ITEM


@pytest.mark.asyncio
async def test_previous_response_id_sends_a_deferred_mcp_result_back_with_the_turn_that_received_it() -> None:
    anthropic: Final = _RecordingRawAnthropicMessages(TURN_THREE_MESSAGE)
    stored_turn_two: Final = _stored_deferred_mcp_turn(
        "resp-deferred-mcp-2",
        [_LOCAL_TIME_OUTPUT_ITEM],
        _stored_chat_completion(
            "stop",
            {
                "role": "assistant",
                "content": TURN_TWO_TEXT,
                "provider_specific_fields": {"tool_results": [DEFERRED_MCP_TOOL_RESULT]},
            },
        ),
    )

    with patch(
        "litellm.proxy.proxy_server.prisma_client",
        _FakePrismaClient([_STORED_DEFERRED_MCP_TURN_ONE, stored_turn_two]),
    ):
        await litellm.aresponses(
            model=_MODEL,
            previous_response_id="resp-deferred-mcp-2",
            input=DEFERRED_MCP_FOLLOW_UP,
            tools=list(_DEFERRED_MCP_TOOLS),
            api_key="sk-ant-fake",
            client=AsyncHTTPHandler(transport=httpx.MockTransport(anthropic)),
        )

    assert _assistant_turns(anthropic.body) == [
        [MCP_TOOL_USE, CLIENT_TOOL_USE],
        [DEFERRED_MCP_TOOL_RESULT, _TURN_TWO_TEXT_BLOCK],
    ]
