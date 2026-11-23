import json
from itertools import chain
from collections.abc import AsyncIterable, Iterable, Mapping, Sequence
from typing import Final

import httpx
import pytest
import respx
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm
from litellm.types.llms.openai import ResponseCompletedEvent, ResponsesAPIResponse
from tests.unit.proxy.conftest import httpx_transport

pytestmark: Final = pytest.mark.usefixtures(httpx_transport.__name__)
_GEMINI_URL: Final = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:(?:generateContent|streamGenerateContent).*"
)
_GEMINI_3_8_FLASH_MODEL: Final = "gemini/gemini-3.8-flash"
_GEMINI_3_8_FLASH_URL: Final = (
    r"https://generativelanguage\.googleapis\.com/[^/]+/models/gemini-3\.8-flash:generateContent.*"
)
_GEMINI_3_8_FLASH_REPLY: Final = {
    "candidates": [
        {"content": {"role": "model", "parts": [{"text": "How can I help?"}]}, "finishReason": "STOP", "index": 0}
    ],
    "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 4, "totalTokenCount": 8},
}
_JSON_OBJECT: Final = TypeAdapter(dict[str, object])


def _function_call_response(signature: str) -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [
                        {
                            "functionCall": {
                                "name": "get_weather",
                                "args": {"location": "San Francisco"},
                            },
                            "thoughtSignature": signature,
                        }
                    ],
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 4,
            "candidatesTokenCount": 2,
            "totalTokenCount": 6,
        },
    }


@pytest.mark.asyncio
async def test_web_search_preview_is_sent_as_google_search() -> None:
    response_body: Final = {
        "candidates": [
            {
                "content": {"role": "model", "parts": [{"text": "Search completed"}]},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 2,
            "totalTokenCount": 4,
        },
    }

    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_URL).mock(
            return_value=httpx.Response(status_code=200, json=response_body)
        )
        response: Final = await litellm.aresponses(
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            input="Find current weather",
            tools=[{"type": "web_search_preview", "search_context_size": "low"}],
        )
        requests: Final = tuple(route.calls)

    assert isinstance(response, ResponsesAPIResponse)
    assert response.output_text == "Search completed"
    assert len(requests) == 1
    request_body: Final = _JSON_OBJECT.validate_json(requests[0].request.content)
    assert request_body["tools"] == [{"googleSearch": {}}]


@pytest.mark.asyncio
async def test_gemini_function_call_preserves_thought_signature() -> None:
    signature: Final = "gemini-thought-signature"
    response_body: Final = _function_call_response(signature)

    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_URL).mock(
            return_value=httpx.Response(status_code=200, json=response_body)
        )
        response: Final = await litellm.aresponses(
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            input="What is the weather in San Francisco?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )
        requests: Final = tuple(route.calls)

    assert isinstance(response, ResponsesAPIResponse)
    function_calls: Final = tuple(item for item in response.output if item.type == "function_call")
    assert len(function_calls) == 1
    assert function_calls[0].name == "get_weather"
    assert function_calls[0].provider_specific_fields == {"thought_signature": signature}
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_gemini_streaming_function_call_preserves_thought_signature() -> None:
    signature: Final = "gemini-stream-thought-signature"
    response_body: Final = _function_call_response(signature)
    event_stream: Final = f"data: {json.dumps(response_body)}\n\n"

    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_URL).mock(
            return_value=httpx.Response(
                status_code=200,
                content=event_stream,
                headers={"content-type": "text/event-stream"},
            )
        )
        stream: Final = await litellm.aresponses(
            model="gemini/gemini-2.5-flash",
            api_key="test-key",
            input="What is the weather in San Francisco?",
            stream=True,
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                }
            ],
        )
        events: Final = tuple([event async for event in stream])
        requests: Final = tuple(route.calls)

    completed_events: Final = tuple(event for event in events if isinstance(event, ResponseCompletedEvent))
    assert len(completed_events) == 1
    function_calls: Final = tuple(item for item in completed_events[0].response.output if item.type == "function_call")
    assert len(function_calls) == 1
    assert function_calls[0].name == "get_weather"
    assert function_calls[0].provider_specific_fields == {"thought_signature": signature}
    assert len(requests) == 1


async def _gemini_contents_for(input_items: list[dict[str, object]], **kwargs: object) -> object:
    with respx.mock() as mock_router:
        route: Final = mock_router.post(url__regex=_GEMINI_3_8_FLASH_URL).mock(
            return_value=httpx.Response(status_code=200, json=_GEMINI_3_8_FLASH_REPLY)
        )
        response: Final = await litellm.aresponses(
            model=_GEMINI_3_8_FLASH_MODEL, api_key="test-key", input=input_items, **kwargs
        )
        requests: Final = tuple(route.calls)

    assert isinstance(response, ResponsesAPIResponse)
    assert len(requests) == 1
    return _JSON_OBJECT.validate_json(requests[0].request.content)["contents"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_user_item",
    [
        pytest.param({"role": "user", "content": ""}, id="empty_string"),
        pytest.param({"role": "user", "content": []}, id="empty_list"),
        pytest.param({"role": "user", "content": None}, id="null"),
        pytest.param({"role": "user"}, id="missing"),
    ],
)
async def test_an_empty_trailing_user_item_reaches_gemini_as_a_user_turn(empty_user_item: dict[str, object]) -> None:
    contents: Final = await _gemini_contents_for(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, empty_user_item]
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "Hi"}]},
        {"role": "model", "parts": [{"text": "Hello!"}]},
        {"role": "user", "parts": [{"text": " "}]},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "empty_user_item",
    [
        pytest.param({"role": "user", "content": ""}, id="empty_string"),
        pytest.param({"role": "user", "content": []}, id="empty_list"),
        pytest.param({"role": "user", "content": None}, id="null"),
        pytest.param({"role": "user"}, id="missing"),
    ],
)
async def test_an_empty_trailing_user_item_reaches_an_alternating_gemini_deployment_as_the_configured_continue_message(
    empty_user_item: dict[str, object],
) -> None:
    contents: Final = await _gemini_contents_for(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}, empty_user_item],
        ensure_alternating_roles=True,
        user_continue_message={"role": "user", "content": "Keep going"},
    )

    assert contents == [
        {"role": "user", "parts": [{"text": "Hi"}]},
        {"role": "model", "parts": [{"text": "Hello!"}]},
        {"role": "user", "parts": [{"text": "Keep going"}]},
    ]


_GEMINI_3_URL: Final = "https://generativelanguage.googleapis.com/v1alpha/models/gemini-3-flash-preview"
_SEARCH_PROMPT: Final = "Find the newest LiteLLM release, then echo it"
_SEARCH_AND_SHELL_TOOLS: Final = [
    {"type": "web_search"},
    {
        "type": "function",
        "name": "shell",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
    },
]
_SEARCHED_RESPONSE_ID: Final = "searched-response-1"
_SEARCHED_USAGE: Final = {"promptTokenCount": 30, "candidatesTokenCount": 9, "totalTokenCount": 39}
_SHELL_SIGNATURE: Final = "c2hlbGwtc2lnbmF0dXJl"
_SHELL_CALL_PART: Final = {
    "functionCall": {"name": "shell", "args": {"command": "echo 1.2.3"}},
    "thoughtSignature": _SHELL_SIGNATURE,
}
_TEXT_PART: Final = {"text": "Checking the release."}
_THOUGHT_PART: Final = {"text": "The release is on PyPI.", "thought": True}
_SIGNED_THOUGHT_PART: Final = {**_THOUGHT_PART, "thoughtSignature": "dGhvdWdodC1zaWduYXR1cmU="}
_SEARCH_PARTS: Final = (
    {
        "toolCall": {
            "toolType": "GOOGLE_SEARCH_WEB",
            "args": {"queries": ["litellm latest release", "litellm pypi version"]},
            "id": "call_1",
        },
        "thoughtSignature": "c2VhcmNoLXNpZ25hdHVyZQ==",
    },
    {"toolResponse": {"toolType": "GOOGLE_SEARCH_WEB", "response": {}, "id": "call_1"}},
)
_GROUNDING: Final = {"webSearchQueries": ["litellm pypi version"]}
_SEARCHED_QUERIES: Final = ("litellm latest release", "litellm pypi version")


class _SearchAction(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str
    query: str | None = None


class _OutputItem(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str
    id: str | None = None
    status: str | None = None
    action: _SearchAction | None = None


class _ResponseOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    output: tuple[_OutputItem, ...]


class _StreamEvent(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: str
    output_index: int | None = None
    item_id: str | None = None
    item: _OutputItem | None = None
    response: _ResponseOutput | None = None

    def is_about(self, call_id: str) -> bool:
        return call_id in (self.item_id, self.item.id if self.item else None)


_STREAM_EVENT: Final = TypeAdapter(_StreamEvent)
_JSON_OBJECT_LIST: Final = TypeAdapter(list[dict[str, object]])


class _GeminiContent(BaseModel):
    model_config = ConfigDict(frozen=True)
    role: str
    parts: tuple[dict[str, object], ...]


class _GeminiContents(BaseModel):
    model_config = ConfigDict(frozen=True)
    contents: tuple[_GeminiContent, ...]


_GEMINI_CONTENTS: Final = TypeAdapter(_GeminiContents)


def _searched_body(parts: Sequence[Mapping[str, object]], grounding: Mapping[str, object] | None) -> dict[str, object]:
    grounding_fields: Final = {} if grounding is None else {"groundingMetadata": grounding}
    return {
        "candidates": [
            {"content": {"role": "model", "parts": list(parts)}, "finishReason": "STOP", "index": 0, **grounding_fields}
        ],
        "usageMetadata": _SEARCHED_USAGE,
        "responseId": _SEARCHED_RESPONSE_ID,
    }


def _partial_body(parts: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {
        "candidates": [{"content": {"role": "model", "parts": list(parts)}, "index": 0}],
        "responseId": _SEARCHED_RESPONSE_ID,
    }


def _sse(*events: Mapping[str, object]) -> httpx.Response:
    return httpx.Response(
        200,
        content="".join(f"data: {json.dumps(event)}\n\n" for event in events),
        headers={"content-type": "text/event-stream"},
    )


def _output_items(response: object) -> tuple[_OutputItem, ...]:
    assert isinstance(response, ResponsesAPIResponse)
    return _ResponseOutput.model_validate_json(response.model_dump_json()).output


def _search_items(items: Sequence[_OutputItem]) -> tuple[_OutputItem, ...]:
    return tuple(item for item in items if item.type == "web_search_call")


@pytest.mark.asyncio
async def test_gemini_web_searches_come_back_as_web_search_call_items_before_the_message() -> None:
    with respx.mock() as mock_router:
        mock_router.post(f"{_GEMINI_3_URL}:generateContent").mock(
            return_value=httpx.Response(
                200, json=_searched_body([*_SEARCH_PARTS, _TEXT_PART, _SHELL_CALL_PART], _GROUNDING)
            )
        )
        response: Final = await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
        )

    items: Final = _output_items(response)
    searches: Final = _search_items(items)
    assert tuple(item.type for item in items) == (
        *("web_search_call",) * len(_SEARCHED_QUERIES),
        "message",
        "function_call",
    )
    assert tuple(search.action.query for search in searches if search.action) == _SEARCHED_QUERIES
    assert {(search.status, search.action.type if search.action else None) for search in searches} == {
        ("completed", "search")
    }
    assert len({search.id for search in searches}) == len(searches)
    assert all(search.id and search.id.startswith("ws_") for search in searches)


@pytest.mark.asyncio
async def test_streamed_gemini_web_searches_arrive_in_order_with_the_non_streaming_ids() -> None:
    with respx.mock() as mock_router:
        mock_router.post(f"{_GEMINI_3_URL}:generateContent").mock(
            return_value=httpx.Response(
                200, json=_searched_body([*_SEARCH_PARTS, _TEXT_PART, _SHELL_CALL_PART], _GROUNDING)
            )
        )
        mock_router.post(f"{_GEMINI_3_URL}:streamGenerateContent?alt=sse").mock(
            return_value=_sse(
                _partial_body(_SEARCH_PARTS),
                _partial_body([_TEXT_PART]),
                _searched_body([_SHELL_CALL_PART], _GROUNDING),
            )
        )
        response: Final = await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
        )
        stream: Final = await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
            stream=True,
        )
        stream_events: Final = TypeAdapter(
            AsyncIterable[object],
            config=ConfigDict(arbitrary_types_allowed=True),
        ).validate_python(stream)
        events: Final = tuple(
            [_STREAM_EVENT.validate_python(event, from_attributes=True) async for event in stream_events]
        )

    completed_output: Final = next(
        event.response.output for event in events if event.response and event.type == "response.completed"
    )
    search_ids: Final = tuple(search.id for search in _search_items(completed_output) if search.id)
    trails: Final = {
        call_id: tuple((event.type, event.output_index) for event in events if event.is_about(call_id))
        for call_id in search_ids
    }
    added_indexes: Final = tuple(event.output_index for event in events if event.type == "response.output_item.added")
    event_types: Final = tuple(event.type for event in events)
    first_text_position: Final = event_types.index("response.output_text.delta")
    added_positions: Final = tuple(
        next(position for position, event in enumerate(events) if event.is_about(call_id)) for call_id in search_ids
    )

    assert search_ids == tuple(search.id for search in _search_items(_output_items(response)))
    assert (
        tuple(search.action.query for search in _search_items(completed_output) if search.action) == _SEARCHED_QUERIES
    )
    assert {
        call_id: tuple(event_type for event_type, _ in trail) for call_id, trail in trails.items()
    } == dict.fromkeys(
        search_ids,
        (
            "response.output_item.added",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
            "response.output_item.done",
        ),
    )
    assert {call_id: {output_index for _, output_index in trail} for call_id, trail in trails.items()} == {
        call_id: {position} for position, item in enumerate(completed_output) if (call_id := item.id) in trails
    }
    assert len(set(added_indexes)) == len(added_indexes)
    assert len(added_positions) == len(_SEARCHED_QUERIES)
    assert max(added_positions) < first_text_position


_CHUNKS_SHARING_SEARCHES_WITH_THE_FIRST_OUTPUT: Final = (
    pytest.param(((*_SEARCH_PARTS, _TEXT_PART), (_SHELL_CALL_PART,)), id="search_then_text"),
    pytest.param(((_THOUGHT_PART, *_SEARCH_PARTS), (_SHELL_CALL_PART,)), id="thought_then_search"),
    pytest.param(((_SIGNED_THOUGHT_PART, *_SEARCH_PARTS), (_SHELL_CALL_PART,)), id="signed_thought_then_search"),
    pytest.param(
        ((_THOUGHT_PART,), (*_SEARCH_PARTS, _TEXT_PART), (_SHELL_CALL_PART,)), id="thought_chunk_then_search_chunk"
    ),
    pytest.param(((_THOUGHT_PART, *_SEARCH_PARTS), (_THOUGHT_PART,)), id="stream_ends_while_thinking"),
)


def _searches_sharing_a_chunk(chunks: Sequence[Sequence[Mapping[str, object]]]) -> httpx.Response:
    return _sse(*(_partial_body(parts) for parts in chunks[:-1]), _searched_body(chunks[-1], _GROUNDING))


def _the_same_reply_unstreamed(chunks: Sequence[Sequence[Mapping[str, object]]]) -> httpx.Response:
    return httpx.Response(200, json=_searched_body([*chain.from_iterable(chunks)], _GROUNDING))


def _assert_items_open_and_close_in_output_order(
    events: Sequence[_StreamEvent], unstreamed_output: Sequence[_OutputItem]
) -> None:
    completed_output: Final = next(
        event.response.output for event in events if event.response and event.type == "response.completed"
    )
    added: Final = tuple(event for event in events if event.type == "response.output_item.added" and event.item)
    done_before_the_answer: Final = tuple(
        event.output_index
        for event in events
        if event.type == "response.output_item.done" and event.item and event.item.type != "message"
    )

    assert tuple(event.item.id for event in added if event.item) == tuple(item.id for item in completed_output)
    assert tuple(event.output_index for event in added) == tuple(range(len(completed_output)))
    assert done_before_the_answer == tuple(sorted(done_before_the_answer))
    assert tuple(item.type for item in completed_output) == tuple(item.type for item in unstreamed_output)


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", _CHUNKS_SHARING_SEARCHES_WITH_THE_FIRST_OUTPUT)
async def test_streamed_gemini_searches_sharing_a_chunk_with_the_first_output_open_in_output_order(
    chunks: tuple[tuple[Mapping[str, object], ...], ...],
) -> None:
    with respx.mock() as mock_router:
        mock_router.post(f"{_GEMINI_3_URL}:streamGenerateContent?alt=sse").mock(
            return_value=_searches_sharing_a_chunk(chunks)
        )
        mock_router.post(f"{_GEMINI_3_URL}:generateContent").mock(
            return_value=_the_same_reply_unstreamed(chunks)
        )
        response: Final = await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
        )
        stream: Final = await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
            stream=True,
        )
        stream_events: Final = TypeAdapter(
            AsyncIterable[object],
            config=ConfigDict(arbitrary_types_allowed=True),
        ).validate_python(stream)
        events: Final = tuple(
            [_STREAM_EVENT.validate_python(event, from_attributes=True) async for event in stream_events]
        )

    _assert_items_open_and_close_in_output_order(events, _output_items(response))


@pytest.mark.parametrize("chunks", _CHUNKS_SHARING_SEARCHES_WITH_THE_FIRST_OUTPUT)
def test_sync_streamed_gemini_searches_sharing_a_chunk_with_the_first_output_open_in_output_order(
    chunks: tuple[tuple[Mapping[str, object], ...], ...],
) -> None:
    with respx.mock() as mock_router:
        mock_router.post(f"{_GEMINI_3_URL}:streamGenerateContent?alt=sse").mock(
            return_value=_searches_sharing_a_chunk(chunks)
        )
        mock_router.post(f"{_GEMINI_3_URL}:generateContent").mock(
            return_value=_the_same_reply_unstreamed(chunks)
        )
        response: Final = litellm.responses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
        )
        stream: Final = litellm.responses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
            stream=True,
        )
        stream_events: Final = TypeAdapter(
            Iterable[object],
            config=ConfigDict(arbitrary_types_allowed=True),
        ).validate_python(stream)
        events: Final = tuple(_STREAM_EVENT.validate_python(event, from_attributes=True) for event in stream_events)

    _assert_items_open_and_close_in_output_order(events, _output_items(response))


@pytest.mark.asyncio
async def test_a_turn_replaying_gemini_web_search_calls_sends_valid_gemini_contents() -> None:
    with respx.mock() as mock_router:
        route: Final = mock_router.post(f"{_GEMINI_3_URL}:generateContent").mock(
            side_effect=[
                httpx.Response(200, json=_searched_body([*_SEARCH_PARTS, _TEXT_PART, _SHELL_CALL_PART], _GROUNDING)),
                httpx.Response(200, json=_searched_body([{"text": "LiteLLM 1.2.3 is the newest release."}], None)),
            ]
        )
        first_turn: Final = await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
        )
        assert isinstance(first_turn, ResponsesAPIResponse)
        replayed_output: Final = _JSON_OBJECT_LIST.validate_json(
            json.dumps(first_turn.model_dump(mode="json", exclude_none=True)["output"])
        )
        shell_call_id: Final = next(str(item["call_id"]) for item in replayed_output if item["type"] == "function_call")
        await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=[
                {"role": "user", "content": _SEARCH_PROMPT},
                *replayed_output,
                {"type": "function_call_output", "call_id": shell_call_id, "output": "1.2.3"},
            ],
            tools=_SEARCH_AND_SHELL_TOOLS,
        )
        contents: Final = _GEMINI_CONTENTS.validate_json(route.calls.last.request.content).contents

    assert sum(item["type"] == "web_search_call" for item in replayed_output) == len(_SEARCHED_QUERIES)
    assert all(content.parts for content in contents)
    assert tuple(content.role for content in contents) == ("user", "model", "user")
    assert [part.get("thoughtSignature") for part in contents[1].parts if "function_call" in part] == [_SHELL_SIGNATURE]


@pytest.mark.asyncio
async def test_bridged_gemini_search_tool_calls_bill_like_chat_completions() -> None:
    searched_body: Final = _searched_body([*_SEARCH_PARTS, _SHELL_CALL_PART], None)
    unsearched_body: Final = _searched_body([_SHELL_CALL_PART], None)
    with respx.mock() as mock_router:
        mock_router.post(f"{_GEMINI_3_URL}:generateContent").mock(
            side_effect=[httpx.Response(200, json=body) for body in (searched_body, searched_body, unsearched_body)]
        )
        bridged: Final = await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
        )
        chat: Final = await litellm.acompletion(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            messages=[{"role": "user", "content": _SEARCH_PROMPT}],
            tools=[
                {"type": "web_search"},
                {
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                    },
                },
            ],
        )
        unsearched: Final = await litellm.aresponses(
            model="gemini/gemini-3-flash-preview",
            api_key="test-key",
            input=_SEARCH_PROMPT,
            tools=_SEARCH_AND_SHELL_TOOLS,
        )

    bridged_cost: Final = litellm.completion_cost(completion_response=bridged)
    assert bridged_cost == pytest.approx(litellm.completion_cost(completion_response=chat))
    assert bridged_cost > litellm.completion_cost(completion_response=unsearched)
