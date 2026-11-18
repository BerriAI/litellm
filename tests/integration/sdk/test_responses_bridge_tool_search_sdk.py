import asyncio
import uuid
from collections.abc import Sequence
from typing import Final, cast

import pytest
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support import tool_search as ts
from integration._support.wire import Wire, wire_server
from pydantic import JsonValue

import litellm
from litellm import Router
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolMessage, ChatCompletionToolParam
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, Message, ModelResponse

pytestmark: Final = pytest.mark.timeout(120)

_API_KEY: Final = "sk-scripted-provider"


def _tools(*tools: JsonValue) -> list[ChatCompletionToolParam]:
    return cast(list[ChatCompletionToolParam], list(tools))


def _messages(*messages: object) -> list[AllMessageValues]:
    return cast(list[AllMessageValues], list(messages))


def _user(marker: str, *words: str) -> dict[str, JsonValue]:
    return {"role": "user", "content": " ".join((pcb.prompt(marker), *words))}


def _replay(marker: str) -> list[dict[str, JsonValue]]:
    return [*ts.searched(marker, "a", ts.WEATHER), ts.reasoning(marker, "b"), ts.replayed_call(marker)]


def _message(response: object) -> Message:
    assert isinstance(response, ModelResponse), type(response)
    choice: Final = response.choices[0]
    assert isinstance(choice, Choices), type(choice)
    return choice.message


def _assert_turn_one(message: Message, marker: str) -> None:
    fields: Final = rv.JSON_OBJECT.validate_python(message.provider_specific_fields or {})
    assert fields.get("tool_search_items") == [*ts.searched(marker, "a", ts.WEATHER)], message
    calls: Final = message.tool_calls or []
    call_fields: Final = [
        rv.JSON_OBJECT.validate_python(getattr(call, "provider_specific_fields", None) or {}) for call in calls
    ]
    assert call_fields == [{"namespace": ts.WEATHER}], message


def _kwargs(wire: Wire) -> dict[str, str]:
    return {"api_base": f"{wire.url}/v1", "api_key": _API_KEY}


def test_completion_returns_and_replays_tool_search_items() -> None:
    marker: Final = uuid.uuid4().hex
    tools: Final = _tools(*ts.chat_tools(ts.WEATHER))
    with wire_server(ts.scripted) as wire:
        user: Final = _user(marker)
        first: Final = _message(
            litellm.completion(model=pcb.MODEL, messages=_messages(user), tools=tools, **_kwargs(wire))
        )
        _assert_turn_one(first, marker)
        pcb.posted(wire, marker)
        second: Final = _message(
            litellm.completion(
                model=pcb.MODEL,
                messages=_messages(user, first, ts.tool_result(marker)),
                tools=tools,
                **_kwargs(wire),
            )
        )
        assert second.content == rv.answer(marker), second
        assert ts.replayed(pcb.posted(wire, marker)) == _replay(marker)


def test_streamed_completion_rebuilt_by_stream_chunk_builder_replays_tool_search_items() -> None:
    marker: Final = uuid.uuid4().hex
    tools: Final = _tools(*ts.chat_tools(ts.WEATHER))

    async def run(wire: Wire) -> tuple[Message, Message]:
        user: Final = _user(marker)
        stream: Final = await litellm.acompletion(
            model=pcb.MODEL, messages=_messages(user), tools=tools, stream=True, **_kwargs(wire)
        )
        chunks: Final = [chunk async for chunk in stream]  # pyright: ignore[reportGeneralTypeIssues]  # stream wrapper
        built: Final = _message(litellm.stream_chunk_builder(chunks, messages=_messages(user)))
        pcb.posted(wire, marker)
        answered: Final = _message(
            await litellm.acompletion(
                model=pcb.MODEL,
                messages=_messages(user, built.model_dump(), ts.tool_result(marker)),
                tools=tools,
                **_kwargs(wire),
            )
        )
        return built, answered

    with wire_server(ts.scripted) as wire:
        built, answered = asyncio.run(run(wire))
        _assert_turn_one(built, marker)
        assert [item["id"] for item in built.reasoning_items or []] == [f"rs_a_{marker}", f"rs_b_{marker}"], built
        assert answered.content == rv.answer(marker), answered
        assert ts.replayed(pcb.posted(wire, marker)) == _replay(marker)


def test_streamed_turn_without_tool_search_replays_its_reasoning() -> None:
    marker: Final = uuid.uuid4().hex
    tools: Final = _tools({"type": "function", "function": ts.function(ts.WEATHER)})
    with wire_server(ts.scripted) as wire:
        user: Final = _user(marker)
        stream: Final = litellm.completion(
            model=pcb.MODEL, messages=_messages(user), tools=tools, stream=True, **_kwargs(wire)
        )
        chunks: Final = list(stream)  # pyright: ignore[reportArgumentType]  # stream wrapper
        built: Final = _message(litellm.stream_chunk_builder(chunks, messages=_messages(user)))
        pcb.posted(wire, marker)
        answered: Final = _message(
            litellm.completion(
                model=pcb.MODEL,
                messages=_messages(user, built.model_dump(), ts.tool_result(marker)),
                tools=tools,
                **_kwargs(wire),
            )
        )
        assert answered.content == rv.answer(marker), answered
        assert ts.replayed(pcb.posted(wire, marker)) == [
            ts.reasoning(marker, "a"),
            ts.reasoning(marker, "b"),
            ts.replayed_call(marker),
        ]


def test_router_completion_replays_tool_search_items() -> None:
    marker: Final = uuid.uuid4().hex
    tools: Final = _tools(*ts.chat_tools(ts.WEATHER))
    with wire_server(ts.scripted) as wire:
        router: Final = Router(
            model_list=[{"model_name": "bridged", "litellm_params": {"model": pcb.MODEL, **_kwargs(wire)}}],
            num_retries=0,
        )
        user: Final = _user(marker)
        first: Final = _message(router.completion(model="bridged", messages=_messages(user), tools=tools))
        _assert_turn_one(first, marker)
        pcb.posted(wire, marker)
        second: Final = _message(
            router.completion(model="bridged", messages=_messages(user, first, ts.tool_result(marker)), tools=tools)
        )
        assert second.content == rv.answer(marker), second
        assert ts.replayed(pcb.posted(wire, marker)) == _replay(marker)


def _sunny(tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
    return {"role": "tool", "tool_call_id": tool_call.id, "content": "sunny"}


async def _async_sunny(tool_call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
    return _sunny(tool_call)


def _posted_in_order(wire: Wire, marker: str) -> Sequence[list[dict[str, JsonValue]]]:
    return [ts.replayed(request) for request in pcb.with_marker(pcb.drained_posts(wire), marker)]


def test_run_tool_loop_keeps_tool_search_items_and_namespace_between_rounds() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire:
        answer: Final = litellm.run_tool_loop(
            model=pcb.MODEL,
            messages=_messages(_user(marker)),
            tools=_tools(*ts.chat_tools(ts.WEATHER)),
            execute_tool=_sunny,
            **_kwargs(wire),
        )
        assert answer == rv.answer(marker)
        assert _posted_in_order(wire, marker) == [[], _replay(marker)]


def test_async_run_tool_loop_keeps_two_searches_in_order() -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire:
        answer: Final = asyncio.run(
            litellm.arun_tool_loop(
                model=pcb.MODEL,
                messages=_messages(_user(marker, ts.TWO_SEARCHES)),
                tools=_tools(*ts.chat_tools(ts.WEATHER, ts.TIME)),
                execute_tool=_async_sunny,
                **_kwargs(wire),
            )
        )
        assert answer == rv.answer(marker)
        assert _posted_in_order(wire, marker) == [
            [],
            _replay(marker),
            [*_replay(marker), *ts.searched(marker, "c", ts.TIME), ts.replayed_call(marker, ts.TIME)],
        ]
