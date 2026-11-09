import uuid
from collections.abc import Awaitable, Callable, Sequence
from queue import SimpleQueue
from typing import Final, cast

import litellm
from integration._support.wire import wire_server
from integration.providers._tool_search_support import (
    MODEL,
    SEARCH_RESULT,
    SEARCH_TURN,
    SEARCH_USE,
    TOOLS,
    WEATHER_ANSWER,
    WEATHER_ID,
    WEATHER_USE,
    Event,
    answer,
    answered_ids,
    first_messages,
    scripted_peer,
    wire_assistant,
)
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolMessage, ChatCompletionToolParam
from litellm.types.utils import ChatCompletionMessageToolCall

_API_KEY: Final = "scripted-anthropic-key"
_TOOLS: Final = cast(list[ChatCompletionToolParam], TOOLS)


def _messages(marker: str) -> list[AllMessageValues]:
    return cast(list[AllMessageValues], first_messages(marker))


def _answer(call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
    return {"role": "tool", "tool_call_id": call.id, "content": WEATHER_ANSWER}


def _recording(executed: SimpleQueue[str]) -> Callable[[ChatCompletionMessageToolCall], ChatCompletionToolMessage]:
    def execute(call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executed.put(str(call.function.name))
        return _answer(call)

    return execute


def _async_recording(
    executed: SimpleQueue[str],
) -> Callable[[ChatCompletionMessageToolCall], Awaitable[ChatCompletionToolMessage]]:
    async def execute(call: ChatCompletionMessageToolCall) -> ChatCompletionToolMessage:
        executed.put(str(call.function.name))
        return _answer(call)

    return execute


def _drained(executed: SimpleQueue[str]) -> tuple[str, ...]:
    return tuple(executed.get() for _ in range(executed.qsize()))


def _run(marker: str, wire_url: str, executed: SimpleQueue[str]) -> str | None:
    return litellm.run_tool_loop(
        model=f"anthropic/{MODEL}",
        messages=_messages(marker),
        tools=_TOOLS,
        execute_tool=_recording(executed),
        api_base=wire_url,
        api_key=_API_KEY,
        max_tokens=64,
    )


def test_run_tool_loop_executes_only_the_client_tool_and_replays_the_kept_search() -> None:
    marker: Final = uuid.uuid4().hex
    executed: Final[SimpleQueue[str]] = SimpleQueue()
    with wire_server(scripted_peer()) as wire:
        content: Final = _run(marker, wire.url, executed)
        received: Final = wire.drain()
    assert content == answer(marker), content
    assert _drained(executed) == ("get_weather",)
    assert len(received) == 2, received
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body


async def test_arun_tool_loop_executes_only_the_client_tool_and_replays_the_kept_search() -> None:
    marker: Final = uuid.uuid4().hex
    executed: Final[SimpleQueue[str]] = SimpleQueue()
    with wire_server(scripted_peer()) as wire:
        content: Final = await litellm.arun_tool_loop(
            model=f"anthropic/{MODEL}",
            messages=_messages(marker),
            tools=_TOOLS,
            execute_tool=_async_recording(executed),
            api_base=wire.url,
            api_key=_API_KEY,
            max_tokens=64,
        )
        received: Final = wire.drain()
    assert content == answer(marker), content
    assert _drained(executed) == ("get_weather",)
    assert len(received) == 2, received
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body


def test_run_tool_loop_returns_a_turn_that_only_used_the_server_search() -> None:
    marker: Final = uuid.uuid4().hex
    closing: Final = f"No weather tool was needed for marker-{marker}"
    turn: Final[Sequence[Event]] = (SEARCH_USE, SEARCH_RESULT, {"type": "text", "text": closing})
    executed: Final[SimpleQueue[str]] = SimpleQueue()
    with wire_server(scripted_peer(turn)) as wire:
        content: Final = _run(marker, wire.url, executed)
        received: Final = wire.drain()
    assert content == closing, content
    assert _drained(executed) == ()
    assert len(received) == 1, received


def test_run_tool_loop_replays_a_client_only_turn_as_a_plain_tool_call() -> None:
    marker: Final = uuid.uuid4().hex
    executed: Final[SimpleQueue[str]] = SimpleQueue()
    with wire_server(scripted_peer((WEATHER_USE,))) as wire:
        content: Final = _run(marker, wire.url, executed)
        received: Final = wire.drain()
    assert content == answer(marker), content
    assert _drained(executed) == ("get_weather",)
    assert len(received) == 2, received
    assert wire_assistant(received[1]) == [WEATHER_USE], received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body
