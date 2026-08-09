"""Tests for the streamed output-item lifecycle of the Responses API completion bridge.

A Responses client builds its item list from the event stream: an item exists only once
``response.output_item.added`` has announced it, every content event must arrive inside that
item's ``added``..``done`` window, and it must carry that item's id. A client that trusts the
protocol therefore drops -- or fails on -- content that belongs to an item it was never told
about, or that names an id it has never seen.

These tests drive the iterator over synthetic chat-completion chunks, once through the async
implementation and once through the sync one, and assert those invariants for the six content
shapes real providers produce.
"""

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, NamedTuple

import pytest

from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.types.llms.openai import (
    BaseLiteLLMOpenAIResponseObject,
    ResponsesAPIStreamEvents,
)
from litellm.types.utils import (
    Delta,
    ModelResponseStream,
    StreamingChoices,
)

ITEM_SCOPED_PREFIXES = (
    "response.output_text",
    "response.content_part",
    "response.reasoning_summary",
    "response.function_call_arguments",
)

MAX_EVENTS = 500

Events = list[BaseLiteLLMOpenAIResponseObject]
Driver = Callable[[list[ModelResponseStream]], Events | Awaitable[Events]]


def chunk(
    *,
    text: str | None = None,
    reasoning: str | None = None,
    thinking_blocks: list[dict[str, Any]] | None = None,
    tool: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    chunk_id: str = "chatcmpl-lifecycle",
) -> ModelResponseStream:
    """One chat-completion chunk carrying at most one kind of content."""
    delta = Delta(
        content=text,
        reasoning_content=reasoning,
        tool_calls=[tool] if tool is not None else None,
    )
    if thinking_blocks is not None:
        delta.thinking_blocks = thinking_blocks
    return ModelResponseStream(
        id=chunk_id,
        created=1700000000,
        model="test-model",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=delta,
                finish_reason=finish_reason,
            )
        ],
    )


def tool_call(
    call_id: str = "call_lifecycle_1",
    name: str = "get_weather",
    arguments: str = '{"city":"Paris"}',
) -> dict[str, Any]:
    return {
        "index": 0,
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


class FakeStream:
    """Stands in for ``CustomStreamWrapper``, supporting sync and async iteration."""

    logging_obj = None

    def __init__(self, chunks: Sequence[ModelResponseStream]) -> None:
        self._chunks = list(chunks)

    def __aiter__(self) -> "FakeStream":
        return self

    async def __anext__(self) -> ModelResponseStream:
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    def __iter__(self) -> "FakeStream":
        return self

    def __next__(self) -> ModelResponseStream:
        if not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)


def build_iterator(chunks: Sequence[ModelResponseStream]) -> LiteLLMCompletionStreamingIterator:
    return LiteLLMCompletionStreamingIterator(
        model="test-model",
        litellm_custom_stream_wrapper=FakeStream(chunks),  # pyright: ignore[reportArgumentType]  # test double
        request_input="Test input",
        responses_api_request={},
    )


async def collect_async(chunks: list[ModelResponseStream]) -> Events:
    """Drive ``__anext__`` to exhaustion and return every event it emitted."""
    iterator = build_iterator(chunks)
    events: Events = []
    while len(events) < MAX_EVENTS:
        try:
            events.append(await iterator.__anext__())
        except StopAsyncIteration:
            return events
    raise AssertionError(f"the async iterator emitted more than {MAX_EVENTS} events without finishing")


def collect_sync(chunks: list[ModelResponseStream]) -> Events:
    """Drive ``__next__`` to exhaustion and return every event it emitted."""
    iterator = build_iterator(chunks)
    events: Events = []
    while len(events) < MAX_EVENTS:
        try:
            events.append(iterator.__next__())
        except StopIteration:
            return events
    raise AssertionError(f"the sync iterator emitted more than {MAX_EVENTS} events without finishing")


async def collect(driver: Driver, chunks: list[ModelResponseStream]) -> Events:
    result = driver(chunks)
    return await result if inspect.isawaitable(result) else result


class Lifecycle(NamedTuple):
    added_indexes: list[int]
    done_indexes: list[int]
    orphans: list[str]
    id_mismatches: list[str]
    unclosed: list[int]
    missing_fields: list[str]


def event_type(event: BaseLiteLLMOpenAIResponseObject) -> str:
    return str(getattr(event.type, "value", event.type))


def analyse(events: Sequence[BaseLiteLLMOpenAIResponseObject]) -> Lifecycle:
    """Replay the stream the way a client does and report where it breaks its own contract."""
    open_items: dict[int, str | None] = {}
    added_indexes: list[int] = []
    done_indexes: list[int] = []
    orphans: list[str] = []
    id_mismatches: list[str] = []
    missing_fields: list[str] = []

    for event in events:
        name = event_type(event)
        output_index = getattr(event, "output_index", None)

        if name.startswith("response.reasoning_summary") and not hasattr(event, "summary_index"):
            missing_fields.append(f"{name} carries no summary_index, so a client discards it")

        if name == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value:
            added_indexes.append(output_index)
            open_items[output_index] = getattr(event.item, "id", None)
            if getattr(event.item, "type", None) == "reasoning" and not isinstance(
                getattr(event.item, "summary", None), list
            ):
                missing_fields.append(
                    "a reasoning output_item.added carries no summary array, so the item fails to deserialise"
                )
            continue

        if name == ResponsesAPIStreamEvents.OUTPUT_ITEM_DONE.value:
            done_indexes.append(output_index)
            open_items.pop(output_index, None)
            continue

        if not name.startswith(ITEM_SCOPED_PREFIXES):
            continue

        if output_index not in open_items:
            orphans.append(f"{name} at output_index {output_index}, which no open item holds")
            continue

        item_id = getattr(event, "item_id", None)
        if item_id is not None and item_id != open_items[output_index]:
            id_mismatches.append(
                f"{name} carries item_id {item_id!r}, "
                f"but output_index {output_index} was announced as {open_items[output_index]!r}"
            )

    return Lifecycle(
        added_indexes=added_indexes,
        done_indexes=done_indexes,
        orphans=orphans,
        id_mismatches=id_mismatches,
        unclosed=sorted(open_items),
        missing_fields=missing_fields,
    )


def assert_well_formed(events: Sequence[BaseLiteLLMOpenAIResponseObject], expected_items: int) -> None:
    lifecycle = analyse(events)
    assert lifecycle.orphans == []
    assert lifecycle.id_mismatches == []
    assert lifecycle.unclosed == []
    assert lifecycle.missing_fields == []
    assert sorted(lifecycle.added_indexes) == list(range(expected_items))
    assert sorted(lifecycle.done_indexes) == list(range(expected_items))


def assert_reasoning_events_address_the_reasoning_item(
    events: Sequence[BaseLiteLLMOpenAIResponseObject],
) -> None:
    announced = {
        event.item.id
        for event in events
        if event_type(event) == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value
        and getattr(event.item, "type", None) == "reasoning"
    }
    streamed = {
        getattr(event, "item_id", None)
        for event in events
        if event_type(event).startswith("response.reasoning_summary")
    }
    assert len(announced) == 1
    assert streamed == announced


def text_only() -> list[ModelResponseStream]:
    return [
        chunk(text="Hel"),
        chunk(text="lo, "),
        chunk(text="world"),
        chunk(finish_reason="stop"),
    ]


def text_then_tool() -> list[ModelResponseStream]:
    return [
        chunk(text="Let me "),
        chunk(text="check."),
        chunk(tool=tool_call()),
        chunk(finish_reason="tool_calls"),
    ]


def reasoning_then_text() -> list[ModelResponseStream]:
    return [
        chunk(reasoning="The user "),
        chunk(reasoning="wants a "),
        chunk(reasoning="greeting."),
        chunk(text="Hel"),
        chunk(text="lo, "),
        chunk(text="world"),
        chunk(finish_reason="stop"),
    ]


def reasoning_then_text_then_tool() -> list[ModelResponseStream]:
    return [
        chunk(reasoning="The user "),
        chunk(reasoning="wants the "),
        chunk(reasoning="weather."),
        chunk(text="Let me "),
        chunk(text="check."),
        chunk(tool=tool_call()),
        chunk(finish_reason="tool_calls"),
    ]


def text_then_reasoning() -> list[ModelResponseStream]:
    return [
        chunk(text="Hel"),
        chunk(text="lo, world"),
        chunk(reasoning="That "),
        chunk(reasoning="was "),
        chunk(reasoning="easy."),
        chunk(finish_reason="stop"),
    ]


def tool_only() -> list[ModelResponseStream]:
    return [
        chunk(tool=tool_call()),
        chunk(finish_reason="tool_calls"),
    ]


def text_tool_text() -> list[ModelResponseStream]:
    return [
        chunk(text="Let me check."),
        chunk(tool=tool_call()),
        chunk(text="It is "),
        chunk(text="sunny."),
        chunk(finish_reason="stop"),
    ]


def text_then_reasoning_then_text() -> list[ModelResponseStream]:
    return [
        chunk(text="Let me check."),
        chunk(reasoning="The tool "),
        chunk(reasoning="said sunny."),
        chunk(text="It is sunny."),
        chunk(finish_reason="stop"),
    ]


def reasoning_text_tool_reasoning() -> list[ModelResponseStream]:
    return [
        chunk(reasoning="I need the "),
        chunk(reasoning="weather."),
        chunk(text="Let me check."),
        chunk(tool=tool_call()),
        chunk(reasoning="Now I can "),
        chunk(reasoning="answer."),
        chunk(finish_reason="stop"),
    ]


drivers = pytest.mark.parametrize("driver", [collect_async, collect_sync], ids=["async", "sync"])


@drivers
async def test_text_only_announces_one_message_item(driver: Driver) -> None:
    """Plain text is one message item, and its deltas belong to it."""
    assert_well_formed(await collect(driver, text_only()), expected_items=1)


@drivers
async def test_text_then_tool_announces_a_message_and_a_function_call(driver: Driver) -> None:
    """Text followed by a tool call is two items, each with its own output_index."""
    assert_well_formed(await collect(driver, text_then_tool()), expected_items=2)


@drivers
async def test_reasoning_then_text_announces_a_reasoning_item_and_a_message(driver: Driver) -> None:
    """Reasoning followed by text is two items; the text may not be delivered inside the
    reasoning item's window, nor after it has closed with no message announced."""
    events = await collect(driver, reasoning_then_text())
    assert_well_formed(events, expected_items=2)
    assert_reasoning_events_address_the_reasoning_item(events)


@drivers
async def test_reasoning_then_text_then_tool_announces_all_three_items(driver: Driver) -> None:
    """The full assistant turn -- reasoning, then text, then a tool call -- is three items."""
    events = await collect(driver, reasoning_then_text_then_tool())
    assert_well_formed(events, expected_items=3)
    assert_reasoning_events_address_the_reasoning_item(events)


@drivers
async def test_text_then_reasoning_announces_a_message_and_a_reasoning_item(driver: Driver) -> None:
    """Some providers stream text before reasoning; the reasoning still needs its own item
    rather than being delivered inside the message's window."""
    events = await collect(driver, text_then_reasoning())
    assert_well_formed(events, expected_items=2)
    assert_reasoning_events_address_the_reasoning_item(events)


@drivers
async def test_text_tool_text_announces_a_second_message_item(driver: Driver) -> None:
    """Text resuming after a tool call is a new message item, not a continuation of the first."""
    assert_well_formed(await collect(driver, text_tool_text()), expected_items=3)


@drivers
async def test_tool_only_turn_emits_no_message_done_events(driver: Driver) -> None:
    """A turn whose only content is a tool call has no message item, so the message's done
    events must not be emitted: they would describe an item the client was never told about."""
    events = await collect(driver, tool_only())
    assert_well_formed(events, expected_items=1)
    emitted = {event_type(event) for event in events}
    assert ResponsesAPIStreamEvents.OUTPUT_TEXT_DONE.value not in emitted
    assert ResponsesAPIStreamEvents.CONTENT_PART_DONE.value not in emitted
    assert ResponsesAPIStreamEvents.CONTENT_PART_ADDED.value not in emitted


@drivers
async def test_text_then_reasoning_then_text_announces_three_items(driver: Driver) -> None:
    """Reasoning between two answers closes the first message item and opens a second."""
    events = await collect(driver, text_then_reasoning_then_text())
    assert_well_formed(events, expected_items=3)


@drivers
async def test_two_reasoning_blocks_in_one_turn_are_two_items(driver: Driver) -> None:
    """A turn that reasons again after a tool call gets a second reasoning item, with its own
    id -- reusing the first item's id would merge two blocks into one for the client."""
    events = await collect(driver, reasoning_text_tool_reasoning())
    assert_well_formed(events, expected_items=4)
    reasoning_ids = [
        str(event.item.id)
        for event in events
        if event_type(event) == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value
        and getattr(event.item, "type", None) == "reasoning"
    ]
    assert len(reasoning_ids) == 2
    assert len(set(reasoning_ids)) == 2


@drivers
async def test_reasoning_summary_part_added_precedes_its_deltas(driver: Driver) -> None:
    """A client opens its summary part on part.added and drops any delta that arrives first."""
    events = await collect(driver, reasoning_then_text())
    names = [event_type(event) for event in events]
    part_added = names.index("response.reasoning_summary_part.added")
    first_delta = names.index(ResponsesAPIStreamEvents.REASONING_SUMMARY_TEXT_DELTA.value)
    assert part_added < first_delta

    reasoning_id = announced_item_ids_by_type(events)["reasoning"]
    assert events[part_added].item_id == reasoning_id
    assert events[part_added].summary_index == events[first_delta].summary_index


@drivers
async def test_message_content_part_is_a_text_part(driver: Driver) -> None:
    """The message item's content part describes the message's own text. A reasoning part here
    describes an item this one is not, and overwrites the answer with the thinking."""
    events = await collect(driver, reasoning_then_text())
    parts = [
        event.part
        for event in events
        if event_type(event)
        in (
            ResponsesAPIStreamEvents.CONTENT_PART_ADDED.value,
            ResponsesAPIStreamEvents.CONTENT_PART_DONE.value,
        )
    ]
    assert parts
    assert {getattr(part, "type", None) for part in parts} == {"output_text"}


def announced_item_ids_by_type(events: Sequence[BaseLiteLLMOpenAIResponseObject]) -> dict[str, str]:
    return {
        str(getattr(event.item, "type", None)): str(getattr(event.item, "id", None))
        for event in events
        if event_type(event) == ResponsesAPIStreamEvents.OUTPUT_ITEM_ADDED.value
    }


def completed_item_ids_by_type(events: Sequence[BaseLiteLLMOpenAIResponseObject]) -> dict[str, str]:
    completed = [event for event in events if event_type(event) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED.value]
    assert len(completed) == 1
    return {str(item.type): str(item.id) for item in completed[0].response.output}


@drivers
async def test_completed_response_carries_the_ids_the_stream_announced(driver: Driver) -> None:
    """A client reconciles the items it built from the stream against the completed response by
    id, so an item must not be renamed between the two."""
    events = await collect(driver, reasoning_then_text())
    announced = announced_item_ids_by_type(events)
    completed = completed_item_ids_by_type(events)
    assert completed["reasoning"] == announced["reasoning"]
    assert completed["message"] == announced["message"]


@drivers
async def test_reasoning_item_ids_are_minted_per_turn_not_derived_from_content(driver: Driver) -> None:
    """Two turns that reason identically are still two different items. Deriving the id from the
    content also made it depend on hash randomisation, so it differed across workers."""
    first = announced_item_ids_by_type(await collect(driver, reasoning_then_text()))
    second = announced_item_ids_by_type(await collect(driver, reasoning_then_text()))
    assert first["reasoning"] != second["reasoning"]


@drivers
async def test_completed_response_holds_no_item_the_stream_never_announced(driver: Driver) -> None:
    """A turn that produced no assistant text has no message item. Putting an empty one in the
    completed response gives a reconciling client an item it never saw announced, which reads as
    an item it missed rather than one that does not exist."""
    chunks = [
        chunk(tool=tool_call(call_id="call_no_message_item")),
        chunk(finish_reason="tool_calls"),
    ]
    events = await collect(driver, chunks)
    completed = [event for event in events if event_type(event) == ResponsesAPIStreamEvents.RESPONSE_COMPLETED.value]
    assert len(completed) == 1
    assert [str(item.type) for item in completed[0].response.output] == ["function_call"]


