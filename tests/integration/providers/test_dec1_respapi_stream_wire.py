"""Streamed `/v1/responses` built by the bridge from chat completion chunks (gaps left by the stream suites).

`LiteLLMCompletionStreamingIterator` (`litellm/responses/litellm_completion_transformation/streaming_iterator.py`)
turns chat chunks into Responses events. The text lifecycle, sequence numbers, a single tool call's argument
events and the `length` to `response.incomplete` ending are pinned in `streaming/test_dec1_stream_responses_events.py`.
This file pins what is left: chat `reasoning_content` becomes `response.reasoning_summary_text.delta` events on an
`rs_` item at an output index before the message, closed by `response.reasoning_summary_text.done` with the joined
text; the `response.created` snapshot echoes the request's `instructions`, `tools`, `metadata` and `user` with
`status: in_progress`; the terminal `response.completed` carries the chat usage renamed to input and output
tokens and the full output; two streamed tool calls get two `function_call` items with distinct output indexes;
and the same lifecycle comes out of an Anthropic stream. The response id stays the same on every snapshot.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value, string_value
from integration._support.wire import Reply
from integration.providers.dec1_respapi_support import (
    COMPLETION_TOKENS,
    PROMPT_TOKENS,
    REPLY_TEXT,
    STREAM_PIECES,
    Call,
    anthropic,
    chat_bridge,
    chat_chunk,
    chat_stream,
    chat_usage_chunk,
    deepseek,
    one_event,
    peer,
    stream_events,
    text_stream,
    types_of,
)
from pydantic import JsonValue

WEATHER_TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
}


def _text(call: Call) -> Reply:
    return text_stream()


def _reasoning_then_text(call: Call) -> Reply:
    identity: Final = "chatcmpl-dec1reason"
    return chat_stream(
        (
            chat_chunk(identity, {"role": "assistant", "content": ""}),
            chat_chunk(identity, {"reasoning_content": "first "}),
            chat_chunk(identity, {"reasoning_content": "thought"}),
            chat_chunk(identity, {"content": REPLY_TEXT}),
            chat_chunk(identity, {}, "stop"),
            chat_usage_chunk(identity),
        )
    )


def _two_tool_calls(call: Call) -> Reply:
    identity: Final = "chatcmpl-dec1tools"

    def call_delta(index: int, call_id: str, arguments: str) -> dict[str, JsonValue]:
        return {
            "tool_calls": [
                {
                    "index": index,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }
            ]
        }

    return chat_stream(
        (
            chat_chunk(identity, {"role": "assistant", "content": None}),
            chat_chunk(identity, call_delta(0, "call_dec1s1", '{"city":"Oslo"}')),
            chat_chunk(identity, call_delta(1, "call_dec1s2", '{"city":"Rome"}')),
            chat_chunk(identity, {}, "tool_calls"),
            chat_usage_chunk(identity),
        )
    )


def test_streamed_reasoning_becomes_reasoning_summary_text_deltas(gateway: Gateway) -> None:
    """DeepSeek streams `reasoning_content` `first ` then `thought`, then the text.

    The stream must carry two `response.reasoning_summary_text.delta` events with deltas `first ` and
    `thought`, both on one `item_id` that starts with `rs_` (`mint_reasoning_item_id`).
    """
    with peer(_reasoning_then_text) as provider, gateway.scenario() as scenario:
        model: Final = deepseek(scenario, provider)
        events: Final = stream_events(gateway, model)
        deltas: Final = [event for event in events if event["type"] == "response.reasoning_summary_text.delta"]
        assert [event["delta"] for event in deltas] == ["first ", "thought"]
        item_ids: Final = {string_value(event["item_id"]) for event in deltas}
        assert len(item_ids) == 1 and next(iter(item_ids)).startswith("rs_"), item_ids


def test_streamed_reasoning_is_closed_with_the_joined_summary_text(gateway: Gateway) -> None:
    """The same reasoning stream. One `response.reasoning_summary_text.done` must carry `text: "first thought"`
    and `summary_index: 0` on the same `rs_` item (`streaming_iterator.py:989`)."""
    with peer(_reasoning_then_text) as provider, gateway.scenario() as scenario:
        model: Final = deepseek(scenario, provider)
        events: Final = stream_events(gateway, model)
        done: Final = one_event(events, "response.reasoning_summary_text.done")
        assert (done["text"], done["summary_index"]) == ("first thought", 0)
        assert string_value(done["item_id"]).startswith("rs_")


def test_streamed_reasoning_takes_the_output_index_before_the_message(gateway: Gateway) -> None:
    """Reasoning arrives before text, so the reasoning item allocates output index 0 and the message item index 1.

    Every `response.reasoning_summary_text.delta` must have `output_index: 0` and every
    `response.output_text.delta` must have `output_index: 1` (`_reasoning_index`, `_message_index`).
    """
    with peer(_reasoning_then_text) as provider, gateway.scenario() as scenario:
        model: Final = deepseek(scenario, provider)
        events: Final = stream_events(gateway, model)
        assert {
            event["output_index"] for event in events if event["type"] == "response.reasoning_summary_text.delta"
        } == {0}
        assert {event["output_index"] for event in events if event["type"] == "response.output_text.delta"} == {1}


def test_the_completed_snapshot_holds_the_reasoning_item_then_the_message(gateway: Gateway) -> None:
    """The terminal `response.completed` of the reasoning stream must list `output` types `reasoning, message`,
    and the message text must be `REPLY_TEXT` with no reasoning text in it."""
    with peer(_reasoning_then_text) as provider, gateway.scenario() as scenario:
        model: Final = deepseek(scenario, provider)
        completed: Final = object_value(one_event(stream_events(gateway, model), "response.completed")["response"])
        output: Final = [object_value(item) for item in list_value(completed["output"])]
        assert [item["type"] for item in output] == ["reasoning", "message"]
        assert REPLY_TEXT in json.dumps(output[1]) and "first thought" not in json.dumps(output[1])


def test_the_created_snapshot_echoes_the_request_and_is_in_progress(gateway: Gateway) -> None:
    """The caller streams with `instructions: "Be brief."`, `user: dec1-user`, `metadata: {"dec1": "yes"}` and the
    `get_weather` tool.

    `response.created` must carry `status: in_progress`, `output: []`, `usage: null`, and echo `instructions`,
    `user`, `metadata` and `tools[0].name` (`_default_response_created_event_data`, `streaming_iterator.py:785`).
    """
    with peer(_text) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        created: Final = object_value(
            one_event(
                stream_events(
                    gateway,
                    model,
                    instructions="Be brief.",
                    user="dec1-user",
                    metadata={"dec1": "yes"},
                    tools=[WEATHER_TOOL],
                ),
                "response.created",
            )["response"]
        )
        assert (created["status"], created["output"], created.get("usage")) == ("in_progress", [], None)
        assert (created["instructions"], created["user"], created["metadata"]) == (
            "Be brief.",
            "dec1-user",
            {"dec1": "yes"},
        )
        assert object_value(list_value(created["tools"])[0])["name"] == "get_weather"


def test_the_created_snapshot_echoes_store_false(gateway: Gateway) -> None:
    """The caller streams with `store: false`.

    The `response.created` snapshot should say `store: false`, as the non streamed answer and OpenAI do.
    """
    pytest.skip(
        "BUG: the bridged response.created snapshot hardcodes store: true and parallel_tool_calls: true "
        "(streaming_iterator.py:799) whatever the caller sent"
    )
    with peer(_text) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        created: Final = object_value(
            one_event(stream_events(gateway, model, store=False), "response.created")["response"]
        )
        assert created["store"] is False


def test_every_snapshot_shares_one_managed_response_id(gateway: Gateway) -> None:
    """`response.created`, `response.in_progress` and `response.completed` each carry a `response` snapshot.

    All three `response.id` values must be equal and start with `resp_`, so a client can read the id from the
    first event and use it for `previous_response_id` after the stream ends.
    """
    with peer(_text) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        events: Final = stream_events(gateway, model)
        ids: Final = {
            string_value(object_value(event["response"])["id"])
            for event in events
            if event["type"] in ("response.created", "response.in_progress", "response.completed")
        }
        assert len(ids) == 1 and next(iter(ids)).startswith("resp_"), ids


def test_the_completed_event_carries_the_chat_usage_renamed(gateway: Gateway) -> None:
    """The chat stream's usage chunk reports 11 prompt and 7 completion tokens.

    `response.completed.response.usage` must hold `input_tokens: 11`, `output_tokens: 7`, `total_tokens: 18`.
    """
    with peer(_text) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        completed: Final = object_value(one_event(stream_events(gateway, model), "response.completed")["response"])
        usage: Final = object_value(completed["usage"])
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (
            PROMPT_TOKENS,
            COMPLETION_TOKENS,
            PROMPT_TOKENS + COMPLETION_TOKENS,
        )


def test_the_completed_snapshot_holds_the_full_streamed_text(gateway: Gateway) -> None:
    """The three streamed pieces `dec1 `, `bridged `, `reply` must be joined into the `output_text` of the one
    message item in `response.completed`, so a client that only reads the last event still sees the answer."""
    with peer(_text) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        completed: Final = object_value(one_event(stream_events(gateway, model), "response.completed")["response"])
        messages: Final = [
            object_value(item) for item in list_value(completed["output"]) if object_value(item)["type"] == "message"
        ]
        assert len(messages) == 1
        texts: Final = [object_value(part)["text"] for part in list_value(messages[0]["content"])]
        assert texts == ["".join(STREAM_PIECES)]


def test_two_streamed_tool_calls_become_two_function_call_items_with_distinct_indexes(gateway: Gateway) -> None:
    """The chat stream announces tool calls `call_dec1s1` (index 0) and `call_dec1s2` (index 1).

    The stream must carry two `response.output_item.added` events whose items are `function_call`s with those
    call ids and different `output_index` values, and `response.completed` must list both calls in that order.
    """
    with peer(_two_tool_calls) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        events: Final = stream_events(gateway, model, tools=[WEATHER_TOOL])
        added: Final = [
            event
            for event in events
            if event["type"] == "response.output_item.added" and object_value(event["item"])["type"] == "function_call"
        ]
        assert [object_value(event["item"])["call_id"] for event in added] == ["call_dec1s1", "call_dec1s2"]
        assert len({event["output_index"] for event in added}) == 2
        completed: Final = object_value(one_event(events, "response.completed")["response"])
        calls: Final = [
            object_value(item)["call_id"]
            for item in list_value(completed["output"])
            if object_value(item)["type"] == "function_call"
        ]
        assert calls == ["call_dec1s1", "call_dec1s2"]


def test_each_streamed_tool_call_ends_with_its_full_arguments(gateway: Gateway) -> None:
    """For the same two tool calls, each `response.function_call_arguments.done` must carry the call's whole
    argument string: `{"city":"Oslo"}` then `{"city":"Rome"}`."""
    with peer(_two_tool_calls) as provider, gateway.scenario() as scenario:
        model: Final = chat_bridge(scenario, provider)
        events: Final = stream_events(gateway, model, tools=[WEATHER_TOOL])
        done: Final = [
            event["arguments"] for event in events if event["type"] == "response.function_call_arguments.done"
        ]
        assert done == ['{"city":"Oslo"}', '{"city":"Rome"}']


def test_an_anthropic_stream_produces_the_same_text_lifecycle(gateway: Gateway) -> None:
    """An Anthropic deployment streams `message_start`, one text block with two deltas, `message_delta` with
    `stop_reason: end_turn` and `message_stop`.

    The caller's stream must start with `response.created`, end with `response.completed`, and its
    `response.output_text.delta` events must join to `dec1 claude`.
    """

    def respond(call: Call) -> Reply:
        frames: Final[list[tuple[str, dict[str, JsonValue]]]] = [
            (
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_dec1stream",
                        "type": "message",
                        "role": "assistant",
                        "model": call.body.get("model"),
                        "content": [],
                        "stop_reason": None,
                        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": 1},
                    },
                },
            ),
            (
                "content_block_start",
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "dec1 "}},
            ),
            (
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "claude"}},
            ),
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": COMPLETION_TOKENS},
                },
            ),
            ("message_stop", {"type": "message_stop"}),
        ]
        return Reply(
            content_type="text/event-stream",
            chunks=tuple(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode() for name, data in frames),
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        model: Final = anthropic(scenario, provider)
        events: Final = stream_events(gateway, model)
        kinds: Final = types_of(events)
        assert kinds[0] == "response.created" and kinds[-1] == "response.completed", kinds
        assert "".join(str(event["delta"]) for event in events if event["type"] == "response.output_text.delta") == (
            "dec1 claude"
        )
