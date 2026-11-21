"""The Responses event stream the bridge builds for providers without a native Responses API.

A streamed `/v1/responses` call on a bridged deployment is answered by
`LiteLLMCompletionStreamingIterator` (`litellm/responses/litellm_completion_transformation/streaming_iterator.py`),
which turns chat completion chunks into the OpenAI Responses event grammar. Clients such as the OpenAI Agents SDK
and Codex drive their state machines from that grammar, so the order of events, the ids they carry and the
indexes they name are the contract: `response.created` and `response.in_progress` first, then for a text answer
`output_item.added` (a `msg_` message item, `in_progress`, empty content), `content_part.added` (an empty
`output_text` part), one `output_text.delta` per text chunk, `output_text.done`, `content_part.done`,
`output_item.done`, and finally `response.completed` (or `response.incomplete` when the model ran out of tokens).
Every event carries a `sequence_number` stamped at emission (`streaming_iterator.py:1334`). Tool calls become a
`function_call` item whose argument string is cut into 10 character `function_call_arguments.delta` events
(`streaming_iterator.py:557`, `:635`). Round 1 pinned reasoning deltas, snapshot echoes, completed usage and two
parallel tool calls; this file pins the exact grammar, on the chat carrier, on Anthropic and on Gemini.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value, string_value
from integration._support.wire import Reply
from integration.providers.dec1_respapi2_support import (
    GEMINI_STREAM_PATH,
    WEATHER_TOOL,
    every_respond,
    gemini,
    gemini_stream,
)
from integration.providers.dec1_respapi_support import (
    STREAM_PIECES,
    Call,
    anthropic,
    chat_bridge,
    chat_chunk,
    chat_stream,
    chat_usage_chunk,
    default_respond,
    one_event,
    peer,
    stream_events,
    text_stream,
    tool_call,
    types_of,
)
from pydantic import JsonValue

TEXT_LIFECYCLE: Final = [
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.delta",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
]
LONG_ARGUMENTS: Final = '{"city":"Copenhagen"}'


def _streamed_tool_call(call_id: str = "call_dec1r2", arguments: str = LONG_ARGUMENTS) -> Reply:
    """A chat stream that announces one tool call and sends its whole argument string in one chunk."""
    stream_id: Final = "chatcmpl-dec1r2tool"
    return chat_stream(
        (
            chat_chunk(stream_id, {"role": "assistant", "content": None}),
            chat_chunk(
                stream_id,
                {"tool_calls": [{"index": 0, **tool_call(call_id, "get_weather", arguments)}]},
            ),
            chat_chunk(stream_id, {}, "tool_calls"),
            chat_usage_chunk(stream_id),
        )
    )


def _tool_respond(call: Call) -> Reply:
    return _streamed_tool_call() if call.method == "POST" else default_respond(call)


def _of_type(events: Sequence[Mapping[str, JsonValue]], kind: str) -> list[dict[str, JsonValue]]:
    return [dict(event) for event in events if event.get("type") == kind]


def _text_events(gateway: Gateway) -> list[dict[str, JsonValue]]:
    with peer() as provider, gateway.scenario() as scenario:
        return stream_events(gateway, chat_bridge(scenario, provider))


def test_a_bridged_text_stream_follows_the_exact_responses_event_order(gateway: Gateway) -> None:
    """The chat carrier streams a role chunk, three text pieces, a `stop` chunk and a usage chunk.

    The caller's event types must be exactly the eleven event lifecycle in `TEXT_LIFECYCLE`, in that order
    (`streaming_iterator.py:1196` initial events, `:1525` deltas, `:1162` done events, `:1672` terminal event).
    """
    assert types_of(_text_events(gateway)) == TEXT_LIFECYCLE


def test_sequence_numbers_count_up_from_zero_without_gaps(gateway: Gateway) -> None:
    """Every emitted event is stamped with the next `sequence_number` (`streaming_iterator.py:1334`), so the
    numbers on the plain text stream must be exactly `0, 1, ..., n-1` in arrival order."""
    events: Final = _text_events(gateway)
    assert [event["sequence_number"] for event in events] == list(range(len(events)))


def test_the_message_item_is_announced_in_progress_with_empty_content(gateway: Gateway) -> None:
    """The one `response.output_item.added` must carry `output_index: 0` and an item with `type: message`,
    `role: assistant`, `status: in_progress`, `content: []` and an id that starts with `msg_`
    (`streaming_iterator.py:886`)."""
    added: Final = one_event(_text_events(gateway), "response.output_item.added")
    item: Final = object_value(added["item"])
    assert added["output_index"] == 0
    assert (item["type"], item["role"], item["status"], item["content"]) == ("message", "assistant", "in_progress", [])
    assert string_value(item["id"]).startswith("msg_")


def test_the_content_part_is_announced_as_an_empty_output_text(gateway: Gateway) -> None:
    """`response.content_part.added` must carry `content_index: 0` and the part
    `{"type": "output_text", "text": "", "annotations": []}` (`streaming_iterator.py:868`)."""
    added: Final = one_event(_text_events(gateway), "response.content_part.added")
    assert added["content_index"] == 0
    assert added["part"] == {"type": "output_text", "text": "", "annotations": []}


def test_every_text_event_names_the_announced_message_item(gateway: Gateway) -> None:
    """Each `content_part.added`, `output_text.delta`, `output_text.done` and `content_part.done` must carry the
    `item_id` the `output_item.added` announced, with `output_index: 0` and `content_index: 0`, so a client can
    attach each delta to the right item."""
    events: Final = _text_events(gateway)
    item_id: Final = object_value(one_event(events, "response.output_item.added")["item"])["id"]
    scoped: Final = [
        event
        for event in events
        if event["type"]
        in {
            "response.content_part.added",
            "response.output_text.delta",
            "response.output_text.done",
            "response.content_part.done",
        }
    ]
    assert len(scoped) == 6, types_of(events)
    assert {(event["item_id"], event["output_index"], event["content_index"]) for event in scoped} == {(item_id, 0, 0)}


def test_each_chat_text_chunk_is_relayed_as_one_delta_in_order(gateway: Gateway) -> None:
    """The three chat pieces `dec1 `, `bridged `, `reply` must arrive as three `output_text.delta` events whose
    `delta` values are exactly those pieces in that order; the bridge does not merge or split text deltas."""
    deltas: Final = _of_type(_text_events(gateway), "response.output_text.delta")
    assert [event["delta"] for event in deltas] == list(STREAM_PIECES)


def test_the_done_events_carry_the_whole_joined_text(gateway: Gateway) -> None:
    """`output_text.done.text`, `content_part.done.part.text` and the single `output_text` part of
    `output_item.done` must each be the joined text `dec1 bridged reply`, and the done item is `completed`
    (`streaming_iterator.py:1048`, `:1060`, `:1087`)."""
    events: Final = _text_events(gateway)
    joined: Final = "".join(STREAM_PIECES)
    assert one_event(events, "response.output_text.done")["text"] == joined
    assert object_value(one_event(events, "response.content_part.done")["part"])["text"] == joined
    done_item: Final = object_value(one_event(events, "response.output_item.done")["item"])
    assert done_item["status"] == "completed"
    assert [object_value(part)["text"] for part in list_value(done_item["content"])] == [joined]


def test_the_completed_snapshot_reuses_the_streamed_message_item_id(gateway: Gateway) -> None:
    """The message item in `response.completed.response.output` must keep the `msg_` id the stream announced
    (`_output_with_streamed_item_ids`, `streaming_iterator.py:1595`), so a client that keys its state by item id
    does not see a second message."""
    events: Final = _text_events(gateway)
    streamed_id: Final = object_value(one_event(events, "response.output_item.added")["item"])["id"]
    completed: Final = object_value(one_event(events, "response.completed")["response"])
    assert [object_value(item)["id"] for item in list_value(completed["output"])] == [streamed_id]


def test_a_length_finish_ends_the_stream_with_response_incomplete(gateway: Gateway) -> None:
    """The chat stream ends with `finish_reason: length`. The terminal event must be `response.incomplete`
    whose snapshot has `status: incomplete` and `incomplete_details.reason: max_output_tokens`, and no
    `response.completed` is sent (`streaming_iterator.py:1693`, `transformation.py:217`)."""
    stream_id: Final = "chatcmpl-dec1r2len"

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        return chat_stream(
            (
                chat_chunk(stream_id, {"role": "assistant", "content": "cut "}),
                chat_chunk(stream_id, {"content": "short"}),
                chat_chunk(stream_id, {}, "length"),
                chat_usage_chunk(stream_id),
            )
        )

    with peer(respond) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, chat_bridge(scenario, provider), max_output_tokens=16)
    assert types_of(events)[-1] == "response.incomplete"
    assert "response.completed" not in types_of(events)
    snapshot: Final = object_value(events[-1]["response"])
    assert snapshot["status"] == "incomplete"
    assert object_value(snapshot["incomplete_details"])["reason"] == "max_output_tokens"


def test_a_streamed_tool_call_is_announced_as_an_in_progress_function_call_item(gateway: Gateway) -> None:
    """The chat stream announces tool call `call_dec1r2` to `get_weather`. The `output_item.added` for it must
    carry `type: function_call`, `id: fc_call_dec1r2`, `call_id: call_dec1r2`, `name: get_weather`,
    `arguments: ""` and `status: in_progress` (`custom_tools.py:124`, `streaming_iterator.py:549`)."""
    with peer(_tool_respond) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, chat_bridge(scenario, provider), tools=[WEATHER_TOOL])
    added: Final = one_event(events, "response.output_item.added")
    assert object_value(added["item"]) == {
        "type": "function_call",
        "id": "fc_call_dec1r2",
        "call_id": "call_dec1r2",
        "name": "get_weather",
        "arguments": "",
        "status": "in_progress",
    }


def test_a_long_argument_chunk_is_cut_into_ten_character_deltas(gateway: Gateway) -> None:
    """The whole argument string `{"city":"Copenhagen"}` (21 characters) arrives in one chat chunk. The bridge
    splits it into `function_call_arguments.delta` events of at most 10 characters (`chunk_size = 10`,
    `streaming_iterator.py:557`): exactly `{"city":"C`, `openhagen"`, `}`, all naming item `fc_call_dec1r2`.
    A rewrite that relays the chunk whole would change what token by token clients render."""
    with peer(_tool_respond) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, chat_bridge(scenario, provider), tools=[WEATHER_TOOL])
    deltas: Final = _of_type(events, "response.function_call_arguments.delta")
    assert [event["delta"] for event in deltas] == ['{"city":"C', 'openhagen"', "}"]
    assert {event["item_id"] for event in deltas} == {"fc_call_dec1r2"}


def test_a_streamed_tool_call_ends_with_its_arguments_and_a_completed_item(gateway: Gateway) -> None:
    """After the deltas the stream must carry one `function_call_arguments.done` with
    `arguments: {"city":"Copenhagen"}` and one `output_item.done` whose function call item is `completed` with
    the same arguments, both on the output index the item was announced at (`streaming_iterator.py:652`)."""
    with peer(_tool_respond) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, chat_bridge(scenario, provider), tools=[WEATHER_TOOL])
    added_index: Final = one_event(events, "response.output_item.added")["output_index"]
    done: Final = one_event(events, "response.function_call_arguments.done")
    assert (done["arguments"], done["output_index"]) == (LONG_ARGUMENTS, added_index)
    item_done: Final = one_event(events, "response.output_item.done")
    assert item_done["output_index"] == added_index
    assert {key: object_value(item_done["item"])[key] for key in ("type", "status", "arguments")} == {
        "type": "function_call",
        "status": "completed",
        "arguments": LONG_ARGUMENTS,
    }


def test_a_tool_only_stream_ends_completed_with_the_function_call_in_the_snapshot(gateway: Gateway) -> None:
    """The tool only stream must still end with `response.completed`, whose `output` is exactly one
    `function_call` item with `call_id: call_dec1r2` and the full arguments, and the usage 11 and 7 tokens."""
    with peer(_tool_respond) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, chat_bridge(scenario, provider), tools=[WEATHER_TOOL])
    completed: Final = object_value(one_event(events, "response.completed")["response"])
    output: Final = [object_value(item) for item in list_value(completed["output"])]
    assert [(item["type"], item["call_id"], item["arguments"]) for item in output] == [
        ("function_call", "call_dec1r2", LONG_ARGUMENTS)
    ]
    usage: Final = object_value(completed["usage"])
    assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def _anthropic_sse(event: str, data: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _anthropic_tool_stream(call: Call) -> Reply:
    if call.method != "POST":
        return default_respond(call)
    usage: Final[dict[str, JsonValue]] = {"input_tokens": 11, "output_tokens": 0}
    return Reply(
        content_type="text/event-stream",
        chunks=(
            _anthropic_sse(
                "message_start",
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_dec1r2",
                        "type": "message",
                        "role": "assistant",
                        "model": call.body.get("model"),
                        "content": [],
                        "stop_reason": None,
                        "usage": usage,
                    },
                },
            ),
            _anthropic_sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "tool_use", "id": "toolu_dec1r2", "name": "get_weather", "input": {}},
                },
            ),
            _anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
                },
            ),
            _anthropic_sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '"Oslo"}'},
                },
            ),
            _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
            _anthropic_sse(
                "message_delta",
                {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}},
            ),
            _anthropic_sse("message_stop", {"type": "message_stop"}),
        ),
    )


def test_an_anthropic_tool_use_stream_becomes_function_call_events_with_the_toolu_id(gateway: Gateway) -> None:
    """Anthropic streams a `tool_use` block `toolu_dec1r2` whose input arrives as two `input_json_delta` parts.
    The caller must see one function call item `fc_toolu_dec1r2` with `call_id: toolu_dec1r2`, deltas that join
    to `{"city":"Oslo"}`, and `function_call_arguments.done` with exactly that string, so a Claude tool call
    streams the same way an OpenAI one does."""
    with peer(_anthropic_tool_stream) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, anthropic(scenario, provider), tools=[WEATHER_TOOL])
    item: Final = object_value(one_event(events, "response.output_item.added")["item"])
    assert (item["id"], item["call_id"], item["name"]) == ("fc_toolu_dec1r2", "toolu_dec1r2", "get_weather")
    joined: Final = "".join(
        string_value(event["delta"]) for event in _of_type(events, "response.function_call_arguments.delta")
    )
    assert joined == '{"city":"Oslo"}'
    assert one_event(events, "response.function_call_arguments.done")["arguments"] == '{"city":"Oslo"}'


def test_a_gemini_stream_produces_the_text_lifecycle_and_the_completed_usage(gateway: Gateway) -> None:
    """A Gemini deployment streams three SSE frames `dec1 `, `gemini `, `reply`, the last with `finishReason:
    STOP` and usage 11 and 7. The caller must see the same eleven event lifecycle as on the chat carrier, the
    proxy must have called `:streamGenerateContent`, and `response.completed` must report
    `input_tokens: 11` and `output_tokens: 7`."""
    with peer(every_respond) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, gemini(scenario, provider))
        assert provider.only().path == GEMINI_STREAM_PATH
    assert types_of(events) == TEXT_LIFECYCLE
    usage: Final = object_value(object_value(one_event(events, "response.completed")["response"])["usage"])
    assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def test_a_gemini_safety_stop_mid_stream_ends_incomplete_for_content_filter(gateway: Gateway) -> None:
    """Gemini ends its stream with `finishReason: SAFETY`, which maps to the chat `content_filter` finish. The
    terminal event must be `response.incomplete` with `incomplete_details.reason: content_filter`
    (`transformation.py:217`)."""

    def respond(call: Call) -> Reply:
        return gemini_stream(("partial ", "answer"), finish="SAFETY") if call.method == "POST" else every_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, gemini(scenario, provider))
    assert types_of(events)[-1] == "response.incomplete"
    assert object_value(object_value(events[-1]["response"])["incomplete_details"])["reason"] == "content_filter"


def test_the_created_snapshot_echoes_the_requested_reasoning(gateway: Gateway) -> None:
    """The caller streams with `reasoning: {"effort": "low"}`. The `response.created` snapshot should echo the
    request like the non streamed answer does (`reasoning.effort == "low"`), but `_default_response_created_event_data`
    hardcodes `reasoning: {"effort": null, "summary": null}` (`streaming_iterator.py:803`)."""
    pytest.skip("BUG: bridged response.created hardcodes reasoning {effort: null, summary: null}")
    with peer() as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, chat_bridge(scenario, provider), reasoning={"effort": "low"})
    created: Final = object_value(one_event(events, "response.created")["response"])
    assert object_value(created["reasoning"])["effort"] == "low"


def test_the_created_snapshot_defaults_top_p_to_one_and_tools_to_empty(gateway: Gateway) -> None:
    """A bare streamed call sends neither `top_p` nor `tools`. The `response.created` snapshot fills
    `top_p: 1.0`, `tools: []` and `tool_choice: "auto"` (`streaming_iterator.py:810-825`), while the non streamed
    answer leaves `top_p` null, so clients reading the first snapshot see these defaults."""
    created: Final = object_value(one_event(_text_events(gateway), "response.created")["response"])
    assert (created["top_p"], created["tools"], created["tool_choice"]) == (1.0, [], "auto")


def test_a_text_stream_from_a_fresh_chat_id_does_not_leak_the_chat_id(gateway: Gateway) -> None:
    """The chat chunks carry id `chatcmpl-dec1r2leak`. No event the caller sees may carry that raw id as the
    response id: every `response` snapshot id starts with `resp_` and differs from the chat id."""

    def respond(call: Call) -> Reply:
        return text_stream("chatcmpl-dec1r2leak") if call.method == "POST" else default_respond(call)

    with peer(respond) as provider, gateway.scenario() as scenario:
        events: Final = stream_events(gateway, chat_bridge(scenario, provider))
    snapshot_ids: Final = {
        string_value(object_value(event["response"])["id"]) for event in events if "response" in event
    }
    assert snapshot_ids and all(identity.startswith("resp_") for identity in snapshot_ids), snapshot_ids
    assert "chatcmpl-dec1r2leak" not in snapshot_ids
