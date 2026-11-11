"""Event sequence on a streamed `/v1/responses` answer.

A Responses stream is a series of typed events, each a `data:` frame whose JSON has a `type` and a
`sequence_number`. For an OpenAI deployment the provider's events are relayed. For a chat only
deployment the proxy builds the events from chat completion chunks: `response.created`,
`response.in_progress`, `response.output_item.added`, `response.content_part.added`,
`response.output_text.delta` (one per text piece), `response.output_text.done`,
`response.content_part.done`, `response.output_item.done`, and a terminal `response.completed` (or
`response.incomplete`). The OpenAI SDK's `responses.stream()` helper validates the order, the monotonic
`sequence_number` and the final response snapshot, and there is no `[DONE]` sentinel on this route.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.openai_wire import responses_stream
from integration.streaming.dec1_stream_support import (
    DONE,
    UPSTREAM_MODEL,
    chunks_of,
    delta,
    marker,
    scripted,
    serving,
    sse,
    text_frames,
    usage_only,
)

_BRIDGED: Final = "deepseek/dec1-chat-backend"
"""A chat only provider. `hosted_vllm/` would not do: it has a native Responses config
(`HostedVLLMResponsesAPIConfig`), so its `/v1/responses` calls go to the peer's own `/v1/responses` route
instead of through the chat completions bridge."""
_TEXT_LIFECYCLE: Final = [
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
]


def _collapsed(events: list[dict]) -> list[str]:
    """Event types in order with runs of the same type collapsed to one entry."""
    types: Final = [str(item["type"]) for item in events]
    return [kind for position, kind in enumerate(types) if position == 0 or types[position - 1] != kind]


def test_bridged_text_stream_emits_the_full_responses_lifecycle_in_order(gateway: Gateway) -> None:
    """A chat only deployment streams three text pieces. The caller of `/v1/responses` receives, in
    order: created, in_progress, output_item.added, content_part.added, the text deltas,
    output_text.done, content_part.done, output_item.done, completed. Status is 200 with content type
    `text/event-stream`."""
    with scripted(
        gateway, serving(text_frames(f"chatcmpl-{marker()}", ("one ", "two ", "three"))), model=_BRIDGED
    ) as rig:
        response: Final = rig.responses()
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].split(";")[0] == "text/event-stream"
        assert _collapsed(chunks_of(response.text)) == _TEXT_LIFECYCLE


def test_bridged_sequence_numbers_start_at_zero_and_rise_by_one(gateway: Gateway) -> None:
    """Every event of a bridged Responses stream carries `sequence_number`, the first is 0, and each
    later event is exactly one higher than the one before it. The OpenAI SDK uses the number to resume
    and to detect dropped events."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("a", "b", "c", "d"))), model=_BRIDGED) as rig:
        events: Final = chunks_of(rig.responses().text)
        assert [item["sequence_number"] for item in events] == list(range(len(events)))


def test_bridged_text_deltas_and_done_events_agree_on_the_text(gateway: Gateway) -> None:
    """A chat deployment streams `Hel`, `lo `, `there`. The `response.output_text.delta` events carry
    those pieces in order, `response.output_text.done.text` is `Hello there`, and the completed
    response's first output item holds the same text. All three views must agree because clients pick
    different ones."""
    pieces: Final = ("Hel", "lo ", "there")
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", pieces)), model=_BRIDGED) as rig:
        events: Final = chunks_of(rig.responses().text)
        deltas: Final = [item["delta"] for item in events if item["type"] == "response.output_text.delta"]
        assert "".join(str(piece) for piece in deltas) == "Hello there"
        done: Final = next(item for item in events if item["type"] == "response.output_text.done")
        assert done["text"] == "Hello there"
        final: Final = object_value(events[-1]["response"])
        message: Final = object_value(final["output"][0])  # type: ignore[index]
        assert object_value(message["content"][0])["text"] == "Hello there"  # type: ignore[index]


def test_bridged_events_share_one_response_id_and_one_item_id(gateway: Gateway) -> None:
    """In a bridged Responses stream the response id in `response.created`, `response.in_progress` and
    `response.completed` is the same string starting `resp_`, and every delta, the item added event and
    the item done event name the same message item id. UNVERIFIED on the `resp_` prefix: read from
    `_with_encoded_response_id` in the bridge's streaming iterator."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("same ", "ids"))), model=_BRIDGED) as rig:
        events: Final = chunks_of(rig.responses().text)
        response_ids: Final = {object_value(item["response"])["id"] for item in events if "response" in item}
        assert len(response_ids) == 1 and str(next(iter(response_ids))).startswith("resp_")
        added: Final = next(item for item in events if item["type"] == "response.output_item.added")
        item_id: Final = object_value(added["item"])["id"]
        assert {item["item_id"] for item in events if item["type"] == "response.output_text.delta"} == {item_id}
        closed: Final = next(item for item in events if item["type"] == "response.output_item.done")
        assert object_value(closed["item"])["id"] == item_id


def test_bridged_completed_event_reports_status_and_usage(gateway: Gateway) -> None:
    """The chat provider reports 11 prompt and 4 completion tokens. The terminal event is
    `response.completed`, its response has `status == "completed"` and
    `usage == {input_tokens: 11, output_tokens: 4, total_tokens: 15}` (other usage keys may be present),
    and the SpendLogs row has the same counts with `call_type == "aresponses"`."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("use", "age"))), model=_BRIDGED) as rig:
        final: Final = chunks_of(rig.responses().text)[-1]
        assert final["type"] == "response.completed"
        response: Final = object_value(final["response"])
        usage: Final = object_value(response["usage"])
        assert response["status"] == "completed"
        assert (usage["input_tokens"], usage["output_tokens"], usage["total_tokens"]) == (11, 4, 15)
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["call_type"]) == (11, 4, "aresponses")


def test_bridged_length_finish_ends_with_response_incomplete(gateway: Gateway) -> None:
    """A chat deployment ends its stream with `finish_reason: length`. The terminal event of the
    Responses stream is `response.incomplete` rather than `response.completed`, and its response has
    `status == "incomplete"`. UNVERIFIED on `incomplete_details.reason == "max_output_tokens"`, which is
    asserted from the OpenAI Responses convention; the event type is read from
    `_emit_terminal_response_event`."""
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("cut ", "short"), finish="length")
    with scripted(gateway, serving(frames), model=_BRIDGED) as rig:
        final: Final = chunks_of(rig.responses().text)[-1]
        assert final["type"] == "response.incomplete"
        response: Final = object_value(final["response"])
        assert response["status"] == "incomplete"
        assert object_value(response["incomplete_details"])["reason"] == "max_output_tokens"


def test_bridged_tool_call_streams_function_call_argument_events(gateway: Gateway) -> None:
    """A chat deployment streams one tool call in two argument fragments. The Responses stream contains
    an `output_item.added` whose item is a `function_call` named `get_weather`,
    `response.function_call_arguments.delta` events whose deltas concatenate to `{"city":"Paris"}`, one
    `response.function_call_arguments.done` with the same full string, and the completed response lists
    the function call in `output`."""
    identity: Final = f"chatcmpl-{marker()}"
    opening: Final = {
        "index": 0,
        "id": "call_dec1",
        "type": "function",
        "function": {"name": "get_weather", "arguments": ""},
    }
    frames: Final = (
        delta(identity, {"role": "assistant", "content": None, "tool_calls": [opening]}),
        delta(identity, {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
        delta(identity, {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
        delta(identity, {}, finish="tool_calls"),
        usage_only(identity),
        DONE,
    )
    tools: Final = [
        {
            "type": "function",
            "name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    with scripted(gateway, serving(frames), model=_BRIDGED) as rig:
        events: Final = chunks_of(rig.responses(tools=tools).text)
        added: Final = [object_value(item["item"]) for item in events if item["type"] == "response.output_item.added"]
        assert [(item["type"], item.get("name")) for item in added if item["type"] == "function_call"] == [
            ("function_call", "get_weather")
        ]
        pieces: Final = [
            str(item["delta"]) for item in events if item["type"] == "response.function_call_arguments.delta"
        ]
        assert json.loads("".join(pieces)) == {"city": "Paris"}
        done: Final = [item for item in events if item["type"] == "response.function_call_arguments.done"]
        assert [json.loads(str(item["arguments"])) for item in done] == [{"city": "Paris"}]
        output: Final = [object_value(item) for item in object_value(events[-1]["response"])["output"]]  # type: ignore[union-attr]
        assert "function_call" in [item["type"] for item in output]


def test_responses_stream_has_no_done_sentinel(gateway: Gateway) -> None:
    """A Responses stream ends with its terminal `response.completed` frame. `data: [DONE]` never
    appears on this route (`ends_with_openai_done` is false for Responses streams), since the OpenAI
    SDK's Responses reader does not expect it."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("no ", "done"))), model=_BRIDGED) as rig:
        body: Final = rig.responses().text
        assert "[DONE]" not in body
        assert chunks_of(body)[-1]["type"] == "response.completed"


def test_bridged_request_reaches_the_chat_provider_as_a_streaming_chat_call(gateway: Gateway) -> None:
    """The caller posts `/v1/responses` with `input` text, `instructions` and `max_output_tokens: 33`. The
    chat provider receives `stream: true`, a `system` message with the instructions, a `user` message
    with the input text, and a token cap of 33."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("ok",))), model=_BRIDGED) as rig:
        prompt: Final = f"dec1 responses probe {marker()}"
        assert rig.responses(prompt, instructions="Be terse.", max_output_tokens=33).status_code == 200
        body: Final = rig.upstream_bodies()[0]
        messages: Final = [object_value(message) for message in body["messages"]]  # type: ignore[union-attr]
        assert [message["role"] for message in messages] == ["system", "user"]
        assert prompt in json.dumps(messages[1]["content"])
        assert body["stream"] is True
        assert 33 in (body.get("max_tokens"), body.get("max_completion_tokens"))


def test_native_openai_events_are_relayed_in_order_with_their_sequence_numbers(gateway: Gateway) -> None:
    """An OpenAI deployment answers `/v1/responses` with created, output_item.added, one text delta,
    output_item.done and completed, numbered 0 to 4. The caller receives the same five event types in
    the same order with sequence numbers 0 to 4 and the delta text unchanged. The provider was called on
    its `/v1/responses` route, not on chat completions."""
    identity: Final = f"resp_{marker()}"
    with scripted(gateway, lambda request: sse(responses_stream(identity, UPSTREAM_MODEL, "native text"))) as rig:
        events: Final = chunks_of(rig.responses().text)
        assert [item["type"] for item in events] == [
            "response.created",
            "response.output_item.added",
            "response.output_text.delta",
            "response.output_item.done",
            "response.completed",
        ]
        assert [item["sequence_number"] for item in events] == [0, 1, 2, 3, 4]
        assert events[2]["delta"] == "native text"
        assert [request.target for request in rig.wire.drain() if request.method == "POST"] == ["/v1/responses"]


def test_native_response_snapshots_show_the_public_model_name(gateway: Gateway) -> None:
    """The provider's `response.created` and `response.completed` snapshots say `model: gpt-4o-mini`. The
    caller sees the public model name it sent in both snapshots (`_restamp_nested_response_model`), and
    the provider was asked for `gpt-4o-mini`."""
    identity: Final = f"resp_{marker()}"
    with scripted(gateway, lambda request: sse(responses_stream(identity, UPSTREAM_MODEL, "alias"))) as rig:
        events: Final = chunks_of(rig.responses().text)
        models: Final = {object_value(item["response"])["model"] for item in events if "response" in item}
        assert models == {rig.model}
        assert rig.upstream_bodies()[0]["model"] == UPSTREAM_MODEL


def test_native_responses_stream_is_billed_from_the_completed_event(gateway: Gateway) -> None:
    """The provider's `response.completed` reports `input_tokens: 5` and `output_tokens: 3`. One SpendLogs
    row lands with 5 prompt tokens, 3 completion tokens, `call_type == "aresponses"`,
    `status == "success"`, and at 0.001 and 0.002 per token a spend of 0.011."""
    identity: Final = f"resp_{marker()}"
    with scripted(
        gateway,
        lambda request: sse(responses_stream(identity, UPSTREAM_MODEL, "billed")),
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    ) as rig:
        assert rig.responses().status_code == 200
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"], row["call_type"], row["status"]) == (
            5,
            3,
            "aresponses",
            "success",
        )
        assert float(str(row["spend"])) == pytest.approx(5 * 0.001 + 3 * 0.002)


def test_native_stream_cut_before_completed_ends_with_a_failed_or_error_event(gateway: Gateway) -> None:
    """The provider sends `response.created` and one delta, then the connection drops before
    `response.completed`. The caller's stream (HTTP 200) does not end silently: its last event is
    `response.failed`, `response.incomplete` or `error`. A client waiting for a terminal event would
    otherwise treat the truncated text as still in progress forever."""
    identity: Final = f"resp_{marker()}"
    frames: Final = responses_stream(identity, UPSTREAM_MODEL, "cut")
    with scripted(gateway, lambda request: sse(frames, abort_after=2)) as rig:
        response: Final = rig.responses()
        assert response.status_code == 200
        assert chunks_of(response.text)[-1]["type"] in ("response.failed", "response.incomplete", "error")
