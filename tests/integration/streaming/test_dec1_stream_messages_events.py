"""Anthropic event sequence on a streamed `/v1/messages` answer.

`/v1/messages` serves two kinds of deployment. For a native Anthropic deployment the provider's events
are relayed (with the model name in `message_start` rewritten to the public name). For any other
provider the proxy calls chat completions and synthesises the Anthropic events itself. Either way an
Anthropic SDK needs the same grammar: `message_start`, then for each content block
`content_block_start`, one or more `content_block_delta`, `content_block_stop`, then one
`message_delta` with `stop_reason` and usage, then `message_stop`. Each frame is an `event:` line naming
the type and a `data:` line whose JSON `type` matches. There is no `[DONE]` sentinel.

The bridged cases use a `hosted_vllm/` deployment (OpenAI compatible chat only) so the request is known
to take the chat completions bridge rather than the Responses bridge.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.anthropic_sse import CONTENT_BLOCK_START, CONTENT_TAIL, PING, message_start, text_delta
from integration._support.client import Gateway, object_value
from integration.streaming.dec1_stream_support import (
    CLAUDE,
    DETAILED_USAGE,
    DONE,
    delta,
    event,
    events_of,
    marker,
    scripted,
    serving,
    sse,
    text_frames,
    usage_only,
)

_BRIDGED: Final = "hosted_vllm/dec1-chat-backend"
_TOOLS: Final = [
    {
        "name": "get_weather",
        "description": "Look up the weather",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    }
]


def _types(events: list[tuple[str, dict]]) -> list[str]:
    return [str(body["type"]) for _, body in events]


def _blocks(events: list[tuple[str, dict]]) -> list[dict]:
    return [object_value(body["content_block"]) for _, body in events if body["type"] == "content_block_start"]


def _native(identity: str, *pieces: str) -> tuple[bytes, ...]:
    return (message_start(identity, CLAUDE), CONTENT_BLOCK_START, PING, *map(text_delta, pieces), CONTENT_TAIL)


def test_bridged_text_stream_follows_the_anthropic_event_grammar(gateway: Gateway) -> None:
    """A chat only deployment streams two text pieces, a `stop` finish and usage. The caller of
    `/v1/messages` receives exactly `message_start`, `content_block_start`, two `content_block_delta`,
    `content_block_stop`, `message_delta`, `message_stop`, in that order, with status 200 and content
    type `text/event-stream`."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("Hel", "lo"))), model=_BRIDGED) as rig:
        response: Final = rig.messages()
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].split(";")[0] == "text/event-stream"
        assert _types(events_of(response.text)) == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]


def test_bridged_frames_name_their_type_on_the_event_line(gateway: Gateway) -> None:
    """Every frame of a bridged messages stream has an `event:` line equal to the `type` inside its
    `data:` JSON. The Anthropic SDK dispatches on the event line, other clients on the JSON type, so the
    two must agree."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("ev", "ent"))), model=_BRIDGED) as rig:
        events: Final = events_of(rig.messages().text)
        assert events and all(name == body["type"] for name, body in events)


def test_bridged_stream_has_no_done_sentinel(gateway: Gateway) -> None:
    """A bridged messages stream ends with the `message_stop` frame. The OpenAI `data: [DONE]` sentinel
    never appears, because the Anthropic SDK would try to parse it as JSON."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("no ", "done"))), model=_BRIDGED) as rig:
        body: Final = rig.messages().text
        assert "[DONE]" not in body
        assert events_of(body)[-1][0] == "message_stop"


def test_bridged_message_start_is_an_empty_assistant_message(gateway: Gateway) -> None:
    """The first event of a bridged stream is `message_start` whose `message` has an `id` starting
    `msg_`, `type: message`, `role: assistant`, `content: []`, `stop_reason: null`,
    `stop_sequence: null` and a `usage` with `output_tokens: 0`."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("st", "art"))), model=_BRIDGED) as rig:
        message: Final = object_value(events_of(rig.messages().text)[0][1]["message"])
        assert str(message["id"]).startswith("msg_")
        assert (message["type"], message["role"], message["content"]) == ("message", "assistant", [])
        assert message["stop_reason"] is None and message["stop_sequence"] is None
        assert object_value(message["usage"])["output_tokens"] == 0


def test_bridged_text_block_opens_empty_at_index_zero_and_deltas_carry_the_text(gateway: Gateway) -> None:
    """The text block of a bridged stream opens as `{"type": "text", "text": ""}` at index 0, every
    delta is a `text_delta` at index 0, the delta texts are the provider's pieces in order, and the
    block's `content_block_stop` names index 0. Text must not be duplicated into the start event."""
    pieces: Final = ("Bon", "jour ", "monde")
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", pieces)), model=_BRIDGED) as rig:
        events: Final = [body for _, body in events_of(rig.messages().text)]
        start: Final = next(body for body in events if body["type"] == "content_block_start")
        assert start["index"] == 0 and start["content_block"] == {"type": "text", "text": ""}
        deltas: Final = [body for body in events if body["type"] == "content_block_delta"]
        assert [(body["index"], object_value(body["delta"])["type"]) for body in deltas] == [(0, "text_delta")] * 3
        assert [object_value(body["delta"])["text"] for body in deltas] == list(pieces)
        assert [body["index"] for body in events if body["type"] == "content_block_stop"] == [0]


@pytest.mark.parametrize(
    ("finish", "stop_reason"),
    [("stop", "end_turn"), ("length", "max_tokens"), ("tool_calls", "tool_use"), ("content_filter", "refusal")],
)
def test_bridged_finish_reason_maps_to_the_anthropic_stop_reason(gateway: Gateway, finish: str, stop_reason: str) -> None:
    """A chat deployment ends its stream with each OpenAI finish reason. The single `message_delta` event
    carries the Anthropic equivalent: `stop` becomes `end_turn`, `length` becomes `max_tokens`,
    `tool_calls` becomes `tool_use`, `content_filter` becomes `refusal`
    (`_translate_openai_finish_reason_to_anthropic`)."""
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("map ", finish), finish=finish)
    with scripted(gateway, serving(frames), model=_BRIDGED) as rig:
        deltas: Final = [body for _, body in events_of(rig.messages().text) if body["type"] == "message_delta"]
        assert [object_value(body["delta"])["stop_reason"] for body in deltas] == [stop_reason]


def test_bridged_message_delta_reports_provider_usage(gateway: Gateway) -> None:
    """The chat provider reports 11 prompt and 4 completion tokens in a trailing usage chunk. The bridged
    stream's `message_delta` has `usage == {"input_tokens": 11, "output_tokens": 4}`, and the SpendLogs
    row records the same counts."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("us", "age"))), model=_BRIDGED) as rig:
        events: Final = [body for _, body in events_of(rig.messages().text)]
        usage: Final = object_value(next(body for body in events if body["type"] == "message_delta")["usage"])
        assert (usage["input_tokens"], usage["output_tokens"]) == (11, 4)
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 4)


def test_bridged_cached_prompt_tokens_surface_as_cache_read_input_tokens(gateway: Gateway) -> None:
    """The chat provider reports 20 prompt tokens of which 8 were cached. The `message_delta` usage
    reports `cache_read_input_tokens: 8` and `input_tokens: 12`, following Anthropic's convention that
    `input_tokens` excludes cache reads (`translate_openai_usage_to_anthropic_usage_delta` subtracts
    them). Claude Code shows cache savings from this field."""
    frames: Final = text_frames(f"chatcmpl-{marker()}", ("ca", "che"), usage=DETAILED_USAGE)
    with scripted(gateway, serving(frames), model=_BRIDGED) as rig:
        events: Final = [body for _, body in events_of(rig.messages().text)]
        usage: Final = object_value(next(body for body in events if body["type"] == "message_delta")["usage"])
        assert usage["cache_read_input_tokens"] == 8
        assert usage["input_tokens"] == 12
        assert usage["output_tokens"] == 9


def test_bridged_tool_call_becomes_a_tool_use_block_with_input_json_deltas(gateway: Gateway) -> None:
    """A chat deployment streams one tool call in fragments. The caller receives a `content_block_start`
    whose block is `{"type": "tool_use", "id": "call_dec1", "name": "get_weather", "input": {}}`, then
    `input_json_delta` events whose `partial_json` pieces concatenate to `{"city":"Paris"}`, then
    `content_block_stop`, and `message_delta.stop_reason == "tool_use"`."""
    identity: Final = f"chatcmpl-{marker()}"
    opening: Final = {"index": 0, "id": "call_dec1", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
    frames: Final = (
        delta(identity, {"role": "assistant", "content": None, "tool_calls": [opening]}),
        delta(identity, {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":'}}]}),
        delta(identity, {"tool_calls": [{"index": 0, "function": {"arguments": '"Paris"}'}}]}),
        delta(identity, {}, finish="tool_calls"),
        usage_only(identity),
        DONE,
    )
    with scripted(gateway, serving(frames), model=_BRIDGED) as rig:
        events: Final = events_of(rig.messages(tools=_TOOLS).text)
        blocks: Final = _blocks(events)
        assert [(block["type"], block.get("id"), block.get("name"), block.get("input")) for block in blocks] == [
            ("tool_use", "call_dec1", "get_weather", {})
        ]
        partial: Final = "".join(
            str(object_value(body["delta"]).get("partial_json") or "")
            for _, body in events
            if body["type"] == "content_block_delta" and object_value(body["delta"])["type"] == "input_json_delta"
        )
        assert json.loads(partial) == {"city": "Paris"}
        assert [object_value(body["delta"])["stop_reason"] for _, body in events if body["type"] == "message_delta"] == ["tool_use"]


def test_bridged_text_then_tool_call_closes_the_text_block_before_opening_the_tool_block(gateway: Gateway) -> None:
    """A chat deployment streams text and then a tool call. The caller sees block 0 (`text`) start, its
    deltas, `content_block_stop` for index 0, and only then block 1 (`tool_use`) start, its deltas and
    its stop. Two blocks are never open at once and indexes are 0 then 1."""
    identity: Final = f"chatcmpl-{marker()}"
    opening: Final = {"index": 0, "id": "call_after", "type": "function", "function": {"name": "get_weather", "arguments": ""}}
    frames: Final = (
        delta(identity, {"role": "assistant", "content": "Checking. "}),
        delta(identity, {"tool_calls": [opening]}),
        delta(identity, {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":"Rome"}'}}]}),
        delta(identity, {}, finish="tool_calls"),
        usage_only(identity),
        DONE,
    )
    with scripted(gateway, serving(frames), model=_BRIDGED) as rig:
        events: Final = events_of(rig.messages(tools=_TOOLS).text)
        structure: Final = [(body["type"], body["index"]) for _, body in events if body["type"] in ("content_block_start", "content_block_stop")]
        assert structure == [("content_block_start", 0), ("content_block_stop", 0), ("content_block_start", 1), ("content_block_stop", 1)]
        assert [block["type"] for block in _blocks(events)] == ["text", "tool_use"]


def test_bridged_parallel_tool_calls_become_one_tool_use_block_each(gateway: Gateway) -> None:
    """A chat deployment streams two tool calls. The caller receives two `tool_use` blocks at consecutive
    indexes with ids `call_a` and `call_b`, each closed by its own `content_block_stop`, and each
    block's `partial_json` belongs to its own call."""
    identity: Final = f"chatcmpl-{marker()}"

    def opening(index: int, call_id: str) -> dict:
        return {"index": index, "id": call_id, "type": "function", "function": {"name": "get_weather", "arguments": ""}}

    frames: Final = (
        delta(identity, {"role": "assistant", "content": None, "tool_calls": [opening(0, "call_a")]}),
        delta(identity, {"tool_calls": [{"index": 0, "function": {"arguments": '{"city":"Oslo"}'}}]}),
        delta(identity, {"tool_calls": [opening(1, "call_b")]}),
        delta(identity, {"tool_calls": [{"index": 1, "function": {"arguments": '{"city":"Lima"}'}}]}),
        delta(identity, {}, finish="tool_calls"),
        usage_only(identity),
        DONE,
    )
    with scripted(gateway, serving(frames), model=_BRIDGED) as rig:
        events: Final = events_of(rig.messages(tools=_TOOLS).text)
        assert [block.get("id") for block in _blocks(events)] == ["call_a", "call_b"]
        by_index: Final = {}
        for _, body in events:
            if body["type"] == "content_block_delta":
                by_index[body["index"]] = by_index.get(body["index"], "") + str(object_value(body["delta"]).get("partial_json") or "")
        assert [json.loads(by_index[index]) for index in sorted(by_index)] == [{"city": "Oslo"}, {"city": "Lima"}]


def test_bridged_reasoning_then_answer_is_a_thinking_block_then_a_text_block(gateway: Gateway) -> None:
    """A chat deployment streams `reasoning_content` and then `content`. The caller receives a `thinking`
    block at index 0 with `thinking_delta` events, then a `text` block at index 1 with `text_delta`
    events. Reasoning text never appears inside a `text_delta`."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "reasoning_content": "weigh options"}),
        delta(identity, {"content": "final answer"}),
        delta(identity, {}, finish="stop"),
        usage_only(identity),
        DONE,
    )
    with scripted(gateway, serving(frames), model=_BRIDGED) as rig:
        events: Final = events_of(rig.messages().text)
        assert [block["type"] for block in _blocks(events)] == ["thinking", "text"]
        deltas: Final = [object_value(body["delta"]) for _, body in events if body["type"] == "content_block_delta"]
        assert "".join(str(item.get("thinking") or "") for item in deltas) == "weigh options"
        assert "".join(str(item.get("text") or "") for item in deltas) == "final answer"


def test_bridged_request_reaches_the_chat_provider_as_openai_messages(gateway: Gateway) -> None:
    """The caller posts an Anthropic body with a top level `system` string and `max_tokens: 64`. The chat
    provider receives `stream: true`, a `system` role message with that text ahead of the user message,
    and a token cap of 64 (`max_tokens` or `max_completion_tokens`)."""
    with scripted(gateway, serving(text_frames(f"chatcmpl-{marker()}", ("ok",))), model=_BRIDGED) as rig:
        assert rig.messages(system="You are terse.").status_code == 200
        body: Final = rig.upstream_bodies()[0]
        roles: Final = [object_value(message)["role"] for message in body["messages"]]  # type: ignore[union-attr]
        assert roles == ["system", "user"]
        assert body["stream"] is True
        assert 64 in (body.get("max_tokens"), body.get("max_completion_tokens"))


def test_native_stream_relays_every_provider_event_in_order_including_ping(gateway: Gateway) -> None:
    """A native Anthropic deployment streams `message_start`, `content_block_start`, `ping`, two text
    deltas, `content_block_stop`, `message_delta`, `message_stop`. The caller receives the same event
    names in the same order, the ping included, and the delta texts unchanged."""
    identity: Final = f"msg_{marker()}"
    with scripted(gateway, lambda request: sse(_native(identity, "na", "tive")), model=f"anthropic/{CLAUDE}", base_suffix="") as rig:
        events: Final = events_of(rig.messages().text)
        assert [name for name, _ in events] == [
            "message_start",
            "content_block_start",
            "ping",
            "content_block_delta",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        texts: Final = [object_value(body["delta"])["text"] for _, body in events if body["type"] == "content_block_delta"]
        assert texts == ["na", "tive"]


def test_native_message_start_keeps_the_provider_id_and_shows_the_public_model_name(gateway: Gateway) -> None:
    """A native Anthropic stream's `message_start` carries the provider's message id unchanged, while
    `message.model` is the public model name the caller sent instead of the provider's `claude-...` id
    (`AnthropicStreamModelRestamper`). The provider was asked for the real model id."""
    identity: Final = f"msg_{marker()}"
    with scripted(gateway, lambda request: sse(_native(identity, "id")), model=f"anthropic/{CLAUDE}", base_suffix="") as rig:
        message: Final = object_value(events_of(rig.messages().text)[0][1]["message"])
        assert message["id"] == identity
        assert message["model"] == rig.model
        assert rig.upstream_bodies()[0]["model"] == CLAUDE


def test_native_stream_request_reaches_anthropic_with_its_headers_and_stream_flag(gateway: Gateway) -> None:
    """The caller streams `/v1/messages` against a native Anthropic deployment. The provider receives one
    POST to `/v1/messages` with `stream: true`, `max_tokens: 64`, an `x-api-key` header holding the
    deployment's key (not the caller's virtual key) and an `anthropic-version` header."""
    identity: Final = f"msg_{marker()}"
    with scripted(gateway, lambda request: sse(_native(identity, "hdr")), model=f"anthropic/{CLAUDE}", base_suffix="") as rig:
        assert rig.messages().status_code == 200
        requests: Final = [request for request in rig.wire.drain() if request.method == "POST"]
        assert [request.target.split("?")[0] for request in requests] == ["/v1/messages"]
        body: Final = json.loads(requests[0].body)
        assert body["stream"] is True and body["max_tokens"] == 64
        assert requests[0].headers["x-api-key"] == "synthetic-provider-key"
        assert requests[0].headers.get("anthropic-version")
        assert rig.key not in json.dumps(dict(requests[0].headers))


def test_native_stream_is_billed_from_the_usage_in_its_events(gateway: Gateway) -> None:
    """A native Anthropic stream reports `input_tokens: 5` in `message_start` and `output_tokens: 3` in
    `message_delta`. The SpendLogs row records 5 prompt and 3 completion tokens and
    `custom_llm_provider == "anthropic"`, and at 0.001 and 0.002 per token the spend is 0.011."""
    identity: Final = f"msg_{marker()}"
    with scripted(
        gateway,
        lambda request: sse(_native(identity, "bill")),
        model=f"anthropic/{CLAUDE}",
        base_suffix="",
        input_cost_per_token=0.001,
        output_cost_per_token=0.002,
    ) as rig:
        assert rig.messages().status_code == 200
        row: Final = rig.spend_rows()[0]
        assert (row["prompt_tokens"], row["completion_tokens"]) == (5, 3)
        assert row["custom_llm_provider"] == "anthropic"
        assert float(str(row["spend"])) == pytest.approx(5 * 0.001 + 3 * 0.002)


def test_native_mid_stream_error_event_reaches_the_caller_as_an_error_event(gateway: Gateway) -> None:
    """A native Anthropic deployment sends `message_start`, one text delta and then
    `event: error` with `{"type": "error", "error": {"type": "overloaded_error", "message":
    "Overloaded"}}`. The caller receives the text delta and then an `error` event whose `error.type` is
    `overloaded_error`; no `message_stop` follows. The Anthropic SDK raises on this event."""
    identity: Final = f"msg_{marker()}"
    failure: Final = event("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    frames: Final = (message_start(identity, CLAUDE), CONTENT_BLOCK_START, text_delta("before"), failure)
    with scripted(gateway, lambda request: sse(frames), model=f"anthropic/{CLAUDE}", base_suffix="") as rig:
        events: Final = events_of(rig.messages().text)
        names: Final = [name for name, _ in events]
        assert "content_block_delta" in names and names[-1] == "error"
        assert object_value(events[-1][1]["error"])["type"] == "overloaded_error"
        assert "message_stop" not in names


def test_provider_error_on_messages_stream_open_is_an_anthropic_shaped_http_error(gateway: Gateway) -> None:
    """A native Anthropic deployment refuses the streaming request with HTTP 529 and an Anthropic error
    body. The caller receives a non 200 status and a JSON body shaped
    `{"type": "error", "error": {"type", "message"}}`, the Anthropic error envelope rather than the
    OpenAI one. UNVERIFIED on the relayed status for 529; this asserts only that it is an error status
    of 500 or above."""
    from integration._support.wire import Reply

    body: Final = json.dumps({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}).encode()
    with scripted(gateway, lambda request: Reply(status=529, body=body), model=f"anthropic/{CLAUDE}", base_suffix="") as rig:
        response: Final = rig.messages()
        assert response.status_code >= 500, response.text
        payload: Final = response.json()
        assert payload["type"] == "error"
        assert {"type", "message"} <= set(payload["error"])
