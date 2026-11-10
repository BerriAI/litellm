"""Streaming basics of `POST /v1/messages` with `stream: true` for an Anthropic deployment.

The provider streams Anthropic server-sent events (`message_start`, `content_block_start`,
`content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`). The proxy relays the
same named events, in the same order, with the same JSON data, as `text/event-stream`. The single
intended change is in `message_start`: `message.model` reports the proxy model name the caller
asked for (`anthropic_endpoints/streaming_model_restamp.py`). Anthropic streams have no `[DONE]`
sentinel and the proxy must not add one. Usage arrives the Anthropic way: input tokens in
`message_start.message.usage`, final output tokens in `message_delta.usage`.

Failure modes of streams (provider error frames, truncated streams, retries before first content)
are already pinned by the existing `providers/anthropic/test_anthropic_messages_*` files; this file
covers the success path a rewrite has to reproduce first.
"""

from __future__ import annotations

import re
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply
from integration.messages_endpoint.dec1_messages_support import (
    BACKEND,
    JSON_OBJECT,
    NO_CACHE,
    PRICES,
    always,
    deployment,
    marker,
    message_reply,
    sse,
    sse_events,
    stream_chunks,
    stream_reply,
)
from pydantic import JsonValue

_UUID: Final = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_EVENT_ORDER: Final = (
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
)


def _expected_events(message_id: str, model: str, **fields: object) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    """The provider's events with `message_start.message.model` renamed to the proxy model."""
    provider_events: Final = sse_events(b"".join(stream_chunks(message_id, **fields)).decode())  # type: ignore[arg-type]  # test-local keyword passthrough
    return tuple(
        (event, {**data, "message": {**object_value(data["message"]), "model": model}})
        if event == "message_start"
        else (event, data)
        for event, data in provider_events
    )


def test_a_stream_is_answered_200_as_text_event_stream(gateway: Gateway) -> None:
    """A `stream: true` request whose provider streams normally must be answered HTTP 200 with a
    `content-type` that starts with `text/event-stream`."""
    with deployment(gateway, always(stream_reply(f"msg_{marker()}"))) as rig:
        response, _text = rig.stream(rig.body(stream=True))
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")


def test_the_six_anthropic_events_arrive_in_provider_order(gateway: Gateway) -> None:
    """The provider sends the six standard events once each. Ignoring keepalive pings, the caller must
    receive exactly those six event names in the same order, each as an `event:` line followed by a
    `data:` line holding JSON whose `type` equals the event name."""
    with deployment(gateway, always(stream_reply(f"msg_{marker()}"))) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        assert tuple(event for event, _data in events) == _EVENT_ORDER
        assert all(data["type"] == event for event, data in events)


def test_every_event_payload_is_relayed_unchanged_except_the_model_name(gateway: Gateway) -> None:
    """Comparing decoded JSON, every event the caller receives must equal the provider's event, with
    the single difference that `message_start.message.model` is the proxy model name. Ids, indexes,
    delta text, stop reason and usage numbers are untouched."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(stream_reply(message_id))) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        assert sse_events(text) == _expected_events(message_id, rig.model)


def test_message_start_reports_the_requested_proxy_model(gateway: Gateway) -> None:
    """The provider's `message_start` says `model: claude-sonnet-4-5-20250929`. The caller's
    `message_start.message.model` must be the proxy model name it requested, the same value a
    non-streaming response reports, and must not be the backend name."""
    with deployment(gateway, always(stream_reply(f"msg_{marker()}"))) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        message: Final = object_value(dict(sse_events(text))["message_start"]["message"])
        assert message["model"] == rig.model
        assert message["model"] != BACKEND


def test_the_stream_has_no_openai_done_sentinel(gateway: Gateway) -> None:
    """An Anthropic stream ends with `message_stop`. The body must not contain a `data: [DONE]` line,
    which Anthropic SDKs fail to parse as JSON, and the last event must be `message_stop`."""
    with deployment(gateway, always(stream_reply(f"msg_{marker()}"))) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        assert "[DONE]" not in text
        assert sse_events(text)[-1][0] == "message_stop"


def test_several_text_deltas_arrive_in_order_and_concatenate_to_the_full_text(gateway: Gateway) -> None:
    """The provider splits its answer over four `text_delta` events, one of them with non-ASCII text
    and one with a newline. The caller must receive four deltas in the same order whose texts join to
    exactly `Hel` + `lo, ` + `wörld é` + `\\nbye`."""
    pieces: Final = ("Hel", "lo, ", "wörld é", "\nbye")
    base: Final = stream_chunks(f"msg_{marker()}")
    chunks: Final = (
        *base[:2],
        *(
            sse(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
            )
            for piece in pieces
        ),
        *base[3:],
    )
    with deployment(gateway, always(Reply(content_type="text/event-stream", chunks=chunks))) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        deltas: Final = tuple(
            object_value(data["delta"])["text"] for event, data in sse_events(text) if event == "content_block_delta"
        )
        assert deltas == pieces


def test_a_streamed_tool_use_block_relays_its_start_and_json_deltas(gateway: Gateway) -> None:
    """The provider streams a `tool_use` block: a `content_block_start` with the tool id and name,
    two `input_json_delta` fragments, and `message_delta` with `stop_reason: "tool_use"`. The caller
    must receive the same start block, the two partial JSON strings unchanged and in order (they join
    to `{"city": "Paris"}`), and the `tool_use` stop reason."""
    base: Final = stream_chunks(f"msg_{marker()}")
    tool_start: Final[dict[str, JsonValue]] = {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "tool_use", "id": "toolu_synthetic_stream", "name": "get_weather", "input": {}},
    }
    fragments: Final = ('{"city": ', '"Paris"}')
    chunks: Final = (
        base[0],
        sse("content_block_start", tool_start),
        *(
            sse(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": fragment},
                },
            )
            for fragment in fragments
        ),
        base[3],
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 9},
            },
        ),
        base[5],
    )
    with deployment(gateway, always(Reply(content_type="text/event-stream", chunks=chunks))) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = sse_events(text)
        assert next(data for event, data in events if event == "content_block_start") == tool_start
        partials: Final = tuple(
            object_value(data["delta"])["partial_json"] for event, data in events if event == "content_block_delta"
        )
        assert partials == fragments
        assert object_value(dict(events)["message_delta"]["delta"])["stop_reason"] == "tool_use"


def test_streamed_usage_is_relayed_in_message_start_and_message_delta(gateway: Gateway) -> None:
    """The provider reports 23 input tokens in `message_start` and 17 final output tokens in
    `message_delta`. The caller must see `message_start.message.usage.input_tokens == 23` and
    `message_delta.usage == {"output_tokens": 17}` with no added `cost` or `total_tokens` key, since
    cost injection is off unless the caller or the config asks for it."""
    reply: Final = stream_reply(f"msg_{marker()}", input_tokens=23, output_tokens=17)
    with deployment(gateway, always(reply)) as rig:
        _response, text = rig.stream(rig.body(stream=True))
        events: Final = dict(sse_events(text))
        start_usage: Final = object_value(object_value(events["message_start"]["message"])["usage"])
        assert start_usage["input_tokens"] == 23
        assert events["message_delta"]["usage"] == {"output_tokens": 17}


def test_a_streamed_response_carries_the_proxy_headers(gateway: Gateway) -> None:
    """Response headers go out before the first event. A streamed response must still carry
    `x-litellm-call-id` (a UUID, or the caller's own value when sent) and a non-empty
    `x-litellm-model-id`, like a non-streaming one."""
    call_id: Final = marker("call")
    with deployment(gateway, always(stream_reply(f"msg_{marker()}"))) as rig:
        generated, _text = rig.stream(rig.body(stream=True))
        assert _UUID.match(generated.headers["x-litellm-call-id"])
        assert generated.headers["x-litellm-model-id"]
        echoed, _text2 = rig.stream(rig.body(stream=True), headers={"x-litellm-call-id": call_id})
        assert echoed.headers["x-litellm-call-id"] == call_id


def test_a_stream_reaches_the_provider_once(gateway: Gateway) -> None:
    """One streamed caller request must be exactly one provider `POST /v1/messages` with
    `"stream": true`. The proxy must not issue a second, non-streaming call to compute usage or cost."""
    with deployment(gateway, always(stream_reply(f"msg_{marker()}")), **PRICES) as rig:
        rig.stream(rig.body(stream=True))
        assert rig.one_body()["stream"] is True


def test_stream_false_returns_json_even_when_the_caller_accepts_event_stream(gateway: Gateway) -> None:
    """Streaming is chosen by the body's `stream` field, not by the `Accept` header. A request with
    `stream: false` and `Accept: text/event-stream` must be answered as one `application/json`
    message."""
    with deployment(gateway, always(message_reply("msg_plain"))) as rig:
        response: Final = rig.post(rig.body(stream=False), headers={"accept": "text/event-stream"})
        assert response.status_code == 200, response.text
        assert response.headers["content-type"].startswith("application/json")
        assert JSON_OBJECT.validate_json(response.content)["id"] == "msg_plain"


@pytest.mark.parametrize("uncached", (False, True), ids=("cache_default", "no_cache"))
def test_a_stream_is_the_same_with_and_without_the_response_cache(gateway: Gateway, uncached: bool) -> None:
    """A streamed call on the default path and one sent with `cache: {"no-cache": true}` (the shape
    eligible for the Rust route) must both deliver the provider's six events unchanged apart from the
    proxy model name in `message_start`."""
    message_id: Final = f"msg_{marker()}"
    with deployment(gateway, always(stream_reply(message_id))) as rig:
        response, text = rig.stream(rig.body(stream=True, **(NO_CACHE if uncached else {})))
        assert response.status_code == 200
        assert sse_events(text) == _expected_events(message_id, rig.model)
