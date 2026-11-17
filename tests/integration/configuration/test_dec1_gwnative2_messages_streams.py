"""How the native Rust gateway preview frames `/v1/messages` streams, per provider.

The contract, read from `litellm/rust/crates/inference-messages/src/handler.rs` (`streaming_response`,
`decoded_chunks`), `relay.rs`, `llms/src/bedrock/messages/invoke_transformations/stream.rs` and
`host-http/src/sse.rs`:

Anthropic-speaking providers (Anthropic, Vertex AI Claude, Azure AI) have no stream decoder, so their SSE bytes
are relayed to the caller exactly as the provider wrote them: every event, pings and anything after
`message_stop` included, and a stream that ends cleanly without `message_stop` simply ends. Bedrock invoke has a
decoder: each AWS event-stream `chunk` is decoded and re-encoded as `event: <type>\\ndata: <Python json.dumps>`,
Bedrock's `amazon-bedrock-invocationMetrics` on `message_stop` is folded into `usage`, `message_delta` is held
until `message_stop` so it can carry `input_tokens`, and a decoded stream that ends without `message_stop` or a
provider error event gets one closing `api_error` event (Python's `async_sse_wrapper`, which only Bedrock uses).
The stream head carries `content-type: text/event-stream` and none of the provider's response headers.

A rewrite that decodes and re-encodes Anthropic streams, or that drops the incomplete-stream event, changes
what Claude Code and the Anthropic SDKs see on a broken stream.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.wire import Reply
from tests.integration.configuration.dec1_gwnative2_support import (
    INCOMPLETE_STREAM_EVENT,
    INVOCATION_METRICS_KEY,
    bedrock_claude_events,
    bedrock_stream,
    python_sse,
    sse_blocks,
)
from tests.integration.configuration.dec1_gwnative_support import (
    NativeGateway,
    ScriptedProvider,
    anthropic_deployment,
    anthropic_stream_chunks,
    bedrock_deployment,
    marker,
    master_key_config,
    messages_body,
    native_gateway,
    scripted_provider,
    sse_events,
)

_PING: Final = b'event: ping\ndata: {"type": "ping"}\n\n'


@pytest.fixture(scope="module")
def provider() -> Iterator[ScriptedProvider]:
    with scripted_provider() as scripted:
        yield scripted


@pytest.fixture(scope="module")
def gateway(provider: ScriptedProvider, tmp_path_factory: pytest.TempPathFactory) -> Iterator[NativeGateway]:
    url: Final = provider.url
    config: Final = master_key_config([anthropic_deployment("claude", url), bedrock_deployment("bedrock-claude", url)])
    with native_gateway(config, tmp_path_factory.mktemp("gwnative2-streams")) as booted:
        yield booted


def _sse_reply(chunks: tuple[bytes, ...], **headers: str) -> Reply:
    return Reply(content_type="text/event-stream", chunks=chunks, headers=headers)


# Anthropic: raw relay


def test_an_anthropic_stream_is_relayed_byte_for_byte(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The Anthropic provider streams six events with a `ping` after `message_start`. The caller's body must
    equal the concatenation of the provider's chunks exactly, because a stream without a decoder is relayed as
    raw bytes (`decoder: None` in `streaming_response`, `encode_chunk` returns its input in `sse.rs:51`)."""
    case: Final = marker()
    chunks: Final = (anthropic_stream_chunks(case)[0], _PING, *anthropic_stream_chunks(case)[1:])
    provider.script(case, lambda _: _sse_reply(chunks))
    response: Final = gateway.post("/v1/messages", messages_body(case, stream=True))
    assert response.status_code == 200, response.text
    assert response.content == b"".join(chunks)


def test_an_anthropic_stream_that_ends_without_message_stop_just_ends(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The Anthropic provider sends `message_start` through `content_block_stop` and then ends the chunked body
    cleanly, with no `message_stop`. The caller must get exactly those four events and nothing appended: the
    incomplete-stream `api_error` is added only on decoded (Bedrock) streams (`decoded_chunks` in
    `handler.rs:1216`; Python calls `async_sse_wrapper` only from the Bedrock invoke config)."""
    case: Final = marker()
    sent: Final = anthropic_stream_chunks(case)[:4]
    provider.script(case, lambda _: _sse_reply(sent))
    response: Final = gateway.post("/v1/messages", messages_body(case, stream=True))
    assert response.status_code == 200, response.text
    assert response.content == b"".join(sent)
    assert "error" not in [name for name, _ in sse_events(response.text)]


def test_an_anthropic_in_stream_error_event_is_relayed_verbatim(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """After `message_start` the Anthropic provider sends `event: error` with an `overloaded_error` object and
    ends. The caller must receive that error block unchanged as the last block and no second error event."""
    case: Final = marker()
    error: Final = (
        b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "dec1 busy"}}\n\n'
    )
    provider.script(case, lambda _: _sse_reply((anthropic_stream_chunks(case)[0], error)))
    response: Final = gateway.post("/v1/messages", messages_body(case, stream=True))
    blocks: Final = sse_blocks(response.text)
    assert blocks[-1].encode() == error
    assert sum(1 for block in blocks if block.startswith("event: error")) == 1


def test_events_after_message_stop_are_still_relayed(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The Anthropic provider sends a whole stream and then one more `ping` after `message_stop`. The caller's
    last block must be that ping: the raw relay does not cut the stream at `message_stop`."""
    case: Final = marker()
    provider.script(case, lambda _: _sse_reply((*anthropic_stream_chunks(case), _PING)))
    response: Final = gateway.post("/v1/messages", messages_body(case, stream=True))
    assert sse_blocks(response.text)[-1].encode() == _PING


def test_anthropic_stream_usage_reaches_the_caller_unchanged(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider's `message_start` reports `usage.input_tokens` 5 and its `message_delta` reports
    `usage.output_tokens` 3. The caller must see the same two numbers in the same events; nothing recounts or
    restamps usage on the raw relay."""
    case: Final = marker()
    provider.script(case, lambda _: _sse_reply(anthropic_stream_chunks(case)))
    events: Final = dict(sse_events(gateway.post("/v1/messages", messages_body(case, stream=True)).text))
    start: Final = events["message_start"]["message"]
    assert isinstance(start, dict)
    assert start["usage"] == {"input_tokens": 5, "output_tokens": 3}
    assert events["message_delta"]["usage"] == {"output_tokens": 3}


def test_crlf_framed_provider_events_reach_the_caller_with_crlf(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider frames every event with `\\r\\n` line endings. The caller's body must equal the provider's
    bytes, CRLFs included; the relay does not normalise line endings (raw `Bytes` relay)."""
    case: Final = marker()
    chunks: Final = tuple(chunk.replace(b"\n", b"\r\n") for chunk in anthropic_stream_chunks(case))
    provider.script(case, lambda _: _sse_reply(chunks))
    response: Final = gateway.post("/v1/messages", messages_body(case, stream=True))
    assert response.content == b"".join(chunks)


def test_a_stream_head_is_text_event_stream_without_provider_headers(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider's stream reply carries `request-id: req_dec1_stream` and
    `anthropic-ratelimit-requests-remaining: 9`. The caller's response must have `content-type`
    `text/event-stream` and neither provider header, because `encode_stream_head` builds a fresh head with only
    the content type (`host-http/src/sse.rs:43-49`)."""
    case: Final = marker()
    provider.script(
        case,
        lambda _: _sse_reply(
            anthropic_stream_chunks(case),
            **{"request-id": "req_dec1_stream", "anthropic-ratelimit-requests-remaining": "9"},
        ),
    )
    response: Final = gateway.post("/v1/messages", messages_body(case, stream=True))
    assert response.headers["content-type"] == "text/event-stream"
    assert "request-id" not in response.headers
    assert "anthropic-ratelimit-requests-remaining" not in response.headers


# Bedrock invoke: decoded relay


def test_a_bedrock_stream_is_re_encoded_as_python_json_sse(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The Bedrock provider streams six Claude events as AWS event-stream `chunk` messages. The caller must get
    `content-type` `text/event-stream` and the `content_block_stop` event exactly as
    `event: content_block_stop\\ndata: {"type": "content_block_stop", "index": 0}\\n\\n` (Python `json.dumps`
    spacing, `sse_frame` in `bedrock/messages/invoke_transformations/stream.rs`)."""
    case: Final = marker()
    events: Final = bedrock_claude_events(case)
    provider.script(case, lambda _: bedrock_stream(events))
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True))
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "text/event-stream"
    assert python_sse(events[3]) in sse_blocks(response.text)


def test_a_bedrock_stream_keeps_the_anthropic_event_order(gateway: NativeGateway, provider: ScriptedProvider) -> None:
    """The same six-event Bedrock stream. The caller's event names must be exactly `message_start`,
    `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, with
    no error appended (the stream ended with `message_stop`, `relay.rs:108`)."""
    case: Final = marker()
    provider.script(case, lambda _: bedrock_stream(bedrock_claude_events(case)))
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True))
    assert [name for name, _ in sse_events(response.text)] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


def test_bedrock_invocation_metrics_become_usage_on_message_stop(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """Bedrock's `message_stop` carries `amazon-bedrock-invocationMetrics` with `inputTokenCount` 11 and
    `outputTokenCount` 4. The caller's `message_stop` must have no metrics key and `usage` equal to
    `{"input_tokens": 11, "output_tokens": 4}` (`chunk_parser` and `METRICS_USAGE_KEYS` in `stream.rs`)."""
    case: Final = marker()
    provider.script(case, lambda _: bedrock_stream(bedrock_claude_events(case, input_tokens=11, output_tokens=4)))
    events: Final = dict(
        sse_events(gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True)).text)
    )
    assert INVOCATION_METRICS_KEY not in events["message_stop"]
    assert events["message_stop"]["usage"] == {"input_tokens": 11, "output_tokens": 4}


def test_bedrock_message_delta_gains_the_input_tokens_from_the_metrics(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The provider's `message_delta` has only `usage.output_tokens` 4. Because the relay holds `message_delta`
    until `message_stop` and copies the metrics' `input_tokens` into it (`UsagePromoter::completed`), the
    caller's `message_delta.usage` must be `{"output_tokens": 4, "input_tokens": 11}`. Spend for Claude Code on
    Bedrock is computed from this event."""
    case: Final = marker()
    provider.script(case, lambda _: bedrock_stream(bedrock_claude_events(case, input_tokens=11, output_tokens=4)))
    events: Final = dict(
        sse_events(gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True)).text)
    )
    assert events["message_delta"]["usage"] == {"output_tokens": 4, "input_tokens": 11}


def test_a_bedrock_stream_cut_before_message_stop_ends_with_the_incomplete_stream_error(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The Bedrock provider sends the first four events and ends its body cleanly. The caller's last block must
    be exactly the incomplete-stream event `event: error` / `api_error` / `Provider stream ended before
    emitting a message_stop event; ...` (`incomplete_stream_error_event`, `relay.rs:13-17`), and it must
    appear once."""
    case: Final = marker()
    provider.script(case, lambda _: bedrock_stream(bedrock_claude_events(case)[:4]))
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True))
    blocks: Final = sse_blocks(response.text)
    assert blocks[-1] == INCOMPLETE_STREAM_EVENT
    assert blocks.count(INCOMPLETE_STREAM_EVENT) == 1


def test_a_bedrock_message_delta_without_a_stop_is_flushed_before_the_incomplete_error(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The Bedrock provider sends everything up to `message_delta` and ends without `message_stop`. The held
    `message_delta` must still reach the caller (`UsagePromoter::finish`), followed by the incomplete-stream
    error as the final event."""
    case: Final = marker()
    provider.script(case, lambda _: bedrock_stream(bedrock_claude_events(case)[:5]))
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True))
    names: Final = [name for name, _ in sse_events(response.text)]
    assert names[-2:] == ["message_delta", "error"]
    assert sse_blocks(response.text)[-1] == INCOMPLETE_STREAM_EVENT


def test_a_bedrock_error_event_with_an_error_object_is_terminal(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The Bedrock provider sends `message_start` then a chunk `{"type": "error", "error": {"type":
    "overloaded_error", "message": "dec1 busy"}}` and ends. That chunk is a terminal event
    (`is_provider_error_chunk`, `relay.rs:64-76`), so the caller must get it as the last event and no
    incomplete-stream error."""
    case: Final = marker()
    failure: Final = {"type": "error", "error": {"type": "overloaded_error", "message": "dec1 busy"}}
    provider.script(case, lambda _: bedrock_stream((bedrock_claude_events(case)[0], failure)))
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True))
    events: Final = sse_events(response.text)
    assert events[-1] == ("error", failure)
    assert INCOMPLETE_STREAM_EVENT not in sse_blocks(response.text)


def test_a_bedrock_error_event_without_an_error_object_is_not_terminal(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """The Bedrock provider sends `message_start` then `{"type": "error", "error": "dec1 overloaded"}` (a string,
    not an object) and ends. That chunk is relayed but is not terminal (unit case
    `error_without_an_error_object`, `relay.rs:159`), so the incomplete-stream error must follow it."""
    case: Final = marker()
    provider.script(
        case, lambda _: bedrock_stream((bedrock_claude_events(case)[0], {"type": "error", "error": "dec1 overloaded"}))
    )
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True))
    blocks: Final = sse_blocks(response.text)
    assert json.loads(blocks[-2].split("data: ", 1)[1]) == {"type": "error", "error": "dec1 overloaded"}
    assert blocks[-1] == INCOMPLETE_STREAM_EVENT


def test_a_bedrock_error_before_the_stream_is_a_json_anthropic_error(
    gateway: NativeGateway, provider: ScriptedProvider
) -> None:
    """A streaming Bedrock call whose provider answers 400 `{"message": "dec1 bad stream request"}` before any
    frame. No stream head goes out, so the caller must get 400 `application/json` with
    `{"type": "error", "error": {"type": "invalid_request_error", "message": "dec1 bad stream request"}}`."""
    case: Final = marker()
    provider.script(case, lambda _: Reply(status=400, body=b'{"message": "dec1 bad stream request"}'))
    response: Final = gateway.post("/v1/messages", messages_body(case, "bedrock-claude", stream=True))
    assert response.status_code == 400, response.text
    assert response.headers["content-type"] == "application/json"
    assert response.json() == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": "dec1 bad stream request"},
    }
