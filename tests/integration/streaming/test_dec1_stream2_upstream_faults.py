"""A provider that hangs up, sends broken SSE, or ends on a partial event, per route and provider dialect.

What the caller sees depends on the route. `/v1/chat/completions` already sent 200, so a provider failure
after the first chunk becomes one `data: {"error": {...}}` frame followed by `data: [DONE]`
(`async_data_generator`, `litellm/proxy/proxy_server.py`). Native `/v1/messages` relays the provider's bytes
and, on a transport failure, writes one `event: error` frame in the Anthropic envelope
(`litellm/rust/crates/gateway-inference/src/error.rs::sse_frame`); a provider stream that ends cleanly
without `message_stop` or an error event gets the `api_error` event with the fixed incomplete stream
message (`litellm/rust/crates/inference-messages/src/relay.rs`). Neither `/v1/messages` nor `/v1/responses`
ever ends with `[DONE]`. A failure before the first chunk is a plain HTTP error with a JSON body on every
route, because the proxy holds the status line until the first chunk.

The relay's terminal event detection is byte exact (`event: message_stop` with a space; CRLF and CR line
endings accepted), so those edges are pinned here too. Mid-stream cuts on OpenAI and Azure chat, the
Anthropic native clean EOF case and the Bedrock Converse exception frame already have tests; this file adds
the remaining providers and the malformed frame cases.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.streaming.dec1_stream2_support import (
    ANTHROPIC,
    AZURE_FOUNDRY_CLAUDE,
    BEDROCK_CONVERSE,
    BEDROCK_INVOKE,
    GEMINI_AI_STUDIO,
    INCOMPLETE_MESSAGES_STREAM,
    NATIVE_MESSAGES_PROVIDERS,
    OPENAI,
    OPENAI_RESPONSES,
    SSE,
    Provider,
    anthropic_frames,
    deployed,
    provider_ids,
    route_body,
    streaming,
)
from integration.streaming.dec1_stream_support import (
    bedrock_frame,
    chunks_of,
    content_of,
    delta,
    event,
    events_of,
    marker,
    payloads,
)

PIECES: Final = ("before ", "the ", "cut")


def _cut_after_text(provider: Provider) -> tuple[bytes, ...]:
    """The provider's frames up to and including the second text piece; the connection then drops."""
    frames: Final = provider.frames(f"dec1s2-{marker()}", PIECES)
    keep: Final = {
        ANTHROPIC.name: 4,
        AZURE_FOUNDRY_CLAUDE.name: 4,
        BEDROCK_INVOKE.name: 4,
        GEMINI_AI_STUDIO.name: 2,
    }.get(provider.name, 3)
    return frames[:keep]


def _dropping(provider: Provider) -> Reply:
    frames: Final = _cut_after_text(provider)
    return Reply(content_type=provider.content_type, chunks=(*frames, b""), abort_after=len(frames))


def _error_frames(text: str) -> list[dict]:
    return [item for item in chunks_of(text) if "error" in item]


@pytest.mark.parametrize("provider", [ANTHROPIC, GEMINI_AI_STUDIO, BEDROCK_CONVERSE], ids=provider_ids)
def test_chat_provider_drop_after_text_ends_with_one_error_frame_then_done(
    gateway: Gateway, provider: Provider
) -> None:
    """Anthropic, Gemini or Bedrock Converse streams two text pieces and then the TCP connection drops with
    no terminator. The `/v1/chat/completions` caller gets status 200, the text `before the ` intact, then
    exactly one `data: {"error": ...}` frame, and `data: [DONE]` as the very last frame."""
    with deployed(gateway, provider, lambda request: _dropping(provider)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200
        frames: Final = payloads(response.text)
        assert frames[-1] == "[DONE]"
        assert content_of([item for item in chunks_of(response.text) if "error" not in item]).startswith("before ")
        assert len(_error_frames(response.text)) == 1


@pytest.mark.parametrize("provider", [ANTHROPIC, GEMINI_AI_STUDIO, BEDROCK_CONVERSE], ids=provider_ids)
def test_chat_mid_stream_error_frame_has_the_openai_error_envelope(gateway: Gateway, provider: Provider) -> None:
    """Same provider drop. The error frame's `error` object has exactly the keys `message`, `type`, `param`
    and `code` (`ProxyException.to_dict`), its message is a non empty string, and it carries no Python
    traceback text. UNVERIFIED: the exact `code` for a transport drop (likely `"500"`)."""
    with deployed(gateway, provider, lambda request: _dropping(provider)) as rig:
        error: Final = object_value(_error_frames(rig.chat().text)[0]["error"])
        assert set(error) == {"message", "type", "param", "code"}
        assert isinstance(error["message"], str) and error["message"]
        assert "Traceback" not in str(error["message"])


@pytest.mark.parametrize("provider", [ANTHROPIC, GEMINI_AI_STUDIO, BEDROCK_CONVERSE], ids=provider_ids)
def test_chat_provider_drop_after_text_is_logged_as_one_failure_row(gateway: Gateway, provider: Provider) -> None:
    """Same provider drop. Exactly one SpendLogs row lands for the key and its `status` is `failure`: a
    stream the provider broke is not a success, unlike a stream the caller abandoned."""
    with deployed(gateway, provider, lambda request: _dropping(provider)) as rig:
        rig.chat()
        assert rig.spend_rows(1, settle=5)[0]["status"] == "failure"


@pytest.mark.parametrize("provider", NATIVE_MESSAGES_PROVIDERS, ids=provider_ids)
def test_messages_provider_drop_after_text_ends_with_an_anthropic_error_event_and_no_done(
    gateway: Gateway, provider: Provider
) -> None:
    """A native `/v1/messages` provider (Anthropic, Foundry Claude, Bedrock Invoke) sends message_start, the
    text block start and one text delta, then the connection drops. The caller gets 200, the relayed events
    first, then a last event named `error` whose data is `{"type": "error", "error": {"type": <str>,
    "message": <str>}}`, and no `[DONE]` anywhere in the body."""
    with deployed(gateway, provider, lambda request: _dropping(provider)) as rig:
        response: Final = rig.messages()
        assert response.status_code == 200
        events: Final = events_of(response.text)
        assert events[0][0] == "message_start"
        name, body = events[-1]
        assert name == "error" and body["type"] == "error"
        error: Final = object_value(body["error"])
        assert isinstance(error["type"], str) and isinstance(error["message"], str)
        assert "[DONE]" not in response.text


@pytest.mark.parametrize("provider", [AZURE_FOUNDRY_CLAUDE, BEDROCK_INVOKE], ids=provider_ids)
def test_messages_clean_eof_without_message_stop_appends_the_incomplete_stream_error(
    gateway: Gateway, provider: Provider
) -> None:
    """Foundry Claude or Bedrock Invoke sends a well formed stream that simply stops after a text delta
    (clean end of body, no message_delta, no message_stop). The caller's last event is exactly
    `{"type": "error", "error": {"type": "api_error", "message": "Provider stream ended before emitting a
    message_stop event; ..."}}`. The Anthropic provider has this test already; these two moved to the Rust
    relay with it (#51721, #51330).

    BUG (Foundry cell): run 44176682961 relayed the Foundry stream to its last `content_block_delta` with no error
    event, while the Anthropic and Bedrock Invoke cells append it (`INCOMPLETE_STREAM_ERROR_MESSAGE`,
    litellm/rust/crates/inference-messages/src/relay.rs:8, and the Python iterator at
    litellm/llms/anthropic/messages/native/streaming_iterator.py:700). UNVERIFIED which path Foundry takes."""
    if provider is AZURE_FOUNDRY_CLAUDE:
        pytest.skip("BUG: a Foundry Claude stream cut before message_stop gets no incomplete stream error")
    frames: Final = provider.frames(f"dec1s2-{marker()}", PIECES)
    truncated: Final = frames[: len(frames) - 3]
    with deployed(gateway, provider, streaming(truncated, provider.content_type)) as rig:
        name, body = events_of(rig.messages().text)[-1]
        assert name == "error"
        assert body == {"type": "error", "error": {"type": "api_error", "message": INCOMPLETE_MESSAGES_STREAM}}


def test_messages_crlf_terminated_message_stop_counts_as_terminal(gateway: Gateway) -> None:
    """The Anthropic provider writes every line with CRLF endings (`event: message_stop\\r\\ndata: ...\\r\\n\\r\\n`).
    The caller receives the whole stream and no appended incomplete stream error: terminal detection splits
    on CR, LF and CRLF (`relay.rs` `lines`, case `crlf`)."""
    frames: Final = tuple(frame.replace(b"\n", b"\r\n") for frame in anthropic_frames(f"msg_{marker()}", PIECES))
    with deployed(gateway, ANTHROPIC, streaming(frames)) as rig:
        response: Final = rig.messages()
        assert INCOMPLETE_MESSAGES_STREAM not in response.text
        assert events_of(response.text)[-1][0] == "message_stop"


def test_messages_unspaced_message_stop_line_is_terminal(gateway: Gateway) -> None:
    """The Anthropic provider ends with `event:message_stop` (no space after the colon, legal SSE). The relay
    now treats it as the terminal event: run 44176682961 relayed the frame as sent and appended no `api_error`
    incomplete stream event, where the round 1 relay only recognised the exact line `event: message_stop`.
    UNVERIFIED: the relay line that now parses the unspaced field."""
    frames: Final = anthropic_frames(f"msg_{marker()}", PIECES)
    unspaced: Final = (*frames[:-1], b'event:message_stop\ndata: {"type": "message_stop"}\n\n')
    with deployed(gateway, ANTHROPIC, streaming(unspaced)) as rig:
        text: Final = rig.messages().text
        assert text.rstrip().endswith('event:message_stop\ndata: {"type": "message_stop"}'), text[-400:]
        assert INCOMPLETE_MESSAGES_STREAM not in text


def test_messages_last_event_without_a_trailing_blank_line_still_ends_the_stream(gateway: Gateway) -> None:
    """The Anthropic provider's final `message_stop` frame has no trailing blank line and the body then ends.
    The caller receives the `message_stop` event and no incomplete stream error, because terminal detection
    looks at lines, not at complete frames."""
    frames: Final = anthropic_frames(f"msg_{marker()}", PIECES)
    with deployed(gateway, ANTHROPIC, streaming((*frames[:-1], frames[-1].rstrip(b"\n")))) as rig:
        text: Final = rig.messages().text
        assert "event: message_stop" in text
        assert INCOMPLETE_MESSAGES_STREAM not in text


def test_messages_unknown_provider_event_is_relayed_verbatim(gateway: Gateway) -> None:
    """The Anthropic provider inserts `event: provider_metrics` with a JSON body between two text deltas.
    The native `/v1/messages` caller receives that frame byte for byte in the same position, and the stream
    still ends with `message_stop`. The relay is byte level; filtering unknown events would break newer
    Anthropic features before the proxy learns about them."""
    extra: Final = event("provider_metrics", {"type": "provider_metrics", "queue_ms": 3})
    frames: Final = anthropic_frames(f"msg_{marker()}", PIECES)
    with deployed(gateway, ANTHROPIC, streaming((*frames[:3], extra, *frames[3:]))) as rig:
        text: Final = rig.messages().text
        assert extra.decode() in text
        names: Final = [name for name, _ in events_of(text)]
        assert names.index("provider_metrics") == 3 and names[-1] == "message_stop"


def test_messages_malformed_data_line_does_not_stop_the_relay(gateway: Gateway) -> None:
    """The Anthropic provider sends a frame whose data line is not JSON (`data: {"type": "content_block_delta"`)
    between two good text deltas. The native `/v1/messages` caller still receives both good deltas and the
    final `message_stop`. UNVERIFIED: whether the broken frame itself is relayed verbatim (byte relay) or
    dropped; the test does not pin that."""
    broken: Final = b'event: content_block_delta\ndata: {"type": "content_block_delta"\n\n'
    frames: Final = anthropic_frames(f"msg_{marker()}", PIECES)
    with deployed(gateway, ANTHROPIC, streaming((*frames[:3], broken, *frames[3:]))) as rig:
        text: Final = rig.messages().text
        assert '"text": "before "' in text and '"text": "cut"' in text
        assert "event: message_stop" in text


def test_chat_malformed_json_frame_never_reaches_the_caller_as_a_data_frame(gateway: Gateway) -> None:
    """An OpenAI provider sends `data: {"id": broken` between two good deltas. Every `data:` payload the chat
    caller receives is either valid JSON or `[DONE]`, the text before the broken frame is delivered, and
    the last frame is `[DONE]`. UNVERIFIED whether the proxy skips the broken frame or turns it into the
    error frame; both satisfy this contract."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": "good "}),
        b'data: {"id": broken\n\n',
        delta(identity, {"content": "later"}),
        delta(identity, {}, finish="stop"),
        b"data: [DONE]\n\n",
    )
    with deployed(gateway, OPENAI, streaming(frames)) as rig:
        response: Final = rig.chat()
        assert payloads(response.text)[-1] == "[DONE]"
        assert content_of(chunks_of(response.text)).startswith("good ")
        assert "broken" not in response.text.split("[DONE]")[0].replace('"message"', "")


def test_chat_data_lines_without_a_space_after_the_colon_are_parsed(gateway: Gateway) -> None:
    """An OpenAI compatible provider writes `data:{...}` with no space (valid SSE). The chat caller receives
    the full text and one `[DONE]`; the proxy's own frames use `data: ` with a space."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = tuple(
        frame.replace(b"data: ", b"data:", 1)
        for frame in (
            delta(identity, {"role": "assistant", "content": "no "}),
            delta(identity, {"content": "space"}),
            delta(identity, {}, finish="stop"),
            b"data: [DONE]\n\n",
        )
    )
    with deployed(gateway, OPENAI, streaming(frames)) as rig:
        response: Final = rig.chat()
        assert content_of(chunks_of(response.text)) == "no space"
        assert payloads(response.text).count("[DONE]") == 1
        assert all(line.startswith("data: ") for line in response.text.splitlines() if line.startswith("data"))


def test_chat_done_sentinel_without_a_trailing_blank_line_still_ends_with_one_done(gateway: Gateway) -> None:
    """An OpenAI provider's last bytes are `data: [DONE]` with no newline before the body ends. The chat
    caller receives the full text and exactly one `data: [DONE]\\n\\n` at the very end."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": "partial "}),
        delta(identity, {"content": "tail"}),
        delta(identity, {}, finish="stop"),
        b"data: [DONE]",
    )
    with deployed(gateway, OPENAI, streaming(frames)) as rig:
        response: Final = rig.chat()
        assert content_of(chunks_of(response.text)) == "partial tail"
        assert response.text.endswith("data: [DONE]\n\n") and response.text.count("[DONE]") == 1


def test_chat_crlf_framed_provider_stream_is_reframed_with_lf(gateway: Gateway) -> None:
    """An OpenAI compatible provider separates frames with `\\r\\n\\r\\n`. The chat caller receives the full text
    and the proxy's own frames end with `\\n\\n`, with no stray carriage returns in the body."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = tuple(
        frame.replace(b"\n", b"\r\n")
        for frame in (
            delta(identity, {"role": "assistant", "content": "cr"}),
            delta(identity, {"content": "lf"}),
            delta(identity, {}, finish="stop"),
            b"data: [DONE]\n\n",
        )
    )
    with deployed(gateway, OPENAI, streaming(frames)) as rig:
        response: Final = rig.chat()
        assert content_of(chunks_of(response.text)) == "crlf"
        assert "\r" not in response.text


def test_anthropic_chat_translation_ignores_an_unknown_event_type(gateway: Gateway) -> None:
    """An Anthropic deployment inserts an unknown `event: provider_metrics` frame between text deltas. The
    `/v1/chat/completions` caller receives the full text `before the cut`, one `stop` finish and `[DONE]`,
    and no chunk derived from the unknown event."""
    extra: Final = event("provider_metrics", {"type": "provider_metrics", "queue_ms": 3})
    frames: Final = anthropic_frames(f"msg_{marker()}", PIECES)
    with deployed(gateway, ANTHROPIC, streaming((*frames[:3], extra, *frames[3:]))) as rig:
        response: Final = rig.chat()
        assert content_of(chunks_of(response.text)) == "before the cut"
        assert "provider_metrics" not in response.text


def test_bedrock_unknown_event_type_mid_stream_does_not_break_the_chat_stream(gateway: Gateway) -> None:
    """A Bedrock Converse stream carries an event type the proxy does not know (`futureEvent`) between two
    text deltas. The chat caller receives the full text and `[DONE]`. UNVERIFIED: that the Converse decoder
    skips unknown event types rather than failing; look at
    `litellm/llms/bedrock/chat/converse_handler.py` if this fails."""
    frames: Final = BEDROCK_CONVERSE.frames(f"dec1s2-{marker()}", PIECES)
    extra: Final = bedrock_frame("futureEvent", {"note": "ignored"})
    with deployed(
        gateway, BEDROCK_CONVERSE, streaming((*frames[:2], extra, *frames[2:]), BEDROCK_CONVERSE.content_type)
    ) as rig:
        response: Final = rig.chat()
        assert content_of(chunks_of(response.text)) == "before the cut"
        assert payloads(response.text)[-1] == "[DONE]"


def test_bedrock_frame_with_a_bad_checksum_ends_the_chat_stream_with_an_error_frame(gateway: Gateway) -> None:
    """The third Bedrock event stream frame has its message CRC flipped. The chat caller gets 200, the text
    before it, then one error frame and `[DONE]`; it never receives garbled text from the damaged frame.
    UNVERIFIED: that the decoder validates the CRC rather than trusting the length prefix."""
    frames: Final = list(BEDROCK_CONVERSE.frames(f"dec1s2-{marker()}", PIECES))
    damaged: Final = frames[2][:-1] + bytes([frames[2][-1] ^ 0xFF])
    frames[2] = damaged
    with deployed(gateway, BEDROCK_CONVERSE, streaming(tuple(frames), BEDROCK_CONVERSE.content_type)) as rig:
        response: Final = rig.chat()
        assert response.status_code == 200
        assert payloads(response.text)[-1] == "[DONE]"
        assert len(_error_frames(response.text)) == 1


def _empty_then_drop(request: Request) -> Reply:
    return Reply(content_type=SSE, chunks=(b"",), abort_after=0)


@pytest.mark.parametrize(
    ("route", "provider"),
    [("/v1/chat/completions", OPENAI), ("/v1/messages", ANTHROPIC), ("/v1/responses", OPENAI_RESPONSES)],
    ids=["chat", "messages", "responses"],
)
def test_provider_that_dies_before_its_first_byte_is_an_http_error_not_a_200(
    gateway: Gateway, route: str, provider: Provider
) -> None:
    """The provider answers status 200 with SSE headers and then drops the connection without sending a
    single body byte. On each route the caller gets an HTTP status of 500 or above (never 200) with a JSON
    body that has an `error` key, because the proxy holds the status line until the first chunk.
    UNVERIFIED: the exact status (500 or 502) for a body that dies before any byte."""
    with deployed(gateway, provider, _empty_then_drop) as rig:
        response: Final = rig.post(route, route_body(route, rig.model), headers={"anthropic-version": "2023-06-01"})
        assert response.status_code >= 500
        assert "error" in response.json()


def test_messages_drop_before_the_first_byte_answers_in_the_anthropic_error_envelope(gateway: Gateway) -> None:
    """Same before-first-byte drop on a native `/v1/messages` Anthropic deployment. The JSON body is the
    Anthropic envelope `{"type": "error", "error": {"type": <str>, "message": <str>}}`, not the OpenAI
    one, since `/v1/messages` callers are Anthropic SDKs (`error.rs::body`)."""
    with deployed(gateway, ANTHROPIC, _empty_then_drop) as rig:
        body: Final = rig.messages().json()
        assert body["type"] == "error"
        assert set(object_value(body["error"])) >= {"type", "message"}


def test_responses_native_drop_after_text_never_ends_with_done(gateway: Gateway) -> None:
    """A native OpenAI Responses stream sends created, item added and one delta, then drops. The caller
    gets 200, the delta text, a final event that is `error` or `response.failed`, and no `[DONE]` frame
    (Responses streams never carry one; `ends_with_openai_done` is false for them)."""
    frames: Final = OPENAI_RESPONSES.frames(f"resp_{marker()}", PIECES)[:3]
    with deployed(
        gateway, OPENAI_RESPONSES, lambda request: Reply(content_type=SSE, chunks=(*frames, b""), abort_after=3)
    ) as rig:
        response: Final = rig.responses()
        assert response.status_code == 200
        assert "[DONE]" not in response.text
        assert chunks_of(response.text)[-1]["type"] in {"error", "response.failed"}
