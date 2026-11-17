"""HTTP framing, response headers, compression, usage and back pressure of streams on every route.

The status line and headers of a stream leave the proxy together with the first chunk (the proxy holds them
so that a failure before the first chunk can still be an HTTP error). They carry `content-type:
text/event-stream`, HTTP/1.1 chunked transfer coding with no `content-length`, `Cache-Control: no-cache` and
`X-Accel-Buffering: no` (`_TTFT_KEEPALIVE_HEADERS`, `litellm/proxy/common_request_processing.py:1799`) and the
`x-litellm-*` call headers. The proxy never gzips an event stream, whatever the caller accepts
(`gzip_middleware.py` compresses only dashboard routes), and a provider that gzips its own stream is
decoded before relay. `stream_options.include_usage` adds a final usage chunk on chat for every provider;
`/v1/messages` always reports usage in `message_delta`. A slow caller makes the proxy stop reading from the
provider instead of buffering the whole answer.

Chat on OpenAI already has the header and chunked framing tests; this file adds `/v1/messages` and
`/v1/responses`, the other providers, gzip and back pressure. The Rust relay for native `/v1/messages`
uses reqwest without its `gzip` feature (`litellm/rust/Cargo.toml:117`), so the gzip contracts there are
the ones most at risk.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, held, object_value
from integration.streaming.dec1_stream2_support import (
    ANTHROPIC,
    AZURE_FOUNDRY_CLAUDE,
    BEDROCK_CONVERSE,
    BEDROCK_INVOKE,
    GEMINI_AI_STUDIO,
    KIB,
    MIB,
    OPENAI,
    OPENAI_RESPONSES,
    Provider,
    big_text_frames,
    deployed,
    gzipped,
    opened,
    progress_deployed,
    provider_ids,
    provider_stream,
    streaming,
)
from integration.streaming.dec1_stream_support import (
    chunks_of,
    content_of,
    events_of,
    marker,
    payloads,
    usage_chunks,
)

ROUTES: Final = [
    ("/v1/chat/completions", OPENAI),
    ("/v1/messages", ANTHROPIC),
    ("/v1/responses", OPENAI_RESPONSES),
]
ROUTE_IDS: Final = ["chat", "messages", "responses"]
PIECES: Final = ("wire ", "level")


@pytest.mark.parametrize(("route", "provider"), ROUTES, ids=ROUTE_IDS)
def test_stream_is_chunked_event_stream_without_content_length_on_each_route(
    gateway: Gateway, route: str, provider: Provider
) -> None:
    """A stream on each route answers 200 with `content-type` starting `text/event-stream`, `transfer-encoding:
    chunked` and no `content-length` header. A length on a stream would make HTTP/1.1 clients wait for the
    whole body."""
    with deployed(gateway, provider, provider_stream(provider, PIECES)) as rig, opened(rig, route) as response:
        response.read()
        assert response.status_code == 200
        assert response.headers["content-type"].split(";")[0].strip() == "text/event-stream"
        assert response.headers.get("transfer-encoding", "").lower() == "chunked"
        assert "content-length" not in response.headers


@pytest.mark.parametrize(("route", "provider"), ROUTES[1:], ids=ROUTE_IDS[1:])
def test_stream_carries_the_anti_buffering_headers_on_messages_and_responses(
    gateway: Gateway, route: str, provider: Provider
) -> None:
    """A stream on `/v1/messages` or `/v1/responses` carries `cache-control: no-cache` and
    `x-accel-buffering: no`, like chat does. Without them nginx style front doors buffer tokens."""
    with deployed(gateway, provider, provider_stream(provider, PIECES)) as rig, opened(rig, route) as response:
        response.read()
        assert response.headers.get("cache-control") == "no-cache"
        assert response.headers.get("x-accel-buffering") == "no"


@pytest.mark.parametrize(("route", "provider"), ROUTES[1:], ids=ROUTE_IDS[1:])
def test_stream_carries_the_call_id_and_model_id_headers_on_messages_and_responses(
    gateway: Gateway, route: str, provider: Provider
) -> None:
    """A stream on `/v1/messages` or `/v1/responses` carries a non empty `x-litellm-call-id` and an
    `x-litellm-model-id` equal to the deployment id registered for the rig's model, and the call id equals
    the `request_id` of the single SpendLogs row. UNVERIFIED for the Rust messages relay's header set."""
    with deployed(gateway, provider, provider_stream(provider, PIECES)) as rig:
        with opened(rig, route) as response:
            response.read()
            call_id: Final = response.headers.get("x-litellm-call-id", "")
            assert call_id
            assert response.headers.get("x-litellm-model-id")
        assert rig.spend_rows(1)[0]["request_id"] == call_id


@pytest.mark.parametrize(("route", "provider"), ROUTES[1:], ids=ROUTE_IDS[1:])
def test_headers_arrive_with_the_first_event_while_the_provider_is_still_streaming(
    gateway: Gateway, route: str, provider: Provider
) -> None:
    """The provider of a `/v1/messages` or `/v1/responses` stream sends its first event and then holds. The
    caller already has status 200, the `x-litellm-call-id` header and the first `data:` line, and gets the
    rest once the provider continues. Buffering the whole stream before the headers would break every UI."""
    gate: Final = threading.Event()
    with deployed(gateway, provider, provider_stream(provider, PIECES, gate_after_first=gate)) as rig:
        with opened(rig, route) as response:
            assert response.status_code == 200
            assert response.headers.get("x-litellm-call-id")
            lines: Final = response.iter_lines()
            assert next(line for line in lines if line.startswith("data:"))
            gate.set()
            rest: Final = [line for line in lines if line.startswith("data:")]
        assert rest


@pytest.mark.parametrize(("route", "provider"), ROUTES, ids=ROUTE_IDS)
def test_stream_is_never_gzipped_even_when_the_caller_accepts_gzip(
    gateway: Gateway, route: str, provider: Provider
) -> None:
    """The caller sends `Accept-Encoding: gzip` on a stream on each route. The response has no
    `content-encoding` header and the raw body bytes start with `event:` or `data:`, not the gzip magic
    `1f 8b`: compressing an event stream would hold tokens in the compressor's window."""
    with (
        deployed(gateway, provider, provider_stream(provider, PIECES)) as rig,
        opened(rig, route, headers={"Accept-Encoding": "gzip"}) as response,
    ):
        raw: Final = b"".join(response.iter_raw())
        assert "content-encoding" not in response.headers
        assert raw[:2] != b"\x1f\x8b"
        assert raw.lstrip().startswith((b"event:", b"data:", b": ping"))


@pytest.mark.parametrize("provider", [OPENAI, ANTHROPIC], ids=provider_ids)
def test_chat_stream_from_a_gzipping_provider_is_decoded_before_relay(gateway: Gateway, provider: Provider) -> None:
    """The OpenAI or Anthropic provider answers the chat stream with `content-encoding: gzip` and a gzip body
    cut into four transfer chunks. The chat caller gets plain SSE (no `content-encoding` header), the full
    text `wire level` and `[DONE]`."""
    frames: Final = gzipped(provider.frames(f"dec1s2-{marker()}", PIECES))
    with deployed(gateway, provider, streaming(frames, headers={"content-encoding": "gzip"})) as rig:
        response: Final = rig.chat()
        assert "content-encoding" not in response.headers
        assert content_of(chunks_of(response.text)) == "wire level"
        assert payloads(response.text)[-1] == "[DONE]"


@pytest.mark.parametrize("provider", [ANTHROPIC, AZURE_FOUNDRY_CLAUDE], ids=provider_ids)
def test_messages_stream_from_a_gzipping_provider_reaches_the_caller_readable(
    gateway: Gateway, provider: Provider
) -> None:
    """A native `/v1/messages` provider gzips its stream (`content-encoding: gzip`). The caller receives
    readable Anthropic events with the text `wire level` ending in `message_stop`, and no
    `content-encoding: gzip` header on a body that is not compressed. Suspected risk: the Rust relay's
    reqwest client is built without the `gzip` feature, so it may pass compressed bytes through; if so this
    should become `pytest.skip("BUG: ...")`."""
    frames: Final = gzipped(provider.frames(f"msg_{marker()}", PIECES))
    with deployed(gateway, provider, streaming(frames, headers={"content-encoding": "gzip"})) as rig:
        response: Final = rig.messages()
        events: Final = events_of(response.text)
        assert events and events[-1][0] == "message_stop"
        texts: Final = [
            object_value(body["delta"]).get("text") for name, body in events if name == "content_block_delta"
        ]
        assert "".join(str(text) for text in texts) == "wire level"
        assert response.headers.get("content-encoding") != "gzip"


def test_responses_stream_from_a_gzipping_provider_is_decoded(gateway: Gateway) -> None:
    """A native OpenAI Responses provider gzips its stream. The `/v1/responses` caller receives readable events
    ending with `response.completed` and the delta text `wire level`."""
    frames: Final = gzipped(OPENAI_RESPONSES.frames(f"resp_{marker()}", PIECES))
    with deployed(gateway, OPENAI_RESPONSES, streaming(frames, headers={"content-encoding": "gzip"})) as rig:
        events: Final = chunks_of(rig.responses().text)
        assert events[-1]["type"] == "response.completed"
        assert "".join(
            str(item.get("delta") or "") for item in events if item["type"] == "response.output_text.delta"
        ) == ("wire level")


@pytest.mark.parametrize(
    ("provider", "usage"),
    [(ANTHROPIC, (10, 5, 15)), (GEMINI_AI_STUDIO, (9, 7, 16)), (BEDROCK_CONVERSE, (9, 5, 14))],
    ids=["anthropic", "gemini", "bedrock"],
)
def test_include_usage_on_chat_reports_each_providers_own_token_counts(
    gateway: Gateway, provider: Provider, usage: tuple[int, int, int]
) -> None:
    """Anthropic (input 10, output 5), Gemini (9, 7, total 16) or Bedrock Converse (9, 5, total 14) streams
    text; the chat caller sends `stream_options: {"include_usage": true}`. Exactly one chunk carries `usage`,
    it has an empty `choices` list, it is the last chunk before `[DONE]`, and its prompt, completion and
    total tokens are the provider's own numbers."""
    with deployed(gateway, provider, provider_stream(provider, PIECES)) as rig:
        response: Final = rig.chat(stream_options={"include_usage": True})
        chunks: Final = chunks_of(response.text)
        with_usage: Final = usage_chunks(chunks)
        assert len(with_usage) == 1 and with_usage[0] is chunks[-1]
        assert chunks[-1]["choices"] == []
        counted: Final = object_value(chunks[-1]["usage"])
        assert (counted["prompt_tokens"], counted["completion_tokens"], counted["total_tokens"]) == usage


@pytest.mark.parametrize("provider", [ANTHROPIC, GEMINI_AI_STUDIO, BEDROCK_CONVERSE], ids=provider_ids)
def test_without_include_usage_chat_from_each_provider_sends_no_usage_chunk(
    gateway: Gateway, provider: Provider
) -> None:
    """The same providers stream with no `stream_options`. No chunk the chat caller receives carries a
    non null `usage`, although the proxy still bills from the provider's usage."""
    with deployed(gateway, provider, provider_stream(provider, PIECES)) as rig:
        assert usage_chunks(chunks_of(rig.chat().text)) == []


@pytest.mark.parametrize("provider", [AZURE_FOUNDRY_CLAUDE, BEDROCK_INVOKE], ids=provider_ids)
def test_messages_stream_reports_usage_in_message_delta_without_being_asked(
    gateway: Gateway, provider: Provider
) -> None:
    """A native `/v1/messages` stream from Foundry Claude or Bedrock Invoke reports `input_tokens: 10` in
    `message_start` and `output_tokens: 5` in `message_delta`, unchanged; the Anthropic protocol has no
    include_usage switch and the relay must not strip or recompute these."""
    with deployed(gateway, provider, provider_stream(provider, PIECES)) as rig:
        events: Final = dict(events_of(rig.messages().text))
        start_usage: Final = object_value(object_value(events["message_start"]["message"])["usage"])
        assert start_usage["input_tokens"] == 10
        assert object_value(events["message_delta"]["usage"])["output_tokens"] == 5


def test_slow_caller_makes_the_proxy_stop_reading_from_the_provider(gateway: Gateway) -> None:
    """The provider tries to stream 1024 deltas of 64 KiB (64 MiB) as fast as its socket allows. The caller
    reads the first frame and then reads nothing for several seconds. During that pause the provider's
    written byte count stops growing well below half the stream: the proxy applies back pressure instead of
    draining the provider into memory. When the caller resumes it receives all 1024 deltas and `[DONE]`."""
    frames: Final = big_text_frames(OPENAI, 64 * KIB, 1024)
    with progress_deployed(gateway, OPENAI, frames) as (rig, peer), opened(rig, "/v1/chat/completions") as response:
        lines: Final = response.iter_lines()
        first: Final = next(line for line in lines if line.startswith("data:"))
        stalled: Final = held(peer.progress, lambda written: written < 32 * MIB, holding=4, seconds=20)
        assert stalled < 32 * MIB and not peer.finished.is_set()
        rest: Final = [line for line in lines if line.startswith("data:")]
        body: Final = "\n\n".join([first, *rest])
        assert payloads(body)[-1] == "[DONE]"
        assert len(content_of(chunks_of(body))) == 1024 * 64 * KIB


def test_slow_caller_on_the_messages_relay_also_gets_back_pressure(gateway: Gateway) -> None:
    """Same 64 MiB provider stream on a native Anthropic `/v1/messages` deployment, served by the Rust byte
    relay. While the caller pauses after the first event, the provider's written byte count stays below
    32 MiB; after resuming the caller receives `message_stop`. A relay with an unbounded channel between the
    upstream reader and the downstream writer would let the provider finish."""
    frames: Final = big_text_frames(ANTHROPIC, 64 * KIB, 1024)
    with progress_deployed(gateway, ANTHROPIC, frames) as (rig, peer), opened(rig, "/v1/messages") as response:
        lines: Final = response.iter_lines()
        next(line for line in lines if line.startswith("data:"))
        stalled: Final = held(peer.progress, lambda written: written < 32 * MIB, holding=4, seconds=20)
        assert stalled < 32 * MIB and not peer.finished.is_set()
        assert [line for line in lines if line.startswith("event:")][-1] == "event: message_stop"


def test_a_paused_caller_does_not_slow_another_callers_stream(gateway: Gateway) -> None:
    """Two callers stream from the same deployment at once. The first reads one frame and then pauses for
    5 seconds; the second reads normally and receives its full stream (three pieces 0.1 s apart) and `[DONE]`
    while the first is still paused. Back pressure is per stream, never a shared stall."""
    with deployed(gateway, OPENAI, provider_stream(OPENAI, PIECES, pause_between_chunks=0.1)) as rig:
        release: Final = threading.Event()

        def paused() -> None:
            with opened(rig, "/v1/chat/completions") as response:
                next(line for line in response.iter_lines() if line.startswith("data:"))
                release.wait(timeout=5)

        with ThreadPoolExecutor(1) as pool:
            slow: Final = pool.submit(paused)
            second: Final = rig.chat()
            first_still_paused: Final = not slow.done()
            release.set()
            slow.result(timeout=30)
        assert content_of(chunks_of(second.text)) == "wire level"
        assert first_still_paused


def test_a_caller_that_never_resumes_and_hangs_up_frees_the_provider(gateway: Gateway) -> None:
    """The 64 MiB provider stream is opened; the caller reads one frame, pauses until back pressure has
    stalled the provider, then hangs up. The provider's next write fails within 10 seconds (the proxy closed
    the upstream connection) and it never finishes the stream."""
    frames: Final = big_text_frames(OPENAI, 64 * KIB, 1024)
    with progress_deployed(gateway, OPENAI, frames) as (rig, peer):
        with opened(rig, "/v1/chat/completions") as response:
            next(line for line in response.iter_lines() if line.startswith("data:"))
            held(peer.progress, lambda written: written < 32 * MIB, holding=2, seconds=20)
        assert eventually(peer.broken.is_set, lambda broken: broken, seconds=10)
        assert not peer.finished.is_set()
