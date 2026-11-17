"""Keepalive pings while a stream is idle, and the timeouts that bound a stream.

Keepalive: a deployment's `keepalive_seconds` (or the global `litellm_settings.sse_keepalive_ping_interval_seconds`,
unset by default) makes the proxy write `: ping` SSE comments whenever the provider is silent for that long,
before and after the first chunk (`_iter_with_keepalive`, `_resolve_keepalive_seconds` in
`litellm/proxy/proxy_server.py`). The interval is clamped to 1..300 seconds. `keepalive_seconds: 0` on the
deployment is a hard disable. A caller may pick its own interval (body `keepalive_seconds` or header
`x-litellm-keepalive-seconds`) only when the deployment sets `allow_client_keepalive_override: true`. Pings
are not provider output: they never become chunks and never count as the first chunk for billing.

Timeouts: `stream_timeout` bounds each wait for the provider, before and between chunks, not the stream as
a whole. `total_timeout` bounds only non-streaming requests: in the body of a streaming request it is refused
with 400, in the `x-litellm-total-timeout` header of a streaming request it is ignored
(`total_timeout_for`, `litellm/proxy/common_request_processing.py:3787`).
"""

from __future__ import annotations

import threading
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, object_value
from integration.streaming.dec1_stream2_support import (
    ANTHROPIC,
    OPENAI,
    OPENAI_RESPONSES,
    Provider,
    deployed,
    opened,
    provider_stream,
    route_body,
)
from integration.streaming.dec1_stream_support import Rig, chunks_of, content_of, marker, payloads

PIECES: Final = ("idle ", "then ", "done")
TOTAL_TIMEOUT_REFUSAL: Final = (
    "total_timeout applies only to non-streaming, non-background /v1/chat/completions, /v1/messages and "
    "/v1/responses requests"
)


def _lines_with_a_pause(
    rig: Rig, route: str, gate: threading.Event, pings_before_release: int, **extra: object
) -> list[str]:
    """Read a stream whose provider holds after its first chunk; release it once `pings_before_release`
    ping comments arrived (or at once when 0, after a fixed number of lines), and return every line."""
    lines: list[str] = []
    with opened(rig, route, **extra) as response:  # type: ignore[arg-type]
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            lines.append(line)
            if pings_before_release and lines.count(": ping") >= pings_before_release:
                gate.set()
    gate.set()
    return lines


def test_mid_stream_silence_is_filled_with_ping_comments(gateway: Gateway) -> None:
    """A deployment with `keepalive_seconds: 1` sends its first chunk and then goes silent until the caller
    has seen two `: ping` lines. The chat caller receives the first data frame, then `: ping` comment lines,
    then the rest of the text, and `[DONE]` last; the concatenated content is `idle then done`."""
    gate: Final = threading.Event()
    with deployed(gateway, OPENAI, provider_stream(OPENAI, PIECES, gate_after_first=gate), keepalive_seconds=1) as rig:
        lines: Final = _lines_with_a_pause(rig, "/v1/chat/completions", gate, 2)
        first_data: Final = next(index for index, line in enumerate(lines) if line.startswith("data:"))
        assert ": ping" in lines[first_data + 1 :]
        assert lines[-1] == "data: [DONE]"
        text: Final = "\n\n".join(line for line in lines if line.startswith("data:"))
        assert content_of(chunks_of(text)) == "idle then done"


def test_ping_comments_are_never_turned_into_chunks(gateway: Gateway) -> None:
    """Same paused stream with `keepalive_seconds: 1`. Every non empty line the caller receives is either a
    `data:` frame or exactly `: ping`; no `data:` frame has an empty `choices` list or empty delta created
    for a ping."""
    gate: Final = threading.Event()
    with deployed(gateway, OPENAI, provider_stream(OPENAI, PIECES, gate_after_first=gate), keepalive_seconds=1) as rig:
        lines: Final = [line for line in _lines_with_a_pause(rig, "/v1/chat/completions", gate, 2) if line]
        assert all(line.startswith("data:") or line == ": ping" for line in lines)
        data_frames: Final = chunks_of("\n\n".join(line for line in lines if line.startswith("data:")))
        assert len(data_frames) == len([frame for frame in data_frames if frame.get("choices") or frame.get("usage")])


def test_no_pings_without_any_keepalive_setting(gateway: Gateway) -> None:
    """A deployment without `keepalive_seconds` (the shared proxy sets no global interval) pauses for 2.5
    seconds after its first chunk. The caller receives no `: ping` line at all; a silent provider stays
    silent on the wire."""
    gate: Final = threading.Event()
    timer: Final = threading.Timer(2.5, gate.set)
    with deployed(gateway, OPENAI, provider_stream(OPENAI, PIECES, gate_after_first=gate)) as rig:
        timer.start()
        lines: Final = _lines_with_a_pause(rig, "/v1/chat/completions", gate, 0)
        assert ": ping" not in lines


def test_keepalive_zero_on_the_deployment_cannot_be_reenabled_by_the_caller_header(gateway: Gateway) -> None:
    """A deployment with `keepalive_seconds: 0` and `allow_client_keepalive_override: true` pauses 2.5 seconds;
    the caller sends `x-litellm-keepalive-seconds: 1`. No `: ping` line arrives: an operator's explicit 0 is a
    hard disable (`_is_explicit_keepalive_disable`)."""
    gate: Final = threading.Event()
    timer: Final = threading.Timer(2.5, gate.set)
    with deployed(
        gateway,
        OPENAI,
        provider_stream(OPENAI, PIECES, gate_after_first=gate),
        keepalive_seconds=0,
        allow_client_keepalive_override=True,
    ) as rig:
        timer.start()
        lines: Final = _lines_with_a_pause(
            rig, "/v1/chat/completions", gate, 0, headers={"x-litellm-keepalive-seconds": "1"}
        )
        assert ": ping" not in lines


def test_caller_keepalive_header_is_ignored_without_the_deployment_opt_in(gateway: Gateway) -> None:
    """A deployment with no keepalive and no `allow_client_keepalive_override` pauses 2.5 seconds; the caller
    sends `x-litellm-keepalive-seconds: 1`. No `: ping` line arrives: a client cannot switch heartbeats on
    (and dodge an idle timeout) for a deployment that never opted in."""
    gate: Final = threading.Event()
    timer: Final = threading.Timer(2.5, gate.set)
    with deployed(gateway, OPENAI, provider_stream(OPENAI, PIECES, gate_after_first=gate)) as rig:
        timer.start()
        lines: Final = _lines_with_a_pause(
            rig, "/v1/chat/completions", gate, 0, headers={"x-litellm-keepalive-seconds": "1"}
        )
        assert ": ping" not in lines


def test_caller_keepalive_header_is_honoured_with_the_deployment_opt_in(gateway: Gateway) -> None:
    """A deployment with `allow_client_keepalive_override: true` and no interval of its own pauses after its
    first chunk; the caller sends `x-litellm-keepalive-seconds: 1`. `: ping` lines arrive during the pause
    and the stream then completes with the full text."""
    gate: Final = threading.Event()
    with deployed(
        gateway, OPENAI, provider_stream(OPENAI, PIECES, gate_after_first=gate), allow_client_keepalive_override=True
    ) as rig:
        lines: Final = _lines_with_a_pause(
            rig, "/v1/chat/completions", gate, 2, headers={"x-litellm-keepalive-seconds": "1"}
        )
        assert lines.count(": ping") >= 2
        assert lines[-1] == "data: [DONE]"


def test_caller_keepalive_body_field_is_honoured_and_never_reaches_the_provider(gateway: Gateway) -> None:
    """With the deployment opt in, the caller sends `keepalive_seconds: 1` in the JSON body. Pings arrive
    during the pause, and the provider's request body has no `keepalive_seconds` field."""
    gate: Final = threading.Event()
    with deployed(
        gateway, OPENAI, provider_stream(OPENAI, PIECES, gate_after_first=gate), allow_client_keepalive_override=True
    ) as rig:
        lines: Final = _lines_with_a_pause(rig, "/v1/chat/completions", gate, 2, keepalive_seconds=1)
        assert lines.count(": ping") >= 2
        assert all("keepalive_seconds" not in body for body in rig.upstream_bodies())


def test_a_sub_second_keepalive_is_clamped_to_one_second(gateway: Gateway) -> None:
    """A deployment asks for `keepalive_seconds: 0.05` and pauses 2.5 seconds after its first chunk. The
    caller receives at most 3 `: ping` lines, not the ~50 a 50 ms interval would produce, because the
    interval is clamped to `_KEEPALIVE_MIN_SECONDS` (1.0)."""
    gate: Final = threading.Event()
    timer: Final = threading.Timer(2.5, gate.set)
    with deployed(
        gateway, OPENAI, provider_stream(OPENAI, PIECES, gate_after_first=gate), keepalive_seconds=0.05
    ) as rig:
        timer.start()
        lines: Final = _lines_with_a_pause(rig, "/v1/chat/completions", gate, 0)
        assert 1 <= lines.count(": ping") <= 3


def test_responses_stream_silence_is_filled_with_ping_comments(gateway: Gateway) -> None:
    """A native OpenAI Responses deployment with `keepalive_seconds: 1` pauses after its first event. The
    `/v1/responses` caller receives `: ping` comment lines between events, every event still parses, and no
    `[DONE]` is added (the Responses route shares the keepalive generator)."""
    gate: Final = threading.Event()
    with deployed(
        gateway, OPENAI_RESPONSES, provider_stream(OPENAI_RESPONSES, PIECES, gate_after_first=gate), keepalive_seconds=1
    ) as rig:
        lines: Final = _lines_with_a_pause(rig, "/v1/responses", gate, 2)
        assert lines.count(": ping") >= 2
        assert "data: [DONE]" not in lines
        events: Final = chunks_of("\n\n".join(line for line in lines if line.startswith("data:")))
        assert events[-1]["type"] == "response.completed"


def test_messages_stream_pings_keep_the_anthropic_grammar(gateway: Gateway) -> None:
    """A native Anthropic `/v1/messages` deployment with `keepalive_seconds: 1` pauses after message_start.
    Every frame the proxy adds during the pause is either the comment `: ping` or the Anthropic
    `event: ping` with data `{"type": "ping"}` (`STREAM_SSE_KEEPALIVE_PING_CHUNK`), the stream ends with
    message_stop, and the Anthropic SDK grammar is never broken. UNVERIFIED: which of the two forms the Rust
    messages relay writes mid stream, and whether it honours `keepalive_seconds` at all."""
    gate: Final = threading.Event()
    timer: Final = threading.Timer(2.5, gate.set)
    with deployed(
        gateway, ANTHROPIC, provider_stream(ANTHROPIC, PIECES, gate_after_first=gate), keepalive_seconds=1
    ) as rig:
        timer.start()
        lines: Final = [line for line in _lines_with_a_pause(rig, "/v1/messages", gate, 0) if line]
        assert all(line.startswith(("event:", "data:")) or line == ": ping" for line in lines)
        assert lines[-2:] == ["event: message_stop", 'data: {"type": "message_stop"}']


def test_a_stream_of_only_pings_when_the_caller_leaves_charges_no_output(gateway: Gateway) -> None:
    """A deployment with `keepalive_seconds: 1` holds its first chunk; the caller reads two `: ping` lines
    (so the 200 and headers are already out) and hangs up. No output is charged: the key's stored spend
    stays 0 on a deployment that only prices output, because pings are not provider output
    (`provider_output_started` stays false)."""
    gate: Final = threading.Event()
    with deployed(
        gateway,
        OPENAI,
        provider_stream(OPENAI, PIECES, gate_before_first=gate),
        keepalive_seconds=1,
        input_cost_per_token=0,
        output_cost_per_token=0.002,
    ) as rig:
        with opened(rig, "/v1/chat/completions") as response:
            assert response.status_code == 200
            seen = 0
            for line in response.iter_lines():
                seen += line == ": ping"
                if seen == 2:
                    break
        assert eventually(rig.wire.open_connections, lambda count: count == 0, seconds=10) == 0
        gate.set()
        assert rig.key_spend() == 0.0


@pytest.mark.parametrize(
    ("route", "provider"),
    [("/v1/chat/completions", OPENAI), ("/v1/messages", ANTHROPIC), ("/v1/responses", OPENAI_RESPONSES)],
    ids=["chat", "messages", "responses"],
)
def test_total_timeout_in_the_body_of_a_streaming_request_is_refused_with_400(
    gateway: Gateway, route: str, provider: Provider
) -> None:
    """A streaming request on each route carries `total_timeout: 5` in its body. The caller gets 400 with
    `error.type == "invalid_request_error"`, `error.param == "total_timeout"` and the exact message
    `total_timeout applies only to non-streaming, non-background /v1/chat/completions, /v1/messages and
    /v1/responses requests`, and the provider is never called."""
    with deployed(gateway, provider, provider_stream(provider, PIECES)) as rig:
        response: Final = rig.post(route, route_body(route, rig.model, total_timeout=5))
        assert response.status_code == 400
        error: Final = object_value(response.json()["error"])
        assert (error["type"], error["param"]) == ("invalid_request_error", "total_timeout")
        assert error["message"] == TOTAL_TIMEOUT_REFUSAL
        assert rig.upstream_bodies() == []


def test_total_timeout_header_on_a_streaming_request_is_ignored(gateway: Gateway) -> None:
    """A streaming chat request carries `x-litellm-total-timeout: 1` and the provider takes about 2.4 seconds
    in all (three pieces 0.8 s apart). The caller still receives the full text and `[DONE]`: a header total
    timeout (often set globally on a client) does not bound a stream."""
    with deployed(gateway, OPENAI, provider_stream(OPENAI, PIECES, pause_between_chunks=0.8)) as rig:
        response: Final = rig.chat(headers={"x-litellm-total-timeout": "1"})
        assert response.status_code == 200
        assert content_of(chunks_of(response.text)) == "idle then done"
        assert payloads(response.text)[-1] == "[DONE]"


def test_invalid_total_timeout_header_on_a_streaming_request_is_also_ignored(gateway: Gateway) -> None:
    """A streaming chat request carries `x-litellm-total-timeout: soon`. The header is ignored before it is
    parsed for a request it cannot bound, so the caller gets 200 and the stream rather than the 400 a
    non-streaming request with that header gets."""
    with deployed(gateway, OPENAI, provider_stream(OPENAI, PIECES)) as rig:
        response: Final = rig.chat(headers={"x-litellm-total-timeout": "soon"})
        assert response.status_code == 200
        assert content_of(chunks_of(response.text)) == "idle then done"


@pytest.mark.parametrize(
    ("route", "provider"),
    [("/v1/messages", ANTHROPIC), ("/v1/responses", OPENAI_RESPONSES)],
    ids=["messages", "responses"],
)
def test_stream_timeout_bounds_each_wait_not_the_whole_stream(gateway: Gateway, route: str, provider: Provider) -> None:
    """A deployment with `stream_timeout: 1.5` streams quickly to first byte and then one event every 0.8 s,
    about 5 seconds in all. The caller of `/v1/messages` or `/v1/responses` receives the whole stream with its
    normal terminal event: no single wait exceeded the stream timeout, so the total length does not matter."""
    pieces: Final = tuple(f"p{number} " for number in range(4))
    with deployed(
        gateway, provider, provider_stream(provider, pieces, pause_between_chunks=0.8), stream_timeout=1.5
    ) as rig:
        response: Final = rig.post(route, route_body(route, rig.model), headers={"anthropic-version": "2023-06-01"})
        assert response.status_code == 200
        last: Final = chunks_of(response.text)[-1]["type"]
        assert last in {"message_stop", "response.completed"}


def test_messages_stream_silent_past_stream_timeout_before_the_first_byte_is_an_http_error(gateway: Gateway) -> None:
    """A native Anthropic deployment with `stream_timeout: 1` accepts the request and holds its first event
    for 10 seconds. The `/v1/messages` caller gets an HTTP error status (408 or 5xx, never 200) within a few
    seconds, with the Anthropic error envelope. UNVERIFIED: the exact status the Rust relay maps an upstream
    first byte timeout to."""
    gate: Final = threading.Event()
    timer: Final = threading.Timer(10, gate.set)
    with deployed(
        gateway, ANTHROPIC, provider_stream(ANTHROPIC, PIECES, gate_before_first=gate), stream_timeout=1
    ) as rig:
        timer.start()
        try:
            response: Final = rig.messages(f"dec1 stream2 timeout {marker()}")
        finally:
            gate.set()
        assert response.status_code != 200
        assert response.status_code in {408, 500, 502, 503, 504}
        assert response.json()["type"] == "error"
