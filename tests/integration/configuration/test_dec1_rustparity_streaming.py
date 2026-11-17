"""Streaming parity: Server-Sent Events through the Python only proxy versus through the Rust forwarder.

Contract: a streamed `/v1/chat/completions`, `/v1/messages` or `/v1/responses` call must reach the caller with the
same status, the same `content-type: text/event-stream`, the same chunked framing, the same sequence of SSE events
(names and normalised data) and the same terminator (`data: [DONE]` for chat, none for messages) in both modes. The
first event must arrive while the provider is still streaming, a provider that drops mid-stream must end the
caller's stream the same way, and a caller that hangs up must make the proxy hang up on the provider in both
modes.

Why it matters for the migration: the forwarder never buffers a body (gateway-forward/AGENTS.md, "Stream both
bodies"), re-frames Python's chunked body with its own chunking after stripping `transfer-encoding`
(`rewrite.rs`), and must close the upstream connection when the client goes away, because that is the only signal
Python has to stop a provider stream. A buffering or re-chunking bug would merge or split SSE events, delay the
first token, or keep paying for an abandoned stream.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import eventually
from tests.integration.configuration.dec1_rustparity_support import (
    ANTHROPIC,
    BODY_FAILED,
    CHAT,
    MASTER_KEY,
    MESSAGES,
    OPENAI,
    RESPONSES,
    STREAM_CELLS,
    Booted,
    Cell,
    Pair,
    Rig,
    comparable_error,
    directive,
    litellm_header_names,
    marker,
    parity_rig,
    provider_body,
    sse_events,
    sse_lines_shape,
)

pytestmark: Final = pytest.mark.timeout(900)

CELLS: Final = pytest.mark.parametrize("cell", list(STREAM_CELLS.values()), ids=list(STREAM_CELLS))
CHAT_CELLS: Final = pytest.mark.parametrize(
    "cell",
    [cell for cell in STREAM_CELLS.values() if cell.path == CHAT],
    ids=[name for name, cell in STREAM_CELLS.items() if cell.path == CHAT],
)
MESSAGES_CELLS: Final = pytest.mark.parametrize(
    "cell",
    [cell for cell in STREAM_CELLS.values() if cell.path == MESSAGES],
    ids=[name for name, cell in STREAM_CELLS.items() if cell.path == MESSAGES],
)
ABORT_CELLS: Final = pytest.mark.parametrize(
    "cell",
    [Cell(CHAT, OPENAI), Cell(MESSAGES, ANTHROPIC), Cell(RESPONSES, OPENAI)],
    ids=["chat-openai", "messages-anthropic", "responses-openai"],
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with parity_rig(tmp_path_factory.mktemp("dec1-rustparity-streaming")) as booted:
        yield booted


@CELLS
def test_a_stream_answers_200_with_the_same_event_sequence_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The master key posts the cell's request with `stream: true` to both proxies. Both must answer 200, and the
    list of SSE events (event name plus JSON data, per-call ids removed, marker normalised) must be identical in
    order and content: no event merged, split, dropped or reordered by the hop."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    assert (seen.python.status, seen.rust.status) == (200, 200), (seen.python.body[:300], seen.rust.body[:300])
    assert sse_events(seen.rust) == sse_events(seen.python)


@CELLS
def test_a_stream_is_labelled_text_event_stream_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same streamed call on both proxies. The `content-type` must start with `text/event-stream` in both and
    be byte-identical between them (including any `charset` parameter)."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    assert seen.python.headers.get("content-type", "").startswith("text/event-stream"), seen.python.headers
    assert seen.rust.headers.get("content-type") == seen.python.headers.get("content-type")


@CELLS
def test_a_stream_is_chunked_without_a_content_length_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same streamed call on both proxies. Both answers must use `transfer-encoding: chunked` and carry no
    `content-length`: Python's chunked framing is a hop-by-hop header Rust strips, and hyper must re-chunk the
    body because its size is unknown, not buffer it to compute a length."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    for observed in (seen.python, seen.rust):
        assert observed.headers.get("transfer-encoding") == "chunked", observed.headers
        assert "content-length" not in observed.headers, observed.headers


@CELLS
def test_the_sse_line_structure_is_identical_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same streamed call on both proxies. The sequence of line kinds (`event`, `data`, blank separator) must
    be identical, so an SSE parser that splits on blank lines sees the same blocks: a hop that rewrote `\\n\\n`
    or coalesced lines would change it even when the events still parse."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    assert sse_lines_shape(seen.rust) == sse_lines_shape(seen.python)


@CHAT_CELLS
def test_a_chat_stream_ends_with_the_done_sentinel_in_both_modes(rig: Rig, cell: Cell) -> None:
    """A streamed chat call on both proxies. The last SSE event must be `data: [DONE]` in both: OpenAI SDKs stop
    reading on it, and a hop that ended the body one frame early would leave them hanging."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    assert (sse_events(seen.python)[-1], sse_events(seen.rust)[-1]) == (("", "[DONE]"), ("", "[DONE]"))


@MESSAGES_CELLS
def test_a_messages_stream_ends_with_message_stop_and_no_done_in_both_modes(rig: Rig, cell: Cell) -> None:
    """A streamed `/v1/messages` call on both proxies. The last event must be named `message_stop` and no
    `[DONE]` sentinel may appear in either: the Anthropic wire format has none."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    for observed in (seen.python, seen.rust):
        events = sse_events(observed)  # rebind-ok: one per mode
        assert events[-1][0] == "message_stop", events
        assert all(data != "[DONE]" for _, data in events), events


@CELLS
def test_a_stream_sets_the_same_x_litellm_header_names_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same streamed call on both proxies. The `x-litellm-*` response header names must be equal: on a stream
    they are sent with the head, before any event, and the hop must forward the head as soon as Python sends it."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    assert litellm_header_names(seen.rust) == litellm_header_names(seen.python)


@CELLS
def test_a_stream_sends_the_same_provider_request_in_both_modes(rig: Rig, cell: Cell) -> None:
    """The same streamed call on both proxies. The provider must be called once per proxy with equal JSON bodies
    once the marker is normalised (including `stream: true` and any `stream_options` litellm adds)."""
    seen: Final = rig.send(cell.path, cell.body(stream=True), stream=True)
    sent: Final = rig.provider_requests(seen)
    assert (len(sent.python), len(sent.rust)) == (1, 1), sent
    assert provider_body(sent.rust[0], seen.rust.case) == provider_body(sent.python[0], seen.python.case)


def _first_chunk_while_held(rig: Rig, proxy: Booted) -> bytes:
    rig.provider.release.clear()
    rig.provider.arrived.clear()
    body: Final = Cell(CHAT, OPENAI).body(stream=True)(f"{marker()} hold-at-provider")
    try:
        with proxy.gateway.client.stream(
            "POST", CHAT, json=dict(body), headers={"Authorization": f"Bearer {MASTER_KEY}"}, timeout=60
        ) as response:
            first: Final = next(response.iter_raw())
            assert rig.provider.arrived.is_set() and not rig.provider.release.is_set()
            return first
    finally:
        rig.provider.release.set()


def test_the_first_event_reaches_the_caller_while_the_provider_is_still_streaming_in_both_modes(rig: Rig) -> None:
    """The provider sends the first chat chunk and then holds the stream open until the test releases it. On both
    proxies the caller must receive a first body piece starting with `data: ` before the release: neither Python
    nor the forwarder may wait for the end of the stream (forward.rs relays frames as they arrive)."""
    pieces: Final = rig.both(lambda proxy: _first_chunk_while_held(rig, proxy))
    assert (pieces.python.startswith(b"data: "), pieces.rust.startswith(b"data: ")) == (True, True), pieces


@ABORT_CELLS
def test_a_provider_that_drops_mid_stream_ends_the_caller_stream_the_same_way_in_both_modes(
    rig: Rig, cell: Cell
) -> None:
    """The provider sends one stream chunk and then closes the connection without the terminating chunk. Both
    proxies must answer 200 (the head was already sent) and the caller must see the same events and the same ending
    in both modes: either both bodies break off (marked `BODY_FAILED` by the reader) or both end cleanly after the
    same error event. The forwarder must never turn Python's broken body into a clean end, or the other way round
    (gateway-forward/AGENTS.md: "An upstream that dies mid-response fails the client body")."""
    seen: Final = rig.send(cell.path, directive(cell.body(stream=True), "abort-mid-stream"), stream=True)
    assert (seen.python.status, seen.rust.status) == (200, 200)
    assert seen.rust.body.endswith(BODY_FAILED) == seen.python.body.endswith(BODY_FAILED)
    assert sse_events(seen.rust) == sse_events(seen.python)


def _hang_up_reaches_provider(rig: Rig, proxy: Booted) -> bool:
    before: Final = rig.provider.hung_up()
    body: Final = Cell(CHAT, OPENAI).body(stream=True)(f"{marker()} endless-stream")
    with proxy.gateway.client.stream(
        "POST", CHAT, json=dict(body), headers={"Authorization": f"Bearer {MASTER_KEY}"}, timeout=60
    ) as response:
        assert response.status_code == 200
        next(response.iter_raw())
    return eventually(rig.provider.hung_up, lambda count: count > before, seconds=30) > before


def test_a_client_that_hangs_up_mid_stream_makes_the_proxy_hang_up_on_the_provider_in_both_modes(rig: Rig) -> None:
    """The provider streams chat chunks every 100 ms for a minute. The caller reads the first piece and closes its
    connection. In both modes the provider must see its connection to the proxy drop within 30 seconds: through
    Rust, the client's disconnect has to close the forwarder's upstream connection so Python cancels the provider
    call (gateway-forward/AGENTS.md: "A client disconnect closes the upstream connection at every stage")."""
    assert rig.both(lambda proxy: _hang_up_reaches_provider(rig, proxy)) == Pair(True, True)


@pytest.mark.parametrize(
    "cell", [Cell(CHAT, OPENAI), Cell(MESSAGES, ANTHROPIC)], ids=["chat-openai", "messages-anthropic"]
)
def test_a_provider_error_before_the_first_event_is_the_same_in_both_modes(rig: Rig, cell: Cell) -> None:
    """A streamed call whose provider answers 500 before sending any event. Status, `content-type` and the JSON
    error body (OpenAI envelope for chat, Anthropic envelope for messages) must be identical in both modes: the
    error is decided by Python before the stream starts, so the hop only relays it."""
    seen: Final = rig.send(cell.path, directive(cell.body(stream=True), "fail=500"), stream=True)
    assert seen.rust.status == seen.python.status, (seen.python.body[:300], seen.rust.body[:300])
    assert seen.rust.headers.get("content-type") == seen.python.headers.get("content-type")
    assert comparable_error(seen.rust) == comparable_error(seen.python)
