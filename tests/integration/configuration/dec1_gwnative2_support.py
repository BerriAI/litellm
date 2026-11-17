"""Round 2 additions to the `dec1_gwnative_support` rig for the native Rust gateway preview.

Everything that boots the binary, scripts the provider and parses replies lives in the round 1 helper
`dec1_gwnative_support.py`, which this module imports and never changes. Added here: Bedrock invoke
stream frames (AWS event-stream messages carrying base64 Anthropic events, the shape
`llms/src/bedrock/messages/invoke_transformations/stream.rs` decodes), a scripted provider that serves
HTTPS with a self-signed certificate (for the upstream TLS contracts), raw SSE block parsing that keeps the
exact bytes, and helpers for streaming calls that the caller abandons part way.
"""

from __future__ import annotations

import base64
import json
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
from pydantic import JsonValue

from tests.integration._support.tls import server_context, write_self_signed_cert
from tests.integration._support.upstream import aws_event_stream_frame
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.configuration.dec1_gwnative_support import (
    GATEWAY_KEY,
    NativeGateway,
    anthropic_message,
    anthropic_reply,
)

EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
# `INCOMPLETE_STREAM_ERROR_MESSAGE` in `inference-messages/src/relay.rs:8`, Python's
# `_incomplete_stream_error_sse_event` in `llms/anthropic/messages/native/streaming_iterator.py:260`.
INCOMPLETE_STREAM_MESSAGE: Final = (
    "Provider stream ended before emitting a message_stop event; the response is incomplete and any partial "
    "content (e.g. tool_use input JSON) may be truncated."
)
INCOMPLETE_STREAM_EVENT: Final = (
    'event: error\ndata: {"type": "error", "error": {"type": "api_error", "message": "'
    + INCOMPLETE_STREAM_MESSAGE
    + '"}}\n\n'
)
INVOCATION_METRICS_KEY: Final = "amazon-bedrock-invocationMetrics"


def python_sse(event: Mapping[str, JsonValue]) -> str:
    """One SSE block as Python's `json.dumps` writes it (`", "` and `": "` separators), which is how the Bedrock
    relay re-encodes every decoded chunk (`sse_frame` in `bedrock/messages/invoke_transformations/stream.rs`)."""
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def bedrock_chunk(event: Mapping[str, JsonValue]) -> bytes:
    """One `chunk` event-stream message whose payload is `{"bytes": base64(event JSON)}`."""
    encoded: Final = base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode()
    return aws_event_stream_frame(
        {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"},
        json.dumps({"bytes": encoded}).encode(),
    )


def bedrock_claude_events(
    text: str, *, input_tokens: int = 11, output_tokens: int = 4
) -> tuple[dict[str, JsonValue], ...]:
    """A whole Claude invoke stream; `message_stop` carries Bedrock's invocation metrics."""
    start: Final = {**anthropic_message("", model="claude-dec1-bedrock"), "content": [], "stop_reason": None}
    start["usage"] = {"input_tokens": input_tokens, "output_tokens": 1}
    return (
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": output_tokens},
        },
        {
            "type": "message_stop",
            INVOCATION_METRICS_KEY: {
                "inputTokenCount": input_tokens,
                "outputTokenCount": output_tokens,
                "invocationLatency": 120,
                "firstByteLatency": 40,
            },
        },
    )


def bedrock_stream(events: tuple[Mapping[str, JsonValue], ...], **reply: object) -> Reply:
    """The events as one chunked event-stream reply, one AWS message per HTTP chunk."""
    return Reply(content_type=EVENT_STREAM, chunks=tuple(bedrock_chunk(event) for event in events), **reply)  # type: ignore[arg-type]


def sse_blocks(text: str) -> tuple[str, ...]:
    """The raw SSE blocks of a body, each with its trailing blank line, in order and byte for byte."""
    return tuple(f"{block}\n\n" for block in text.split("\n\n") if block)


def disconnected_targets(wire: Wire) -> tuple[str, ...]:
    """Targets whose reply the scripted peer could not finish writing because the client went away."""
    return tuple(wire.disconnected.get_nowait() for _ in range(wire.disconnected.qsize()))


@dataclass
class TlsProvider:
    """An HTTPS peer with a self-signed certificate for 127.0.0.1 that answers every call as Anthropic."""

    wire: Wire
    certificate: Path
    seen: list[Request] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return self.wire.url

    def requests(self) -> tuple[Request, ...]:
        with self.lock:
            self.seen.extend(self.wire.drain())
            return tuple(self.seen)


@contextmanager
def tls_provider(directory: Path) -> Iterator[TlsProvider]:
    certificate, key = write_self_signed_cert(directory)
    with wire_server(lambda _: anthropic_reply("dec1 over tls"), tls=server_context(certificate, key)) as wire:
        yield TlsProvider(wire, certificate)


def open_stream(
    gateway: NativeGateway,
    path: str,
    body: Mapping[str, JsonValue],
    headers: Mapping[str, str] = MappingProxyType({}),
) -> tuple[AbstractContextManager[httpx.Client], httpx.Response]:
    """Start a streaming call and return a closer for its client and the open response; the caller enters both.

    The client is already open once `send` ran, and httpx refuses `with client` on an open client
    (`Cannot open a client instance more than once`), so the caller gets `closing(client)` instead."""
    client: Final = gateway.client(timeout=60)
    request: Final = client.build_request(
        "POST", path, json=dict(body), headers={"Authorization": f"Bearer {GATEWAY_KEY}", **headers}
    )
    return closing(client), client.send(request, stream=True)


def first_block(response: httpx.Response) -> str:
    """Read the response until one whole SSE block has arrived and return it."""
    seen = ""
    for piece in response.iter_text():
        seen += piece
        if "\n\n" in seen:
            return seen.split("\n\n", 1)[0] + "\n\n"
    return seen


def in_threads(calls: tuple[Callable[[], httpx.Response], ...]) -> tuple[httpx.Response, ...]:
    """Run each call on its own thread and return the responses in call order."""
    results: list[httpx.Response | None] = [None] * len(calls)

    def run(index: int) -> None:
        results[index] = calls[index]()

    threads: Final = tuple(threading.Thread(target=run, args=(index,)) for index in range(len(calls)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert all(result is not None for result in results), "a concurrent call did not finish"
    return tuple(result for result in results if result is not None)
