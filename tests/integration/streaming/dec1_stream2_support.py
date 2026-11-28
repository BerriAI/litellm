"""Shared rig for the `test_dec1_stream2_*` streaming resilience contracts.

These contracts pin what a caller observes when a stream is stressed rather than when it is well formed:
either side hanging up mid stream, a provider sending broken or unusual SSE, keepalive pings, very large and
very small frames, multibyte text cut at transfer boundaries, timeouts, headers and HTTP framing, gzip from
the provider, and a caller that reads slowly. The streaming path is moving to Rust (byte level relay for
native `/v1/messages`, SSE coalescing), so everything here is black box: an owned TCP peer plays the
provider, a deployment that points at it is registered through `/model/new`, a key scoped to it is minted,
and the test talks to the shared proxy over HTTP.

The frame builders reuse `dec1_stream_support` where it already has one and add the provider dialects it
lacks (Anthropic native, Gemini, Bedrock Converse and Invoke, native OpenAI Responses). `ProgressPeer` is a
small raw socket provider that records how many bytes it managed to write, which is the only way to see
from outside whether the proxy keeps reading from the provider while its own caller has stopped reading.
"""

from __future__ import annotations

import base64
import gzip
import json
import socket
import socketserver
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import httpx
from integration._support.client import Gateway
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, wire_server
from integration.streaming.dec1_stream_support import (
    BEDROCK_CLAUDE,
    CLAUDE,
    DONE,
    EVENT_STREAM,
    GEMINI,
    PROVIDER_KEY,
    UPSTREAM_MODEL,
    Rig,
    _registered,
    bedrock_frame,
    data,
    delta,
    event,
    marker,
    usage_only,
)
from pydantic import JsonValue

SSE: Final = "text/event-stream"
KIB: Final = 1024
MIB: Final = 1024 * 1024
ANTHROPIC_USAGE_IN: Final = 10
ANTHROPIC_USAGE_OUT: Final = 5
INCOMPLETE_MESSAGES_STREAM: Final = (
    "Provider stream ended before emitting a message_stop event; the response is incomplete and any partial "
    "content (e.g. tool_use input JSON) may be truncated."
)
"""`INCOMPLETE_STREAM_ERROR_MESSAGE` in `litellm/rust/crates/inference-messages/src/relay.rs:8`, mirrored from
`litellm/llms/anthropic/messages/native/utils.py:33`."""
CLIENT_DISCONNECT_ERROR_INFORMATION: Final[Mapping[str, str]] = {
    "error_code": "499",
    "error_message": "Client disconnected the request",
    "error_class": "ClientDisconnected",
}
"""`_CLIENT_DISCONNECTED_ERROR_INFORMATION`, `litellm/proxy/common_request_processing.py:388`."""


def openai_frames(identity: str, pieces: Sequence[str]) -> tuple[bytes, ...]:
    """Role plus first piece, the rest, a `stop` finish chunk, a usage chunk and `[DONE]`."""
    first: Final = delta(identity, {"role": "assistant", "content": pieces[0]})
    rest: Final = tuple(delta(identity, {"content": piece}) for piece in pieces[1:])
    return (first, *rest, delta(identity, {}, finish="stop"), usage_only(identity), DONE)


def anthropic_event_dicts(identity: str, pieces: Sequence[str], *, terminal: bool = True) -> list[dict[str, JsonValue]]:
    """The Anthropic Messages streaming grammar as dicts; `terminal=False` drops message_delta and message_stop."""
    opening: Final[list[dict[str, JsonValue]]] = [
        {
            "type": "message_start",
            "message": {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": CLAUDE,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": ANTHROPIC_USAGE_IN, "output_tokens": 1},
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    ]
    body: Final[list[dict[str, JsonValue]]] = [
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}} for piece in pieces
    ]
    closing: Final[list[dict[str, JsonValue]]] = [
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": ANTHROPIC_USAGE_OUT},
        },
        {"type": "message_stop"},
    ]
    return [*opening, *body, *(closing if terminal else [])]


def anthropic_frames(identity: str, pieces: Sequence[str], *, terminal: bool = True) -> tuple[bytes, ...]:
    return tuple(event(str(item["type"]), item) for item in anthropic_event_dicts(identity, pieces, terminal=terminal))


def invoke_frames(identity: str, pieces: Sequence[str], *, terminal: bool = True) -> tuple[bytes, ...]:
    """Bedrock InvokeModelWithResponseStream: each Anthropic event base64 wrapped in an event stream `chunk`."""
    return tuple(
        aws_event_stream_frame(
            {":event-type": "chunk", ":content-type": "application/json", ":message-type": "event"},
            json.dumps({"bytes": base64.b64encode(json.dumps(item).encode()).decode()}).encode(),
        )
        for item in anthropic_event_dicts(identity, pieces, terminal=terminal)
    )


GEMINI_USAGE: Final[Mapping[str, JsonValue]] = {"promptTokenCount": 9, "candidatesTokenCount": 7, "totalTokenCount": 16}


def gemini_frames(_identity: str, pieces: Sequence[str]) -> tuple[bytes, ...]:
    """One candidate per piece; the last carries `finishReason: STOP` and `usageMetadata`."""

    def frame(position: int, piece: str) -> bytes:
        last: Final = position == len(pieces) - 1
        candidate: Final[dict[str, JsonValue]] = {
            "content": {"parts": [{"text": piece}], "role": "model"},
            "index": 0,
            **({"finishReason": "STOP"} if last else {}),
        }
        return data({"candidates": [candidate], **({"usageMetadata": dict(GEMINI_USAGE)} if last else {})})

    return tuple(frame(position, piece) for position, piece in enumerate(pieces))


def converse_frames(_identity: str, pieces: Sequence[str]) -> tuple[bytes, ...]:
    """Bedrock ConverseStream binary frames: start, one delta per piece, stops and metadata usage."""
    return (
        bedrock_frame("messageStart", {"role": "assistant"}),
        *(bedrock_frame("contentBlockDelta", {"delta": {"text": piece}, "contentBlockIndex": 0}) for piece in pieces),
        bedrock_frame("contentBlockStop", {"contentBlockIndex": 0}),
        bedrock_frame("messageStop", {"stopReason": "end_turn"}),
        bedrock_frame(
            "metadata", {"usage": {"inputTokens": 9, "outputTokens": 5, "totalTokens": 14}, "metrics": {"latencyMs": 1}}
        ),
    )


def responses_frames(identity: str, pieces: Sequence[str], *, terminal: bool = True) -> tuple[bytes, ...]:
    """A native OpenAI Responses stream: created, item added, one text delta per piece, item done, completed."""
    text: Final = "".join(pieces)
    item_id: Final = f"msg_{identity}"
    snapshot: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "response",
        "created_at": 1700000000,
        "model": UPSTREAM_MODEL,
        "status": "in_progress",
        "output": [],
    }
    item: Final[dict[str, JsonValue]] = {
        "type": "message",
        "id": item_id,
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    events: list[dict[str, JsonValue]] = [
        {"type": "response.created", "response": snapshot},
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        *(
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": piece,
            }
            for piece in pieces
        ),
    ]
    if terminal:
        done_item: Final[dict[str, JsonValue]] = {
            **item,
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        events += [
            {"type": "response.output_item.done", "output_index": 0, "item": done_item},
            {
                "type": "response.completed",
                "response": {
                    **snapshot,
                    "status": "completed",
                    "output": [done_item],
                    "usage": {"input_tokens": 8, "output_tokens": 6, "total_tokens": 14},
                },
            },
        ]
    return tuple(event(str(body["type"]), {**body, "sequence_number": number}) for number, body in enumerate(events))


@dataclass(frozen=True, slots=True)
class Provider:
    """One provider dialect: how to register it and how it streams `pieces` of text."""

    name: str
    model: str
    base_suffix: str
    content_type: str
    frames: Callable[[str, Sequence[str]], tuple[bytes, ...]]
    params: Mapping[str, JsonValue] = field(default_factory=dict)


OPENAI: Final = Provider("openai", f"openai/{UPSTREAM_MODEL}", "/v1", SSE, openai_frames)
AZURE: Final = Provider("azure", f"azure/{UPSTREAM_MODEL}", "", SSE, openai_frames, {"api_version": "2024-10-21"})
ANTHROPIC: Final = Provider("anthropic", f"anthropic/{CLAUDE}", "", SSE, anthropic_frames)
GEMINI_AI_STUDIO: Final = Provider("gemini", f"gemini/{GEMINI}", "", SSE, gemini_frames)
BEDROCK_CONVERSE: Final = Provider(
    "bedrock", f"bedrock/converse/{BEDROCK_CLAUDE}", "", EVENT_STREAM, converse_frames, {"aws_region_name": "us-east-1"}
)
BEDROCK_INVOKE: Final = Provider(
    "bedrock_invoke",
    f"bedrock/invoke/{BEDROCK_CLAUDE}",
    "",
    EVENT_STREAM,
    invoke_frames,
    {"aws_region_name": "us-east-1"},
)
AZURE_FOUNDRY_CLAUDE: Final = Provider("azure_ai", "azure_ai/claude-sonnet-4-6", "", SSE, anthropic_frames)
OPENAI_RESPONSES: Final = Provider("openai_responses", f"openai/{UPSTREAM_MODEL}", "/v1", SSE, responses_frames)

CHAT_PROVIDERS: Final = (OPENAI, AZURE, ANTHROPIC, GEMINI_AI_STUDIO, BEDROCK_CONVERSE)
"""The big providers as `/v1/chat/completions` sees them. Vertex Gemini shares the Gemini SSE dialect and
parser with Google AI Studio; it is reached here through `gemini/` because Vertex needs service account
credentials the harness does not mint (see `test_dec1_vertex_*` for Vertex specific wiring)."""
NATIVE_MESSAGES_PROVIDERS: Final = (ANTHROPIC, AZURE_FOUNDRY_CLAUDE, BEDROCK_INVOKE)
"""Providers whose `/v1/messages` stream is served on Rust only since #51119, #51721 and #51330."""


def provider_ids(provider: Provider) -> str:
    return provider.name


@contextmanager
def deployed(
    gateway: Gateway,
    provider: Provider,
    respond: Callable[[Request], Reply],
    *,
    key_fields: Mapping[str, JsonValue] | None = None,
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Register `provider` against an owned peer that answers with `respond`, and a key scoped to it.

    Bedrock deployments point both `aws_bedrock_runtime_endpoint` and `api_base` at the peer: in CI run
    44176682961 the native `/v1/messages` Bedrock invoke path ignored `aws_bedrock_runtime_endpoint` and called
    the suite's shared upstream ("Unknown scenario"), while `api_base` reaches the peer on every Bedrock path.
    Every other provider uses `api_base`. Extra keywords become `litellm_params` (prices, `keepalive_seconds`,
    `stream_timeout`)."""
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        endpoint: Final[dict[str, JsonValue]] = (
            {"aws_bedrock_runtime_endpoint": wire.url, "api_base": wire.url, "api_key": "synthetic-bedrock-token"}
            if provider.content_type == EVENT_STREAM
            else {"api_base": wire.url + provider.base_suffix, "api_key": PROVIDER_KEY}
        )
        name: Final = scenario.model(model=provider.model, **endpoint, **dict(provider.params), **litellm_params)
        _registered(gateway, name)
        yield Rig(gateway, wire, name, scenario.key(models=[name], **(key_fields or {})))


def streaming(frames: Sequence[bytes], content_type: str = SSE, **reply: object) -> Callable[[Request], Reply]:
    """Answer every POST with `frames` as a chunked body of `content_type`; Reply keywords pass through."""

    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        return Reply(content_type=content_type, chunks=tuple(frames), **reply)  # type: ignore[arg-type]

    return respond


def provider_stream(provider: Provider, pieces: Sequence[str], **reply: object) -> Callable[[Request], Reply]:
    return streaming(provider.frames(f"dec1s2-{marker()}", pieces), provider.content_type, **reply)


ROUTE_BODIES: Final[Mapping[str, Callable[[str, str], dict[str, JsonValue]]]] = {
    "/v1/chat/completions": lambda model, text: {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "stream": True,
    },
    "/v1/messages": lambda model, text: {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": text}],
        "stream": True,
    },
    "/v1/responses": lambda model, text: {"model": model, "input": text, "stream": True},
}


def route_body(route: str, model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {**ROUTE_BODIES[route](model, f"dec1 stream2 probe {marker()}"), **extra}


def auth(rig: Rig, **extra: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {rig.key}", "anthropic-version": "2023-06-01", **extra}


@contextmanager
def opened(
    rig: Rig, route: str, *, headers: Mapping[str, str] | None = None, **extra: JsonValue
) -> Iterator[httpx.Response]:
    """An open streaming response on `route` that the test reads (or abandons) itself."""
    with rig.gateway.client.stream(
        "POST",
        route,
        json=route_body(route, rig.model, **extra),
        headers={**auth(rig), **(headers or {})},
        timeout=60,
    ) as response:
        yield response


def first_frame(response: httpx.Response) -> str:
    """Read until the first `data:` line and return it; the caller then closes to simulate a hang up."""
    return next(line for line in response.iter_lines() if line.startswith("data:"))


def gzipped(frames: Sequence[bytes], pieces: int = 4) -> tuple[bytes, ...]:
    """The whole body gzip compressed, cut into `pieces` transfer chunks."""
    packed: Final = gzip.compress(b"".join(frames))
    size: Final = max(1, len(packed) // pieces + 1)
    return tuple(packed[start : start + size] for start in range(0, len(packed), size))


def cut_every(frames: Sequence[bytes], size: int) -> tuple[bytes, ...]:
    """The concatenated frames re-cut into transfer chunks of `size` bytes, ignoring frame boundaries."""
    joined: Final = b"".join(frames)
    return tuple(joined[start : start + size] for start in range(0, len(joined), size))


class ProgressPeer:
    """A raw socket SSE provider that streams `frames` and records how far it got.

    `written` is the number of body bytes the kernel accepted. With a small send buffer, a proxy that stops
    reading from the provider while its caller is not reading makes `written` stall well short of the
    total; a proxy that drains the provider into memory lets it reach the total."""

    def __init__(self, frames: Sequence[bytes], *, send_buffer: int = 16 * KIB) -> None:
        self.frames: Final = tuple(frames)
        self.total: Final = sum(len(frame) for frame in frames)
        self.send_buffer: Final = send_buffer
        self.written = 0
        self.finished = threading.Event()
        self.broken = threading.Event()
        self.requests = 0
        self._lock: Final = threading.Lock()

    def serve(self, connection: socket.socket) -> None:
        connection.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.send_buffer)
        reader: Final = connection.makefile("rb")
        length = 0
        while (line := reader.readline()) not in (b"\r\n", b"\n", b""):
            name, _, value = line.decode("latin-1").partition(":")
            if name.strip().lower() == "content-length":
                length = int(value.strip())
        reader.read(length)
        with self._lock:
            self.requests += 1
        head: Final = (
            b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n"
            b"connection: close\r\n\r\n"
        )
        try:
            connection.sendall(head)
            for frame in self.frames:
                connection.sendall(b"%x\r\n%s\r\n" % (len(frame), frame))
                with self._lock:
                    self.written += len(frame)
            connection.sendall(b"0\r\n\r\n")
            self.finished.set()
        except OSError:
            self.broken.set()

    def progress(self) -> int:
        with self._lock:
            return self.written


@contextmanager
def progress_peer(frames: Sequence[bytes], *, send_buffer: int = 16 * KIB) -> Iterator[tuple[ProgressPeer, str]]:
    peer: Final = ProgressPeer(frames, send_buffer=send_buffer)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self) -> None:
            peer.serve(self.request)

    class Server(socketserver.ThreadingTCPServer):
        daemon_threads = True
        allow_reuse_address = True

    with Server(("127.0.0.1", 0), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        try:
            yield peer, f"http://127.0.0.1:{server.server_address[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=6)


@contextmanager
def progress_deployed(
    gateway: Gateway, provider: Provider, frames: Sequence[bytes], **litellm_params: JsonValue
) -> Iterator[tuple[Rig, ProgressPeer]]:
    """`deployed`, but the provider is a `ProgressPeer`. `Rig.wire` is unused and left as a dummy wire."""
    with (
        progress_peer(frames) as (peer, url),
        wire_server(lambda request: Reply(status=404)) as unused,
        gateway.scenario() as scenario,
    ):
        name: Final = scenario.model(
            model=provider.model,
            api_base=url + provider.base_suffix,
            api_key=PROVIDER_KEY,
            **dict(provider.params),
            **litellm_params,
        )
        _registered(gateway, name)
        yield Rig(gateway, unused, name, scenario.key(models=[name])), peer


def big_text_frames(provider: Provider, piece_size: int, count: int) -> tuple[bytes, ...]:
    """`count` deltas of `piece_size` ASCII characters each, numbered so order and loss are visible."""
    pieces: Final = tuple(f"{index:08d}" + "x" * (piece_size - 8) for index in range(count))
    return provider.frames(f"dec1s2-{marker()}", pieces)


__all__ = [
    "ANTHROPIC",
    "AZURE",
    "AZURE_FOUNDRY_CLAUDE",
    "BEDROCK_CONVERSE",
    "BEDROCK_INVOKE",
    "CHAT_PROVIDERS",
    "CLIENT_DISCONNECT_ERROR_INFORMATION",
    "GEMINI_AI_STUDIO",
    "INCOMPLETE_MESSAGES_STREAM",
    "KIB",
    "MIB",
    "NATIVE_MESSAGES_PROVIDERS",
    "OPENAI",
    "OPENAI_RESPONSES",
    "SSE",
    "ProgressPeer",
    "Provider",
    "anthropic_event_dicts",
    "anthropic_frames",
    "auth",
    "big_text_frames",
    "converse_frames",
    "cut_every",
    "deployed",
    "first_frame",
    "gemini_frames",
    "gzipped",
    "invoke_frames",
    "openai_frames",
    "opened",
    "progress_deployed",
    "provider_ids",
    "provider_stream",
    "responses_frames",
    "route_body",
    "streaming",
]
