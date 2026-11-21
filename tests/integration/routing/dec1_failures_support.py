"""Misbehaving upstream peers for the `test_dec1_failures_upstream_*` files.

`raw_peer` is an HTTP/1.1 peer that hands the test full control over the bytes it writes back, which the
harness `wire_server` deliberately does not: a status line with no body, a reset in the middle of the body, a
`100 Continue`, a body shorter than its `content-length`, headers that arrive late, a body that trickles.
Every script is an async function of the parsed request and the connection. The peer records each request
and whether the client hung up before the script finished.

`unaccepting_listener` is a TCP port whose accept queue is full, so a new connection attempt gets no
SYN-ACK and the client's connect timeout is what ends the call (Linux drops the SYN of a full queue).

`upstream_error` builds the OpenAI-shaped error body every status test uses. Its `type` is deliberately not
`invalid_request_error`, which litellm's OpenAI error mapper treats specially.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

from pydantic import JsonValue

from tests.integration._support.async_server import LoopbackServer
from tests.integration._support.ports import allocate_port

UPSTREAM_ERROR_TYPE: Final = "dec1_upstream_error"
UPSTREAM_ERROR_MESSAGE: Final = "dec1 scripted upstream failure"
REPLY_TEXT: Final = "dec1 raw peer reply"
_POLL_SECONDS: Final = 0.02


@dataclass(frozen=True, slots=True)
class Exchange:
    method: str
    target: str
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> dict[str, JsonValue]:
        loaded: Final = json.loads(self.body or b"{}")
        assert isinstance(loaded, dict), loaded
        return loaded


@dataclass(frozen=True, slots=True)
class Connection:
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter
    hung_up: threading.Event

    async def send(self, data: bytes) -> None:
        self.writer.write(data)
        await self.writer.drain()

    async def wait(self, gate: threading.Event) -> bool:
        """Wait for `gate`; returns False if the client closed the connection first."""
        while not gate.is_set():
            if self.reader.at_eof():
                self.hung_up.set()
                return False
            try:
                leftover = await asyncio.wait_for(self.reader.read(1), timeout=_POLL_SECONDS)
            except TimeoutError:
                continue
            if not leftover:
                self.hung_up.set()
                return False
        return True

    def reset(self) -> None:
        """Close with a TCP reset instead of a clean FIN."""
        transport: Final = self.writer.transport
        if transport is not None:
            transport.abort()


Script = Callable[[Exchange, Connection], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RawPeer:
    url: str
    port: int
    hung_up: threading.Event
    seen: list[Exchange] = field(default_factory=list)  # mutable-ok: appended on the peer's loop thread

    @property
    def api_base(self) -> str:
        return f"{self.url}/v1"

    def posts(self) -> tuple[Exchange, ...]:
        return tuple(exchange for exchange in tuple(self.seen) if exchange.method == "POST")


async def _read_exchange(reader: asyncio.StreamReader) -> Exchange | None:
    try:
        head: Final = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
        return None
    lines: Final = head.decode("latin-1").split("\r\n")
    method, target, _version = lines[0].split(" ", 2)
    headers: Final = {
        name.strip().lower(): value.strip() for name, _, value in (line.partition(":") for line in lines[1:] if line)
    }
    try:
        body: Final = await reader.readexactly(int(headers.get("content-length", "0")))
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    return Exchange(method, target, headers, body)


@contextmanager
def raw_peer(script: Script) -> Iterator[RawPeer]:
    """Serve `script` for every POST; a GET (model discovery) gets an empty model list."""
    port: Final = allocate_port()
    hung_up: Final = threading.Event()
    seen: Final[list[Exchange]] = []

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        exchange: Final = await _read_exchange(reader)
        if exchange is None:
            return
        seen.append(exchange)
        connection: Final = Connection(reader, writer, hung_up)
        try:
            if exchange.method != "POST":
                await connection.send(http_response(200, b'{"object":"list","data":[]}'))
                return
            await script(exchange, connection)
        except (ConnectionError, OSError):
            hung_up.set()

    server: Final = LoopbackServer(port, serve)
    try:
        server.start()
        yield RawPeer(f"http://127.0.0.1:{port}", port, hung_up, seen)
    finally:
        server.stop()


def http_response(
    status: int,
    body: bytes,
    *,
    content_type: str | None = "application/json",
    headers: Mapping[str, str] | None = None,
    declared_length: int | None = None,
    reason: str = "Scripted",
) -> bytes:
    """One complete HTTP/1.1 response. `declared_length` overrides the real `content-length`."""
    lines: Final = [
        f"HTTP/1.1 {status} {reason}",
        *((f"content-type: {content_type}",) if content_type is not None else ()),
        f"content-length: {len(body) if declared_length is None else declared_length}",
        "connection: close",
        *(f"{name}: {value}" for name, value in (headers or {}).items()),
    ]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body


def chat_body(text: str = REPLY_TEXT, model: str = "gpt-4o-mini") -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
    ).encode()


def embedding_body(model: str = "text-embedding-3-small") -> bytes:
    return json.dumps(
        {
            "object": "list",
            "model": model,
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.5, 0.75]}],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
    ).encode()


def success_body(exchange: Exchange) -> bytes:
    """A valid 200 body for whichever OpenAI route the request was sent to."""
    return embedding_body() if exchange.target.endswith("/embeddings") else chat_body()


def upstream_error(status: int, message: str = UPSTREAM_ERROR_MESSAGE) -> bytes:
    return json.dumps(
        {"error": {"message": f"{message} {status}", "type": UPSTREAM_ERROR_TYPE, "param": None, "code": None}}
    ).encode()


def answering(response: Callable[[Exchange], bytes]) -> Script:
    """Write the bytes `response` builds for each request, then close cleanly."""

    async def script(exchange: Exchange, connection: Connection) -> None:
        await connection.send(response(exchange))

    return script


def ok() -> Script:
    return answering(lambda exchange: http_response(200, success_body(exchange)))


def status_reply(status: int, *, headers: Mapping[str, str] | None = None) -> Script:
    return answering(lambda _exchange: http_response(status, upstream_error(status), headers=headers))


def closing_without_a_reply() -> Script:
    async def script(_exchange: Exchange, _connection: Connection) -> None:
        return None

    return script


def resetting_after(partial: Callable[[Exchange], bytes]) -> Script:
    """Write `partial` (for example headers and half a body) and then reset the connection."""

    async def script(exchange: Exchange, connection: Connection) -> None:
        await connection.send(partial(exchange))
        connection.reset()

    return script


def silent_until(gate: threading.Event) -> Script:
    """Send nothing, not even a status line, until `gate` is set; then answer 200."""

    async def script(exchange: Exchange, connection: Connection) -> None:
        if await connection.wait(gate):
            await connection.send(http_response(200, success_body(exchange)))

    return script


def headers_then_body_after(gate: threading.Event) -> Script:
    """Send the status line and headers at once and the body only when `gate` is set."""

    async def script(exchange: Exchange, connection: Connection) -> None:
        body: Final = success_body(exchange)
        await connection.send(http_response(200, b"", declared_length=len(body)))
        if await connection.wait(gate):
            await connection.send(body)

    return script


def trickling_body(pieces: int, gap_seconds: float) -> Script:
    """Send headers at once and the body in `pieces` parts with `gap_seconds` between them."""

    async def script(exchange: Exchange, connection: Connection) -> None:
        body: Final = success_body(exchange)
        await connection.send(http_response(200, b"", declared_length=len(body)))
        size: Final = max(1, len(body) // pieces)
        for start in range(0, len(body), size):
            await asyncio.sleep(gap_seconds)
            await connection.send(body[start : start + size])

    return script


def delayed(seconds: float, then: Script | None = None) -> Script:
    """Wait `seconds` before running `then` (a plain 200 by default)."""

    async def script(exchange: Exchange, connection: Connection) -> None:
        await asyncio.sleep(seconds)
        await (then or ok())(exchange, connection)

    return script


def in_turn(*scripts: Script) -> Script:
    """Answer the first POST with the first script, the second with the second, and the last one from then on."""
    served: Final[list[int]] = []  # mutable-ok: one entry per POST, appended on the peer's loop thread

    async def script(exchange: Exchange, connection: Connection) -> None:
        served.append(1)
        await scripts[min(len(served), len(scripts)) - 1](exchange, connection)

    return script


@contextmanager
def unaccepting_listener() -> Iterator[str]:
    """The base URL of a port that completes no new TCP handshake, so only a connect timeout ends a call."""
    port: Final = allocate_port()
    listener: Final = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    fillers: Final[list[socket.socket]] = []
    try:
        listener.bind(("127.0.0.1", port))
        listener.listen(0)
        for _ in range(16):
            filler = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            filler.setblocking(False)
            filler.connect_ex(("127.0.0.1", port))
            fillers.append(filler)
        yield f"http://127.0.0.1:{port}"
    finally:
        for filler in fillers:
            filler.close()
        listener.close()


@contextmanager
def plain_http_port() -> Iterator[int]:
    """A port that answers the way a plain HTTP/1.1 server answers bytes it cannot parse: whatever arrives
    first gets `400 Bad Request` and the connection is closed. A TLS client therefore receives a reply that
    is not TLS at once, instead of waiting for a server that is still looking for the end of a request head,
    which is what `raw_peer` would do with a ClientHello."""
    port: Final = allocate_port()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await reader.read(4096)
            writer.write(http_response(400, b"", content_type=None, reason="Bad Request"))
            await writer.drain()
        except (ConnectionError, OSError):
            return

    server: Final = LoopbackServer(port, serve)
    try:
        server.start()
        yield port
    finally:
        server.stop()


def unbound_api_base() -> str:
    """An API base on a loopback port nothing listens on: every connection attempt is refused at once."""
    return f"http://127.0.0.1:{allocate_port()}/v1"
