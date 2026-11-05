from __future__ import annotations

import io
import selectors
import socket
import ssl
import threading
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import SimpleQueue
from types import MappingProxyType
from typing import TYPE_CHECKING, BinaryIO, Final

if TYPE_CHECKING:
    from collections.abc import Buffer

_POLL_SECONDS: Final = 0.05
_WRITE_PIECE_BYTES: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    target: str
    headers: Mapping[str, str]
    body: bytes
    header_lines: tuple[tuple[str, str], ...] = ()


class AbortedBody(Exception):
    """The client closed the connection before the body it announced was complete."""


def exactly(stream: BinaryIO, size: int) -> bytes:
    data: Final = stream.read(size)
    if len(data) < size:
        raise AbortedBody
    return data


def chunked_body(stream: BinaryIO) -> Iterator[bytes]:
    while True:
        size_line: Final = stream.readline()
        if not size_line:
            raise AbortedBody
        size: Final = int(size_line.split(b";")[0].strip(), 16)
        if size == 0:
            while stream.readline().strip():
                pass
            return
        yield exactly(stream, size)
        stream.readline()


def read_body(headers: Mapping[str, str], stream: BinaryIO) -> bytes:
    if headers.get("transfer-encoding", "").lower() == "chunked":
        return b"".join(chunked_body(stream))
    return exactly(stream, int(headers.get("content-length", "0")))


@dataclass(frozen=True, slots=True)
class Reply:
    status: int = 200
    body: bytes = b"{}"
    content_type: str = "application/json"
    chunks: tuple[bytes, ...] | None = None
    abort_after: int | None = None
    gate_before_first: threading.Event | None = None
    gate_after_first: threading.Event | None = None
    pause_between_chunks: float = 0
    heartbeat: bytes | None = None
    headers: Mapping[str, str] = MappingProxyType({})
    drop_connection: bool = False


@dataclass(frozen=True, slots=True)
class Wire:
    url: str
    received: SimpleQueue[Request]
    disconnected: SimpleQueue[str]
    connected: SimpleQueue[str]
    closed: SimpleQueue[str]

    def drain(self) -> tuple[Request, ...]:
        return tuple(self.received.get_nowait() for _ in range(self.received.qsize()))

    def connections(self) -> int:
        return self.connected.qsize()

    def open_connections(self) -> int:
        return self.connected.qsize() - self.closed.qsize()


def _await_release(
    gate: threading.Event, closing: threading.Event, beat: Callable[[], None] | None = None, every: float = 0.05
) -> bool:
    while not closing.is_set():
        if gate.wait(timeout=every):
            return True
        if beat is not None:
            beat()
    return gate.is_set()


class _RequestReader(io.BufferedReader):
    cut = False

    def readline(self, size: int | None = -1, /) -> bytes:
        line: Final = super().readline(size)
        self.cut = self.cut or not line.endswith(b"\n")
        return line


@dataclass(frozen=True, slots=True)
class _Blocked:
    since: float
    writing: bool


class _Connection(io.BufferedIOBase):
    blocked: _Blocked | None = None

    def __init__(self, connection: socket.socket, closing: threading.Event) -> None:
        super().__init__()
        self.connection = connection
        self.closing = closing
        self.selector = selectors.DefaultSelector()
        self.selector.register(connection, selectors.EVENT_READ)

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def readinto(self, buffer: Buffer, /) -> int:
        if not self._wait_for_bytes():
            return 0
        with self._blocking(writing=False):
            return self.connection.recv_into(buffer)

    def write(self, buffer: Buffer, /) -> int:
        view: Final = memoryview(buffer).cast("B")
        for start in range(0, view.nbytes, _WRITE_PIECE_BYTES):
            with self._blocking(writing=True):
                self.connection.sendall(view[start : start + _WRITE_PIECE_BYTES])
        return view.nbytes

    def close(self) -> None:
        self.selector.close()
        super().close()

    def stalled(self, grace: float, teardown_started: float) -> _Blocked | None:
        blocked: Final = self.blocked
        if blocked is None or time.monotonic() - max(blocked.since, teardown_started) < grace:
            return None
        return blocked

    def shut(self) -> None:
        with suppress(OSError):
            self.connection.shutdown(socket.SHUT_RDWR)

    def _wait_for_bytes(self) -> bool:
        while not self.closing.is_set():
            if self._has_bytes(_POLL_SECONDS):
                return True
        return self._has_bytes(0)

    def _has_bytes(self, timeout: float) -> bool:
        decrypted: Final = isinstance(self.connection, ssl.SSLSocket) and self.connection.pending() > 0
        return decrypted or bool(self.selector.select(timeout))

    @contextmanager
    def _blocking(self, writing: bool) -> Iterator[None]:
        self.blocked = _Blocked(time.monotonic(), writing)
        try:
            yield
        finally:
            self.blocked = None


def _close_unless_stalled(
    close: Callable[[], None], connections: SimpleQueue[_Connection], grace: float, errors: SimpleQueue[Exception]
) -> None:
    teardown_started: Final = time.monotonic()
    closer: Final = threading.Thread(target=close)
    closer.start()
    while closer.is_alive():
        for connection in tuple(connections.get_nowait() for _ in range(connections.qsize())):
            stalled: Final = connection.stalled(grace, teardown_started)
            if stalled is None:
                connections.put(connection)
                continue
            if stalled.writing:
                errors.put(TimeoutError(f"a reply write made no progress for {grace} s"))
            connection.shut()
        closer.join(timeout=_POLL_SECONDS)


@contextmanager
def wire_server(
    respond: Callable[[Request], Reply],
    tls: ssl.SSLContext | None = None,
    port: int = 0,
    keep_alive: bool = False,
    teardown_grace: float = 10,
) -> Generator[Wire, None, None]:
    """Owned TCP peer; requests traverse the real HTTP client and serialization. With `keep_alive` the
    peer honours HTTP/1.1 persistent connections so `connections()` counts the client's TCP sessions."""
    received: Final[SimpleQueue[Request]] = SimpleQueue()
    errors: Final[SimpleQueue[Exception]] = SimpleQueue()
    disconnected: Final[SimpleQueue[str]] = SimpleQueue()
    connected: Final[SimpleQueue[str]] = SimpleQueue()
    closed: Final[SimpleQueue[str]] = SimpleQueue()
    streams: Final[SimpleQueue[_Connection]] = SimpleQueue()
    closing: Final = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        request: socket.socket
        reader: _RequestReader

        def setup(self) -> None:
            stream: Final = _Connection(self.request, closing)
            self.reader = _RequestReader(stream)
            self.rfile = self.reader
            self.wfile = stream
            streams.put(stream)
            connected.put(f"{self.client_address[0]}:{self.client_address[1]}")

        def finish(self) -> None:
            super().finish()
            closed.put(f"{self.client_address[0]}:{self.client_address[1]}")

        def write_chunk(self, chunk: bytes) -> None:
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()

        def respond(self) -> None:
            if self.reader.cut:
                self.close_connection = True
                return
            headers: Final = {name.lower(): value for name, value in self.headers.items()}
            try:
                body: Final = read_body(headers, self.rfile)
            except AbortedBody:
                self.close_connection = True
                disconnected.put(self.path)
                return
            request: Final = Request(self.command, self.path, headers, body, tuple(self.headers.items()))
            received.put(request)
            try:
                reply = respond(request)
            except Exception as error:
                errors.put(error)
                reply = Reply(status=500)
            if reply.drop_connection:
                self.close_connection = True
                disconnected.put(request.target)
                return
            self.send_response(reply.status)
            self.send_header("content-type", reply.content_type)
            for name, value in reply.headers.items():
                self.send_header(name, value)
            if reply.chunks is None:
                self.send_header("content-length", str(len(reply.body)))
            else:
                self.send_header("transfer-encoding", "chunked")
            if not keep_alive:
                self.send_header("connection", "close")
            self.end_headers()
            try:
                if self.command == "HEAD":
                    self.wfile.flush()
                elif reply.chunks is None:
                    self.wfile.write(reply.body)
                else:
                    for index, chunk in enumerate(reply.chunks):
                        if reply.abort_after == index:
                            break
                        if (
                            index == 0
                            and reply.gate_before_first is not None
                            and not _await_release(reply.gate_before_first, closing)
                        ):
                            break
                        self.write_chunk(chunk)
                        if (
                            index == 0
                            and reply.gate_after_first is not None
                            and not (
                                _await_release(reply.gate_after_first, closing)
                                if reply.heartbeat is None
                                else _await_release(
                                    reply.gate_after_first,
                                    closing,
                                    partial(self.write_chunk, reply.heartbeat),
                                    reply.pause_between_chunks,
                                )
                            )
                        ):
                            break
                        if reply.pause_between_chunks and index + 1 < len(reply.chunks):
                            time.sleep(reply.pause_between_chunks)
                    else:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                disconnected.put(request.target)
            except Exception as error:
                errors.put(error)
            self.close_connection = not keep_alive

        do_POST = respond
        do_PUT = respond
        do_GET = respond
        do_DELETE = respond
        do_PATCH = respond
        do_HEAD = respond

        def log_message(self, format: str, *args: object) -> None:
            pass

    class OwnedHTTPServer(ThreadingHTTPServer):
        daemon_threads = False
        request_queue_size = 128

        def server_bind(self) -> None:
            super().server_bind()
            if tls is not None:
                self.socket = tls.wrap_socket(self.socket, server_side=True)

    with OwnedHTTPServer(("127.0.0.1", port), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": _POLL_SECONDS})
        thread.start()
        try:
            yield Wire(
                f"{'https' if tls is not None else 'http'}://127.0.0.1:{server.server_port}",
                received,
                disconnected,
                connected,
                closed,
            )
        finally:
            closing.set()
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "Owned HTTP server survived cleanup"
            _close_unless_stalled(server.server_close, streams, teardown_grace, errors)
            failure: Final = None if errors.empty() else errors.get_nowait()
            assert failure is None, f"Owned HTTP peer failed: {failure!r}"
