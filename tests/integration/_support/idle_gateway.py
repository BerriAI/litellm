from __future__ import annotations

import itertools
import re
import socket
import struct
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BufferedIOBase
from queue import SimpleQueue
from socketserver import BaseServer
from typing import Final

from integration._support.wire import Reply, Request

MARKER: Final = re.compile(rb"idle-[0-9a-f]{16}")
_LOOPBACK: Final = "127.0.0.1"


def _exactly(stream: BufferedIOBase, size: int) -> bytes | None:
    data: Final = stream.read(size)
    return data if len(data) == size else None


def _body(headers: Mapping[str, str], stream: BufferedIOBase) -> bytes | None:
    if headers.get("transfer-encoding", "").lower() == "chunked":
        return None
    return _exactly(stream, int(headers.get("content-length", "0")))


class Drop(Enum):
    CLOSE = "close"
    RESET = "reset"


@dataclass(frozen=True, slots=True)
class Arrival:
    marker: str
    target: str
    connection: int
    dropped: bool


@dataclass(frozen=True, slots=True)
class IdleGateway:
    """A keep-alive HTTP/1.1 peer that never closes an idle connection itself and, like a load balancer whose
    idle timer already fired, swallows a request that reuses a connection idle for `drop_after` seconds."""

    url: str
    port: int
    arrivals: SimpleQueue[Arrival]

    def pending(self) -> int:
        return self.arrivals.qsize()

    def drain(self) -> tuple[Arrival, ...]:
        return tuple(self.arrivals.get_nowait() for _ in range(self.arrivals.qsize()))


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    request_queue_size = 128

    def __init__(
        self,
        port: int,
        respond: Callable[[Request], Reply],
        drop_after: float | None,
        drop: Drop,
        arrivals: SimpleQueue[Arrival],
    ) -> None:
        super().__init__((_LOOPBACK, port), _Handler)
        self.respond = respond
        self.drop_after = drop_after
        self.drop = drop
        self.arrivals = arrivals
        self.connections = itertools.count(1)
        self.sockets: SimpleQueue[socket.socket] = SimpleQueue()
        self.failures: SimpleQueue[Exception] = SimpleQueue()

    def close_connections(self) -> None:
        for client in tuple(self.sockets.get_nowait() for _ in range(self.sockets.qsize())):
            with suppress(OSError):
                client.shutdown(socket.SHUT_RDWR)

    def handle_error(self, request: object, client_address: object) -> None:
        return


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(
        self, request: socket.socket | tuple[bytes, socket.socket], client_address: object, server: BaseServer
    ) -> None:
        assert isinstance(server, _Server)
        assert isinstance(request, socket.socket)
        self.gateway = server
        self.client = request
        super().__init__(request, client_address, server)

    def setup(self) -> None:
        super().setup()
        self.connection_id = next(self.gateway.connections)
        self.idle_since: float | None = None
        self.arrived_at = time.monotonic()
        self.gateway.sockets.put(self.client)

    def parse_request(self) -> bool:
        self.arrived_at = time.monotonic()
        return super().parse_request()

    def _stale(self) -> bool:
        return (
            self.gateway.drop_after is not None
            and self.idle_since is not None
            and self.arrived_at - self.idle_since >= self.gateway.drop_after
        )

    def answer(self) -> None:
        headers: Final = {name.lower(): value for name, value in self.headers.items()}
        body: Final = _body(headers, self.rfile)
        if body is None:
            self.gateway.failures.put(ValueError(f"unreadable body for {self.path}: {headers}"))
            self.close_connection = True
            return
        found: Final = MARKER.search(body) or MARKER.search(self.path.encode())
        stale: Final = self._stale()
        self.gateway.arrivals.put(
            Arrival(found.group().decode() if found else "", self.path, self.connection_id, stale)
        )
        if stale:
            self._drop()
            return
        try:
            reply: Final = self.gateway.respond(Request(self.command, self.path, headers, body))
        except (ValueError, KeyError, TypeError) as error:
            self.gateway.failures.put(error)
            self._write(Reply(status=500))
            return
        self._write(reply)
        self.idle_since = time.monotonic()

    do_POST = answer
    do_GET = answer

    def _drop(self) -> None:
        self.close_connection = True
        match self.gateway.drop:
            case Drop.CLOSE:
                with suppress(OSError):
                    self.client.shutdown(socket.SHUT_RDWR)
            case Drop.RESET:
                self.client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                self.rfile.close()
                self.client.close()

    def _write(self, reply: Reply) -> None:
        self.send_response(reply.status)
        self.send_header("content-type", reply.content_type)
        for name, value in reply.headers.items():
            self.send_header(name, value)
        if reply.chunks is None:
            self.send_header("content-length", str(len(reply.body)))
        else:
            self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        if reply.chunks is None:
            self.wfile.write(reply.body)
            return
        for index, chunk in enumerate(reply.chunks):
            if index and reply.pause_between_chunks:
                time.sleep(reply.pause_between_chunks)
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
        self.wfile.write(b"0\r\n\r\n")

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def idle_gateway(
    respond: Callable[[Request], Reply],
    *,
    drop_after: float | None,
    drop: Drop = Drop.CLOSE,
    port: int = 0,
) -> Generator[IdleGateway]:
    arrivals: Final[SimpleQueue[Arrival]] = SimpleQueue()
    with _Server(port, respond, drop_after, drop, arrivals) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield IdleGateway(f"http://{_LOOPBACK}:{server.server_port}", server.server_port, arrivals)
        finally:
            server.shutdown()
            thread.join(timeout=6)
            server.close_connections()
        assert not thread.is_alive(), "Idle gateway survived cleanup"
        failure: Final = None if server.failures.empty() else server.failures.get_nowait()
        assert failure is None, f"Idle gateway failed to answer: {failure!r}"
