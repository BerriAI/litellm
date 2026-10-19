from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import SimpleQueue
from typing import Final


@dataclass(frozen=True, slots=True)
class Tunnel:
    method: str
    target: str
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class ForwardProxy:
    url: str
    port: int
    received: SimpleQueue[Tunnel]

    def drain(self) -> tuple[Tunnel, ...]:
        return tuple(self.received.get_nowait() for _ in range(self.received.qsize()))

    def targets(self) -> tuple[str, ...]:
        return tuple(tunnel.target for tunnel in self.drain())


@contextmanager
def refusing_forward_proxy(port: int = 0) -> Generator[ForwardProxy, None, None]:
    """Owned HTTP forward proxy: records the host each CONNECT asks for and refuses the tunnel with 403.

    A litellm proxy booted with ``HTTPS_PROXY`` pointed here makes the host it dials for a provider
    observable without a byte leaving the box; the caller sees litellm's own connection error.
    """
    received: Final[SimpleQueue[Tunnel]] = SimpleQueue()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 5

        def refuse(self) -> None:
            received.put(Tunnel(self.command, self.path, {name.lower(): value for name, value in self.headers.items()}))
            self.send_response(403)
            self.send_header("content-length", "0")
            self.send_header("connection", "close")
            self.end_headers()
            self.close_connection = True

        do_CONNECT = refuse
        do_GET = refuse
        do_POST = refuse
        do_PUT = refuse
        do_DELETE = refuse

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield ForwardProxy(f"http://127.0.0.1:{server.server_port}", server.server_port, received)
        finally:
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "Owned forward proxy survived cleanup"
            server.server_close()


def _pipe(source: socket.socket, sink: socket.socket) -> None:
    with contextlib.suppress(OSError):
        for chunk in iter(lambda: source.recv(65536), b""):
            sink.sendall(chunk)
    with contextlib.suppress(OSError):
        sink.shutdown(socket.SHUT_WR)


@contextmanager
def tunnelling_forward_proxy(destination_port: int, port: int = 0) -> Generator[ForwardProxy, None, None]:
    received: Final[SimpleQueue[Tunnel]] = SimpleQueue()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        rbufsize = 0
        timeout = 30
        request: socket.socket

        def do_CONNECT(self) -> None:
            received.put(Tunnel(self.command, self.path, {name.lower(): value for name, value in self.headers.items()}))
            self.close_connection = True
            with socket.create_connection(("127.0.0.1", destination_port), timeout=30) as destination:
                self.send_response(200, "Connection established")
                self.end_headers()
                outbound: Final = threading.Thread(target=_pipe, args=(self.request, destination))
                outbound.start()
                _pipe(destination, self.request)
                outbound.join(timeout=35)

        def log_message(self, format: str, *args: object) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", port), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield ForwardProxy(f"http://127.0.0.1:{server.server_port}", server.server_port, received)
        finally:
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "Owned forward proxy survived cleanup"
            server.server_close()
