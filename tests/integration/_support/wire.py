from __future__ import annotations

import ssl
import threading
import time
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class Request:
    method: str
    target: str
    headers: Mapping[str, str]
    body: bytes


@dataclass(frozen=True, slots=True)
class Reply:
    status: int = 200
    body: bytes = b"{}"
    content_type: str = "application/json"
    chunks: tuple[bytes, ...] | None = None
    abort_after: int | None = None
    gate_after_first: threading.Event | None = None
    pause_between_chunks: float = 0
    headers: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Wire:
    url: str
    received: SimpleQueue[Request]
    disconnected: SimpleQueue[str]

    def drain(self) -> tuple[Request, ...]:
        return tuple(self.received.get_nowait() for _ in range(self.received.qsize()))


@contextmanager
def wire_server(respond: Callable[[Request], Reply], tls: ssl.SSLContext | None = None) -> Generator[Wire, None, None]:
    """Owned TCP peer; requests traverse the real HTTP client and serialization."""
    received: Final[SimpleQueue[Request]] = SimpleQueue()
    errors: Final[SimpleQueue[Exception]] = SimpleQueue()
    disconnected: Final[SimpleQueue[str]] = SimpleQueue()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 5

        def respond(self) -> None:
            request: Final = Request(
                self.command,
                self.path,
                {name.lower(): value for name, value in self.headers.items()},
                self.rfile.read(int(self.headers.get("content-length", "0"))),
            )
            received.put(request)
            try:
                reply = respond(request)
            except Exception as error:
                errors.put(error)
                reply = Reply(status=500)
            self.send_response(reply.status)
            self.send_header("content-type", reply.content_type)
            for name, value in reply.headers.items():
                self.send_header(name, value)
            if reply.chunks is None:
                self.send_header("content-length", str(len(reply.body)))
            else:
                self.send_header("transfer-encoding", "chunked")
            self.send_header("connection", "close")
            self.end_headers()
            try:
                if reply.chunks is None:
                    self.wfile.write(reply.body)
                else:
                    for index, chunk in enumerate(reply.chunks):
                        if reply.abort_after == index:
                            break
                        self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                        self.wfile.flush()
                        if index == 0 and reply.gate_after_first is not None:
                            assert reply.gate_after_first.wait(timeout=5), "Stream barrier was never released"
                        if reply.pause_between_chunks and index + 1 < len(reply.chunks):
                            time.sleep(reply.pause_between_chunks)
                    else:
                        self.wfile.write(b"0\r\n\r\n")
                        self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                disconnected.put(request.target)
            except Exception as error:
                errors.put(error)
            self.close_connection = True

        do_POST = respond
        do_PUT = respond
        do_GET = respond
        do_DELETE = respond

        def log_message(self, format: str, *args: object) -> None:
            pass

    class OwnedHTTPServer(ThreadingHTTPServer):
        daemon_threads = False

        def server_bind(self) -> None:
            super().server_bind()
            if tls is not None:
                self.socket = tls.wrap_socket(self.socket, server_side=True)

    with OwnedHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
        thread.start()
        try:
            yield Wire(
                f"{'https' if tls is not None else 'http'}://127.0.0.1:{server.server_port}",
                received,
                disconnected,
            )
        finally:
            server.shutdown()
            thread.join(timeout=6)
            assert not thread.is_alive(), "Owned HTTP server survived cleanup"
            server.server_close()
            failure: Final = None if errors.empty() else errors.get_nowait()
            assert failure is None, f"Owned HTTP peer failed: {failure!r}"
