"""A local HTTP upstream that answers every POST only after a fixed delay, the way a slow model does."""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import SimpleQueue
from socket import socket
from typing import Final


@dataclass(frozen=True, slots=True)
class SlowUpstream:
    base_url: str
    delay_seconds: float
    request_arrivals: SimpleQueue[float]

    def seconds_since_first_request(self) -> float:
        return time.monotonic() - self.request_arrivals.get(timeout=1.0)


class _QuietServer(ThreadingHTTPServer):
    block_on_close = False

    def handle_error(self, request: socket | tuple[bytes, socket], client_address: object) -> None:
        return


@contextmanager
def slow_upstream(delay_seconds: float) -> Iterator[SlowUpstream]:
    request_arrivals: Final[SimpleQueue[float]] = SimpleQueue()

    class DelayedHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            request_arrivals.put(time.monotonic())
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            time.sleep(delay_seconds)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server: Final = _QuietServer(("127.0.0.1", 0), DelayedHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield SlowUpstream(
            base_url=f"http://127.0.0.1:{server.server_address[1]}",
            delay_seconds=delay_seconds,
            request_arrivals=request_arrivals,
        )
    finally:
        server.shutdown()
        server.server_close()
