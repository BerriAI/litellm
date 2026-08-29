from __future__ import annotations

import queue
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from pydantic import JsonValue, TypeAdapter

from tests.test_litellm.parity.models import CapturedRequest

JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
EXCLUDED_REQUEST_HEADERS: Final = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "accept-encoding",
        "user-agent",
        "x-ocr-parity-route",
    }
)
EXCLUDED_RESPONSE_HEADERS: Final = frozenset({"content-length", "transfer-encoding", "connection"})


class ReplayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, status_code: int, headers: tuple[tuple[str, str], ...], body: bytes) -> None:
        super().__init__(("127.0.0.1", 0), _ReplayHandler)
        self.status_code: Final = status_code
        self.headers: Final = headers
        self.body: Final = body
        self.requests: queue.Queue[CapturedRequest] = queue.Queue()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def take_request(self) -> CapturedRequest:
        request_count: Final = self.requests.qsize()
        if request_count != 1:
            raise AssertionError(f"expected exactly one provider request, received {request_count}")
        return self.requests.get_nowait()


class _ReplayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, ReplayServer)
        length: Final = int(self.headers.get("content-length") or "0")
        body: Final = JSON_VALUE.validate_json(self.rfile.read(length))
        headers: Final = tuple(
            sorted(
                (name.lower(), value)
                for name, value in self.headers.raw_items()
                if name.lower() not in EXCLUDED_REQUEST_HEADERS
            )
        )
        provider.requests.put(
            CapturedRequest(
                method=self.command,
                path=self.path,
                headers=headers,
                body=body,
                user_agent=self.headers.get("user-agent"),
            )
        )
        self.send_response_only(provider.status_code)
        for name, value in provider.headers:
            if name.lower() not in EXCLUDED_RESPONSE_HEADERS:
                self.send_header(name, value)
        self.send_header("content-length", str(len(provider.body)))
        self.end_headers()
        self.wfile.write(provider.body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def replay_response(
    status_code: int,
    headers: tuple[tuple[str, str], ...],
    body: bytes,
) -> Generator[ReplayServer]:
    server: Final = ReplayServer(status_code=status_code, headers=headers, body=body)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
