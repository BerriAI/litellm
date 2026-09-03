from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from typing import Final, cast


@dataclass(frozen=True, slots=True)
class MockProviderResponse:
    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


class _MockProviderServer(ThreadingHTTPServer):
    def __init__(self, response: MockProviderResponse) -> None:
        super().__init__(("127.0.0.1", 0), _MockProviderHandler)
        self.response: Final = response
        self._request_count = 0
        self._request_count_lock: Final = Lock()

    def record_request(self) -> None:
        with self._request_count_lock:
            self._request_count += 1

    @property
    def request_count(self) -> int:
        with self._request_count_lock:
            return self._request_count


class _MockProviderHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        content_length: Final = int(self.headers.get("content-length", "0"))
        self.rfile.read(content_length)
        server: Final = cast(_MockProviderServer, self.server)
        server.record_request()
        self.send_response(server.response.status_code)
        for name, value in server.response.headers:
            self.send_header(name, value)
        self.send_header("content-length", str(len(server.response.body)))
        self.end_headers()
        self.wfile.write(server.response.body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002  # matches BaseHTTPRequestHandler
        pass


@contextmanager
def mock_provider(response: MockProviderResponse) -> Generator[str]:
    server: Final = _MockProviderServer(response)
    thread: Final = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = cast(tuple[str, int], server.server_address)
    try:
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    if server.request_count != 1:
        raise AssertionError(f"expected one provider request, received {server.request_count}")
