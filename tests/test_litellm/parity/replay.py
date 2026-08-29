from __future__ import annotations

import queue
import threading
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
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

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ReplayHandler)
        self.responses: queue.Queue[ReplayResponse] = queue.Queue()
        self.requests: queue.Queue[CapturedRequest] = queue.Queue()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def enqueue_response(self, status_code: int, headers: tuple[tuple[str, str], ...], body: bytes) -> None:
        self.responses.put(ReplayResponse(status_code=status_code, headers=headers, body=body))

    def take_request(self) -> CapturedRequest:
        request_count: Final = self.requests.qsize()
        if request_count != 1:
            raise AssertionError(f"expected exactly one provider request, received {request_count}")
        return self.requests.get_nowait()

    def reset(self) -> None:
        while not self.responses.empty():
            self.responses.get_nowait()
        while not self.requests.empty():
            self.requests.get_nowait()


@dataclass(frozen=True, slots=True)
class ReplayResponse:
    status_code: int
    headers: tuple[tuple[str, str], ...]
    body: bytes


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
        try:
            response: Final = provider.responses.get(timeout=5)
        except queue.Empty:
            self.send_error(500, "no replay response queued")
            return
        self.send_response_only(response.status_code)
        for name, value in response.headers:
            if name.lower() not in EXCLUDED_RESPONSE_HEADERS:
                self.send_header(name, value)
        self.send_header("content-length", str(len(response.body)))
        self.end_headers()
        self.wfile.write(response.body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def replay_server() -> Generator[ReplayServer]:
    server: Final = ReplayServer()
    thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
