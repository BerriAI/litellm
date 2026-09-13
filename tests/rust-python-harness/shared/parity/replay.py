from __future__ import annotations

import base64
import queue
from contextlib import AbstractContextManager
from typing import Final

from pydantic import JsonValue, TypeAdapter

from .http import local_response_header
from .local_server import LocalHttpHandler, LocalHttpServer, serve_in_thread
from .models import CapturedRequest
from .recorded_http import RecordedHttpResponse, RecordedHttpStreamResponse, RecordedResponse

JSON_VALUE: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
EXCLUDED_REQUEST_HEADERS: Final = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "accept-encoding",
        "user-agent",
        "x-litellm-parity-route",
    }
)
EXCLUDED_RESPONSE_HEADERS: Final = frozenset({"content-length", "transfer-encoding", "connection"})


def _replay_response_header(name: str, value: str, provider_url: str) -> str:
    if name.lower() == "retry-after":
        return "0"
    return local_response_header(name, value, provider_url)


class ReplayServer(LocalHttpServer):
    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ReplayHandler)
        self.responses: queue.Queue[RecordedResponse] = queue.Queue()
        self.requests: queue.Queue[CapturedRequest] = queue.Queue()

    def enqueue_response(self, response: RecordedResponse) -> None:
        self.responses.put(response)

    def take_requests(self, expected_count: int) -> tuple[CapturedRequest, ...]:
        request_count: Final = self.requests.qsize()
        if request_count != expected_count:
            raise AssertionError(f"expected exactly {expected_count} provider requests, received {request_count}")
        return tuple(self.requests.get_nowait() for _ in range(request_count))

    def reset(self) -> None:
        while not self.responses.empty():
            self.responses.get_nowait()
        while not self.requests.empty():
            self.requests.get_nowait()


class _ReplayHandler(LocalHttpHandler):
    def do_POST(self) -> None:
        self._replay()

    def do_GET(self) -> None:
        self._replay()

    def do_PUT(self) -> None:
        self._replay()

    def do_PATCH(self) -> None:
        self._replay()

    def do_DELETE(self) -> None:
        self._replay()

    def _replay(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, ReplayServer)
        length: Final = int(self.headers.get("content-length") or "0")
        raw_body: Final = self.rfile.read(length) if length else b""
        content_type: Final = self.headers.get("content-type", "")
        body: Final = (
            JSON_VALUE.validate_json(raw_body)
            if raw_body and content_type.lower().startswith("application/json")
            else base64.b64encode(raw_body).decode("ascii")
            if raw_body
            else None
        )
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
        for header in response.headers:
            if header.name.lower() not in EXCLUDED_RESPONSE_HEADERS:
                self.send_header(header.name, _replay_response_header(header.name, header.value, provider.url))
        if isinstance(response, RecordedHttpResponse):
            response_body: Final = response.body_bytes()
            self.send_header("content-length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)
            return
        assert isinstance(response, RecordedHttpStreamResponse)
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        self.write_chunked(chunk.data_bytes() for chunk in response.chunks)


def replay_server() -> AbstractContextManager[ReplayServer]:
    return serve_in_thread(ReplayServer(), poll_interval=0.01)
