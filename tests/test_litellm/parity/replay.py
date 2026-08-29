from __future__ import annotations

import json
import queue
import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

from pydantic import TypeAdapter

from tests.test_litellm.parity.models import CapturedRequest, ReplayResponse

JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class JsonReplayServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, expected_path: str, response: ReplayResponse) -> None:
        super().__init__(("127.0.0.1", 0), _JsonReplayHandler)
        self.expected_path: Final = expected_path
        self.response: Final = response
        self.response_body: Final = json.dumps(response.body, sort_keys=True, separators=(",", ":")).encode()
        self.requests: queue.Queue[CapturedRequest] = queue.Queue()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def take_requests(self) -> tuple[CapturedRequest, ...]:
        try:
            first: Final = self.requests.get(timeout=5)
        except queue.Empty as error:
            raise AssertionError("expected at least one provider request, received none") from error
        remaining_count: Final = self.requests.qsize()
        remaining: Final = tuple(self.requests.get_nowait() for _ in range(remaining_count))
        return (first, *remaining)


class _JsonReplayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, JsonReplayServer)
        length: Final = int(self.headers.get("content-length") or "0")
        body: Final = JSON_OBJECT.validate_json(self.rfile.read(length))
        content_type_header: Final = self.headers.get("content-type")
        content_type: Final = content_type_header.split(";", 1)[0].lower() if content_type_header else None
        provider.requests.put(
            CapturedRequest(
                method=self.command,
                path=self.path,
                authorization=self.headers.get("authorization"),
                content_type=content_type,
                parity_case=self.headers.get("x-parity-case"),
                body=body,
                user_agent=self.headers.get("user-agent"),
            )
        )
        matched: Final = self.path == provider.expected_path
        status_code: Final = provider.response.status_code if matched else 404
        response_body: Final = provider.response_body if matched else b'{"error":"unexpected path"}'
        response_headers: Final = provider.response.headers if matched else {"content-type": "application/json"}
        self.send_response(status_code)
        for name, value in response_headers.items():
            if name.lower() not in {"content-length", "transfer-encoding", "content-encoding"}:
                self.send_header(name, value)
        self.send_header("content-length", str(len(response_body)))
        self.end_headers()
        self.wfile.write(response_body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def replay_json_response(expected_path: str, response: ReplayResponse) -> Generator[JsonReplayServer]:
    server: Final = JsonReplayServer(expected_path=expected_path, response=response)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
