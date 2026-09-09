import asyncio
import copy
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    raw_body: bytes
    body: object | None


@dataclass
class ResponseSpec:
    body: object
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0
    events: tuple[tuple[str, object], ...] = ()
    chunks: tuple[bytes, ...] = ()
    accepted: threading.Event | None = None
    release_before_response: threading.Event | None = None
    release: threading.Event | None = None


@dataclass
class RecordingServer:
    server: ThreadingHTTPServer
    requests: list[RecordedRequest]
    responses: list[ResponseSpec]
    default_response: ResponseSpec
    expected_requests: int | None = 1

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def enqueue(self, response: ResponseSpec) -> None:
        self.responses.append(response)

    async def wait_for_requests(self, count: int) -> None:
        async with asyncio.timeout(2):
            while len(self.requests) < count:
                await asyncio.sleep(0.01)


@contextmanager
def recording_service() -> Iterator[RecordingServer]:
    requests: list[RecordedRequest] = []
    responses: list[ResponseSpec] = []

    class Handler(BaseHTTPRequestHandler):
        def _handle(self) -> None:
            content_length: Final = int(self.headers.get("Content-Length", "0"))
            raw_body: Final = self.rfile.read(content_length) if content_length else b""
            body: Final = json.loads(raw_body) if raw_body else None
            requests.append(
                RecordedRequest(
                    method=self.command,
                    path=self.path,
                    headers={name.lower(): value for name, value in self.headers.items()},
                    raw_body=raw_body,
                    body=body,
                )
            )
            response: Final = responses.pop(0) if responses else copy.deepcopy(recording_server.default_response)
            if response.accepted is not None:
                response.accepted.set()
            if response.release_before_response is not None:
                response.release_before_response.wait(timeout=10)
            if response.delay:
                time.sleep(response.delay)
            payload: Final = (
                b"".join(response.chunks)
                if response.chunks
                else (
                    b"".join(
                        f"event: {event}\ndata: {json.dumps(data)}\n\n".encode() for event, data in response.events
                    )
                    if response.events
                    else json.dumps(response.body).encode()
                )
            )
            self.send_response(response.status)
            self.send_header(
                "Content-Type", "text/event-stream" if response.events or response.chunks else "application/json"
            )
            self.send_header("Content-Length", str(len(payload)))
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            try:
                if response.chunks:
                    for index, chunk in enumerate(response.chunks):
                        if index == 1 and response.release is not None:
                            response.release.wait(timeout=10)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                else:
                    self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_GET = _handle
        do_POST = _handle

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        recording_server = RecordingServer(
            server=server,
            requests=requests,
            responses=responses,
            default_response=ResponseSpec(body={}),
        )
        yield recording_server
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        if recording_server.expected_requests is not None:
            assert len(recording_server.requests) == recording_server.expected_requests
        assert recording_server.responses == []


@pytest.fixture
def recording_server() -> Iterator[RecordingServer]:
    with recording_service() as server:
        yield server
