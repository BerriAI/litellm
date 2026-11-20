import asyncio
import copy
import gzip
import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final


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
    chunk_delay: float = 0
    events: tuple[tuple[str, object], ...] = ()
    # Splits a non-stream body into this many writes, `chunk_delay` apart, so it trickles in.
    body_chunks: int = 1
    cut_after: int | None = None
    cut_delay: float = 0
    gzip: bool = False
    # Writes every stream event in one write, so they reach the client in one read.
    burst: bool = False
    # Sends each stream event as its own HTTP/1.1 chunk instead of under a Content-Length.
    chunked: bool = False

    def payloads(self) -> tuple[bytes, ...]:
        if not self.events:
            raw: Final = self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()
            body: Final = gzip.compress(raw, mtime=0) if self.gzip else raw
            size: Final = max(-(-len(body) // max(self.body_chunks, 1)), 1)
            return tuple(body[start : start + size] for start in range(0, len(body), size)) or (body,)
        frames: Final = tuple(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode() for event, data in self.events)
        if self.gzip:
            return (gzip.compress(b"".join(frames), mtime=0),)
        return (b"".join(frames),) if self.burst and not self.chunked else frames

    def writes(self) -> tuple[bytes, ...]:
        payloads: Final = self.payloads()
        if not self.chunked:
            return payloads
        chunks: Final = (*(b"%x\r\n%s\r\n" % (len(payload), payload) for payload in payloads), b"0\r\n\r\n")
        return (b"".join(chunks),) if self.burst else chunks


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
            body: Final = (
                json.loads(raw_body) if raw_body and self.headers.get_content_type() == "application/json" else None
            )
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
            if response.delay:
                time.sleep(response.delay)
            payloads: Final = response.writes()
            if response.chunked:
                self.protocol_version = "HTTP/1.1"
                self.close_connection = True
            self.send_response(response.status)
            self.send_header("Content-Type", "text/event-stream" if response.events else "application/json")
            if response.chunked:
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Connection", "close")
            else:
                self.send_header("Content-Length", str(sum(len(payload) for payload in payloads)))
            if response.gzip:
                self.send_header("Content-Encoding", "gzip")
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            try:
                for index, payload in enumerate(payloads):
                    if index == response.cut_after:
                        time.sleep(response.cut_delay)
                        self.close_connection = True
                        break
                    if index and response.chunk_delay:
                        time.sleep(response.chunk_delay)
                    self.wfile.write(payload)
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_POST = _handle
        do_GET = _handle
        do_DELETE = _handle

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


def relaxed(server: RecordingServer) -> Iterator[RecordingServer]:
    server.expected_requests = None
    yield server
    server.responses.clear()
