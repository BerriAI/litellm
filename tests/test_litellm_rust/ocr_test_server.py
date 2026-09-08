import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final

import pytest

OCR_RESPONSE: Final = {
    "pages": [{"index": 0, "markdown": "native OCR response", "images": [], "dimensions": None}],
    "model": "mistral-ocr-latest",
    "usage_info": {"pages_processed": 1, "doc_size_bytes": 3},
}


@dataclass
class RecordedRequest:
    method: str
    path: str
    headers: dict[str, str]
    body: object | None


@dataclass
class ResponseSpec:
    status: int = 200
    body: object = field(default_factory=lambda: dict(OCR_RESPONSE))
    headers: dict[str, str] = field(default_factory=dict)
    delay: float = 0


@dataclass
class OCRTestServer:
    server: ThreadingHTTPServer
    requests: list[RecordedRequest]
    responses: list[ResponseSpec]

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def enqueue(self, response: ResponseSpec) -> None:
        self.responses.append(response)


@pytest.fixture
def ocr_server() -> Iterator[OCRTestServer]:
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
                    body=body,
                )
            )
            response: Final = responses.pop(0) if responses else ResponseSpec()
            if response.delay:
                time.sleep(response.delay)
            payload: Final = json.dumps(response.body).encode()
            self.send_response(response.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.end_headers()
            try:
                self.wfile.write(payload)
            except BrokenPipeError:
                pass

        do_GET = _handle
        do_POST = _handle

        def log_message(self, format: str, *args: object) -> None:
            pass

    server: Final = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread: Final = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield OCRTestServer(server=server, requests=requests, responses=responses)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
