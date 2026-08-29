from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast

import httpx

from tests.test_litellm._fixture_recorder import ProviderSpec, record_cases
from tests.test_litellm.ocr.fixture_models import ImageUrlDocument, LiteLLMOcrInput


class _ControlledUpstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _ControlledUpstreamHandler)
        self.lock: Final = threading.Lock()
        self.two_requests_started: Final = threading.Event()
        self.active_requests: int = 0
        self.max_active_requests: int = 0
        self.request_count: int = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def start_request(self) -> None:
        with self.lock:
            self.active_requests += 1
            self.request_count += 1
            self.max_active_requests = max(self.max_active_requests, self.active_requests)
            if self.active_requests == 2:
                self.two_requests_started.set()
        self.two_requests_started.wait(timeout=2)

    def end_tracked_request(self) -> None:
        with self.lock:
            self.active_requests -= 1


class _ControlledUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        upstream: Final = self.server
        assert isinstance(upstream, _ControlledUpstream)
        length: Final = int(self.headers.get("content-length") or "0")
        self.rfile.read(length)
        upstream.start_request()
        try:
            body: Final = b"{}"
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        finally:
            upstream.end_tracked_request()

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _controlled_upstream() -> Generator[_ControlledUpstream]:
    server: Final = _ControlledUpstream()
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _case(identifier: str) -> LiteLLMOcrInput:
    return LiteLLMOcrInput.model_validate(
        {
            "model": "mistral/mistral-ocr-latest",
            "document": ImageUrlDocument(type="image_url", image_url="https://example.com/image.png"),
            "id": identifier,
        }
    )


def _sdk_call(**kwargs: object) -> object:
    api_base: Final = cast(str, kwargs["api_base"])
    return httpx.post(f"{api_base}/v1/ocr", content=b"{}", timeout=5)


def test_record_cases_deduplicates_and_limits_concurrency(tmp_path: Path) -> None:
    case_inputs: Final = (_case("one"), _case("two"), _case("one"), _case("three"))
    with _controlled_upstream() as upstream:
        spec: Final = ProviderSpec(model="mistral/mistral-ocr-latest", upstream_base=upstream.url, api_key="test-key")
        results: Final = record_cases(spec, tmp_path, case_inputs, _sdk_call, max_concurrency=2)

    assert len(results) == 3
    assert upstream.request_count == 3
    assert upstream.max_active_requests == 2
