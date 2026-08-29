from __future__ import annotations

import threading
from collections.abc import Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import httpx
import pytest
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict

from tests.test_litellm._fixture_recorder import (
    FIXTURE_SCHEMA_VERSION,
    ProviderSpec,
    fixture_cache_key,
    generate_case_inputs,
    record_case,
    record_cases,
    recorded_fixtures,
)
from tests.test_litellm._json_fs_cache import JsonFileCache
from tests.test_litellm._recorded_http import (
    HttpHeader,
    RecordedHttpStreamResponse,
    RecordedResponse,
    RecordedStreamChunk,
)
from tests.test_litellm.parity.replay import replay_server

_SSE_CHUNKS: Final = (
    b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
    b"data: [DONE]\n\n",
)


class _FixtureInput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    identifier: str

    def canonical_input(self) -> dict[str, object]:
        return {"identifier": self.identifier}


class _ParityCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    litellm_input: _FixtureInput
    provider_response: RecordedResponse


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
        if self.path == "/v1/chat/completions":
            with upstream.lock:
                upstream.request_count += 1
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for chunk in _SSE_CHUNKS:
                self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                self.wfile.write(chunk)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
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


def _case(identifier: str) -> _FixtureInput:
    return _FixtureInput(identifier=identifier)


def _sdk_call(api_base: str, case_input: _FixtureInput) -> object:
    return httpx.post(f"{api_base}/v1/ocr", content=b"{}", timeout=5)


def _stream_sdk_call(api_base: str, case_input: _FixtureInput) -> object:
    return httpx.post(f"{api_base}/v1/chat/completions", content=b"{}", timeout=5)


def test_generate_case_inputs_is_deterministic() -> None:
    strategy: Final = st.builds(_FixtureInput, identifier=st.integers().map(str))

    assert generate_case_inputs(strategy, examples=4) == generate_case_inputs(strategy, examples=4)


def test_record_cases_deduplicates_and_limits_concurrency(tmp_path: Path) -> None:
    case_inputs: Final = (_case("one"), _case("two"), _case("one"), _case("three"))
    with _controlled_upstream() as upstream:
        spec: Final = ProviderSpec(upstream_base=upstream.url)
        results: Final = record_cases(spec, tmp_path, case_inputs, _sdk_call, _ParityCase, max_concurrency=2)

    assert len(results) == 3
    assert upstream.request_count == 3
    assert upstream.max_active_requests == 2
    assert len(recorded_fixtures(tmp_path, _ParityCase)) == 3
    for fixture_path in tmp_path.glob("*.json"):
        contents = fixture_path.read_text(encoding="utf-8")
        assert f'"schema_version": {FIXTURE_SCHEMA_VERSION}' in contents
        assert '"recorded_at":' in contents


def test_record_case_rejects_stale_fixture_before_provider_call(tmp_path: Path) -> None:
    case_input: Final = _case("stale")
    cache: Final = JsonFileCache(tmp_path)
    fixture_path: Final = cache.put(fixture_cache_key(case_input), {"schema_version": 0})

    with pytest.raises(ValueError, match=f"{fixture_path} has schema_version 0, expected {FIXTURE_SCHEMA_VERSION}"):
        record_case(
            ProviderSpec(upstream_base="http://127.0.0.1:1"),
            tmp_path,
            case_input,
            _sdk_call,
            _ParityCase,
        )


def test_streaming_response_records_and_replays_chunks(tmp_path: Path) -> None:
    with _controlled_upstream() as upstream:
        result: Final = record_case(
            ProviderSpec(upstream_base=upstream.url),
            tmp_path,
            _case("stream"),
            _stream_sdk_call,
            _ParityCase,
        )

    response: Final = result.case.provider_response
    assert isinstance(response, RecordedHttpStreamResponse)
    assert tuple(chunk.data_bytes() for chunk in response.chunks) == _SSE_CHUNKS

    with replay_server() as provider:
        provider.enqueue_response(response)
        with httpx.stream("POST", f"{provider.url}/v1/chat/completions", json={}) as replayed:
            replayed_chunks: Final = tuple(replayed.iter_raw())
        provider.take_request()

    assert replayed_chunks == _SSE_CHUNKS


def test_stream_response_model_rejects_buffered_body() -> None:
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        RecordedHttpStreamResponse.model_validate(
            {
                "kind": "http_stream",
                "status_code": 200,
                "headers": [HttpHeader(name="content-type", value="text/event-stream")],
                "chunks": [RecordedStreamChunk.from_bytes(b"data: [DONE]\n\n")],
                "body_b64": "",
            }
        )
