from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final

import httpx
import pytest
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict

from tests.route_parity.fixtures.pipeline import RecordingTarget, record_fixtures
from tests.route_parity.fixtures.recording import ProviderSpec, record_upstream_responses
from tests.route_parity.fixtures.store import (
    FIXTURE_SCHEMA_VERSION,
    fixture_path,
    recorded_fixtures,
)
from tests.route_parity.recorded_http import (
    HttpHeader,
    RecordedHttpStreamResponse,
    RecordedResponse,
    RecordedStreamChunk,
)
from tests.route_parity.replay import replay_server

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
    provider_responses: tuple[RecordedResponse, ...]


@dataclass(frozen=True, slots=True)
class _Invocation:
    sdk_call: Callable[[str, _FixtureInput], object]

    def execute(self, provider_url: str, case_input: _FixtureInput) -> None:
        self.sdk_call(provider_url, case_input)


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
        if self.path == "/upload":
            self._send_json(200, b'{"file_id":"reducto://fixture.pdf"}')
            return
        if self.path == "/parse":
            self._send_json(200, b'{"result":{"chunks":[]}}')
            return
        if self.path == "/analyze":
            self.send_response(202)
            self.send_header("operation-location", f"{upstream.url}/results/1")
            self.send_header("content-length", "0")
            self.end_headers()
            return
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
        if self.path == "/error":
            self._send_json(429, b'{"error":{"message":"rate limited"}}')
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

    def do_GET(self) -> None:
        if self.path == "/results/1":
            self._send_json(200, b'{"status":"succeeded","analyzeResult":{"pages":[]}}')
            return
        self.send_error(404)

    def _send_json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

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


def _error_sdk_call(api_base: str, case_input: _FixtureInput) -> object:
    response: Final = httpx.post(f"{api_base}/error", content=b"{}", timeout=5)
    response.raise_for_status()
    return response


def _multi_sdk_call(api_base: str, case_input: _FixtureInput) -> object:
    upload: Final = httpx.post(f"{api_base}/upload", json={"document": case_input.identifier}, timeout=5)
    upload.raise_for_status()
    parsed: Final = httpx.post(f"{api_base}/parse", json={"input": upload.json()["file_id"]}, timeout=5)
    parsed.raise_for_status()
    return parsed


def _polling_sdk_call(api_base: str, case_input: _FixtureInput) -> object:
    started: Final = httpx.post(f"{api_base}/analyze", json={"document": case_input.identifier}, timeout=5)
    operation_location: Final = started.headers["operation-location"]
    completed: Final = httpx.get(operation_location, timeout=5)
    completed.raise_for_status()
    return completed


def test_recording_deduplicates_per_target_and_caps_global_concurrency(tmp_path: Path) -> None:
    shared_input: Final = _case("shared")
    with _controlled_upstream() as upstream:
        spec: Final = ProviderSpec(upstream_base=upstream.url)
        targets: Final = (
            RecordingTarget(
                name="first",
                provider_spec=spec,
                strategy=st.just(shared_input),
                invocation=_Invocation(_sdk_call),
                required_inputs=(shared_input, shared_input),
            ),
            RecordingTarget(
                name="second",
                provider_spec=spec,
                strategy=st.just(shared_input),
                invocation=_Invocation(_sdk_call),
                required_inputs=(shared_input,),
            ),
        )
        summary: Final = record_fixtures(targets, tmp_path, examples=1, concurrency=2, case_type=_ParityCase)

    assert len(summary.recorded) == 2
    assert {result.target_name for result in summary.recorded} == {"first", "second"}
    assert summary.cached == ()
    assert summary.failed == ()
    assert upstream.request_count == 2
    assert upstream.max_active_requests == 2
    assert len(recorded_fixtures(tmp_path, _ParityCase)) == 2
    for path in tmp_path.rglob("*.json"):
        contents = path.read_text(encoding="utf-8")
        assert f'"schema_version": {FIXTURE_SCHEMA_VERSION}' in contents
        assert '"recorded_at":' in contents


def test_pipeline_rejects_stale_fixture_before_provider_call(tmp_path: Path) -> None:
    case_input: Final = _case("stale")
    directory: Final = tmp_path / "stale-target"
    directory.mkdir()
    path: Final = fixture_path(directory, case_input)
    path.write_text('{"schema_version": 0}\n', encoding="utf-8")
    target: Final = RecordingTarget(
        name="stale-target",
        provider_spec=ProviderSpec(upstream_base="http://127.0.0.1:1"),
        strategy=st.just(case_input),
        invocation=_Invocation(_sdk_call),
    )

    summary: Final = record_fixtures(
        (target,),
        tmp_path,
        examples=1,
        concurrency=1,
        case_type=_ParityCase,
    )

    assert summary.recorded == ()
    assert summary.cached == ()
    assert len(summary.failed) == 1
    assert str(summary.failed[0].error) == (
        f"fixture {path} has schema_version 0, expected {FIXTURE_SCHEMA_VERSION}; "
        "delete it and regenerate the fixture bundle"
    )


def test_cached_fixture_is_reported_without_provider_call(tmp_path: Path) -> None:
    case_input: Final = _case("cached")
    with _controlled_upstream() as upstream:
        target: Final = RecordingTarget(
            name="cached-target",
            provider_spec=ProviderSpec(upstream_base=upstream.url),
            strategy=st.just(case_input),
            invocation=_Invocation(_sdk_call),
        )
        first: Final = record_fixtures((target,), tmp_path, 1, 1, _ParityCase)
        second: Final = record_fixtures((target,), tmp_path, 1, 1, _ParityCase)

    assert len(first.recorded) == 1
    assert len(second.cached) == 1
    assert upstream.request_count == 1


def test_streaming_response_records_and_replays_chunks() -> None:
    with _controlled_upstream() as upstream:
        responses: Final = record_upstream_responses(
            ProviderSpec(upstream_base=upstream.url),
            _case("stream"),
            _stream_sdk_call,
        )

    response: Final = responses[0]
    assert isinstance(response, RecordedHttpStreamResponse)
    assert tuple(chunk.data_bytes() for chunk in response.chunks) == _SSE_CHUNKS
    assert isinstance(response.model_dump(mode="json")["chunks"], list)

    with replay_server() as provider:
        provider.enqueue_response(response)
        with httpx.stream("POST", f"{provider.url}/v1/chat/completions", json={}) as replayed:
            replayed_chunks: Final = tuple(replayed.iter_raw())
        provider.take_requests(1)

    assert replayed_chunks == _SSE_CHUNKS


def test_non_successful_provider_response_is_recorded() -> None:
    with _controlled_upstream() as upstream:
        responses: Final = record_upstream_responses(
            ProviderSpec(upstream_base=upstream.url),
            _case("provider-error"),
            _error_sdk_call,
        )

    response: Final = responses[0]
    assert response.status_code == 429


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


@pytest.mark.parametrize("sdk_call", (_multi_sdk_call, _polling_sdk_call))
def test_multiple_provider_calls_record_and_replay_in_order(
    sdk_call: Callable[[str, _FixtureInput], object],
) -> None:
    with _controlled_upstream() as upstream:
        responses: Final = record_upstream_responses(
            ProviderSpec(upstream_base=upstream.url),
            _case(sdk_call.__name__),
            sdk_call,
        )

    assert len(responses) == 2
    with replay_server() as provider:
        for response in responses:
            provider.enqueue_response(response)
        sdk_call(provider.url, _case(sdk_call.__name__))
        requests: Final = provider.take_requests(2)

    assert len(requests) == 2
