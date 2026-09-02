from __future__ import annotations

import asyncio
import queue
import threading
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
from hypothesis import strategies as st
from openai._streaming import SSEDecoder
from pydantic import BaseModel, ConfigDict

from tests.route_parity.compare import assert_request_parity
from tests.route_parity.fixtures.pipeline import RecordingTarget, record_fixtures
from tests.route_parity.fixtures.recording import (
    UpstreamEndpoint,
    record_upstream_interactions,
    record_upstream_responses,
)
from tests.route_parity.fixtures.store import (
    FIXTURE_SCHEMA_VERSION,
    fixture_path,
    load_fixture,
    recorded_fixtures,
)
from tests.route_parity.inprocess import InProcessExecution, run_in_process, run_in_process_async
from tests.route_parity.recorded_http import (
    HttpHeader,
    RecordedHttpStreamResponse,
    RecordedResponse,
    RecordedStreamChunk,
)
from tests.route_parity.replay import ReplayServer, replay_server
from tests.route_parity.stream import (
    StreamCompleted,
    StreamFailed,
    StreamOutcome,
    assert_stream_parity,
    consume_async_stream,
    consume_sync_stream,
)

_SSE_CHUNKS: Final = (
    b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
    b"data: [DONE]\n\n",
)


class _StreamEvent(BaseModel):
    kind: Literal["delta", "done", "error"]
    value: str


class _StreamApplicationError(Exception):
    status_code: Final = 400
    code: Final = "invalid_input"
    type: Final = "validation_error"
    param: Final = "input"
    model: Final = "fixture-model"
    llm_provider: Final = "fixture-provider"


def _stream_event(data: str) -> _StreamEvent:
    event: Final = _StreamEvent.model_validate_json(data)
    if event.kind == "error":
        raise _StreamApplicationError(event.value)
    return event


def _event_chunks(failed: bool) -> tuple[bytes, ...]:
    terminal: Final = (
        b'event: error\r\ndata: {"kind":"error","value":"invalid input"}\r\n\r\n'
        if failed
        else b'event: done\r\ndata: {"kind":"done","value":""}\r\n\r\n'
    )
    return (
        b'event: delta\r\ndata: {"kind":"delta",\r\ndata: "value":"caf\xc3',
        b'\xa9"}\r\n',
        b'\r\nevent: delta\r\ndata: {"kind":"delta","value":"second"}\r\n\r\n' + terminal,
    )


def _sync_events(api_base: str, case_input: _FixtureInput) -> Iterator[_StreamEvent]:
    with httpx.stream("POST", f"{api_base}/stream", json={"id": case_input.identifier}, timeout=5) as response:
        response.raise_for_status()
        for event in SSEDecoder().iter_bytes(response.iter_bytes()):
            yield _stream_event(event.data)


async def _async_events(api_base: str, case_input: _FixtureInput) -> AsyncIterator[_StreamEvent]:
    async with httpx.AsyncClient(timeout=5) as client:
        async with client.stream("POST", f"{api_base}/stream", json={"id": case_input.identifier}) as response:
            response.raise_for_status()
            async for event in SSEDecoder().aiter_bytes(response.aiter_bytes()):
                yield _stream_event(event.data)


async def _consume_async_events(api_base: str, case_input: _FixtureInput) -> StreamOutcome:
    async def create() -> AsyncIterator[_StreamEvent]:
        return _async_events(api_base, case_input)

    return await consume_async_stream(create)


async def _replay_events(
    mode: Literal["sync", "async"],
    provider: ReplayServer,
    response: RecordedHttpStreamResponse,
    case_input: _FixtureInput,
) -> InProcessExecution[StreamOutcome]:
    if mode == "sync":
        return run_in_process(
            provider, (response,), lambda url: consume_sync_stream(lambda: _sync_events(url, case_input))
        )
    return await run_in_process_async(provider, (response,), lambda url: _consume_async_events(url, case_input))


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

    def __init__(self, stream_chunks: tuple[bytes, ...]) -> None:
        super().__init__(("127.0.0.1", 0), _ControlledUpstreamHandler)
        self.stream_chunks: Final = stream_chunks
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
        if self.path == "/credentials?api_key=query-secret&api-version=1":
            authorized: Final = self.headers.get("authorization") == "Bearer header-secret"
            self._send_json(200 if authorized else 401, b"{}")
            return
        if self.path == "/upload":
            self._send_json(200, b'{"file_id":"fixture://document.pdf"}')
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
        if self.path in {"/v1/chat/completions", "/stream"}:
            with upstream.lock:
                upstream.request_count += 1
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            for chunk in upstream.stream_chunks:
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
            self.send_header("set-cookie", "session=must-not-be-recorded")
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

    def do_PUT(self) -> None:
        self.do_POST()

    def do_PATCH(self) -> None:
        self.do_POST()

    def do_DELETE(self) -> None:
        self.do_POST()

    def _send_json(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _controlled_upstream(stream_chunks: tuple[bytes, ...] = _SSE_CHUNKS) -> Generator[_ControlledUpstream]:
    server: Final = _ControlledUpstream(stream_chunks)
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
    return httpx.post(f"{api_base}/v1/operation", content=b"{}", timeout=5)


def _stream_sdk_call(api_base: str, case_input: _FixtureInput) -> object:
    return httpx.post(f"{api_base}/v1/chat/completions", content=b"{}", timeout=5)


def _error_sdk_call(api_base: str, case_input: _FixtureInput) -> object:
    response: Final = httpx.post(f"{api_base}/error", content=b"{}", timeout=5)
    response.raise_for_status()
    return response


def _method_sdk_call(method: str) -> Callable[[str, _FixtureInput], object]:
    def call(api_base: str, case_input: _FixtureInput) -> object:
        return httpx.request(method, f"{api_base}/method", json={"id": case_input.identifier}, timeout=5)

    return call


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
        spec: Final = UpstreamEndpoint(base_url=upstream.url)
        targets: Final = (
            RecordingTarget(
                name="first",
                upstream=spec,
                strategy=st.just(shared_input),
                invocation=_Invocation(_sdk_call),
                required_inputs=(shared_input, shared_input),
            ),
            RecordingTarget(
                name="second",
                upstream=spec,
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
    for path in tmp_path.rglob("*.yaml"):
        contents = path.read_text(encoding="utf-8")
        assert f"schema_version: {FIXTURE_SCHEMA_VERSION}" in contents
        assert "recorded_at:" in contents


def test_pipeline_rejects_stale_fixture_before_provider_call(tmp_path: Path) -> None:
    case_input: Final = _case("stale")
    directory: Final = tmp_path / "stale-target"
    directory.mkdir()
    path: Final = fixture_path(directory, case_input).with_suffix(".json")
    path.write_text('{"schema_version": 0}\n', encoding="utf-8")
    target: Final = RecordingTarget(
        name="stale-target",
        upstream=UpstreamEndpoint(base_url="http://127.0.0.1:1"),
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
            upstream=UpstreamEndpoint(base_url=upstream.url),
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
            UpstreamEndpoint(base_url=upstream.url),
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
            UpstreamEndpoint(base_url=upstream.url),
            _case("provider-error"),
            _error_sdk_call,
        )

    response: Final = responses[0]
    assert response.status_code == 429


def test_sensitive_response_headers_are_not_recorded() -> None:
    with _controlled_upstream() as upstream:
        responses: Final = record_upstream_responses(
            UpstreamEndpoint(base_url=upstream.url),
            _case("headers"),
            _sdk_call,
        )

    assert all(header.name.lower() != "set-cookie" for header in responses[0].headers)


def test_recorded_requests_strip_credentials_without_changing_the_live_request() -> None:
    def sdk_call(api_base: str, case_input: _FixtureInput) -> object:
        return httpx.post(
            f"{api_base}/credentials?api_key=query-secret&api-version=1",
            headers={
                "Authorization": "Bearer header-secret",
                "Ocp-Apim-Subscription-Key": "azure-secret",
                "Cookie": "session=cookie-secret",
                "X-Test": case_input.identifier,
            },
            content=b"\xffdocument",
        )

    with _controlled_upstream() as upstream:
        interactions: Final = record_upstream_interactions(
            UpstreamEndpoint(upstream.url), _case("credentials"), sdk_call
        )

    interaction: Final = interactions[0]
    assert interaction.response.status_code == 200
    assert interaction.request.uri == "http://parity-provider.invalid/credentials?api-version=1"
    assert interaction.request.body == b"\xffdocument"
    assert interaction.request.headers["x-test"] == "credentials"
    assert all(
        header not in interaction.request.headers for header in ("authorization", "ocp-apim-subscription-key", "cookie")
    )


@pytest.mark.parametrize("method", ("PUT", "PATCH", "DELETE"))
def test_recording_and_replay_support_mutating_http_methods(method: str) -> None:
    sdk_call: Final = _method_sdk_call(method)
    with _controlled_upstream() as upstream:
        responses: Final = record_upstream_responses(
            UpstreamEndpoint(base_url=upstream.url),
            _case(method),
            sdk_call,
        )
    with replay_server() as provider:
        provider.enqueue_response(responses[0])
        sdk_call(provider.url, _case(method))
        requests: Final = provider.take_requests(1)

    assert requests[0].method == method


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
            UpstreamEndpoint(base_url=upstream.url),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("sync", "async"))
@pytest.mark.parametrize("failed", (False, True), ids=("completed", "application-error"))
async def test_typed_stream_recording_cassette_replay_parity(
    tmp_path: Path, mode: Literal["sync", "async"], failed: bool
) -> None:
    case_input: Final = _case("typed-stream")
    outcomes: Final[queue.SimpleQueue[StreamOutcome]] = queue.SimpleQueue()

    def record(api_base: str, sdk_input: _FixtureInput) -> None:
        outcome: Final = (
            consume_sync_stream(lambda: _sync_events(api_base, sdk_input))
            if mode == "sync"
            else asyncio.run(_consume_async_events(api_base, sdk_input))
        )
        outcomes.put(outcome)

    with _controlled_upstream(_event_chunks(failed)) as upstream:
        target: Final = RecordingTarget(
            name="stream",
            upstream=UpstreamEndpoint(upstream.url),
            strategy=st.just(case_input),
            invocation=_Invocation(record),
        )
        summary: Final = record_fixtures((target,), tmp_path, 1, 1, _ParityCase)

    assert summary.failed == ()
    assert len(summary.recorded) == 1
    recorded: Final = outcomes.get_nowait()
    loaded: Final = load_fixture(tmp_path / "stream", case_input, _ParityCase)
    assert loaded is not None
    response: Final = loaded.provider_responses[0]
    assert isinstance(response, RecordedHttpStreamResponse)
    assert response.status_code == 200
    wire_bytes: Final = b"".join(chunk.data_bytes() for chunk in response.chunks)
    assert wire_bytes == b"".join(_event_chunks(failed))
    coalesced: Final = response.model_copy(update={"chunks": (RecordedStreamChunk.from_bytes(wire_bytes),)})

    with replay_server() as provider:
        first: Final = await _replay_events(mode, provider, response, case_input)
        second: Final = await _replay_events(mode, provider, coalesced, case_input)
    assert_request_parity(first.requests, second.requests)
    assert len(first.requests) == 1
    assert first.requests[0].body == {"id": case_input.identifier}
    assert_stream_parity(recorded, first.response)
    assert_stream_parity(first.response, second.response)
    expected: Final = (_StreamEvent(kind="delta", value="café"), _StreamEvent(kind="delta", value="second"))
    assert first.response.chunks == (expected if failed else (*expected, _StreamEvent(kind="done", value="")))
    if failed:
        assert isinstance(first.response.terminal, StreamFailed)
        assert first.response.terminal.phase == "iteration"
        assert first.response.terminal.exception_type is _StreamApplicationError
        assert first.response.terminal.error.code == "invalid_input"
        assert first.response.terminal.error.message == "invalid input"
    else:
        assert first.response.terminal == StreamCompleted()
