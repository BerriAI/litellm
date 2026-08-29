from __future__ import annotations

import hashlib
import os
import queue
import threading
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, Generic, Protocol, TypeVar, cast

import httpx
import pytest
from hypothesis import given, settings
from hypothesis.strategies import SearchStrategy
from pydantic import AwareDatetime, BaseModel, ConfigDict, ValidationError

from tests.test_litellm._json_fs_cache import JsonFileCache, canonical_json
from tests.test_litellm._recorded_http import (
    HttpHeader,
    RecordedHttpResponse,
    RecordedHttpStreamResponse,
    RecordedResponse,
    RecordedStreamChunk,
)

FIXTURE_SCHEMA_VERSION: Final = 1

_HOP_BY_HOP_HEADERS: Final = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class FixtureInput(Protocol):
    def canonical_input(self) -> dict[str, object]: ...


InputT = TypeVar("InputT", bound=FixtureInput)
CaseT = TypeVar("CaseT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    upstream_base: str


@dataclass(frozen=True, slots=True)
class RecorderResult(Generic[CaseT]):
    case: CaseT
    cache_hit: bool


class FixtureEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    recorded_at: AwareDatetime
    case: dict[str, object]


def _excluded_headers(headers: tuple[tuple[str, str], ...]) -> frozenset[str]:
    connection_values: Final = tuple(value for name, value in headers if name.lower() == "connection")
    connection_headers: Final = frozenset(
        token.strip().lower() for value in connection_values for token in value.split(",") if token.strip()
    )
    return _HOP_BY_HOP_HEADERS | connection_headers


def _end_to_end_headers(headers: httpx.Headers) -> tuple[HttpHeader, ...]:
    decoded: Final = tuple((name.decode("ascii"), value.decode("latin-1")) for name, value in headers.raw)
    excluded: Final = _excluded_headers(decoded) | {"content-encoding", "content-length"}
    return tuple(HttpHeader(name=name, value=value) for name, value in decoded if name.lower() not in excluded)


class _RecordingProvider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, spec: ProviderSpec) -> None:
        super().__init__(("127.0.0.1", 0), _RecordingHandler)
        self.spec: Final = spec
        self.responses: queue.Queue[RecordedResponse] = queue.Queue()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def take_response(self) -> RecordedResponse:
        try:
            return self.responses.get(timeout=5)
        except queue.Empty as error:
            raise RuntimeError("successful SDK call did not produce a recorded response") from error


class _RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, _RecordingProvider)
        length: Final = int(self.headers.get("content-length") or "0")
        request_body: Final = self.rfile.read(length)
        raw_headers: Final = tuple(self.headers.raw_items())
        excluded: Final = _excluded_headers(raw_headers) | {"host", "content-length"}
        forwarded_headers: Final = tuple((name, value) for name, value in raw_headers if name.lower() not in excluded)
        upstream_url: Final = f"{provider.spec.upstream_base.rstrip('/')}{self.path}"

        try:
            with httpx.stream(
                self.command,
                upstream_url,
                headers=forwarded_headers,
                content=request_body,
                timeout=120,
            ) as upstream:
                headers: Final = _end_to_end_headers(upstream.headers)
                recorded_response: Final = self._record_upstream_response(upstream, headers)
        except httpx.HTTPError as error:
            self._send_response(502, (), str(error).encode("utf-8"))
            return

        if 200 <= recorded_response.status_code < 300:
            provider.responses.put(recorded_response)
        if isinstance(recorded_response, RecordedHttpResponse):
            self._send_response(recorded_response.status_code, recorded_response.headers, recorded_response.body_bytes())

    def _record_upstream_response(
        self,
        upstream: httpx.Response,
        headers: tuple[HttpHeader, ...],
    ) -> RecordedResponse:
        content_type: Final = cast(str, upstream.headers.get("content-type", ""))
        if content_type.lower().startswith("text/event-stream"):
            return self._record_stream(upstream, headers)
        response_body: Final = b"".join(upstream.iter_bytes())
        return RecordedHttpResponse.from_bytes(
            status_code=upstream.status_code,
            headers=headers,
            body=response_body,
        )

    def _record_stream(
        self,
        upstream: httpx.Response,
        headers: tuple[HttpHeader, ...],
    ) -> RecordedHttpStreamResponse:
        self.send_response_only(upstream.status_code)
        for header in headers:
            self.send_header(header.name, header.value)
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        chunks: Final = tuple(self._relay_chunks(upstream.iter_bytes()))
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        return RecordedHttpStreamResponse(
            kind="http_stream",
            status_code=upstream.status_code,
            headers=headers,
            chunks=chunks,
        )

    def _relay_chunks(self, chunks: Iterable[bytes]) -> Generator[RecordedStreamChunk, None, None]:
        for chunk in chunks:
            self.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
            self.wfile.write(chunk)
            self.wfile.write(b"\r\n")
            self.wfile.flush()
            yield RecordedStreamChunk.from_bytes(chunk)

    def _send_response(self, status_code: int, headers: tuple[HttpHeader, ...], body: bytes) -> None:
        self.send_response_only(status_code)
        for header in headers:
            self.send_header(header.name, header.value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _recording_provider(spec: ProviderSpec) -> Generator[_RecordingProvider]:
    server: Final = _RecordingProvider(spec)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def generate_case_inputs(strategy: SearchStrategy[InputT], examples: int) -> tuple[InputT, ...]:
    generated: Final[queue.SimpleQueue[InputT | None]] = queue.SimpleQueue()

    @settings(max_examples=examples, deadline=None, derandomize=True)
    @given(case_input=strategy)
    def generate_case(case_input: InputT) -> None:
        generated.put(case_input)

    generate_case()
    generated.put(None)
    return tuple(iter(generated.get, None))


def fixture_cache_key(case_input: FixtureInput) -> dict[str, object]:
    return case_input.canonical_input()


def _load_fixture(raw_fixture: dict[str, object], path: Path, case_type: type[CaseT]) -> CaseT:
    schema_version: Final = raw_fixture.get("schema_version")
    if schema_version != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"fixture {path} has schema_version {schema_version!r}, expected {FIXTURE_SCHEMA_VERSION}; "
            "delete it and regenerate the fixture bundle"
        )
    try:
        envelope: Final = FixtureEnvelope.model_validate(raw_fixture)
        return case_type.model_validate(envelope.case)
    except ValidationError as error:
        raise ValueError(f"invalid parity fixture {path} ({len(error.errors())} validation errors)") from error


def record_case(
    spec: ProviderSpec,
    root: Path,
    case_input: InputT,
    sdk_call: Callable[[str, InputT], object],
    case_type: type[CaseT],
) -> RecorderResult[CaseT]:
    cache: Final = JsonFileCache(root)
    cache_key: Final = fixture_cache_key(case_input)
    cached: Final = cache.get(cache_key)
    if cached is not None:
        return RecorderResult(case=_load_fixture(cached, cache.path_for(cache_key), case_type), cache_hit=True)

    with _recording_provider(spec) as recorder:
        sdk_call(recorder.url, case_input)
        upstream_response: Final = recorder.take_response()

    case: Final = case_type.model_validate({"litellm_input": case_input, "provider_response": upstream_response})
    envelope: Final = FixtureEnvelope(
        schema_version=FIXTURE_SCHEMA_VERSION,
        recorded_at=datetime.now(timezone.utc),
        case=cast(dict[str, object], case.model_dump(mode="json", exclude_unset=True)),
    )
    cache.put(cache_key, cast(dict[str, object], envelope.model_dump(mode="json", exclude_unset=True)))
    return RecorderResult(case=case, cache_hit=False)


def record_cases(
    spec: ProviderSpec,
    root: Path,
    case_inputs: tuple[InputT, ...],
    sdk_call: Callable[[str, InputT], object],
    case_type: type[CaseT],
    max_concurrency: int,
) -> tuple[RecorderResult[CaseT], ...]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    unique_inputs: Final = tuple(
        {canonical_json(fixture_cache_key(case_input)): case_input for case_input in case_inputs}.values()
    )
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: Final = tuple(
            executor.submit(record_case, spec, root, case_input, sdk_call, case_type) for case_input in unique_inputs
        )
        return tuple(future.result() for future in futures)


def fixture_directory(configured: Path | None, env_value: str | None, default: Path) -> Path:
    return (configured or Path(env_value or default)).expanduser()


def recorded_fixtures(directory: Path, case_type: type[CaseT]) -> tuple[CaseT, ...]:
    cache: Final = JsonFileCache(directory)
    return tuple(_load_fixture(raw_fixture, path, case_type) for path, raw_fixture in cache.values_with_paths())


def fixture_id(case_input: FixtureInput, prefix: str) -> str:
    input_json: Final = canonical_json(case_input.canonical_input())
    digest: Final = hashlib.sha256(input_json.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{digest}"


def parametrize_recorded_fixtures(
    metafunc: pytest.Metafunc,
    *,
    fixture_name: str,
    case_type: type[CaseT],
    env_var: str,
    default_directory: Path,
    regeneration_command: str,
    id_builder: Callable[[CaseT], str],
) -> None:
    if fixture_name not in metafunc.fixturenames:
        return
    configured: Final = os.environ.get(env_var)
    if configured == "":
        raise pytest.UsageError(f"{env_var} is set but empty")
    directory: Final = Path(configured).expanduser() if configured is not None else default_directory
    try:
        fixtures: Final = recorded_fixtures(directory, case_type)
    except (ValidationError, ValueError) as error:
        raise pytest.UsageError(
            f"Invalid parity fixture bundle at {directory}. "
            "Each fixture must use the current versioned envelope. "
            f"Record fresh fixtures in an empty directory with: `{regeneration_command}`. "
            f"Validation details: {error}"
        ) from error
    if fixtures:
        metafunc.parametrize(fixture_name, fixtures, ids=tuple(id_builder(fixture) for fixture in fixtures))
        return
    if configured is not None:
        raise pytest.UsageError(f"no recorded fixtures in {directory}")
    metafunc.parametrize(
        fixture_name,
        (
            pytest.param(
                None,
                marks=pytest.mark.skip(reason=f"no recorded fixtures in {directory}"),
                id="no-recorded-fixtures",
            ),
        ),
    )
