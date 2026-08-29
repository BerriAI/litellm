from __future__ import annotations

import argparse
import hashlib
import queue
import threading
from collections.abc import Callable, Generator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast

import httpx
from pydantic import ValidationError

from tests.test_litellm._json_fs_cache import JsonFileCache, canonical_json
from tests.test_litellm.ocr.fixture_models import (
    HttpHeader,
    MistralOcrParityInput,
    OcrParityCase,
    RecordedHttpResponse,
)

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


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    model: str
    upstream_base: str
    api_key: str


@dataclass(frozen=True, slots=True)
class RecorderResult:
    case: OcrParityCase
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class GeneratorArgs:
    concurrency: int
    examples: int
    fixture_dir: Path | None
    model: str


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
        self.responses: queue.Queue[RecordedHttpResponse] = queue.Queue()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def take_response(self) -> RecordedHttpResponse:
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
                response_body: Final = b"".join(upstream.iter_bytes())
                recorded_response: Final = RecordedHttpResponse.from_bytes(
                    status_code=upstream.status_code,
                    headers=_end_to_end_headers(upstream.headers),
                    body=response_body,
                )
        except httpx.HTTPError as error:
            self._send_response(502, (), str(error).encode("utf-8"))
            return

        if 200 <= recorded_response.status_code < 300:
            provider.responses.put(recorded_response)
        self._send_recorded_response(recorded_response)

    def _send_recorded_response(self, response: RecordedHttpResponse) -> None:
        self._send_response(response.status_code, response.headers, response.body_bytes())

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


def fixture_cache_key(case_input: MistralOcrParityInput) -> dict[str, object]:
    return case_input.canonical_input()


def record_case(
    spec: ProviderSpec,
    root: Path,
    case_input: MistralOcrParityInput,
    sdk_call: Callable[..., object],
) -> RecorderResult:
    cache: Final = JsonFileCache(root)
    cache_key: Final = fixture_cache_key(case_input)
    cached: Final = cache.get(cache_key)
    if cached is not None:
        return RecorderResult(case=OcrParityCase.model_validate(cached), cache_hit=True)

    with _recording_provider(spec) as recorder:
        sdk_call(api_base=recorder.url, api_key=spec.api_key, **case_input.as_sdk_kwargs())
        upstream_response: Final = recorder.take_response()

    case: Final = OcrParityCase(litellm_input=case_input, provider_response=upstream_response)
    cache.put(cache_key, cast(dict[str, object], case.model_dump(mode="json", exclude_unset=True)))
    return RecorderResult(case=case, cache_hit=False)


def record_cases(
    spec: ProviderSpec,
    root: Path,
    case_inputs: tuple[MistralOcrParityInput, ...],
    sdk_call: Callable[..., object],
    max_concurrency: int,
) -> tuple[RecorderResult, ...]:
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1")
    unique_inputs: Final = tuple(
        {canonical_json(fixture_cache_key(case_input)): case_input for case_input in case_inputs}.values()
    )
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures: Final = tuple(
            executor.submit(record_case, spec, root, case_input, sdk_call) for case_input in unique_inputs
        )
        return tuple(future.result() for future in futures)


def parse_generator_args() -> GeneratorArgs:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--fixture-dir", type=Path)
    parser.add_argument("--model", default="mistral/mistral-ocr-latest")
    namespace: Final = parser.parse_args()
    return GeneratorArgs(
        concurrency=cast(int, namespace.concurrency),
        examples=cast(int, namespace.examples),
        fixture_dir=cast(Path | None, namespace.fixture_dir),
        model=cast(str, namespace.model),
    )


def fixture_directory(configured: Path | None, env_value: str | None, default: Path) -> Path:
    return (configured or Path(env_value or default)).expanduser()


def recorded_fixtures(directory: Path) -> tuple[OcrParityCase, ...]:
    cache: Final = JsonFileCache(directory)
    fixtures: list[OcrParityCase] = []
    for path, raw_fixture in cache.values_with_paths():
        try:
            fixtures.append(OcrParityCase.model_validate(raw_fixture))
        except ValidationError as error:
            raise ValueError(
                f"invalid OCR parity fixture {path}: expected exactly `litellm_input` and `provider_response` "
                f"({len(error.errors())} validation errors)"
            ) from error
    return tuple(fixtures)


def fixture_id(fixture: OcrParityCase) -> str:
    input_json: Final = canonical_json(fixture.litellm_input.canonical_input())
    digest: Final = hashlib.sha256(input_json.encode("utf-8")).hexdigest()[:8]
    return f"mistral-{fixture.litellm_input.model.rsplit('/', 1)[-1]}-{digest}"
