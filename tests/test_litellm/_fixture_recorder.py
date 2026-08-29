from __future__ import annotations

import argparse
import hashlib
import json
import queue
import threading
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter

from tests.test_litellm._json_fs_cache import JsonFileCache, canonical_json

JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class ProviderWireRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    path: str
    body: dict[str, object]


class FixtureRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str
    sdk_kwargs: dict[str, object]
    provider_request: ProviderWireRequest


class FixtureResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    status_code: int
    headers: dict[str, str]
    body: dict[str, object]


class Fixture(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: FixtureRequest
    response: FixtureResponse


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    name: str
    model: str
    upstream_base: str
    api_key: str | None
    upstream_model: str | None = None


@dataclass(frozen=True, slots=True)
class RecorderResult:
    request: FixtureRequest
    response: FixtureResponse | None
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class GeneratorArgs:
    providers: tuple[str, ...]
    examples: int
    fixture_dir: Path | None
    requests_only: bool
    responses_only: bool


class _RecordingProvider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        spec: ProviderSpec,
        sdk_kwargs: dict[str, object],
        cache: JsonFileCache,
        requests_only: bool,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _RecordingHandler)
        self.spec: Final = spec
        self.sdk_kwargs: Final = sdk_kwargs
        self.cache: Final = cache
        self.requests_only: Final = requests_only
        self.results: queue.Queue[RecorderResult] = queue.Queue()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def take_result(self) -> RecorderResult:
        try:
            return self.results.get(timeout=5)
        except queue.Empty as error:
            raise RuntimeError("successful SDK call did not produce a recorder result") from error


class _RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, _RecordingProvider)
        length: Final = int(self.headers.get("content-length") or "0")
        body: Final = JSON_OBJECT.validate_json(self.rfile.read(length))
        fixture_request: Final = FixtureRequest(
            provider=provider.spec.name,
            sdk_kwargs=provider.sdk_kwargs,
            provider_request=ProviderWireRequest(method=self.command, path=self.path, body=body),
        )
        cache_key: Final = fixture_cache_key(
            provider.spec.name,
            fixture_request.sdk_kwargs,
            fixture_request.provider_request,
        )
        cached_value: Final = provider.cache.get(cache_key)
        if cached_value is not None:
            cached_request: Final = FixtureRequest.model_validate(cached_value["request"])
            raw_cached_response: Final = cached_value.get("response")
            if raw_cached_response is not None:
                cached_response: Final = FixtureResponse.model_validate(raw_cached_response)
                provider.results.put(RecorderResult(request=cached_request, response=cached_response, cache_hit=True))
                self._send_fixture_response(cached_response)
                return
            if provider.requests_only:
                provider.results.put(RecorderResult(request=cached_request, response=None, cache_hit=True))
                self._send_response(200, {"content-type": "application/json"}, b"{}")
                return

        if provider.requests_only:
            provider.results.put(RecorderResult(request=fixture_request, response=None, cache_hit=False))
            self._send_response(200, {"content-type": "application/json"}, b"{}")
            return

        upstream_url: Final = f"{provider.spec.upstream_base.rstrip('/')}{self.path}"
        forwarded_headers: Final = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in {"host", "content-length", "accept-encoding", "x-parity-case"}
        }
        upstream_body: Final = (
            {**body, "model": provider.spec.upstream_model} if provider.spec.upstream_model is not None else body
        )
        try:
            upstream_response: Final = httpx.post(
                upstream_url,
                headers=forwarded_headers,
                content=json.dumps(upstream_body, separators=(",", ":")),
                timeout=120,
            )
        except httpx.HTTPError as error:
            error_body: Final = json.dumps({"error": str(error)}).encode()
            self._send_response(502, {"content-type": "application/json"}, error_body)
            return

        raw_content_type: Final = cast(object, upstream_response.headers.get("content-type", "application/json"))
        content_type: Final = raw_content_type if isinstance(raw_content_type, str) else "application/json"
        response_headers: Final = {"content-type": content_type.split(";", 1)[0]}
        if not upstream_response.is_success:
            self._send_response(upstream_response.status_code, response_headers, upstream_response.content)
            return

        upstream_response_body: Final = JSON_OBJECT.validate_json(upstream_response.content)
        fixture_response: Final = FixtureResponse(
            status_code=upstream_response.status_code,
            headers=response_headers,
            body=upstream_response_body,
        )
        provider.results.put(RecorderResult(request=fixture_request, response=fixture_response, cache_hit=False))
        self._send_fixture_response(fixture_response)

    def _send_fixture_response(self, response: FixtureResponse) -> None:
        response_body: Final = json.dumps(response.body, separators=(",", ":")).encode()
        self._send_response(response.status_code, response.headers, response_body)

    def _send_response(self, status_code: int, headers: Mapping[str, str], body: bytes) -> None:
        self.send_response(status_code)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _recording_provider(
    spec: ProviderSpec,
    sdk_kwargs: dict[str, object],
    cache: JsonFileCache,
    requests_only: bool,
) -> Generator[_RecordingProvider]:
    server: Final = _RecordingProvider(
        spec=spec,
        sdk_kwargs=sdk_kwargs,
        cache=cache,
        requests_only=requests_only,
    )
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def fixture_cache_key(
    provider: str,
    sdk_kwargs: dict[str, object],
    request: ProviderWireRequest,
) -> dict[str, object]:
    return {
        "provider": provider,
        "sdk_kwargs": sdk_kwargs,
        "request": request.model_dump(mode="json"),
    }


def record_case(
    spec: ProviderSpec,
    root: Path,
    sdk_kwargs: dict[str, object],
    requests_only: bool,
    sdk_call: Callable[..., object],
) -> RecorderResult:
    cache: Final = JsonFileCache(root / spec.name)
    with _recording_provider(
        spec=spec,
        sdk_kwargs=sdk_kwargs,
        cache=cache,
        requests_only=requests_only,
    ) as recorder:
        try:
            sdk_call(api_base=recorder.url, api_key=spec.api_key, **sdk_kwargs)
        except Exception:
            if not requests_only:
                raise
        result: Final = recorder.take_result()

    if not result.cache_hit:
        value: Final = (
            Fixture(request=result.request, response=result.response).model_dump(mode="json")
            if result.response is not None
            else {"request": result.request.model_dump(mode="json")}
        )
        cache.put(
            fixture_cache_key(spec.name, result.request.sdk_kwargs, result.request.provider_request),
            value,
        )
    return result


def pending_requests(cache: JsonFileCache) -> tuple[FixtureRequest, ...]:
    return tuple(
        FixtureRequest.model_validate(value["request"]) for value in cache.values() if value.get("response") is None
    )


def fill_missing_responses(
    specs: tuple[ProviderSpec, ...],
    root: Path,
    sdk_call: Callable[..., object],
) -> tuple[RecorderResult, ...]:
    return tuple(
        record_case(spec, root, request.sdk_kwargs, requests_only=False, sdk_call=sdk_call)
        for spec in specs
        for request in pending_requests(JsonFileCache(root / spec.name))
        if request.sdk_kwargs.get("model") == spec.model
    )


def parse_generator_args(provider_names: tuple[str, ...]) -> GeneratorArgs:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--provider", action="append", choices=provider_names)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--fixture-dir", type=Path)
    mode: Final = parser.add_mutually_exclusive_group()
    mode.add_argument("--requests-only", action="store_true", help="record deterministic requests without API calls")
    mode.add_argument("--responses-only", action="store_true", help="fill responses for saved pending requests")
    namespace: Final = parser.parse_args()
    providers: Final = cast(list[str] | None, namespace.provider)
    return GeneratorArgs(
        providers=tuple(providers) if providers else provider_names,
        examples=cast(int, namespace.examples),
        fixture_dir=cast(Path | None, namespace.fixture_dir),
        requests_only=cast(bool, namespace.requests_only),
        responses_only=cast(bool, namespace.responses_only),
    )


def fixture_directory(configured: Path | None, env_value: str | None, default: Path) -> Path:
    return (configured or Path(env_value or default)).expanduser()


def recorded_fixtures(directory: Path) -> tuple[Fixture, ...]:
    return tuple(
        Fixture.model_validate(raw_fixture)
        for raw_fixture in JsonFileCache(directory).values()
        if raw_fixture.get("response") is not None
    )


def fixture_id(fixture: Fixture) -> str:
    raw_model: Final = fixture.request.sdk_kwargs.get("model")
    model: Final = raw_model if isinstance(raw_model, str) else "unknown-model"
    request_json: Final = canonical_json(fixture.request.provider_request.model_dump(mode="json"))
    digest: Final = hashlib.sha256(request_json.encode("utf-8")).hexdigest()[:8]
    return f"{fixture.request.provider}-{model.rsplit('/', 1)[-1]}-{digest}"
