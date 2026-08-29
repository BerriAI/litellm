from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import queue
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast
from urllib.parse import quote

import httpx
from dotenv import load_dotenv
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import TypeAdapter

import litellm
from tests.test_litellm._json_fs_cache import JsonFileCache
from tests.test_litellm.ocr.fixture_models import (
    OcrFixture,
    OcrFixtureRequest,
    OcrFixtureResponse,
    ProviderWireRequest,
)

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
JSON_OBJECT: Final = TypeAdapter(dict[str, object])
PROVIDER_NAMES: Final = ("mistral", "azure_ai", "vertex_ai")
LOGGER: Final = logging.getLogger(__name__)
_TEXT: Final = st.from_regex(r"[A-Za-z0-9 ]{1,24}", fullmatch=True)
_OPTIONS: Final = st.fixed_dictionaries(
    {
        "pages": st.just([0]),
        "include_image_base64": st.booleans(),
        "image_limit": st.integers(min_value=1, max_value=4),
        "image_min_size": st.integers(min_value=0, max_value=64),
        "extract_header": st.booleans(),
        "extract_footer": st.booleans(),
        "table_format": st.sampled_from(("markdown", "html")),
        "confidence_scores_granularity": st.sampled_from(("word", "page")),
        "include_blocks": st.booleans(),
        "id": _TEXT,
    }
)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    name: str
    model: str
    upstream_base: str
    api_key: str | None
    upstream_model: str | None = None
    vertex_project: str | None = None
    vertex_location: str | None = None


@dataclass(frozen=True, slots=True)
class RecorderResult:
    fixture: OcrFixture
    cache_hit: bool


@dataclass(frozen=True, slots=True)
class GeneratorArgs:
    providers: tuple[str, ...]
    examples: int
    fixture_dir: Path | None


class _RecordingProvider(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        spec: ProviderSpec,
        sdk_kwargs: dict[str, object],
        cache: JsonFileCache,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _RecordingHandler)
        self.spec: Final = spec
        self.sdk_kwargs: Final = sdk_kwargs
        self.cache: Final = cache
        self.results: queue.Queue[RecorderResult] = queue.Queue()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"

    def take_result(self) -> RecorderResult:
        try:
            return self.results.get(timeout=5)
        except queue.Empty as error:
            raise RuntimeError("successful OCR call did not produce a recorder result") from error


class _RecordingHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        provider: Final = self.server
        assert isinstance(provider, _RecordingProvider)
        length: Final = int(self.headers.get("content-length") or "0")
        body: Final = JSON_OBJECT.validate_json(self.rfile.read(length))
        fixture_request: Final = OcrFixtureRequest(
            provider=provider.spec.name,
            sdk_kwargs=provider.sdk_kwargs,
            provider_request=ProviderWireRequest(method=self.command, path=self.path, body=body),
        )
        cache_key: Final = _fixture_cache_key(provider.spec.name, fixture_request.provider_request)
        cached_value: Final = provider.cache.get(cache_key)
        if cached_value is not None:
            cached_fixture: Final = OcrFixture.model_validate(cached_value)
            provider.results.put(RecorderResult(fixture=cached_fixture, cache_hit=True))
            self._send_fixture_response(cached_fixture.response)
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
        fixture_response: Final = OcrFixtureResponse(
            status_code=upstream_response.status_code,
            headers=response_headers,
            body=upstream_response_body,
        )
        recorded_fixture: Final = OcrFixture(request=fixture_request, response=fixture_response)
        provider.results.put(RecorderResult(fixture=recorded_fixture, cache_hit=False))
        self._send_fixture_response(fixture_response)

    def _send_fixture_response(self, response: OcrFixtureResponse) -> None:
        response_body: Final = json.dumps(response.body, separators=(",", ":")).encode()
        self._send_response(response.status_code, response.headers, response_body)

    def _send_response(self, status_code: int, headers: dict[str, str], body: bytes) -> None:
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
) -> Generator[_RecordingProvider]:
    server: Final = _RecordingProvider(spec=spec, sdk_kwargs=sdk_kwargs, cache=cache)
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _mistral_upstream_base() -> str:
    configured: Final = os.environ.get("MISTRAL_API_BASE", "https://api.mistral.ai").rstrip("/")
    return configured.removesuffix("/v1")


def _fixture_cache_key(provider: str, request: ProviderWireRequest) -> dict[str, object]:
    return {"provider": provider, "request": request.model_dump(mode="json")}


def _provider_specs(selected: tuple[str, ...]) -> tuple[ProviderSpec, ...]:
    specs: Final[dict[str, ProviderSpec | None]] = {
        "mistral": (
            ProviderSpec(
                name="mistral",
                model=os.environ.get("MISTRAL_OCR_MODEL", "mistral/mistral-ocr-latest"),
                upstream_base=_mistral_upstream_base(),
                api_key=os.environ.get("MISTRAL_API_KEY") or os.environ.get("LITELLM_API_KEY"),
                upstream_model=os.environ.get("MISTRAL_OCR_UPSTREAM_MODEL"),
            )
            if os.environ.get("MISTRAL_API_KEY") or os.environ.get("LITELLM_API_KEY")
            else None
        ),
        "azure_ai": (
            ProviderSpec(
                name="azure_ai",
                model=os.environ.get("AZURE_AI_OCR_MODEL", "azure_ai/mistral-document-ai-2512"),
                upstream_base=os.environ["AZURE_AI_API_BASE"],
                api_key=os.environ["AZURE_AI_API_KEY"],
            )
            if os.environ.get("AZURE_AI_API_BASE") and os.environ.get("AZURE_AI_API_KEY")
            else None
        ),
        "vertex_ai": _vertex_spec(),
    }
    missing: Final = tuple(name for name in selected if specs[name] is None)
    if missing:
        LOGGER.warning("Skipping providers without credentials: %s", ", ".join(missing))
    return tuple(spec for name in selected if (spec := specs[name]) is not None)


def _vertex_spec() -> ProviderSpec | None:
    project: Final = os.environ.get("VERTEXAI_PROJECT")
    credentials_available: Final = bool(os.environ.get("VERTEX_AI_API_KEY") or os.environ.get("VERTEXAI_CREDENTIALS"))
    if project is None or not credentials_available:
        return None
    location: Final = os.environ.get("VERTEXAI_LOCATION", os.environ.get("VERTEX_LOCATION", "us-central1"))
    upstream_base: Final = os.environ.get("VERTEX_AI_API_BASE", f"https://{location}-aiplatform.googleapis.com")
    return ProviderSpec(
        name="vertex_ai",
        model=os.environ.get("VERTEX_AI_OCR_MODEL", "vertex_ai/mistral-ocr-2505"),
        upstream_base=upstream_base,
        api_key=os.environ.get("VERTEX_AI_API_KEY"),
        vertex_project=project,
        vertex_location=location,
    )


def _image_data_uri(text: str, font_size: int) -> str:
    url: Final = f"https://dummyjson.com/image/800x300/ffffff/000000?text={quote(text)}&fontSize={font_size}"
    response: Final = httpx.get(url, timeout=30, follow_redirects=True)
    response.raise_for_status()
    raw_content_type: Final = cast(object, response.headers.get("content-type", "image/png"))
    content_type: Final = raw_content_type if isinstance(raw_content_type, str) else "image/png"
    media_type: Final = content_type.split(";", 1)[0]
    encoded: Final = base64.b64encode(response.content).decode("ascii")
    return f"data:{media_type};base64,{encoded}"


def _sdk_kwargs(
    spec: ProviderSpec,
    image_data_uri: str,
    options: dict[str, object],
) -> dict[str, object]:
    provider_kwargs: Final = (
        {"vertex_project": spec.vertex_project, "vertex_location": spec.vertex_location}
        if spec.name == "vertex_ai"
        else {}
    )
    return {
        "model": spec.model,
        "document": {"type": "image_url", "image_url": image_data_uri},
        **options,
        **provider_kwargs,
    }


def _record_case(spec: ProviderSpec, root: Path, sdk_kwargs: dict[str, object]) -> RecorderResult:
    cache: Final = JsonFileCache(root / spec.name)
    with _recording_provider(spec=spec, sdk_kwargs=sdk_kwargs, cache=cache) as recorder:
        ocr_call: Final = cast(Callable[..., object], litellm.ocr)
        ocr_call(api_base=recorder.url, api_key=spec.api_key, **sdk_kwargs)
        result: Final = recorder.take_result()

    if not result.cache_hit:
        cache.put(
            _fixture_cache_key(spec.name, result.fixture.request.provider_request),
            result.fixture.model_dump(mode="json"),
        )
    return result


def _generate(specs: tuple[ProviderSpec, ...], root: Path, examples: int) -> None:
    @settings(max_examples=examples, deadline=None, derandomize=True)
    @given(text=_TEXT, font_size=st.integers(min_value=12, max_value=36), options=_OPTIONS)
    def generate_case(text: str, font_size: int, options: dict[str, object]) -> None:
        image_data_uri: Final = _image_data_uri(text, font_size)
        for spec in specs:
            _generate_provider_case(spec, root, image_data_uri, options)

    generate_case()


def _generate_provider_case(
    spec: ProviderSpec,
    root: Path,
    image_data_uri: str,
    options: dict[str, object],
) -> None:
    sdk_kwargs: Final = _sdk_kwargs(spec, image_data_uri, options)
    result: Final = _record_case(spec, root, sdk_kwargs)
    state: Final = "cached" if result.cache_hit else "recorded"
    LOGGER.info("%s %s %s", state, spec.name, result.fixture.request.provider_request.path)


def _parse_args() -> GeneratorArgs:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--provider", action="append", choices=PROVIDER_NAMES)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--fixture-dir", type=Path)
    namespace: Final = parser.parse_args()
    providers: Final = cast(list[str] | None, namespace.provider)
    return GeneratorArgs(
        providers=tuple(providers) if providers else PROVIDER_NAMES,
        examples=cast(int, namespace.examples),
        fixture_dir=cast(Path | None, namespace.fixture_dir),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv()
    args: Final = _parse_args()
    root: Final = (
        args.fixture_dir or Path(os.environ.get(FIXTURE_DIR_ENV, Path(__file__).with_name(".fixtures"))).expanduser()
    )
    specs: Final = _provider_specs(args.providers)
    if not specs:
        raise SystemExit("No selected provider has the required credentials")
    _generate(specs, root, args.examples)


if __name__ == "__main__":
    main()
