from __future__ import annotations

import json
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Final, cast

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.mistral.ocr.transformation import MistralOCRConfig
from tests.test_litellm._fixture_recorder import ProviderSpec, fixture_cache_key, record_case, recorded_fixtures
from tests.test_litellm._json_fs_cache import JsonFileCache
from tests.test_litellm.ocr.fixture_models import (
    DocumentUrlDocument,
    ImageUrlDocument,
    MistralOcrParityInput,
    OcrParityCase,
)

_UPSTREAM_BODY: Final = b'{"pages":[],"model":"mistral-ocr-latest","usage_info":{"pages_processed":0}}\n'


class _Upstream(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _UpstreamHandler)
        self.requests: list[tuple[tuple[tuple[str, str], ...], bytes]] = []

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        upstream: Final = self.server
        assert isinstance(upstream, _Upstream)
        body: Final = self.rfile.read(int(self.headers.get("content-length") or "0"))
        upstream.requests.append((tuple(self.headers.raw_items()), body))
        self.send_response_only(200)
        self.send_header("content-type", "application/json")
        self.send_header("set-cookie", "first=1")
        self.send_header("set-cookie", "second=2")
        self.send_header("connection", "keep-alive")
        self.send_header("content-length", str(len(_UPSTREAM_BODY)))
        self.end_headers()
        self.wfile.write(_UPSTREAM_BODY)

    def log_message(self, format: str, *args: object) -> None:
        return


@contextmanager
def _upstream() -> Generator[_Upstream]:
    server: Final = _Upstream()
    thread: Final = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _case_input(model: str = "mistral/mistral-ocr-latest") -> MistralOcrParityInput:
    return MistralOcrParityInput(
        model=model,
        document=ImageUrlDocument(type="image_url", image_url="https://example.test/image.png"),
    )


def _sdk_call(**kwargs: object) -> None:
    api_base: Final = kwargs["api_base"]
    assert isinstance(api_base, str)
    raw_body: Final = b'{ "document" : {"type":"image_url"}, "model" : "wire-model" }'
    response: Final = httpx.post(
        f"{api_base}/v1/ocr",
        content=raw_body,
        headers=[("content-type", "application/json"), ("x-repeat", "one"), ("x-repeat", "two")],
    )
    assert response.content == _UPSTREAM_BODY
    assert response.headers.get_list("set-cookie") == ["first=1", "second=2"]
    response.raise_for_status()


def test_case_schema_has_exact_fields_rejects_extras_and_preserves_null() -> None:
    case_input: Final = MistralOcrParityInput(
        model="mistral/mistral-ocr-latest",
        document=DocumentUrlDocument(type="document_url", document_url="https://example.test/file.pdf"),
        pages=None,
    )
    raw_case: Final[dict[str, object]] = {
        "input": case_input.model_dump(mode="json", exclude_unset=True),
        "upstream_response": {"kind": "http", "status_code": 200, "headers": [], "body_b64": "e30="},
    }

    case: Final = OcrParityCase.model_validate(raw_case)

    assert set(case.model_dump(mode="json", exclude_unset=True)) == {"input", "upstream_response"}
    assert case.input.as_sdk_kwargs()["pages"] is None
    assert "include_blocks" not in case.input.as_sdk_kwargs()
    with pytest.raises(ValidationError):
        OcrParityCase.model_validate({**raw_case, "provider_request": {}})
    with pytest.raises(ValidationError):
        MistralOcrParityInput.model_validate({**case_input.canonical_input(), "unknown": True})


def test_typed_input_fields_match_supported_mistral_params() -> None:
    input_fields: Final = frozenset(MistralOcrParityInput.model_fields) - {"model", "document"}

    supported_params: Final = cast(
        list[str],
        MistralOCRConfig().get_supported_ocr_params(  # pyright: ignore[reportUnknownMemberType]  # legacy API returns an unparameterized list
            model="mistral-ocr-latest"
        ),
    )

    assert input_fields == frozenset(supported_params)


def test_record_case_proxies_raw_request_and_roundtrips_response_bytes_and_headers(tmp_path: Path) -> None:
    with _upstream() as upstream:
        spec: Final = ProviderSpec(
            model="mistral/mistral-ocr-latest",
            upstream_base=upstream.url,
            api_key="test-key",
        )
        result: Final = record_case(spec, tmp_path, _case_input(), cast(Callable[..., object], _sdk_call))

    assert not result.cache_hit
    assert result.case.upstream_response.body_bytes() == _UPSTREAM_BODY
    assert tuple((header.name, header.value) for header in result.case.upstream_response.headers) == (
        ("content-type", "application/json"),
        ("set-cookie", "first=1"),
        ("set-cookie", "second=2"),
    )
    assert len(upstream.requests) == 1
    request_headers, request_body = upstream.requests[0]
    assert request_body == b'{ "document" : {"type":"image_url"}, "model" : "wire-model" }'
    assert tuple(value for name, value in request_headers if name.lower() == "x-repeat") == ("one", "two")
    stored: Final = JsonFileCache(tmp_path).values()
    assert len(stored) == 1
    assert set(stored[0]) == {"input", "upstream_response"}
    assert "provider_request" not in json.dumps(stored[0])
    assert recorded_fixtures(tmp_path) == (result.case,)


def test_cache_identity_uses_only_canonical_unified_input(tmp_path: Path) -> None:
    case_input: Final = _case_input()
    key: Final = fixture_cache_key(case_input)
    assert key == case_input.model_dump(mode="json", exclude_unset=True)

    cache: Final = JsonFileCache(tmp_path)
    cache.put(
        key,
        {
            "input": key,
            "upstream_response": {"kind": "http", "status_code": 200, "headers": [], "body_b64": "e30="},
        },
    )
    unreachable: Final = ProviderSpec(model=case_input.model, upstream_base="http://127.0.0.1:1", api_key="key")

    result: Final = record_case(unreachable, tmp_path, case_input, cast(Callable[..., object], _sdk_call))

    assert result.cache_hit
