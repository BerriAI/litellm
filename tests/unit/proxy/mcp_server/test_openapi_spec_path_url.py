from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path
from typing import Any, Dict, Final

import httpx
import pytest
import yaml
from respx import MockRouter

from litellm.proxy.mcp_server import openapi_to_mcp_generator as gen


class _FakeAsyncHTTPHandler:
    """
    Minimal stand-in for the object returned by get_async_httpx_client().
    openapi_to_mcp_generator.load_openapi_spec_async() calls:

        client = get_async_httpx_client(...)
        r = await client.get(url, timeout=30.0)

    So we must implement async get().
    """

    def __init__(self, response: httpx.Response, expected_url: str):
        self._response = response
        self._expected_url = expected_url
        self.calls = 0

    async def get(self, request_url: str, timeout: float = 30.0):
        self.calls += 1
        assert request_url == self._expected_url
        assert timeout == 30.0
        return self._response


def test_load_openapi_spec_supports_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "http://example.local/openapi.json"
    expected: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {},
    }

    # httpx.Response must include a Request for raise_for_status() to work.
    req = httpx.Request("GET", url)
    resp = httpx.Response(status_code=200, json=expected, request=req)

    calls = {"get_async_httpx_client": 0}
    handler_holder: Dict[str, Any] = {}

    def fake_get_async_httpx_client(*args, **kwargs):
        calls["get_async_httpx_client"] += 1
        h = _FakeAsyncHTTPHandler(resp, expected_url=url)
        handler_holder["handler"] = h
        return h

    # Ensure shared/custom client path is used
    monkeypatch.setattr(gen, "get_async_httpx_client", fake_get_async_httpx_client)

    # Bypass SSRF validation in test (example.local doesn't resolve)
    monkeypatch.setattr(
        gen, "async_safe_get", lambda client, url, **kw: client.get(url)
    )

    # Fail loudly if someone reintroduces direct httpx.get()
    def boom(*args, **kwargs):
        raise AssertionError("Direct httpx.get() must not be used for URL spec loading")

    monkeypatch.setattr(httpx, "get", boom)

    spec = gen.load_openapi_spec(url)

    assert spec == expected
    assert calls["get_async_httpx_client"] == 1
    assert handler_holder["handler"].calls == 1


def test_load_openapi_spec_supports_local_file_path(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "Local API", "version": "1.0.0"},
        "paths": {},
    }

    p = tmp_path / "openapi.json"
    p.write_text(
        '{"openapi":"3.0.0","info":{"title":"Local API","version":"1.0.0"},"paths":{}}',
        encoding="utf-8",
    )

    # For local files, shared client must NOT be used.
    def boom_client(*args, **kwargs):
        raise AssertionError(
            "get_async_httpx_client() must not be called for local file paths"
        )

    monkeypatch.setattr(gen, "get_async_httpx_client", boom_client)

    spec = gen.load_openapi_spec(str(p))
    assert spec == expected


@pytest.mark.parametrize("source", ["url", "file"])
@pytest.mark.parametrize("suffix", [".yaml", ".json", ""])
@pytest.mark.parametrize("document_format", ["json", "yaml"])
@pytest.mark.parametrize("use_async", [False, True])
def test_openapi_formats_preserve_document_across_sources(
    tmp_path: Path, respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch,
    source: str, suffix: str, document_format: str, use_async: bool,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    expected: Final = {
        "openapi": "3.0.0",
        "paths": {"/ping": {"get": {
            "operationId": "on",
            "parameters": [
                {"name": "on", "in": "query", "schema": {"type": "string"}},
                {"name": "off", "in": "query", "schema": {"type": "string"}},
            ],
        }}},
        "x-values": [100.0, "on", "off", "yes", "no", "2026-01-01", "1:20", True, False, None, 12, 10, 10],
    }
    content: Final = (
        json.dumps(expected).replace("100.0", "1e2")
        if document_format == "json"
        else "openapi: 3.0.0\npaths:\n  /ping:\n    get:\n      operationId: on\n"
        "      parameters:\n"
        "        - {name: on, in: query, schema: {type: string}}\n"
        "        - {name: off, in: query, schema: {type: string}}\n"
        "x-values: [1e2, on, off, yes, no, 2026-01-01, 1:20, TRUE, False, null, 012, 0o12, 0xA]\n"
    )
    path: Final = tmp_path / ("spec" + suffix)
    path.write_text(content, encoding="utf-8")
    target: Final = str(path) if source == "file" else "https://93.184.216.34/spec" + suffix
    if source == "url":
        respx_mock.get(target).respond(200, text=content, headers={"content-type": "text/plain"})

    spec: Final = asyncio.run(gen.load_openapi_spec_async(target)) if use_async else gen.load_openapi_spec(target)

    assert spec == expected
    operation: Final = spec["paths"]["/ping"]["get"]
    assert gen.sanitize_openapi_tool_name(operation["operationId"]) == "on"
    assert gen.build_input_schema(operation) == {
        "type": "object",
        "properties": {"on": {"type": "string", "description": ""}, "off": {"type": "string", "description": ""}},
        "required": [],
    }


_LONG_YAML_INTEGER: Final = "1" * 1025


@pytest.mark.parametrize("content,message", [
    ("", "OpenAPI spec must be a JSON or YAML object"),
    ("null", "OpenAPI spec must be a JSON or YAML object"),
    ("[]", "OpenAPI spec must be a JSON or YAML object"),
    ("- item", "OpenAPI spec must be a JSON or YAML object"),
    ("plain text", "OpenAPI spec must be a JSON or YAML object"),
    ("paths: [", "Invalid YAML OpenAPI spec"),
    ("paths: {}\nx: \x01", "Invalid YAML OpenAPI spec"),
    ("paths: " + "[" * 600 + "0" + "]" * 600, "Invalid YAML OpenAPI spec"),
    ("x: !!python/object/apply:builtins.str [unsafe]", "Invalid YAML OpenAPI spec"),
    ("x: &a []\ny: *a", "YAML aliases are not supported in OpenAPI specs"),
    ("x: " + _LONG_YAML_INTEGER, "YAML integer is too long in OpenAPI spec"),
    ("x: !!int " + _LONG_YAML_INTEGER, "YAML integer is too long in OpenAPI spec"),
    ("x: !!int {=: '" + _LONG_YAML_INTEGER + "'}", "Invalid YAML OpenAPI spec"),
    ("x: !!int [1]", "Invalid YAML OpenAPI spec"),
    ("x: !!bool invalid", "Invalid YAML OpenAPI spec"),
    ("x: !!bool []", "Invalid YAML OpenAPI spec"),
    ("x: !!str []", "Invalid YAML OpenAPI spec"),
    ("x: !!int ''", "Invalid YAML OpenAPI spec"),
    ("x: !!float ''", "Invalid YAML OpenAPI spec"),
    ("x: !!float {=: 1.2}", "Invalid YAML OpenAPI spec"),
    ("x: !!timestamp invalid", "Invalid YAML OpenAPI spec"),
    ("x: !!timestamp 2026-01-01", "Invalid YAML OpenAPI spec"),
    ("x: !!set {a: null}", "Invalid YAML OpenAPI spec"),
    ("x: !!binary YQ==", "Invalid YAML OpenAPI spec"),
    ("x: {1: value}", "Invalid YAML OpenAPI spec"),
    ("x: {[key]: value}", "Invalid YAML OpenAPI spec"),
    ("x: .inf", "Invalid YAML OpenAPI spec"),
    ("x: .nan", "Invalid YAML OpenAPI spec"),
    ("x: 1e999", "Invalid YAML OpenAPI spec"),
], ids=[
    "empty", "null", "json-array", "yaml-sequence", "scalar", "syntax", "reader", "nesting",
    "python-tag", "alias", "implicit-int", "explicit-int", "mapping-int", "sequence-int",
    "invalid-bool", "sequence-bool", "sequence-str", "empty-int", "empty-float", "mapping-float", "invalid-timestamp",
    "timestamp", "set", "binary", "non-string-key", "complex-key", "infinity", "nan", "float-overflow",
])
def test_openapi_loader_rejects_invalid_or_unsafe_documents(
    tmp_path: Path, content: str, message: str,
) -> None:
    path: Final = tmp_path / "spec.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="^" + message + "$"):
        gen.load_openapi_spec(str(path))


def test_openapi_yaml_integer_boundary_preserves_quoted_strings(tmp_path: Path) -> None:
    integer: Final = "1" * 1024
    path: Final = tmp_path / "spec.yaml"
    path.write_text("x-integer: " + integer + "\nx-string: '" + integer + "1'", encoding="utf-8")

    assert gen.load_openapi_spec(str(path)) == {"x-integer": int(integer), "x-string": integer + "1"}


def test_openapi_json_integer_preserved_even_with_yaml_extension(tmp_path: Path) -> None:
    path: Final = tmp_path / "spec.yaml"
    path.write_text('{"x-integer":' + _LONG_YAML_INTEGER + "}", encoding="utf-8")

    assert gen.load_openapi_spec(str(path)) == {"x-integer": int(_LONG_YAML_INTEGER)}


@pytest.mark.parametrize("value", ["! " + _LONG_YAML_INTEGER, "! '" + _LONG_YAML_INTEGER + "'"], ids=["plain", "quoted"])
def test_openapi_nonspecific_tag_disables_scalar_resolution(tmp_path: Path, value: str) -> None:
    path: Final = tmp_path / "spec.yaml"
    path.write_text("x: " + value, encoding="utf-8")

    assert gen.load_openapi_spec(str(path)) == {"x": _LONG_YAML_INTEGER}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["url", "file"])
async def test_openapi_yaml_parses_off_the_event_loop(
    tmp_path: Path, respx_mock: MockRouter, monkeypatch: pytest.MonkeyPatch, source: str,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    event_loop_thread: Final = threading.get_ident()
    parser_threads: Final[list[int]] = []
    original_load: Final = yaml.load

    def observed_load(stream: str | bytes, Loader: type[yaml.SafeLoader]) -> object:
        parser_threads.append(threading.get_ident())
        return original_load(stream, Loader=Loader)

    monkeypatch.setattr(yaml, "load", observed_load)
    path: Final = tmp_path / "spec.yaml"
    path.write_text("paths: {}", encoding="utf-8")
    target: Final = str(path) if source == "file" else "https://93.184.216.34/spec.yaml"
    if source == "url":
        respx_mock.get(target).respond(200, text="paths: {}")

    assert await gen.load_openapi_spec_async(target) == {"paths": {}}
    assert len(parser_threads) == 1 and parser_threads[0] != event_loop_thread
