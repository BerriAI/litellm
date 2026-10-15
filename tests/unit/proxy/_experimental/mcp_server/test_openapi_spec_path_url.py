from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Final

import httpx
import pytest
from respx import MockRouter

from litellm.proxy._experimental.mcp_server import openapi_to_mcp_generator as gen


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
    monkeypatch.setattr(gen, "async_safe_get", lambda client, url, **kw: client.get(url))

    # Fail loudly if someone reintroduces direct httpx.get()
    def boom(*args, **kwargs):
        raise AssertionError("Direct httpx.get() must not be used for URL spec loading")

    monkeypatch.setattr(httpx, "get", boom)

    spec = gen.load_openapi_spec(url)

    assert spec == expected
    assert calls["get_async_httpx_client"] == 1
    assert handler_holder["handler"].calls == 1


def test_load_openapi_spec_supports_local_file_path(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        raise AssertionError("get_async_httpx_client() must not be called for local file paths")

    monkeypatch.setattr(gen, "get_async_httpx_client", boom_client)

    spec = gen.load_openapi_spec(str(p))
    assert spec == expected


_OPENAPI_YAML: Final = """openapi: 3.0.3
info:
  title: Example API
  version: 1.0.0
paths:
  /status:
    get:
      operationId: getStatus
      responses:
        '200':
          description: OK
x-enabled: true
x-limit: 100
"""
_OPENAPI_EXPECTED: Final = {
    "openapi": "3.0.3",
    "info": {"title": "Example API", "version": "1.0.0"},
    "paths": {"/status": {"get": {"operationId": "getStatus", "responses": {"200": {"description": "OK"}}}}},
    "x-enabled": True,
    "x-limit": 100,
}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix,content_type",
    (
        (".yaml", "application/yaml"),
        (".yml", "text/yaml"),
        (".yaml", "binary/octet-stream"),
        ("", "text/plain"),
        (".json?download=1", "application/json"),
    ),
)
async def test_remote_yaml_preserves_openapi_document(
    respx_mock: MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    suffix: str,
    content_type: str,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    url: Final = f"https://93.184.216.34/openapi{suffix}"
    respx_mock.get(url).respond(200, text=_OPENAPI_YAML, headers={"content-type": content_type})

    assert await gen.load_openapi_spec_async(url) == _OPENAPI_EXPECTED


@pytest.mark.parametrize("suffix", (".yaml", ".yml", ".json"))
def test_local_yaml_preserves_openapi_document(tmp_path: Path, suffix: str) -> None:
    path: Final = tmp_path / f"openapi{suffix}"
    path.write_text(_OPENAPI_YAML, encoding="utf-8")

    assert gen.load_openapi_spec(str(path)) == _OPENAPI_EXPECTED


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ("utf-8", "utf-16"))
async def test_remote_json_preserves_json_scalar_types(
    respx_mock: MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    url: Final = "https://93.184.216.34/openapi.yaml"
    respx_mock.get(url).respond(200, content=b'{"paths":{},"x-number":1e3,"x-boolean":true}'.decode().encode(encoding))

    assert await gen.load_openapi_spec_async(url) == {"paths": {}, "x-number": 1000.0, "x-boolean": True}


@pytest.mark.parametrize("document", ("", "null", "true", "[]", "plain text", "- item"))
def test_non_object_spec_is_rejected(tmp_path: Path, document: str) -> None:
    path: Final = tmp_path / "openapi.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(ValueError, match="OpenAPI spec must be a JSON or YAML object"):
        gen.load_openapi_spec(str(path))


@pytest.mark.parametrize(
    "document,error_type,message",
    (
        ("openapi: [unclosed", ValueError, "Invalid YAML OpenAPI spec"),
        ("openapi: 3.0.0\npaths: \x00", ValueError, "Invalid YAML OpenAPI spec"),
        ("paths: !!python/object/apply:builtins.eval ['1 + 1']", ValueError, "Invalid YAML OpenAPI spec"),
        ("paths: &paths {self: *paths}", TypeError, "YAML aliases are not supported"),
        ("base: &base {a: 1}\npaths: {<<: *base}", TypeError, "YAML aliases are not supported"),
        ("x-value: 1" + ":11" * 400, ValueError, "YAML integer is too long"),
        ("x-value: !!int '" + "1" * 1025 + "'", ValueError, "YAML integer is too long"),
        ("x-value: ! 1" + ":11" * 400, ValueError, "YAML integer is too long"),
        ("x-value: ! '1" + ":11" * 400 + "'", ValueError, "YAML integer is too long"),
        ("x-value: !!int {=: '1" + ":11" * 400 + "'}", ValueError, "YAML integer must be a scalar"),
        ("x-value: !!int [1]", ValueError, "YAML integer must be a scalar"),
    ),
)
def test_unsafe_or_invalid_yaml_is_rejected(
    tmp_path: Path, document: str, error_type: type[Exception], message: str
) -> None:
    path: Final = tmp_path / "openapi.yaml"
    path.write_text(document, encoding="utf-8")

    with pytest.raises(error_type, match=message):
        gen.load_openapi_spec(str(path))


@pytest.mark.parametrize("value", ("1", "! 1", "!!int '1'"))
def test_yaml_integer_scalar_types_are_preserved(tmp_path: Path, value: str) -> None:
    path: Final = tmp_path / "openapi.yaml"
    path.write_text(f"paths: {{}}\nx-value: {value}", encoding="utf-8")

    assert gen.load_openapi_spec(str(path)) == {"paths": {}, "x-value": 1}


@pytest.mark.parametrize("tag", ("", "!!str "))
def test_long_yaml_strings_are_preserved(tmp_path: Path, tag: str) -> None:
    path: Final = tmp_path / "openapi.yaml"
    value: Final = "1" + ":11" * 400
    path.write_text(f"paths: {{}}\nx-value: {tag}'{value}'", encoding="utf-8")

    assert gen.load_openapi_spec(str(path)) == {"paths": {}, "x-value": value}


@pytest.mark.asyncio
async def test_remote_yaml_preserves_http_failure(
    respx_mock: MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
    url: Final = "https://93.184.216.34/openapi.yaml"
    respx_mock.get(url).respond(403, text=_OPENAPI_YAML)

    with pytest.raises(httpx.HTTPStatusError) as error:
        await gen.load_openapi_spec_async(url)

    assert error.value.response.status_code == 403
