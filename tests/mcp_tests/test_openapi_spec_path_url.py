from __future__ import annotations

from typing import Any, Dict

import httpx
import pytest

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


def test_load_openapi_spec_supports_yaml_file(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YAML OpenAPI spec files should be parsed correctly."""
    expected: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "YAML API", "version": "1.0.0"},
        "paths": {},
    }

    p = tmp_path / "openapi.yaml"
    p.write_text(
        "openapi: '3.0.0'\ninfo:\n  title: YAML API\n  version: '1.0.0'\npaths: {}",
        encoding="utf-8",
    )

    def boom_client(*args, **kwargs):
        raise AssertionError("get_async_httpx_client() must not be called for local file paths")

    monkeypatch.setattr(gen, "get_async_httpx_client", boom_client)

    spec = gen.load_openapi_spec(str(p))
    assert spec == expected


def test_load_openapi_spec_url_yaml_content_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL returning YAML with Content-Type: text/yaml should be parsed as YAML."""
    url = "http://example.local/openapi.yaml"
    yaml_body = "openapi: '3.0.0'\ninfo:\n  title: Remote YAML\n  version: '1.0.0'\npaths: {}"

    req = httpx.Request("GET", url)
    resp = httpx.Response(
        status_code=200,
        content=yaml_body.encode(),
        headers={"content-type": "text/yaml"},
        request=req,
    )

    handler_holder: Dict[str, Any] = {}

    def fake_get_async_httpx_client(*args, **kwargs):
        h = _FakeAsyncHTTPHandler(resp, expected_url=url)
        handler_holder["handler"] = h
        return h

    monkeypatch.setattr(gen, "get_async_httpx_client", fake_get_async_httpx_client)
    monkeypatch.setattr(gen, "async_safe_get", lambda client, url, **kw: client.get(url))

    spec = gen.load_openapi_spec(url)
    assert spec["info"]["title"] == "Remote YAML"


def test_load_openapi_spec_url_yaml_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """URL with .yaml extension but no YAML Content-Type should still parse as YAML."""
    url = "https://raw.githubusercontent.com/firefly-iii/api-docs/refs/heads/v6.6.6/dist/firefly-iii-v6.6.6-v1.yaml"
    yaml_body = "openapi: '3.0.0'\ninfo:\n  title: Firefly\n  version: '6.6.6'\npaths: {}"

    req = httpx.Request("GET", url)
    resp = httpx.Response(
        status_code=200,
        content=yaml_body.encode(),
        headers={"content-type": "text/plain"},
        request=req,
    )

    handler_holder: Dict[str, Any] = {}

    def fake_get_async_httpx_client(*args, **kwargs):
        h = _FakeAsyncHTTPHandler(resp, expected_url=url)
        handler_holder["handler"] = h
        return h

    monkeypatch.setattr(gen, "get_async_httpx_client", fake_get_async_httpx_client)
    monkeypatch.setattr(gen, "async_safe_get", lambda client, url, **kw: client.get(url))

    spec = gen.load_openapi_spec(url)
    assert spec["info"]["title"] == "Firefly"
