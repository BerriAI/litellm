from __future__ import annotations

import asyncio
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


@pytest.mark.parametrize(
    "auth_type, auth_value, expected",
    [
        ("bearer_token", "tok", {"Authorization": "Bearer tok"}),
        ("api_key", "tok", {"Authorization": "ApiKey tok"}),
        ("basic", "tok", {"Authorization": "Basic tok"}),
        ("token", "tok", {"Authorization": "token tok"}),
        ("oauth2", "tok", {}),
        ("bearer_token", None, {}),
        (None, "tok", {}),
    ],
)
def test_build_openapi_auth_headers(auth_type, auth_value, expected) -> None:
    assert gen.build_openapi_auth_headers(auth_type, auth_value) == expected


def test_load_openapi_spec_forwards_auth_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a spec hosted behind the same Bearer credential as its
    endpoints must be fetched WITH the Authorization header, otherwise the
    upstream returns 401 before the server can be added."""
    url = "http://example.local/openapi.json"
    expected: Dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "Protected API", "version": "1.0.0"},
        "paths": {},
    }
    req = httpx.Request("GET", url)
    resp = httpx.Response(status_code=200, json=expected, request=req)

    monkeypatch.setattr(
        gen,
        "get_async_httpx_client",
        lambda *a, **k: _FakeAsyncHTTPHandler(resp, expected_url=url),
    )

    seen: Dict[str, Any] = {}

    async def fake_async_safe_get(client, request_url, **kw):
        seen["headers"] = kw.get("headers")
        return await client.get(request_url)

    monkeypatch.setattr(gen, "async_safe_get", fake_async_safe_get)

    auth_headers = gen.build_openapi_auth_headers("bearer_token", "secret-token")
    spec = asyncio.run(gen.load_openapi_spec_async(url, headers=auth_headers))

    assert spec == expected
    assert seen["headers"] == {"Authorization": "Bearer secret-token"}
