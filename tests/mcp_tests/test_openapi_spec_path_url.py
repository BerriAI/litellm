import httpx
import pytest

from litellm.proxy._experimental.mcp_server.openapi_to_mcp_generator import (
    load_openapi_spec,
)


def test_load_openapi_spec_supports_http_url(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "http://example.local/openapi.json"
    expected = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {},
    }

    def fake_get(request_url: str, timeout: float = 30.0):
        assert request_url == url
        return httpx.Response(status_code=200, json=expected)

    # Patch httpx.get used by load_openapi_spec
    monkeypatch.setattr(httpx, "get", fake_get)

    spec = load_openapi_spec(url)
    assert spec == expected

