from __future__ import annotations

from typing import Any

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
    expected: dict[str, Any] = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0.0"},
        "paths": {},
    }

    # httpx.Response must include a Request for raise_for_status() to work.
    req = httpx.Request("GET", url)
    resp = httpx.Response(status_code=200, json=expected, request=req)

    calls = {"get_async_httpx_client": 0}
    handler_holder: dict[str, Any] = {}

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
    expected: dict[str, Any] = {
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


def test_load_openapi_spec_supports_yaml_file(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """YAML OpenAPI spec files should be parsed correctly."""
    expected: dict[str, Any] = {
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

    handler_holder: dict[str, Any] = {}

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

    handler_holder: dict[str, Any] = {}

    def fake_get_async_httpx_client(*args, **kwargs):
        h = _FakeAsyncHTTPHandler(resp, expected_url=url)
        handler_holder["handler"] = h
        return h

    monkeypatch.setattr(gen, "get_async_httpx_client", fake_get_async_httpx_client)
    monkeypatch.setattr(gen, "async_safe_get", lambda client, url, **kw: client.get(url))

    spec = gen.load_openapi_spec(url)
    assert spec["info"]["title"] == "Firefly"


def test_load_openapi_spec_url_plain_text_body_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 body that is neither JSON nor a YAML mapping must raise, not return a string."""
    url = "https://example.local/openapi.json"

    req = httpx.Request("GET", url)
    resp = httpx.Response(
        status_code=200,
        content=b"secret invalid JSON body",
        headers={"content-type": "text/plain"},
        request=req,
    )

    def fake_get_async_httpx_client(*args, **kwargs):
        return _FakeAsyncHTTPHandler(resp, expected_url=url)

    monkeypatch.setattr(gen, "get_async_httpx_client", fake_get_async_httpx_client)
    monkeypatch.setattr(gen, "async_safe_get", lambda client, url, **kw: client.get(url))

    with pytest.raises(ValueError, match="Expecting value"):
        gen.load_openapi_spec(url)


def test_load_openapi_spec_invalid_yaml_raises(tmp_path) -> None:
    """Invalid YAML in a .yaml file must raise ValueError, not crash."""
    p = tmp_path / "openapi.yaml"
    p.write_text("openapi: [unclosed\n  bad: indent", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML OpenAPI spec"):
        gen.load_openapi_spec(str(p))


def test_load_openapi_spec_non_mapping_yaml_raises(tmp_path) -> None:
    """A YAML document that is not a mapping must raise TypeError."""
    p = tmp_path / "openapi.yaml"
    p.write_text("just a plain string", encoding="utf-8")
    with pytest.raises(TypeError, match="expected a JSON/YAML mapping"):
        gen.load_openapi_spec(str(p))


def test_load_openapi_spec_missing_file_raises(tmp_path) -> None:
    """A missing local spec file must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        gen.load_openapi_spec(str(tmp_path / "does-not-exist.yaml"))


def test_load_openapi_spec_json_file_with_yaml_content(tmp_path) -> None:
    """A .json file containing YAML falls back to the YAML parser."""
    p = tmp_path / "openapi.json"
    p.write_text(
        "openapi: '3.0.0'\ninfo:\n  title: YAML in JSON clothing\n  version: '1.0.0'\npaths: {}",
        encoding="utf-8",
    )
    spec = gen.load_openapi_spec(str(p))
    assert spec["info"]["title"] == "YAML in JSON clothing"


def test_import_yaml_missing_raises_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without PyYAML installed the error must say how to fix it."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "yaml":
            raise ImportError("No module named 'yaml'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="pip install pyyaml"):
        gen._import_yaml()
