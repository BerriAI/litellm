import socket

import pytest

import litellm
from litellm.litellm_core_utils import url_utils
from litellm.proxy.guardrails.guardrail_hooks.custom_code import primitives


class _FakeResponse:
    def __init__(self, status=200, body=None, location=None):
        self.status_code = status
        self._body = body if body is not None else {"ok": True}
        self.headers = {"location": location} if location else {}
        self.is_redirect = 300 <= status < 400

    def json(self):
        return self._body

    @property
    def text(self):
        return str(self._body)


class _RecordingHandler:
    def __init__(self, recorder, response=None):
        self.client = _RecordingClient(recorder, response)


class _RecordingClient:
    def __init__(self, recorder, response=None):
        self._recorder = recorder
        self._response = response or _FakeResponse()

    async def request(self, method, url, headers=None, follow_redirects=False, **kw):
        self._recorder.append(
            {"method": method, "url": url, "host": (headers or {}).get("Host"), "kw": kw}
        )
        return self._response


def _patch_dns(monkeypatch, ip):
    def fake(host, port, *a, **kw):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]

    monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)


@pytest.fixture(autouse=True)
def _enable_validation(monkeypatch):
    monkeypatch.setattr(litellm, "user_url_validation", True)


class TestHttpRequestSSRF:
    """Regression for #32889: http_request() must not reach internal targets."""

    async def test_blocks_loopback(self, monkeypatch):
        _patch_dns(monkeypatch, "127.0.0.1")
        recorder = []
        monkeypatch.setattr(
            primitives, "get_async_httpx_client", lambda **kw: _RecordingHandler(recorder)
        )

        result = await primitives.http_request("http://internal.example.com/secret")

        assert result["success"] is False
        assert "Blocked" in (result["error"] or "")
        assert recorder == []

    async def test_blocks_cloud_metadata(self, monkeypatch):
        _patch_dns(monkeypatch, "169.254.169.254")
        recorder = []
        monkeypatch.setattr(
            primitives, "get_async_httpx_client", lambda **kw: _RecordingHandler(recorder)
        )

        result = await primitives.http_post(
            "http://metadata.example.com/latest/meta-data/", body={"x": 1}
        )

        assert result["success"] is False
        assert "Blocked" in (result["error"] or "")
        assert recorder == []

    async def test_allows_public_host(self, monkeypatch):
        _patch_dns(monkeypatch, "93.184.216.34")
        recorder = []
        monkeypatch.setattr(
            primitives, "get_async_httpx_client", lambda **kw: _RecordingHandler(recorder)
        )

        result = await primitives.http_request(
            "http://example.com/api", method="POST", body={"a": 1}
        )

        assert result["success"] is True
        assert len(recorder) == 1
        assert recorder[0]["method"] == "POST"
        assert recorder[0]["host"] == "example.com"
        assert "93.184.216.34" in recorder[0]["url"]
        assert recorder[0]["kw"]["json"] == {"a": 1}

    async def test_redirect_to_internal_is_blocked(self, monkeypatch):
        resolved = {"example.com": "93.184.216.34", "evil.example.com": "127.0.0.1"}

        def fake(host, port, *a, **kw):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (resolved[host], port))]

        monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake)

        redirect = _FakeResponse(302, location="http://evil.example.com/internal")
        recorder = []
        monkeypatch.setattr(
            primitives,
            "get_async_httpx_client",
            lambda **kw: _RecordingHandler(recorder, response=redirect),
        )

        result = await primitives.http_get("http://example.com/start")

        assert result["success"] is False
        assert "Blocked" in (result["error"] or "")
