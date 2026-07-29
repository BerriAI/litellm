import socket

import pytest

from litellm.litellm_core_utils import url_utils
from litellm.proxy.guardrails.guardrail_hooks.custom_code import primitives


def _mock_dns(monkeypatch, ip: str):
    def fake_getaddrinfo(host, port, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 80))]

    monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake_getaddrinfo)


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, location=None):
        self.status_code = status_code
        self._json = json_body if json_body is not None else {"ok": True}
        self.headers = {"location": location} if location else {}
        self.is_redirect = 300 <= status_code < 400

    def json(self):
        return self._json

    @property
    def text(self):
        return str(self._json)


class _FakeRawClient:
    def __init__(self, response=None):
        self.response = response or _FakeResponse()
        self.calls = []

    async def request(self, method, url, headers=None, **kwargs):
        self.calls.append({"method": method, "url": url, "headers": headers or {}})
        return self.response


class _FakeHandler:
    def __init__(self, raw_client):
        self.client = raw_client


@pytest.mark.asyncio
async def test_http_request_blocks_cloud_metadata(monkeypatch):
    """Regression for the guardrail SSRF: http_request() must run the URL
    through validate_url() so the AWS/GCP/Azure metadata IP is rejected and
    no outbound request is ever issued."""
    _mock_dns(monkeypatch, "169.254.169.254")

    raw_client = _FakeRawClient()
    monkeypatch.setattr(
        primitives, "get_async_httpx_client", lambda **kw: _FakeHandler(raw_client)
    )

    result = await primitives.http_request(
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
    )

    assert result["success"] is False
    assert "SSRF" in result["error"]
    assert raw_client.calls == []


@pytest.mark.asyncio
async def test_http_request_blocks_private_ip(monkeypatch):
    _mock_dns(monkeypatch, "127.0.0.1")

    raw_client = _FakeRawClient()
    monkeypatch.setattr(
        primitives, "get_async_httpx_client", lambda **kw: _FakeHandler(raw_client)
    )

    result = await primitives.http_request("http://localhost:6379/", method="POST")

    assert result["success"] is False
    assert "SSRF" in result["error"]
    assert raw_client.calls == []


@pytest.mark.asyncio
async def test_http_request_does_not_follow_redirect_to_private(monkeypatch):
    """A public host that 302-redirects to a private target must be blocked at
    the redirect hop rather than followed into the internal network."""

    def fake_getaddrinfo(host, port, *args, **kwargs):
        ip = "93.184.216.34" if host == "public.example" else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 80))]

    monkeypatch.setattr(url_utils.socket, "getaddrinfo", fake_getaddrinfo)

    raw_client = _FakeRawClient(
        response=_FakeResponse(status_code=302, location="http://internal.example/")
    )
    monkeypatch.setattr(
        primitives, "get_async_httpx_client", lambda **kw: _FakeHandler(raw_client)
    )

    result = await primitives.http_request("http://public.example/start")

    assert result["success"] is False
    assert "SSRF" in result["error"]
    assert len(raw_client.calls) == 1


@pytest.mark.asyncio
async def test_http_request_allows_public_host(monkeypatch):
    """The happy path still works: a public host resolves, the validated request
    is issued to the resolved IP with the original Host header preserved."""
    _mock_dns(monkeypatch, "93.184.216.34")

    raw_client = _FakeRawClient(
        response=_FakeResponse(status_code=200, json_body={"verdict": "clean"})
    )
    monkeypatch.setattr(
        primitives, "get_async_httpx_client", lambda **kw: _FakeHandler(raw_client)
    )

    result = await primitives.http_request(
        "http://api.example.com/moderate",
        method="POST",
        body={"text": "hello"},
    )

    assert result["success"] is True
    assert result["status_code"] == 200
    assert result["body"] == {"verdict": "clean"}
    assert len(raw_client.calls) == 1
    call = raw_client.calls[0]
    assert call["method"] == "POST"
    assert "93.184.216.34" in call["url"]
    assert call["headers"].get("Host") == "api.example.com"
