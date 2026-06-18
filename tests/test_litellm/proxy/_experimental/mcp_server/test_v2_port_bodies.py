"""Unit tests for the v2 port bodies (imperative-shell adapters).

Covers the client_credentials fetcher's grant shape, token parsing, and error mapping.
"""

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server import v2_port_bodies
from litellm.proxy._experimental.mcp_server.v2_port_bodies import (
    HttpxClientCredentialsFetcher,
)
from litellm.proxy.gateway.mcp.outbound_credentials.types import ClientCredentialsConfig
from litellm.proxy.gateway.mcp.result import Error, Ok

pytestmark = pytest.mark.asyncio


class _FakeResponse:
    def __init__(self, status_code, json_body):
        self.status_code = status_code
        self._json = json_body

    def json(self):
        return self._json


class _FakeClient:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.posted = None

    async def post(self, url, data=None):
        self.posted = {"url": url, "data": data}
        if self._exc is not None:
            raise self._exc
        return self._response


def _cfg(scopes=()):
    return ClientCredentialsConfig(
        client_id="cid",
        client_secret=SecretStr("secret"),
        token_url="https://idp/token",
        scopes=tuple(scopes),
    )


def _patch(monkeypatch, response=None, exc=None):
    fake = _FakeClient(response=response, exc=exc)
    monkeypatch.setattr(v2_port_bodies, "get_async_httpx_client", lambda **kw: fake)
    return fake


async def test_fetch_success_builds_stored_token(monkeypatch):
    fake = _patch(
        monkeypatch, _FakeResponse(200, {"access_token": "tok-123", "expires_in": 3600})
    )
    result = await HttpxClientCredentialsFetcher().fetch(_cfg(scopes=["a", "b"]))
    assert isinstance(result, Ok)
    assert result.ok.access_token.get_secret_value() == "tok-123"
    # grant goes in the POST body, scope is space-joined (mirrors v1)
    assert fake.posted["url"] == "https://idp/token"
    assert fake.posted["data"]["grant_type"] == "client_credentials"
    assert fake.posted["data"]["client_id"] == "cid"
    assert fake.posted["data"]["client_secret"] == "secret"
    assert fake.posted["data"]["scope"] == "a b"


async def test_fetch_omits_scope_when_none(monkeypatch):
    fake = _patch(monkeypatch, _FakeResponse(200, {"access_token": "t"}))
    await HttpxClientCredentialsFetcher().fetch(_cfg())
    assert "scope" not in fake.posted["data"]


async def test_fetch_rejected_grant_is_misconfigured(monkeypatch):
    _patch(monkeypatch, _FakeResponse(400, {"error": "invalid_client"}))
    result = await HttpxClientCredentialsFetcher().fetch(_cfg())
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"


async def test_fetch_server_error_is_upstream_unavailable(monkeypatch):
    _patch(monkeypatch, _FakeResponse(503, {}))
    result = await HttpxClientCredentialsFetcher().fetch(_cfg())
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


async def test_fetch_network_error_is_upstream_unavailable(monkeypatch):
    _patch(monkeypatch, exc=httpx.ConnectError("boom"))
    result = await HttpxClientCredentialsFetcher().fetch(_cfg())
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


async def test_fetch_missing_access_token_is_misconfigured(monkeypatch):
    _patch(monkeypatch, _FakeResponse(200, {"token_type": "bearer"}))
    result = await HttpxClientCredentialsFetcher().fetch(_cfg())
    assert isinstance(result, Error)
    assert result.error.tag == "misconfigured"
