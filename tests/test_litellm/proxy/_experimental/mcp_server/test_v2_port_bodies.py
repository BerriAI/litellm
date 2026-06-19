"""Unit tests for the v2 port bodies (imperative-shell adapters).

Covers the client_credentials fetcher's grant shape, token parsing, and error mapping.
"""

import httpx
import pytest
from pydantic import SecretStr

from litellm.proxy._experimental.mcp_server import v2_port_bodies
from litellm.proxy._experimental.mcp_server.v2_port_bodies import (
    HttpxClientCredentialsFetcher,
    HttpxSigV4Signer,
    V1ByokCredentialStore,
    _classify_sigv4_error,
)
from litellm.proxy.gateway.mcp.outbound_credentials.credential_store import (
    CredentialKey,
)
from litellm.proxy.gateway.mcp.outbound_credentials.types import (
    AwsSigV4Config,
    ClientCredentialsConfig,
    StaticKeys,
)
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


def _sigv4_cfg(session_token=None, region="us-east-1", service="bedrock-agentcore"):
    return AwsSigV4Config(
        region=region,
        service=service,
        credentials=StaticKeys(
            access_key_id="AKIATEST",
            secret_access_key=SecretStr("secret"),
            session_token=SecretStr(session_token) if session_token else None,
        ),
    )


async def test_sigv4_static_keys_signs_request():
    result = await HttpxSigV4Signer().build(_sigv4_cfg())
    assert isinstance(result, Ok)
    req = httpx.Request(
        "POST", "https://svc.us-east-1.amazonaws.com/mcp", content=b"{}"
    )
    signed = next(result.ok.auth_flow(req))
    assert signed.headers["Authorization"].startswith(
        "AWS4-HMAC-SHA256 Credential=AKIATEST/"
    )
    assert "X-Amz-Date" in signed.headers
    assert "X-Amz-Security-Token" not in signed.headers


async def test_sigv4_session_token_adds_security_token():
    result = await HttpxSigV4Signer().build(_sigv4_cfg(session_token="tok"))
    assert isinstance(result, Ok)
    req = httpx.Request("GET", "https://svc.us-east-1.amazonaws.com/mcp")
    signed = next(result.ok.auth_flow(req))
    assert "X-Amz-Security-Token" in signed.headers


async def test_sigv4_region_and_service_in_credential_scope():
    result = await HttpxSigV4Signer().build(_sigv4_cfg(region="eu-west-1"))
    assert isinstance(result, Ok)
    req = httpx.Request("GET", "https://svc.eu-west-1.amazonaws.com/mcp")
    signed = next(result.ok.auth_flow(req))
    assert (
        "/eu-west-1/bedrock-agentcore/aws4_request" in signed.headers["Authorization"]
    )


async def test_classify_sigv4_connection_error_is_upstream_unavailable():
    from botocore.exceptions import EndpointConnectionError

    err = EndpointConnectionError(endpoint_url="https://sts.amazonaws.com")
    assert _classify_sigv4_error(err).tag == "upstream_unavailable"


async def test_classify_sigv4_other_error_is_misconfigured():
    assert _classify_sigv4_error(ValueError("no creds")).tag == "misconfigured"


def _cred_key(subject_id="u1", server_id="s1"):
    return CredentialKey(tenant_id="org1", subject_id=subject_id, server_id=server_id)


async def test_byok_store_returns_user_credential():
    async def reader(subject_id, server_id):
        assert (subject_id, server_id) == ("u1", "s1")
        return "user-byok-key"

    result = await V1ByokCredentialStore(reader=reader).get(_cred_key())
    assert isinstance(result, Ok)
    assert result.ok == "user-byok-key"


async def test_byok_store_missing_credential_is_ok_none():
    async def reader(subject_id, server_id):
        return None

    result = await V1ByokCredentialStore(reader=reader).get(_cred_key())
    assert isinstance(result, Ok)
    assert result.ok is None


async def test_byok_store_empty_subject_skips_the_store():
    async def reader(subject_id, server_id):
        raise AssertionError("store must not be queried for an empty subject")

    result = await V1ByokCredentialStore(reader=reader).get(_cred_key(subject_id=""))
    assert isinstance(result, Ok)
    assert result.ok is None


async def test_byok_store_db_error_is_upstream_unavailable():
    async def reader(subject_id, server_id):
        raise RuntimeError("db down")

    result = await V1ByokCredentialStore(reader=reader).get(_cred_key())
    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
