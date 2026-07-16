"""Tests for the v2 token-endpoint collaborator.

`TokenEndpointClient.fetch` makes one authenticated POST and returns the minted token as a value;
`ExchangedTokenCache` memoizes it with per-key single-flight. These pin the grant/client-auth wire
shape, the private-key-JWT vs client_secret authentication, the error-as-value mapping, and the
cache's hit/single-flight behavior. Each assertion fails under a real mutation of the feature.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt
import litellm
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy._experimental.mcp_server.outbound_credentials.result import (
    Error,
    Ok,
    Result,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_endpoint import (
    CLIENT_ASSERTION_TYPE,
    ExchangedToken,
    ExchangedTokenCache,
    TokenEndpointClient,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.types import (
    ClientSecretAuth,
    CredError,
    PrivateKeyJwtAuth,
)
from pydantic import SecretStr

_PATCH_TARGET = (
    "litellm.proxy._experimental.mcp_server.outbound_credentials."
    "token_endpoint.get_async_httpx_client"
)

_ENDPOINT = "https://idp.example.com/oauth2/token"
_CLIENT_ID = "litellm-client-id"

_RSA_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PRIVATE_PEM = _RSA_KEY.private_bytes(
    serialization.Encoding.PEM,
    serialization.PrivateFormat.PKCS8,
    serialization.NoEncryption(),
).decode()
_PUBLIC_PEM = (
    _RSA_KEY.public_key()
    .public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    .decode()
)


def _resp(token="access", expires_in=3600):
    resp = MagicMock()
    resp.json.return_value = {"access_token": token, "expires_in": expires_in}
    resp.raise_for_status = MagicMock()
    return resp


def _client(response):
    client = AsyncMock()
    client.post.return_value = response
    return client


def _posted_data(client):
    return client.post.call_args.kwargs["data"]


@pytest.mark.asyncio
async def test_fetch_forwards_grant_params_and_client_secret():
    client = _client(_resp("the-token", expires_in=1200))
    with patch(_PATCH_TARGET, return_value=client):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g", "subject_token": "user-jwt"},
            ClientSecretAuth(client_secret=SecretStr("shhh")),
        )

    assert isinstance(result, Ok)
    assert result.ok == ExchangedToken(access_token="the-token", expires_in=1200)
    assert client.post.call_args.args[0] == _ENDPOINT
    data = _posted_data(client)
    assert data["grant_type"] == "g"
    assert data["subject_token"] == "user-jwt"
    assert data["client_id"] == _CLIENT_ID
    assert data["client_secret"] == "shhh"
    assert "client_assertion" not in data


@pytest.mark.asyncio
async def test_fetch_private_key_jwt_client_assertion():
    client = _client(_resp())
    with patch(_PATCH_TARGET, return_value=client):
        await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            PrivateKeyJwtAuth(
                private_key=SecretStr(_PRIVATE_PEM),
                key_id="kid-1",
                signing_alg="RS256",
            ),
        )

    data = _posted_data(client)
    assert data["client_assertion_type"] == CLIENT_ASSERTION_TYPE
    assert "client_secret" not in data
    decoded = jwt.decode(
        data["client_assertion"],
        _PUBLIC_PEM,
        algorithms=["RS256"],
        audience=_ENDPOINT,
    )
    assert decoded["iss"] == _CLIENT_ID
    assert decoded["sub"] == _CLIENT_ID
    assert decoded["aud"] == _ENDPOINT
    assert "exp" in decoded
    assert jwt.get_unverified_header(data["client_assertion"])["kid"] == "kid-1"


@pytest.mark.asyncio
async def test_fetch_http_error_maps_to_upstream_unavailable_with_status():
    error_resp = MagicMock()
    error_resp.status_code = 403
    error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Forbidden", request=MagicMock(), response=error_resp
    )
    with patch(_PATCH_TARGET, return_value=_client(error_resp)):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            ClientSecretAuth(client_secret=SecretStr("s")),
        )

    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
    assert "403" in result.error.summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raised",
    [
        httpx.ConnectError("connection refused", request=MagicMock()),
        httpx.ReadTimeout("timed out", request=MagicMock()),
        litellm.Timeout(
            message="Connection timed out",
            model="default-model-name",
            llm_provider="litellm-httpx-handler",
        ),
    ],
)
async def test_fetch_network_error_maps_to_upstream_unavailable(raised):
    client = AsyncMock()
    client.post.side_effect = raised
    with patch(_PATCH_TARGET, return_value=client):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            ClientSecretAuth(client_secret=SecretStr("s")),
        )

    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
    assert _ENDPOINT not in result.error.summary
    assert "idp.example.com" not in result.error.summary


@pytest.mark.asyncio
async def test_fetch_invalid_json_maps_to_upstream_unavailable():
    bad = MagicMock()
    bad.raise_for_status = MagicMock()
    bad.json.side_effect = json.JSONDecodeError("Expecting value", "<html>", 0)
    with patch(_PATCH_TARGET, return_value=_client(bad)):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            ClientSecretAuth(client_secret=SecretStr("s")),
        )

    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"
    assert _ENDPOINT not in result.error.summary
    assert "idp.example.com" not in result.error.summary


@pytest.mark.asyncio
async def test_fetch_none_response_is_upstream_unavailable():
    with patch(_PATCH_TARGET, return_value=_client(None)):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            ClientSecretAuth(client_secret=SecretStr("s")),
        )

    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_fetch_missing_access_token_is_upstream_unavailable():
    bad = MagicMock()
    bad.json.return_value = {"token_type": "Bearer"}
    bad.raise_for_status = MagicMock()
    with patch(_PATCH_TARGET, return_value=_client(bad)):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            ClientSecretAuth(client_secret=SecretStr("s")),
        )

    assert isinstance(result, Error)
    assert result.error.tag == "upstream_unavailable"


@pytest.mark.asyncio
async def test_fetch_http_error_does_not_leak_endpoint_url():
    error_resp = MagicMock()
    error_resp.status_code = 403
    error_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Forbidden", request=MagicMock(), response=error_resp
    )
    with patch(_PATCH_TARGET, return_value=_client(error_resp)):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            ClientSecretAuth(client_secret=SecretStr("s")),
        )

    assert isinstance(result, Error)
    assert _ENDPOINT not in result.error.summary
    assert "idp.example.com" not in result.error.summary


@pytest.mark.asyncio
async def test_fetch_none_response_does_not_leak_endpoint_url():
    with patch(_PATCH_TARGET, return_value=_client(None)):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            ClientSecretAuth(client_secret=SecretStr("s")),
        )

    assert isinstance(result, Error)
    assert _ENDPOINT not in result.error.summary
    assert "idp.example.com" not in result.error.summary


@pytest.mark.asyncio
async def test_fetch_missing_access_token_does_not_leak_endpoint_url():
    bad = MagicMock()
    bad.json.return_value = {"token_type": "Bearer"}
    bad.raise_for_status = MagicMock()
    with patch(_PATCH_TARGET, return_value=_client(bad)):
        result = await TokenEndpointClient().fetch(
            _ENDPOINT,
            _CLIENT_ID,
            {"grant_type": "g"},
            ClientSecretAuth(client_secret=SecretStr("s")),
        )

    assert isinstance(result, Error)
    assert _ENDPOINT not in result.error.summary
    assert "idp.example.com" not in result.error.summary


def _ok_token(value="cached") -> Result[ExchangedToken, CredError]:
    return Ok(ExchangedToken(access_token=value, expires_in=3600))


@pytest.mark.asyncio
async def test_cache_hit_skips_the_second_compute():
    cache = ExchangedTokenCache()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        return _ok_token("tok")

    first = await cache.get_or_compute("k", compute)
    second = await cache.get_or_compute("k", compute)

    assert isinstance(first, Ok) and first.ok == "tok"
    assert isinstance(second, Ok) and second.ok == "tok"
    assert calls == 1


@pytest.mark.asyncio
async def test_cache_single_flights_concurrent_misses():
    cache = ExchangedTokenCache()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.01)
        return _ok_token("shared")

    results = await asyncio.gather(
        cache.get_or_compute("k", compute),
        cache.get_or_compute("k", compute),
    )

    assert [r.ok for r in results] == ["shared", "shared"]
    assert calls == 1


@pytest.mark.asyncio
async def test_cache_does_not_store_a_failed_compute():
    cache = ExchangedTokenCache()
    calls = 0

    async def compute():
        nonlocal calls
        calls += 1
        if calls == 1:
            return Error(CredError.of_upstream_unavailable("down"))
        return _ok_token("recovered")

    first = await cache.get_or_compute("k", compute)
    second = await cache.get_or_compute("k", compute)

    assert isinstance(first, Error)
    assert isinstance(second, Ok) and second.ok == "recovered"
    assert calls == 2
