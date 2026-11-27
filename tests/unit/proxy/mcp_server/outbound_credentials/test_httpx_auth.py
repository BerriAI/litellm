"""Tests for the concrete httpx2.Auth objects the resolver returns.

NoOpAuth must attach nothing; StaticHeaderAuth must set exactly the configured header. These
pin the header emission the api_key family and passthrough depend on.
"""

from unittest.mock import AsyncMock

import httpx2
import pytest

from litellm.proxy.mcp_server.outbound_credentials import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy.mcp_server.outbound_credentials.httpx_auth import (
    PER_USER_OAUTH_RECOVERY_EXTENSION,
    PerUserOAuthBearerAuth,
)
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.proxy.mcp_server.outbound_credentials.types import AuthorizationCodeConfig, CredError


def _apply(auth: httpx2.Auth, request: httpx2.Request) -> httpx2.Request:
    flow = auth.auth_flow(request)
    sent = next(flow)
    flow.close()
    return sent


def test_noop_auth_attaches_no_authorization_header():
    request = httpx2.Request("GET", "https://upstream.example.com/mcp")
    _apply(NoOpAuth(), request)
    assert "authorization" not in request.headers


def test_static_header_auth_defaults_to_authorization():
    request = httpx2.Request("GET", "https://upstream.example.com/mcp")
    _apply(StaticHeaderAuth("Bearer abc"), request)
    assert request.headers["Authorization"] == "Bearer abc"


def test_static_header_auth_honors_custom_header_name():
    request = httpx2.Request("GET", "https://upstream.example.com/mcp")
    _apply(StaticHeaderAuth("raw-key", header_name="X-API-Key"), request)
    assert request.headers["X-API-Key"] == "raw-key"
    assert "authorization" not in request.headers


def test_static_header_auth_masks_credential_from_introspection():
    auth = StaticHeaderAuth("Bearer super-secret-token")
    assert "super-secret-token" not in repr(auth)
    assert "super-secret-token" not in str(vars(auth))


@pytest.mark.asyncio
async def test_per_user_auth_refreshes_once_after_an_upstream_401_and_retries():
    refresh_now = AsyncMock(return_value=Ok(OAuthToken(access_token="fresh")))
    auth = PerUserOAuthBearerAuth("dead", refresh_now, AuthorizationCodeConfig())
    request = httpx2.Request("POST", "https://upstream.example/mcp")
    flow = auth.async_auth_flow(request)
    outbound = await flow.__anext__()
    assert outbound.headers["authorization"] == "Bearer dead"
    rejected = httpx2.Response(401, request=request)
    retry = await flow.asend(rejected)
    assert retry.headers["authorization"] == "Bearer fresh"
    with pytest.raises(StopAsyncIteration):
        await flow.asend(httpx2.Response(200, request=request))
    refresh_now.assert_awaited_once_with("dead")


@pytest.mark.asyncio
async def test_per_user_auth_marks_the_rejection_when_recovery_is_impossible():
    async def refresh_now(failed_access_token: str):
        return Error(CredError.of_unauthorized("sign in again"))

    auth = PerUserOAuthBearerAuth("dead", refresh_now, AuthorizationCodeConfig())
    request = httpx2.Request("POST", "https://upstream.example/mcp")
    flow = auth.async_auth_flow(request)
    await flow.__anext__()
    rejected = httpx2.Response(401, request=request)
    with pytest.raises(StopAsyncIteration):
        await flow.asend(rejected)
    error = rejected.extensions[PER_USER_OAUTH_RECOVERY_EXTENSION]
    assert error.tag == "unauthorized"


@pytest.mark.asyncio
async def test_per_user_auth_marks_a_second_401_for_reauth():
    async def refresh_now(failed_access_token: str):
        return Ok(OAuthToken(access_token="fresh-but-rejected"))

    auth = PerUserOAuthBearerAuth("dead", refresh_now, AuthorizationCodeConfig())
    request = httpx2.Request("POST", "https://upstream.example/mcp")
    flow = auth.async_auth_flow(request)
    await flow.__anext__()
    retry = await flow.asend(httpx2.Response(401, request=request))
    assert retry.headers["authorization"] == "Bearer fresh-but-rejected"
    second_rejection = httpx2.Response(401, request=request)
    with pytest.raises(StopAsyncIteration):
        await flow.asend(second_rejection)
    error = second_rejection.extensions[PER_USER_OAUTH_RECOVERY_EXTENSION]
    assert error.tag == "unauthorized"
