import httpx2
import pytest

from litellm.proxy.mcp_server.outbound_credentials.httpx_auth import (
    PER_USER_OAUTH_RECOVERY_EXTENSION,
    PerUserOAuthBearerAuth,
)
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import OAuthToken
from litellm.proxy.mcp_server.outbound_credentials.result import Error, Ok
from litellm.proxy.mcp_server.outbound_credentials.types import AuthorizationCodeConfig, CredError


@pytest.mark.asyncio
async def test_per_user_auth_refreshes_once_after_an_upstream_401_and_retries():
    refresh_calls = []

    async def refresh_now(failed_access_token: str):
        refresh_calls.append(failed_access_token)
        return Ok(OAuthToken(access_token="fresh"))

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
    assert refresh_calls == ["dead"]


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
