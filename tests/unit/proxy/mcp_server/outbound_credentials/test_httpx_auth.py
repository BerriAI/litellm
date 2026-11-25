"""Tests for the concrete httpx2.Auth objects the resolver returns.

NoOpAuth must attach nothing; StaticHeaderAuth must set exactly the configured header. These
pin the header emission the api_key family and passthrough depend on.
"""

from typing import Final

import httpx2
import pytest

from litellm.proxy.mcp_server.outbound_credentials import (
    NoOpAuth,
    StaticHeaderAuth,
)
from litellm.proxy.mcp_server.outbound_credentials.types import AuthorizationCodeConfig


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


class _Rejections:
    def __init__(self, error: Exception | None = None) -> None:
        self.reported: tuple[tuple[str, str | None], ...] = ()
        self._error = error

    async def __call__(self, access_token: str, upstream_error: str | None) -> None:
        self.reported = (*self.reported, (access_token, upstream_error))
        if self._error is not None:
            raise self._error


async def _send_with_user_grant(
    response: httpx2.Response, rejections: _Rejections, carrier: AuthorizationCodeConfig | None = None
) -> httpx2.Response:
    from litellm.proxy.mcp_server.outbound_credentials.httpx_auth import UserGrantBearerAuth

    auth: Final = UserGrantBearerAuth("at-1", carrier or AuthorizationCodeConfig(), rejections)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(lambda _request: response), auth=auth) as client:
        return await client.post("https://upstream.example.com/mcp")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, reported",
    [
        (httpx2.Response(200), ()),
        (httpx2.Response(403, headers={"WWW-Authenticate": 'Bearer error="insufficient_scope"'}), ()),
        (httpx2.Response(500), ()),
        (httpx2.Response(401), (("at-1", None),)),
        (
            httpx2.Response(401, headers={"WWW-Authenticate": 'Bearer realm="mcp, v2", error="invalid_token"'}),
            (("at-1", "invalid_token"),),
        ),
    ],
)
async def test_user_grant_auth_reports_only_an_upstream_401_for_the_token_it_sent(
    response: httpx2.Response, reported: tuple[tuple[str, str | None], ...]
) -> None:
    rejections: Final = _Rejections()
    answer: Final = await _send_with_user_grant(response, rejections)
    assert answer.request.headers["Authorization"] == "Bearer at-1"
    assert answer.status_code == response.status_code
    assert rejections.reported == reported


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redirect_to, reported",
    [
        ("https://upstream.example.com/mcp/v2", (("at-1", None),)),
        ("https://other.example.org/mcp", ()),
    ],
)
async def test_user_grant_auth_reports_a_redirected_401_only_when_the_token_travelled(
    redirect_to: str, reported: tuple[tuple[str, str | None], ...]
) -> None:
    from litellm.proxy.mcp_server.outbound_credentials.httpx_auth import UserGrantBearerAuth

    def handler(request: httpx2.Request) -> httpx2.Response:
        if str(request.url) == "https://upstream.example.com/mcp":
            return httpx2.Response(307, headers={"Location": redirect_to})
        return httpx2.Response(401)

    rejections: Final = _Rejections()
    auth: Final = UserGrantBearerAuth("at-1", AuthorizationCodeConfig(), rejections)
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler), auth=auth, follow_redirects=True) as client:
        answer: Final = await client.post("https://upstream.example.com/mcp")
    assert answer.status_code == 401 and str(answer.request.url) == redirect_to
    assert rejections.reported == reported


@pytest.mark.asyncio
async def test_user_grant_auth_reports_the_raw_token_for_a_custom_header_carrier() -> None:
    rejections: Final = _Rejections()
    carrier: Final = AuthorizationCodeConfig(header_name="X-Upstream-Token", value_prefix="")
    answer: Final = await _send_with_user_grant(httpx2.Response(401), rejections, carrier)
    assert answer.request.headers["X-Upstream-Token"] == "at-1" and "authorization" not in answer.request.headers
    assert rejections.reported == (("at-1", None),)


@pytest.mark.asyncio
async def test_user_grant_auth_returns_the_401_when_reporting_it_fails() -> None:
    rejections: Final = _Rejections(RuntimeError("store down"))
    answer: Final = await _send_with_user_grant(httpx2.Response(401), rejections)
    assert answer.status_code == 401 and rejections.reported == (("at-1", None),)


@pytest.mark.asyncio
async def test_user_grant_auth_preview_never_reports_a_rejection() -> None:
    from litellm.proxy.mcp_server.outbound_credentials.httpx_auth import UserGrantBearerAuth

    rejections: Final = _Rejections()
    auth: Final = UserGrantBearerAuth("at-1", AuthorizationCodeConfig(), rejections)
    flow: Final = auth.async_auth_flow(httpx2.Request("POST", "https://upstream.example.com/mcp"))
    previewed: Final = await flow.__anext__()
    await flow.aclose()
    assert previewed.headers["Authorization"] == "Bearer at-1" and auth.header_name == "Authorization"
    assert rejections.reported == ()
