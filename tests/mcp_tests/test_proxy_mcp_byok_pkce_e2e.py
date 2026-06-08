"""Interactive PKCE MCP e2e test against a real proxy at localhost:4000.

Assumes CircleCI (or local dev) has:
  - a litellm proxy running at localhost:4000 with DATABASE_URL and LITELLM_MASTER_KEY set
  - a Postgres database accessible via DATABASE_URL

Flow:
  1. register a BYOK MCP server via the management API
  2. mint a UI session cookie (real litellm auth functions)
  3. POST /v1/mcp/oauth/authorize with a PKCE code_challenge -> authorization code
  4. POST /v1/mcp/oauth/token with the code_verifier -> access token
  5. use master key + access token to call mcp tools via the per-server path
"""

import asyncio
import os
import typing
import uuid
from datetime import timedelta
from typing import cast
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import litellm
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from litellm.constants import LITELLM_PROXY_ADMIN_NAME
from litellm.proxy.auth.login_utils import LoginResult, create_ui_token_object
from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler
from tests.mcp_tests.mcp_server import DEFAULT_OAUTH_ACCESS_TOKEN
from tests.mcp_tests.test_proxy_mcp_auth_e2e import (
    _reserve_port,
    _start_mcp_server_process,
)

DATABASE_URL = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(
    not DATABASE_URL, reason="BYOK PKCE e2e requires a database (DATABASE_URL)"
)

MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
PROXY_BASE_URL = os.getenv("LITELLM_PROXY_BASE_URL", "http://localhost:4000")
REDIRECT_URI = "http://127.0.0.1:8765/callback"


def _pkce_pair() -> tuple[str, str]:
    return SSOAuthenticationHandler.generate_pkce_params()


def _session_cookie() -> str:
    login_result = LoginResult(
        user_id=LITELLM_PROXY_ADMIN_NAME,
        key="byok-pkce-e2e-ui-session",
        user_email="byok-pkce-e2e@test.local",
        user_role="proxy_admin",
        login_method="username_password",
    )
    returned_ui_token_object = create_ui_token_object(
        login_result=login_result,
        general_settings={},
        premium_user=False,
    )
    payload = dict(cast(dict, returned_ui_token_object))
    payload["exp"] = litellm.utils.get_utc_datetime() + timedelta(hours=1)
    return jwt.encode(payload, MASTER_KEY, algorithm="HS256")


@pytest.fixture(scope="module")
def upstream_server() -> typing.Iterator[dict[str, typing.Any]]:
    port = _reserve_port()
    process = _start_mcp_server_process(auth_mode="oauth2", port=port, auth_secret=None)
    try:
        yield {"base_url": f"http://127.0.0.1:{port}"}
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except Exception:
            process.kill()


@pytest.fixture(scope="module")
def byok_server_id(upstream_server: dict[str, typing.Any]) -> str:
    """Register a BYOK MCP server via the proxy management API."""
    server_id = str(uuid.uuid4())
    alias = f"byok_pkce_server_{server_id[:8]}"

    response = httpx.post(
        f"{PROXY_BASE_URL}/v1/mcp/server",
        headers={"Authorization": f"Bearer {MASTER_KEY}"},
        json={
            "server_id": server_id,
            "alias": alias,
            "url": f"{upstream_server['base_url']}/mcp",
            "transport": "http",
            "auth_type": "oauth2",
            "is_byok": True,
            "allow_all_keys": True,
        },
    )
    assert response.status_code == 201, (
        f"Failed to register BYOK server: {response.status_code} {response.text}"
    )
    return alias


@pytest.fixture(scope="module")
def proxy_server_url() -> str:
    return PROXY_BASE_URL


async def _complete_pkce_flow(proxy_server_url: str, server_alias: str) -> str:
    """Run authorize -> token and return the issued access token."""
    verifier, challenge = _pkce_pair()
    cookies = {"token": _session_cookie()}

    async with httpx.AsyncClient(follow_redirects=False) as client:
        authorize = await client.post(
            f"{proxy_server_url}/v1/mcp/oauth/authorize",
            data={
                "redirect_uri": REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "xyz",
                "server_id": server_alias,
                "api_key": DEFAULT_OAUTH_ACCESS_TOKEN,
                "client_id": "byok-client",
            },
            cookies=cookies,
        )
        assert authorize.status_code == 302, authorize.text
        code = parse_qs(urlparse(authorize.headers["location"]).query)["code"][0]

        token = await client.post(
            f"{proxy_server_url}/v1/mcp/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": REDIRECT_URI,
                "client_id": "byok-client",
            },
        )
        assert token.status_code == 200, token.text
        return token.json()["access_token"]


@pytest.mark.asyncio
async def test_byok_pkce_authorize_rejects_wrong_verifier(
    proxy_server_url: str, byok_server_id: str
) -> None:
    """PKCE enforcement: a token request with a mismatched verifier is rejected."""
    _verifier, challenge = _pkce_pair()
    cookies = {"token": _session_cookie()}

    async with httpx.AsyncClient(follow_redirects=False) as client:
        authorize = await client.post(
            f"{proxy_server_url}/v1/mcp/oauth/authorize",
            data={
                "redirect_uri": REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "xyz",
                "server_id": byok_server_id,
                "api_key": DEFAULT_OAUTH_ACCESS_TOKEN,
                "client_id": "byok-client",
            },
            cookies=cookies,
        )
        assert authorize.status_code == 302
        code = parse_qs(urlparse(authorize.headers["location"]).query)["code"][0]

        token = await client.post(
            f"{proxy_server_url}/v1/mcp/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": "wrong-verifier-that-will-not-match-the-challenge",
                "redirect_uri": REDIRECT_URI,
                "client_id": "byok-client",
            },
        )
    assert token.status_code == 400
    assert token.json()["error"] == "invalid_grant"


@pytest.mark.asyncio
async def test_byok_pkce_authorize_requires_session(
    proxy_server_url: str, byok_server_id: str
) -> None:
    """Without a UI session cookie the authorize endpoint must refuse."""
    _verifier, challenge = _pkce_pair()
    async with httpx.AsyncClient(follow_redirects=False) as client:
        response = await client.post(
            f"{proxy_server_url}/v1/mcp/oauth/authorize",
            data={
                "redirect_uri": REDIRECT_URI,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "server_id": byok_server_id,
                "api_key": DEFAULT_OAUTH_ACCESS_TOKEN,
                "client_id": "byok-client",
            },
        )
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_byok_pkce_end_to_end_tool_call(
    proxy_server_url: str, byok_server_id: str
) -> None:
    access_token = await _complete_pkce_flow(proxy_server_url, byok_server_id)

    headers = {
        "x-litellm-api-key": f"Bearer {MASTER_KEY}",
        "Authorization": f"Bearer {access_token}",
    }
    async with asyncio.timeout(20):
        async with streamablehttp_client(
            url=f"{proxy_server_url}/{byok_server_id}/mcp", headers=headers
        ) as (read, write, _get_session_id):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("add", arguments={"a": 5, "b": 6})
                assert result.content
                text = getattr(result.content[0], "text", None)
                assert text == "11"
