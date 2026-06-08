"""
Interactive PKCE e2e tests against a live proxy on localhost:4000 with Postgres.
"""

import asyncio
import os
import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import jwt as pyjwt
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from litellm.proxy.management_endpoints.ui_sso import SSOAuthenticationHandler
from tests.mcp_tests.mcp_server import DEFAULT_OAUTH_ACCESS_TOKEN

MASTER_KEY = os.getenv("LITELLM_MASTER_KEY", "sk-1234")
PROXY_BASE_URL = os.getenv("LITELLM_PROXY_BASE_URL", "http://localhost:4000")
BYOK_UPSTREAM_URL = os.getenv(
    "MCP_BYOK_UPSTREAM_URL", "http://127.0.0.1:63889/mcp"
)

class TestBYOKPKCE:
    @pytest.fixture
    def session_token(self):
        r = httpx.post(
            f"{PROXY_BASE_URL}/v2/login",
            json={"username": "admin", "password": MASTER_KEY},
        )
        assert r.status_code == 200, r.text
        return r.json()["token"]

    @pytest.fixture
    def byok_alias(self):
        alias = f"byok_{uuid.uuid4().hex[:8]}"
        r = httpx.post(
            f"{PROXY_BASE_URL}/v1/mcp/server",
            headers={"Authorization": f"Bearer {MASTER_KEY}"},
            json={
                "alias": alias,
                "url": BYOK_UPSTREAM_URL,
                "transport": "http",
                "auth_type": "oauth2",
                "is_byok": True,
                "allow_all_keys": True,
            },
        )
        assert r.status_code == 201, r.text
        return alias

    @pytest.mark.asyncio
    async def test_login_jwt(self, session_token):
        payload = pyjwt.decode(session_token, options={"verify_signature": False})
        assert payload["user_role"] == "proxy_admin"
        assert payload["login_method"] == "username_password"

    @pytest.mark.asyncio
    async def test_pkce_full_flow(self, session_token, byok_alias):
        verifier, challenge = SSOAuthenticationHandler.generate_pkce_params()
        r = httpx.post(
            f"{PROXY_BASE_URL}/v1/mcp/oauth/authorize",
            data={
                "redirect_uri": "http://127.0.0.1:8765/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "t",
                "server_id": byok_alias,
                "api_key": DEFAULT_OAUTH_ACCESS_TOKEN,
                "client_id": "e2e",
            },
            cookies={"token": session_token},
            follow_redirects=False,
        )
        assert r.status_code == 302, f"authorize: {r.status_code} {r.text}"
        code = parse_qs(urlparse(r.headers["location"]).query)["code"][0]

        r = httpx.post(
            f"{PROXY_BASE_URL}/v1/mcp/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": verifier,
                "redirect_uri": "http://127.0.0.1:8765/callback",
                "client_id": "e2e",
            },
        )
        assert r.status_code == 200, f"token: {r.status_code} {r.text}"
        access_token = r.json()["access_token"]

        h = {
            "x-litellm-api-key": f"Bearer {MASTER_KEY}",
            "Authorization": f"Bearer {access_token}",
        }
        async with asyncio.timeout(20):
            async with streamablehttp_client(
                f"{PROXY_BASE_URL}/{byok_alias}/mcp", headers=h
            ) as (rd, wr, _):
                async with ClientSession(rd, wr) as s:
                    await s.initialize()
                    tools = [t.name for t in (await s.list_tools()).tools]
                    add = next((t for t in tools if t.endswith("add")), None)
                    assert add, f"add not in {tools}"
                    res = await s.call_tool(add, arguments={"a": 5, "b": 6})
                    assert getattr(res.content[0], "text", None) == "11"

    @pytest.mark.asyncio
    async def test_pkce_wrong_verifier(self, session_token, byok_alias):
        _, challenge = SSOAuthenticationHandler.generate_pkce_params()
        r = httpx.post(
            f"{PROXY_BASE_URL}/v1/mcp/oauth/authorize",
            data={
                "redirect_uri": "http://127.0.0.1:8765/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "state": "t",
                "server_id": byok_alias,
                "api_key": DEFAULT_OAUTH_ACCESS_TOKEN,
                "client_id": "e2e",
            },
            cookies={"token": session_token},
            follow_redirects=False,
        )
        assert r.status_code == 302
        code = parse_qs(urlparse(r.headers["location"]).query)["code"][0]
        r = httpx.post(
            f"{PROXY_BASE_URL}/v1/mcp/oauth/token",
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": "wrong",
                "redirect_uri": "http://127.0.0.1:8765/callback",
                "client_id": "e2e",
            },
        )
        assert r.status_code == 400
        assert r.json()["error"] == "invalid_grant"

    @pytest.mark.asyncio
    async def test_authorize_no_session(self, byok_alias):
        _, challenge = SSOAuthenticationHandler.generate_pkce_params()
        r = httpx.post(
            f"{PROXY_BASE_URL}/v1/mcp/oauth/authorize",
            data={
                "redirect_uri": "http://127.0.0.1:8765/callback",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "server_id": byok_alias,
                "api_key": DEFAULT_OAUTH_ACCESS_TOKEN,
                "client_id": "e2e",
            },
            follow_redirects=False,
        )
        assert r.status_code == 401
