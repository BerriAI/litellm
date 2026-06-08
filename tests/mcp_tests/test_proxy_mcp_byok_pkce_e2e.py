"""DB-backed end-to-end test for the interactive PKCE (BYOK) MCP OAuth flow.

This drives LiteLLM acting as the OAuth 2.1 authorization server:
  1. mint a UI session cookie
  2. POST /v1/mcp/oauth/authorize with a PKCE code_challenge -> authorization code
  3. POST /v1/mcp/oauth/token with the code_verifier -> access token (and the
     per-user credential is persisted to the DB)
  4. call the MCP server through the proxy using that access token; the proxy
     loads the stored credential and forwards it upstream

Requires a database (DATABASE_URL). Skipped otherwise. In CI this runs against
the Postgres service wired into the MCP workflow.
"""

import asyncio
import base64
import hashlib
import os
import time
import typing
import uuid
from urllib.parse import parse_qs, urlparse

import httpx
import jwt
import pytest
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from tests.mcp_tests.mcp_server import DEFAULT_OAUTH_ACCESS_TOKEN
from tests.mcp_tests.test_proxy_mcp_auth_e2e import (
    _reserve_port,
    _start_mcp_server_process,
    _start_proxy_server,
)

DATABASE_URL = os.getenv("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="BYOK PKCE e2e requires a database (DATABASE_URL)")

MASTER_KEY = "sk-1234"
BYOK_USER_ID = "byok-pkce-user"
REDIRECT_URI = "http://127.0.0.1:8765/callback"


def _pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(os.urandom(40)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


def _session_cookie() -> str:
    return jwt.encode(
        {
            "user_id": BYOK_USER_ID,
            "login_method": "username_password",
            "exp": int(time.time()) + 3600,
        },
        MASTER_KEY,
        algorithm="HS256",
    )


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
    """Create a BYOK MCP server row directly in the DB and return its id."""
    from litellm.proxy._types import NewMCPServerRequest
    from litellm.proxy._experimental.mcp_server.db import create_mcp_server
    from litellm.proxy.utils import PrismaClient, ProxyLogging
    from litellm.caching import DualCache

    server_id = str(uuid.uuid4())

    async def _create() -> None:
        prisma_client = PrismaClient(
            database_url=DATABASE_URL,
            proxy_logging_obj=ProxyLogging(user_api_key_cache=DualCache()),
        )
        await prisma_client.connect()
        try:
            await create_mcp_server(
                prisma_client,
                NewMCPServerRequest(
                    server_id=server_id,
                    alias="byok_pkce_server",
                    url=f"{upstream_server['base_url']}/mcp",
                    transport="http",
                    auth_type="oauth2",
                    is_byok=True,
                    allow_all_keys=True,
                ),
                touched_by="byok-pkce-e2e",
            )
        finally:
            await prisma_client.disconnect()

    asyncio.run(_create())
    return server_id


@pytest.fixture(scope="module")
def proxy_server_url(tmp_path_factory: pytest.TempPathFactory, byok_server_id: str) -> typing.Iterator[str]:
    os.environ["DATABASE_URL"] = DATABASE_URL  # restore if a sibling test cleared it
    os.environ["LITELLM_MASTER_KEY"] = MASTER_KEY

    config = {
        "general_settings": {"master_key": MASTER_KEY},
        "model_list": [
            {
                "model_name": "fake-model",
                "litellm_params": {"model": "openai/fake", "api_key": "fake-key"},
            }
        ],
    }
    config_path = tmp_path_factory.mktemp("byok_pkce") / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))

    server_url, server, thread, sock = _start_proxy_server(str(config_path))
    yield server_url

    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


async def _complete_pkce_flow(proxy_server_url: str, server_id: str) -> str:
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
                "server_id": server_id,
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
async def test_byok_pkce_authorize_rejects_wrong_verifier(proxy_server_url: str, byok_server_id: str) -> None:
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
async def test_byok_pkce_authorize_requires_session(proxy_server_url: str, byok_server_id: str) -> None:
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
async def test_byok_pkce_end_to_end_tool_call(proxy_server_url: str, byok_server_id: str) -> None:
    """Full interactive PKCE flow: authorize -> token -> call the MCP tool with
    the issued access token. The proxy forwards the stored BYOK credential
    upstream, so the tool call succeeds."""
    access_token = await _complete_pkce_flow(proxy_server_url, byok_server_id)

    headers = {
        "Authorization": f"Bearer {access_token}",
        "x-mcp-servers": "byok_pkce_server",
    }
    async with asyncio.timeout(20):
        async with streamablehttp_client(url=f"{proxy_server_url}/mcp", headers=headers) as (
            read,
            write,
            _get_session_id,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert any(tool.name.endswith("add") for tool in tools.tools)

                result = await session.call_tool("add", arguments={"a": 5, "b": 6})
                assert result.content
                assert getattr(result.content[0], "text", None) == "11"
