"""Shared helpers for e2e tests that exercise the Datadog MCP server through
gateway-managed OAuth2 (authorization_code + PKCE).

Datadog's remote MCP server supports OAuth2.1 with mandatory S256 PKCE.
Authorize/token/register URLs and the MCP endpoint are derived from DD_SITE
(same site resolution as datadog_mcp_url) so non-US1 orgs do not 403.
The authorize endpoint serves an interactive consent page, so the browser
leg is a headless Chromium primed with a saved Datadog browser session
(E2E_DD_STORAGE_STATE).

The gateway discovers the OAuth endpoints via /.well-known metadata, so the
server is registered with auth_type=oauth2, oauth2_flow=authorization_code
and no explicit authorize/token URLs. The per-user token is stored via
POST /v1/mcp/server/{server_id}/oauth-user-credential.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import parse_qsl

import httpx
import pytest
from pydantic import BaseModel, TypeAdapter

from e2e_config import REQUEST_TIMEOUT, datadog_app_origin, datadog_mcp_url
from e2e_http import AuthHeaders, NoBody, unwrap
from models import McpServerCreateBody, McpServerInfo
from proxy_client import ProxyClient

if TYPE_CHECKING:
    from playwright.async_api import Route

OAUTH_CLIENT_REDIRECT_URI = "http://127.0.0.1:53682/e2e/callback"
BROWSER_CONSENT_TIMEOUT = 60.0


def _dd_mcp_url() -> str:
    return datadog_mcp_url(toolsets="core")


def _dd_authorize_url() -> str:
    return f"{datadog_app_origin()}/oauth2/v1/authorize"


def _dd_token_url() -> str:
    return f"{datadog_app_origin()}/api/v2/oauth2/token"


def _dd_register_url() -> str:
    return f"{datadog_app_origin()}/api/v2/oauth2/register"


@dataclass(frozen=True, slots=True)
class PkceChallenge:
    verifier: str
    challenge: str
    state: str


@dataclass(frozen=True, slots=True)
class DcrClient:
    client_id: str


@dataclass(frozen=True, slots=True)
class OAuthToken:
    access_token: str
    token_type: str
    refresh_token: str | None = None


@dataclass(frozen=True, slots=True)
class DatadogMcpOAuthServer:
    server_id: str
    alias: str


class OAuthCredentialBody(BaseModel):
    access_token: str
    refresh_token: str | None = None
    expires_in: int | None = None
    scopes: list[str] | None = None


def assert_dd_oauth_env() -> None:
    path = os.environ.get("E2E_DD_STORAGE_STATE", "")
    if not path or not os.path.exists(path):
        pytest.fail(
            "Datadog MCP OAuth e2e requires E2E_DD_STORAGE_STATE to point at a "
            "saved Datadog browser session. Capture one with mcp/dd_session_capture.py."
        )


def _generate_pkce() -> PkceChallenge:
    verifier = secrets.token_urlsafe(64)
    challenge = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode()).digest()
    ).rstrip(b"=").decode()
    state = secrets.token_urlsafe(32)
    return PkceChallenge(verifier=verifier, challenge=challenge, state=state)


def _dcr_register() -> DcrClient:
    resp = httpx.post(
        _dd_register_url(),
        json={
            "client_name": "e2e-mcp-dd-oauth",
            "redirect_uris": [OAUTH_CLIENT_REDIRECT_URI],
            "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    client_id = TypeAdapter(str).validate_python(resp.json()["client_id"])
    return DcrClient(client_id=client_id)


async def _browser_authorize(
    authorize_url: str, storage_state_path: str
) -> tuple[str, str | None]:
    import asyncio

    from playwright.async_api import async_playwright

    captured: dict[str, str] = {}
    trail: list[str] = []

    def _note_request(request: object) -> None:
        url = getattr(request, "url", "")
        if url.startswith(OAUTH_CLIENT_REDIRECT_URI) and "url" not in captured:
            captured["url"] = url

    async def _swallow_redirect(route: "Route") -> None:
        await route.fulfill(status=200, content_type="text/plain", body="ok")

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(storage_state=storage_state_path)
        await context.route(
            re.compile(re.escape(OAUTH_CLIENT_REDIRECT_URI) + r".*"), _swallow_redirect
        )
        page = await context.new_page()
        page.on("request", _note_request)
        page.on("framenavigated", lambda frame: trail.append(frame.url.split("?", 1)[0]))
        await page.goto(authorize_url, wait_until="domcontentloaded")
        deadline = time.monotonic() + BROWSER_CONSENT_TIMEOUT
        while "url" not in captured and time.monotonic() < deadline:
            try:
                await page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                pass
            if "url" in captured:
                break
            control = page.locator(
                'button[name="action"][value="approve"], button:has-text("Authorize"), '
                'button:has-text("Allow"), button:has-text("@"), a:has-text("@")'
            ).first
            try:
                await control.click(timeout=5000)
            except Exception:
                await asyncio.sleep(0.5)
        final_url = page.url
        await browser.close()

    landing = captured.get("url")
    assert landing is not None, (
        f"consent flow never reached {OAUTH_CLIENT_REDIRECT_URI}; "
        f"final={final_url.split('?', 1)[0]!r}; trail={trail[-6:]}"
    )
    params = dict(parse_qsl(httpx.URL(landing).query.decode()))
    assert "code" in params, f"client redirect_uri carried no code: {landing}"
    return params["code"], params.get("state")


def _exchange_code(
    code: str, pkce: PkceChallenge, client: DcrClient
) -> OAuthToken:
    resp = httpx.post(
        _dd_token_url(),
        data={
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": OAUTH_CLIENT_REDIRECT_URI,
            "client_id": client.client_id,
            "code_verifier": pkce.verifier,
        },
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = TypeAdapter(dict[str, object]).validate_python(resp.json())
    access_token = str(data["access_token"])
    token_type = str(data.get("token_type", "Bearer"))
    refresh_token_raw = data.get("refresh_token")
    refresh_token = str(refresh_token_raw) if refresh_token_raw is not None else None
    return OAuthToken(
        access_token=access_token,
        token_type=token_type,
        refresh_token=refresh_token,
    )


def fetch_dd_oauth_token(storage_state_path: str) -> OAuthToken:
    """Drive the full PKCE dance: DCR, authorize (browser), token exchange."""
    import asyncio

    pkce = _generate_pkce()
    dcr = _dcr_register()

    params = {
        "response_type": "code",
        "client_id": dcr.client_id,
        "redirect_uri": OAUTH_CLIENT_REDIRECT_URI,
        "code_challenge": pkce.challenge,
        "code_challenge_method": "S256",
        "state": pkce.state,
    }
    authorize_url = f"{_dd_authorize_url()}?{urllib.parse.urlencode(params)}"
    code, returned_state = asyncio.run(
        _browser_authorize(authorize_url, storage_state_path)
    )
    assert returned_state == pkce.state, (
        f"OAuth state mismatch: sent {pkce.state!r}, got {returned_state!r}"
    )
    return _exchange_code(code, pkce, dcr)


def register_dd_oauth_server(
    proxy: ProxyClient, alias: str
) -> DatadogMcpOAuthServer:
    """Register the Datadog MCP server with auth_type=oauth2,
    oauth2_flow=authorization_code. The gateway discovers the authorize/token
    endpoints via /.well-known metadata."""
    resp = unwrap(
        proxy.transport.post(
            "/v1/mcp/server",
            headers=proxy.transport.master,
            json=McpServerCreateBody(
                alias=alias,
                url=_dd_mcp_url(),
                transport="http",
                allow_all_keys=False,
                auth_type="oauth2",
                oauth2_flow="authorization_code",
            ),
            response_type=McpServerInfo,
        )
    )
    return DatadogMcpOAuthServer(server_id=resp.server_id, alias=alias)


def store_dd_oauth_token(
    proxy: ProxyClient,
    server_id: str,
    key: str,
    token: OAuthToken,
) -> None:
    """Store the OAuth access token in the gateway's per-user credential vault
    via POST /v1/mcp/server/{server_id}/oauth-user-credential."""
    unwrap(
        proxy.transport.post(
            f"/v1/mcp/server/{server_id}/oauth-user-credential",
            headers=AuthHeaders(authorization=f"Bearer {key}"),
            json=OAuthCredentialBody(
                access_token=token.access_token,
                refresh_token=token.refresh_token,
            ),
            response_type=NoBody,
        )
    )


def delete_dd_oauth_server(proxy: ProxyClient, server_id: str) -> None:
    _ = proxy.transport.delete(
        f"/v1/mcp/server/{server_id}",
        headers=proxy.transport.master,
        json=NoBody(),
        response_type=NoBody,
    )
