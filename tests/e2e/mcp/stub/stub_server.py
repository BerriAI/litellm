"""Deterministic MCP upstreams for the mcp e2e suite.

One process, one port, five streamable-http MCP mounts plus a deterministic
OAuth2 IdP, so the compose stack keeps a single `mcp-stub` service:

- `/mcp` — anonymous. `echo` answers immediately so auth tests can assert an
  exact round-trip; `slow_echo` holds the request open for `sleep_seconds`
  while per-`marker` in-flight and max-in-flight counters track how many calls
  the proxy let through simultaneously (the observable a per-server
  `max_concurrent_requests` cap must bound); `stats` reads those counters back,
  so tests observe upstream concurrency through the proxy itself and the stub
  needs no side-channel port.
- `/second/mcp` — anonymous, with a deliberately disjoint tool set
  (`second_ping`). Aggregate-routing tests register it as a second gateway
  server; a call only this mount can answer proves which upstream served it.
- `/apikey/mcp` — rejects any request whose `X-API-Key` is not exactly
  UPSTREAM_API_KEY, the header the gateway injects for `auth_type: api_key`.
- `/oauth/mcp` — rejects any request whose `Authorization` is not exactly
  `Bearer OAUTH_ACCESS_TOKEN`. That token is only obtainable from
  `/oauth/token`, so a served request proves the gateway ran the
  client_credentials exchange rather than forwarding something it already had.
- `/oauthuser/mcp` — the interactive (authorization_code) sibling: rejects
  anything but `Bearer OAUTH_USER_ACCESS_TOKEN`, which only the
  authorization_code grant hands out, so a served request proves the whole
  browser dance (authorize redirect, code, PKCE-verified token exchange) ran.
- `/oauth/authorize` — the auto-approving authorization endpoint: validates
  client_id/response_type, records the one-time code with its PKCE challenge,
  and 302s straight back to the caller's redirect_uri with code and state (the
  "user" of this IdP always consents instantly).
- `/oauth/token` — the token endpoint for all three grants:
  client_credentials answers with OAUTH_ACCESS_TOKEN;
  authorization_code validates the code, redirect_uri, client credentials,
  and (when a challenge was recorded) the S256 code_verifier, then answers
  with OAUTH_USER_ACCESS_TOKEN + OAUTH_USER_REFRESH_TOKEN; refresh_token
  re-issues OAUTH_USER_ACCESS_TOKEN for the known refresh token.

The guarded mounts record the headers of the most recent authorized request;
their `recorded_headers` tool reads them back through the proxy, so tests can
assert exactly which credentials the gateway attached upstream (and that the
caller's LiteLLM virtual key never left the gateway).

The credential constants are mirrored in tests/e2e/e2e_config.py; keep the two
in sync. Counter updates are plain attribute mutations between awaits, so
asyncio's single-threaded scheduling makes them atomic; markers come from
`unique_marker()` so concurrent test runs never share a counter.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.datastructures import FormData, Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

UPSTREAM_API_KEY = "e2e-stub-upstream-api-key"
OAUTH_CLIENT_ID = "e2e-stub-oauth-client-id"
OAUTH_CLIENT_SECRET = "e2e-stub-oauth-client-secret"
OAUTH_SCOPE = "tools:read"
OAUTH_ACCESS_TOKEN = "e2e-stub-minted-access-token"
OAUTH_USER_CLIENT_ID = "e2e-stub-user-client-id"
OAUTH_USER_CLIENT_SECRET = "e2e-stub-user-client-secret"
OAUTH_USER_ACCESS_TOKEN = "e2e-stub-user-access-token"
OAUTH_USER_REFRESH_TOKEN = "e2e-stub-user-refresh-token"

main_mcp = FastMCP("e2e-stub", host="0.0.0.0", port=8765, stateless_http=True)
second_mcp = FastMCP("e2e-stub-second", host="0.0.0.0", port=8765, stateless_http=True)
apikey_mcp = FastMCP("e2e-stub-apikey", host="0.0.0.0", port=8765, stateless_http=True)
oauth_mcp = FastMCP("e2e-stub-oauth", host="0.0.0.0", port=8765, stateless_http=True)
oauthuser_mcp = FastMCP("e2e-stub-oauthuser", host="0.0.0.0", port=8765, stateless_http=True)


@dataclass
class _MarkerStats:
    in_flight: int = 0
    max_in_flight: int = 0
    completed: int = 0


_stats: dict[str, _MarkerStats] = {}

_last_authorized_headers: dict[str, dict[str, str]] = {}


@main_mcp.tool()
def echo(text: str) -> str:
    """Return `text` unchanged."""
    return text


@main_mcp.tool()
async def slow_echo(text: str, marker: str, sleep_seconds: float) -> str:
    """Return `text` after `sleep_seconds`, recording concurrency under `marker`."""
    stats = _stats.setdefault(marker, _MarkerStats())
    stats.in_flight += 1
    stats.max_in_flight = max(stats.max_in_flight, stats.in_flight)
    try:
        await asyncio.sleep(sleep_seconds)
    finally:
        stats.in_flight -= 1
        stats.completed += 1
    return text


@main_mcp.tool()
def stats(marker: str) -> str:
    """Return the JSON stats recorded for `marker`."""
    recorded = _stats.get(marker, _MarkerStats())
    return json.dumps(
        {
            "marker": marker,
            "max_in_flight": recorded.max_in_flight,
            "completed": recorded.completed,
        }
    )


@second_mcp.tool()
def second_ping() -> str:
    """Identify the /second upstream; no other mount serves this tool."""
    return "pong-from-second"
def _register_guarded_tools(server: FastMCP, mount: str) -> None:
    def echo(text: str) -> str:
        """Return `text` unchanged."""
        return text

    def recorded_headers() -> str:
        """Return the headers of the most recent authorized request as JSON."""
        return json.dumps(_last_authorized_headers.get(mount, {}))

    _ = server.tool()(echo)
    _ = server.tool()(recorded_headers)


_register_guarded_tools(apikey_mcp, "apikey")
_register_guarded_tools(oauth_mcp, "oauth")
_register_guarded_tools(oauthuser_mcp, "oauthuser")


def _require_header(app: ASGIApp, *, mount: str, header: str, expected: str) -> ASGIApp:
    """Serve `app` only to requests carrying `header: expected`; 401 otherwise.
    Authorized requests have their full header map recorded under `mount`."""

    async def guard(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if headers.get(header) != expected:
            response = JSONResponse({"error": "unauthorized", "detail": f"missing or wrong {header}"}, status_code=401)
            await response(scope, receive, send)
            return
        _last_authorized_headers[mount] = dict(headers.items())
        await app(scope, receive, send)

    return guard


_issued_codes: dict[str, dict[str, str]] = {}


async def oauth_authorize(request: Request) -> Response:
    """The auto-approving authorization endpoint: the resource owner of this
    IdP consents instantly, so a headless test can drive the browser leg of
    the authorization_code flow with plain HTTP redirects."""
    params = request.query_params
    redirect_uri = params.get("redirect_uri", "")
    valid = params.get("response_type") == "code" and params.get("client_id") == OAUTH_USER_CLIENT_ID and redirect_uri
    if not valid:
        return JSONResponse(
            {"error": "invalid_request", "detail": "need response_type=code, the user client_id, and a redirect_uri"},
            status_code=400,
        )
    code = f"e2e-stub-code-{uuid.uuid4().hex}"
    _issued_codes[code] = {
        "redirect_uri": redirect_uri,
        "code_challenge": params.get("code_challenge", ""),
    }
    state = params.get("state", "")
    separator = "&" if "?" in redirect_uri else "?"
    location = f"{redirect_uri}{separator}code={code}" + (f"&state={state}" if state else "")
    return RedirectResponse(location, status_code=302)


def _s256(code_verifier: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(code_verifier.encode()).digest()).rstrip(b"=").decode()


def _authorization_code_grant(form: FormData) -> JSONResponse:
    code_value = form.get("code")
    record = _issued_codes.pop(code_value, None) if isinstance(code_value, str) else None
    granted = (
        record is not None
        and form.get("client_id") == OAUTH_USER_CLIENT_ID
        and form.get("client_secret") == OAUTH_USER_CLIENT_SECRET
        and form.get("redirect_uri") == record["redirect_uri"]
    )
    if not granted:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)
    assert record is not None
    if record["code_challenge"]:
        verifier = form.get("code_verifier")
        if not isinstance(verifier, str) or _s256(verifier) != record["code_challenge"]:
            return JSONResponse({"error": "invalid_grant", "detail": "PKCE verification failed"}, status_code=400)
    return JSONResponse(
        {
            "access_token": OAUTH_USER_ACCESS_TOKEN,
            "token_type": "Bearer",
            "expires_in": 3600,
            "refresh_token": OAUTH_USER_REFRESH_TOKEN,
        }
    )


def _refresh_token_grant(form: FormData) -> JSONResponse:
    granted = (
        form.get("refresh_token") == OAUTH_USER_REFRESH_TOKEN
        and form.get("client_id") == OAUTH_USER_CLIENT_ID
        and form.get("client_secret") == OAUTH_USER_CLIENT_SECRET
    )
    if not granted:
        return JSONResponse({"error": "invalid_grant"}, status_code=400)
    return JSONResponse({"access_token": OAUTH_USER_ACCESS_TOKEN, "token_type": "Bearer", "expires_in": 3600})


async def oauth_token(request: Request) -> JSONResponse:
    """The token endpoint for all three grants. Every form field is matched
    exactly so a failure points at the precise field the proxy sent wrong."""
    form = await request.form()
    grant_type = form.get("grant_type")
    if grant_type == "client_credentials":
        granted = (
            form.get("client_id") == OAUTH_CLIENT_ID
            and form.get("client_secret") == OAUTH_CLIENT_SECRET
            and form.get("scope") == OAUTH_SCOPE
        )
        if not granted:
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        return JSONResponse({"access_token": OAUTH_ACCESS_TOKEN, "token_type": "Bearer", "expires_in": 3600})
    if grant_type == "authorization_code":
        return _authorization_code_grant(form)
    if grant_type == "refresh_token":
        return _refresh_token_grant(form)
    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


def build_app() -> Starlette:
    servers = (main_mcp, second_mcp, apikey_mcp, oauth_mcp, oauthuser_mcp)
    apps = {server.name: server.streamable_http_app() for server in servers}

    @contextlib.asynccontextmanager
    async def lifespan(_: Starlette) -> AsyncGenerator[None]:
        async with contextlib.AsyncExitStack() as stack:
            for server in servers:
                await stack.enter_async_context(server.session_manager.run())
            yield

    return Starlette(
        routes=[
            Route("/oauth/token", oauth_token, methods=["POST"]),
            Route("/oauth/authorize", oauth_authorize, methods=["GET"]),
            Mount(
                "/oauthuser",
                app=_require_header(
                    apps["e2e-stub-oauthuser"],
                    mount="oauthuser",
                    header="authorization",
                    expected=f"Bearer {OAUTH_USER_ACCESS_TOKEN}",
                ),
            ),
            Mount(
                "/oauth",
                app=_require_header(
                    apps["e2e-stub-oauth"],
                    mount="oauth",
                    header="authorization",
                    expected=f"Bearer {OAUTH_ACCESS_TOKEN}",
                ),
            ),
            Mount(
                "/apikey",
                app=_require_header(
                    apps["e2e-stub-apikey"],
                    mount="apikey",
                    header="x-api-key",
                    expected=UPSTREAM_API_KEY,
                ),
            ),
            Mount("/second", app=apps["e2e-stub-second"]),
            Mount("/", app=apps["e2e-stub"]),
        ],
        lifespan=lifespan,
    )


if __name__ == "__main__":
    uvicorn.run(build_app(), host="0.0.0.0", port=8765)
