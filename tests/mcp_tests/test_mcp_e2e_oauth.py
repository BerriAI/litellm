"""End-to-end OAuth security tests for the LiteLLM MCP proxy.

Boots two real OAuth2 providers as subprocesses, each an Authorization Server +
Bearer-protected MCP Resource Server in one Starlette app. One process backs the
``oauth_m2m_*`` configs (driven via client_credentials) and a second, isolated
process backs ``oauth_interactive`` (authorization_code + PKCE + refresh_token +
DCR). The suite then drives each flow end-to-end through a real MCP client.

Tokens are observed two ways: the upstream's ``/_test/issued_tokens`` endpoint
(which tokens the provider genuinely minted) and the proxy's own
``mcp_oauth2_token_cache`` (what the proxy stored, keyed by server_id). Tests
invalidate the upstream and flush the proxy cache up front so token counts are
deterministic.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
import typing
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import uvicorn
import yaml
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
    mcp_oauth2_token_cache,
)
from litellm.proxy.proxy_server import (
    app as proxy_app,
    cleanup_router_config_variables,
    initialize,
)

OAUTH_SERVER_SCRIPT = Path("tests/mcp_tests/oauth_mcp_server.py")
CONFIG_TEMPLATE_PATH = Path("tests/mcp_tests/test_configs/test_config_mcp_oauth_e2e.yaml")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
START_TIMEOUT = 30


@pytest.fixture(scope="session", autouse=True)
def _clear_proxy_database_env() -> typing.Iterator[None]:
    mp = pytest.MonkeyPatch()
    mp.delenv("DATABASE_URL", raising=False)
    mp.setenv("LITELLM_MASTER_KEY", "sk-1234")
    try:
        yield
    finally:
        mp.undo()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _spawn_oauth_server(label: str) -> tuple[subprocess.Popen, str]:
    host = "127.0.0.1"
    port = _free_port()
    cmd = [
        sys.executable,
        str(OAUTH_SERVER_SCRIPT),
        "--host",
        host,
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
    )
    start = time.time()
    base = f"http://{host}:{port}"
    while True:
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise RuntimeError(
                f"OAuth MCP server ({label}) exited early.\nSTDOUT:\n{out.decode()}\nSTDERR:\n{err.decode()}"
            )
        try:
            with socket.create_connection((host, port), timeout=0.1):
                break
        except OSError:
            if time.time() - start > START_TIMEOUT:
                proc.terminate()
                raise TimeoutError(f"OAuth MCP server ({label}) did not start")
            time.sleep(0.05)
    return proc, base


def _terminate(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture(scope="session")
def oauth_m2m_server() -> typing.Iterator[str]:
    proc, base = _spawn_oauth_server("m2m")
    try:
        yield base
    finally:
        _terminate(proc)


@pytest.fixture(scope="session")
def oauth_interactive_server() -> typing.Iterator[str]:
    proc, base = _spawn_oauth_server("interactive")
    try:
        yield base
    finally:
        _terminate(proc)


def _reset_upstream(upstream_base: str) -> None:
    """Drop every token the upstream has issued so issued-token counts start at
    zero for the calling test. The provider process is shared across a session,
    so without this, token state leaks between tests."""
    r = httpx.post(f"{upstream_base}/_test/invalidate", timeout=5)
    r.raise_for_status()


def _get_issued_tokens(upstream_base: str) -> list[str]:
    r = httpx.get(f"{upstream_base}/_test/issued_tokens", timeout=5)
    r.raise_for_status()
    return r.json()


@pytest.fixture(scope="session")
def proxy_oauth_url(
    tmp_path_factory: pytest.TempPathFactory,
    oauth_m2m_server: str,
    oauth_interactive_server: str,
) -> typing.Iterator[str]:
    config_dir = tmp_path_factory.mktemp("mcp_oauth_e2e")
    config_path = config_dir / "config.yaml"
    config = yaml.safe_load(CONFIG_TEMPLATE_PATH.read_text())
    for entry in config["mcp_servers"].values():
        base = oauth_interactive_server if entry.get("oauth2_flow") == "authorization_code" else oauth_m2m_server
        entry["url"] = f"{base}/mcp"
        entry["token_url"] = f"{base}/token"
        if "authorization_url" in entry:
            entry["authorization_url"] = f"{base}/authorize"
        if "registration_url" in entry:
            entry["registration_url"] = f"{base}/register"
    config_path.write_text(yaml.safe_dump(config))

    cleanup_router_config_variables()
    asyncio.run(initialize(config=str(config_path), debug=True))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    host, port = sock.getsockname()

    uv_config = uvicorn.Config(proxy_app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(uv_config)

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.serve(sockets=[sock]))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    start = time.time()
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError("Proxy server failed to start")
        if time.time() - start > START_TIMEOUT:
            raise TimeoutError("Proxy did not start")
        time.sleep(0.05)

    yield f"http://{host}:{port}"

    server.should_exit = True
    thread.join(timeout=10)
    sock.close()


@pytest.fixture(autouse=True)
def _reset_proxy_token_cache() -> typing.Iterator[None]:
    """The proxy's MCP OAuth client_credentials cache is process-wide and would
    otherwise leak token state across tests, making /token recording assertions
    order-dependent. Reset before each test for deterministic counts."""
    mcp_oauth2_token_cache.flush_cache()
    yield


def _server_id_for(server_name: str) -> str:
    from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
        global_mcp_server_manager,
    )

    server = global_mcp_server_manager.get_mcp_server_by_name(server_name)
    assert server is not None, f"MCP server {server_name!r} not loaded"
    return server.server_id


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _drive_interactive_oauth(proxy_url: str, server_name: str, scope: str) -> dict:
    """Walk the full interactive authorization_code + PKCE handshake against the
    proxy, following each 302 by hand so every hop is observable.

    Returns a dict with the proxy /token response body plus the verifier,
    challenge, and the client state echoed back through /callback.
    """
    verifier, challenge = _pkce_pair()
    client_state = "client-state-" + secrets.token_urlsafe(8)
    client_redirect = f"http://127.0.0.1:{_free_port()}/callback"

    authorize = httpx.get(
        f"{proxy_url}/{server_name}/authorize",
        params={
            "redirect_uri": client_redirect,
            "client_id": "litellm-test-client",
            "state": client_state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "response_type": "code",
            "scope": scope,
        },
        follow_redirects=False,
        timeout=10,
    )
    assert authorize.is_redirect, (authorize.status_code, authorize.text)
    upstream_authorize_url = authorize.headers["location"]

    upstream = httpx.get(upstream_authorize_url, follow_redirects=False, timeout=10)
    assert upstream.is_redirect, (upstream.status_code, upstream.text)
    callback_url = upstream.headers["location"]

    callback = httpx.get(callback_url, follow_redirects=False, timeout=10)
    assert callback.is_redirect, (callback.status_code, callback.text)
    client_redirect_back = callback.headers["location"]

    returned = parse_qs(urlparse(client_redirect_back).query)
    code = returned["code"][0]

    token = httpx.post(
        f"{proxy_url}/{server_name}/token",
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": verifier,
            "client_id": "litellm-test-client",
            "redirect_uri": client_redirect,
        },
        timeout=10,
    )
    assert token.status_code == 200, token.text

    return {
        "token_response": token.json(),
        "verifier": verifier,
        "challenge": challenge,
        "client_state": client_state,
        "returned_state": returned.get("state", [None])[0],
        "client_redirect": client_redirect,
        "upstream_authorize_url": upstream_authorize_url,
    }


async def _list_tools_with_bearer(resource_url: str, bearer: str) -> list[str]:
    async with streamablehttp_client(
        url=resource_url,
        headers={"Authorization": f"Bearer {bearer}"},
    ) as (read, write, _sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [t.name for t in result.tools]


async def _list_tools_through_proxy(proxy_url: str, server_name: str) -> list[str]:
    async with streamablehttp_client(
        url=f"{proxy_url}/mcp",
        headers={
            "x-litellm-api-key": "Bearer sk-1234",
            "x-mcp-servers": server_name,
        },
    ) as (read, write, _sid):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return [t.name for t in result.tools]


async def _seed_per_user_oauth_token(user_id: str, server_id: str, access_token: str, ttl: int = 3600) -> None:
    """Plant an access_token in the proxy's per-user OAuth cache so the next MCP
    request for (user_id, server_id) finds it via the Redis fast path and skips
    both the preemptive 401 and the DB lookup. Bypasses the prisma-backed
    persistence layer that the unit tests already cover."""
    from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
        mcp_per_user_token_cache,
    )

    await mcp_per_user_token_cache.set(user_id, server_id, access_token, ttl=ttl)


async def _clear_per_user_oauth_token(user_id: str, server_id: str) -> None:
    from litellm.proxy._experimental.mcp_server.oauth2_token_cache import (
        mcp_per_user_token_cache,
    )

    await mcp_per_user_token_cache.delete(user_id, server_id)


class TestOAuthClientCredentialsRoundtrip:
    """Sanity: with OAuth client_credentials configured, a real MCP client can
    list and call tools through the proxy. Proves the AS/RS fixture works and
    the proxy is genuinely fetching+attaching a Bearer token the upstream minted."""

    @pytest.mark.asyncio
    async def test_m2m_handshake_lists_and_calls_tools(self, proxy_oauth_url: str, oauth_m2m_server: str) -> None:
        _reset_upstream(oauth_m2m_server)
        server_id = _server_id_for("oauth_m2m_primary")

        async with asyncio.timeout(30):
            async with streamablehttp_client(
                url=f"{proxy_oauth_url}/mcp",
                headers={
                    "x-litellm-api-key": "Bearer sk-1234",
                    "x-mcp-servers": "oauth_m2m_primary",
                },
            ) as (read, write, _sid):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = {t.name for t in tools.tools}
                    assert any(n.endswith("whoami_tool") for n in names), names

                    whoami = next(n for n in names if n.endswith("whoami_tool"))
                    result = await session.call_tool(whoami, arguments={"label": "ok"})
                    assert result.content
                    text = getattr(result.content[0], "text", None)
                    assert text == "authorized:ok"

        stored = mcp_oauth2_token_cache.get_cache(server_id)
        assert stored is not None and stored.startswith("tk_"), stored
        assert stored in _get_issued_tokens(oauth_m2m_server), (
            "proxy attached a token the upstream never issued via client_credentials"
        )


class TestOAuthTokenCacheIsolation:
    """The proxy caches client_credentials tokens keyed by server_id. Two
    distinct MCP servers — even with identical client_id/secret — must NOT
    share a token. Verifies the cache key is the server, not the (client_id,
    token_url) tuple; failure would mean a token issued for server A could
    be reused against server B even after server B's policy changes."""

    @pytest.mark.asyncio
    async def test_distinct_servers_get_distinct_tokens(self, proxy_oauth_url: str, oauth_m2m_server: str) -> None:
        _reset_upstream(oauth_m2m_server)
        primary_id = _server_id_for("oauth_m2m_primary")
        secondary_id = _server_id_for("oauth_m2m_secondary")

        await _list_tools_through_proxy(proxy_oauth_url, "oauth_m2m_primary")
        await _list_tools_through_proxy(proxy_oauth_url, "oauth_m2m_secondary")

        primary_token = mcp_oauth2_token_cache.get_cache(primary_id)
        secondary_token = mcp_oauth2_token_cache.get_cache(secondary_id)
        assert primary_token and secondary_token, (primary_token, secondary_token)
        assert primary_token != secondary_token, (
            "two distinct MCP servers shared a cached token; the cache key must be "
            "the server, not the (client_id, token_url) tuple"
        )

        issued = _get_issued_tokens(oauth_m2m_server)
        assert {primary_token, secondary_token} <= set(issued), (
            f"cached tokens not both issued upstream: cached={ {primary_token, secondary_token} } issued={set(issued)}"
        )

    @pytest.mark.asyncio
    async def test_same_server_reuses_cached_token(self, proxy_oauth_url: str, oauth_m2m_server: str) -> None:
        _reset_upstream(oauth_m2m_server)
        server_id = _server_id_for("oauth_m2m_primary")

        await _list_tools_through_proxy(proxy_oauth_url, "oauth_m2m_primary")
        first_token = mcp_oauth2_token_cache.get_cache(server_id)
        assert first_token and first_token.startswith("tk_"), first_token

        await _list_tools_through_proxy(proxy_oauth_url, "oauth_m2m_primary")
        await _list_tools_through_proxy(proxy_oauth_url, "oauth_m2m_primary")

        assert mcp_oauth2_token_cache.get_cache(server_id) == first_token, (
            "cached token changed across calls; proxy re-fetched instead of reusing"
        )
        issued = _get_issued_tokens(oauth_m2m_server)
        assert issued == [first_token], (
            f"expected exactly one token minted across 3 MCP calls, got {issued}. "
            f"Proxy is re-fetching the token on every call."
        )


class TestOAuthTokenStoredInBackend:
    """The proxy must actually persist the client_credentials token it fetched,
    not just attach it transiently. After one MCP call, the proxy's token cache
    (keyed by server_id) must hold a token, and that token must be one the
    upstream AS genuinely issued — proving the stored value is the real upstream
    token, not a placeholder."""

    @pytest.mark.asyncio
    async def test_client_credentials_token_persisted_and_matches_upstream(
        self, proxy_oauth_url: str, oauth_m2m_server: str
    ) -> None:
        _reset_upstream(oauth_m2m_server)

        server_id = _server_id_for("oauth_m2m_primary")
        assert mcp_oauth2_token_cache.get_cache(server_id) is None, (
            "token cache should be empty before the first MCP call"
        )

        await _list_tools_through_proxy(proxy_oauth_url, "oauth_m2m_primary")

        stored = mcp_oauth2_token_cache.get_cache(server_id)
        assert stored is not None, "proxy did not persist the fetched token"
        assert stored.startswith("tk_"), stored

        issued = _get_issued_tokens(oauth_m2m_server)
        assert stored in issued, (
            f"token stored in the proxy backend is not one the upstream AS issued. stored={stored!r} issued={issued!r}"
        )


class TestOAuthInteractiveAuthorizationCodeFlow:
    """Drives the full interactive authorization_code + PKCE handshake through
    the proxy's discoverable endpoints: /authorize -> upstream -> /callback ->
    /token. The upstream enforces PKCE (S256(code_verifier) must equal the
    code_challenge it recorded at /authorize), so a successful token exchange
    proves the proxy forwarded both the client's challenge and verifier
    unchanged. Also asserts the proxy round-trips the client's opaque state and
    that the returned access token authenticates against the resource server."""

    @pytest.mark.asyncio
    async def test_full_pkce_handshake_through_proxy(self, proxy_oauth_url: str, oauth_interactive_server: str) -> None:
        _reset_upstream(oauth_interactive_server)

        outcome = _drive_interactive_oauth(proxy_oauth_url, "oauth_interactive", scope="read call")

        body = outcome["token_response"]
        assert body.get("access_token"), body
        assert body["access_token"].startswith("tk_"), body
        assert body.get("token_type", "").lower() == "bearer", body

        assert outcome["returned_state"] == outcome["client_state"], (
            "proxy /callback did not round-trip the client's original state; "
            "a mismatch breaks the client's CSRF protection"
        )

        assert body["access_token"] in _get_issued_tokens(oauth_interactive_server), (
            "token returned to the client was not one the upstream AS issued; the "
            "PKCE-bound code exchange did not actually complete upstream"
        )

    @pytest.mark.asyncio
    async def test_issued_token_authenticates_against_resource_server(
        self, proxy_oauth_url: str, oauth_interactive_server: str
    ) -> None:
        _reset_upstream(oauth_interactive_server)

        outcome = _drive_interactive_oauth(proxy_oauth_url, "oauth_interactive", scope="read call")
        access_token = outcome["token_response"]["access_token"]

        async with asyncio.timeout(30):
            tools = await _list_tools_with_bearer(f"{oauth_interactive_server}/mcp", access_token)
        assert any(n.endswith("whoami_tool") for n in tools), tools

    @pytest.mark.asyncio
    async def test_refresh_token_yields_working_access_token(
        self, proxy_oauth_url: str, oauth_interactive_server: str
    ) -> None:
        """After the interactive handshake, the client exchanges its refresh
        token through the proxy for a fresh access token, and that new token must
        authenticate against the resource server. Proves the proxy mediates the
        refresh_token grant (not just the initial code exchange) and that the
        refreshed credential is genuinely usable upstream."""
        _reset_upstream(oauth_interactive_server)

        outcome = _drive_interactive_oauth(proxy_oauth_url, "oauth_interactive", scope="read call")
        first_access = outcome["token_response"]["access_token"]
        refresh_token = outcome["token_response"].get("refresh_token")
        assert refresh_token, (
            "proxy did not surface the upstream refresh_token to the client; "
            f"token response was {outcome['token_response']}"
        )

        refreshed = httpx.post(
            f"{proxy_oauth_url}/oauth_interactive/token",
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": "litellm-test-client",
            },
            timeout=10,
        )
        assert refreshed.status_code == 200, refreshed.text
        new_access = refreshed.json().get("access_token")
        assert new_access and new_access.startswith("tk_"), refreshed.text
        assert new_access != first_access, "refresh must mint a distinct access token, not echo the old one"
        assert new_access in _get_issued_tokens(oauth_interactive_server), (
            "refreshed token was not minted by the upstream; the proxy did not actually forward a refresh_token grant"
        )

        async with asyncio.timeout(30):
            tools = await _list_tools_with_bearer(f"{oauth_interactive_server}/mcp", new_access)
        assert any(n.endswith("whoami_tool") for n in tools), tools

    @pytest.mark.asyncio
    async def test_callback_rejects_untrusted_client_redirect_uri(
        self, proxy_oauth_url: str, oauth_interactive_server: str
    ) -> None:
        verifier, challenge = _pkce_pair()
        authorize = httpx.get(
            f"{proxy_oauth_url}/oauth_interactive/authorize",
            params={
                "redirect_uri": "https://evil.example.com/steal",
                "client_id": "litellm-test-client",
                "state": "x",
                "code_challenge": challenge,
                "code_challenge_method": "S256",
                "response_type": "code",
                "scope": "read",
            },
            follow_redirects=False,
            timeout=10,
        )
        assert authorize.status_code >= 400, (
            "proxy /authorize accepted an untrusted (non-loopback, non-allowlisted) "
            f"client redirect_uri; status={authorize.status_code}"
        )


class TestInteractivePKCEEndToEndThroughProxy:
    """Closes the interactive-PKCE loop end-to-end:

    1. client completes PKCE through the proxy and receives a real upstream
       access_token (already covered by ``TestOAuthInteractiveAuthorizationCodeFlow``)
    2. that token is stored on litellm against the calling user (in the
       per-user OAuth token cache that the proxy consults on every MCP request)
    3. the client lists tools through litellm presenting ONLY its litellm
       API key — the proxy must look up the stored upstream token and inject
       it as the Bearer header to the upstream MCP server. The client never
       presents the upstream token; it lives inside the gateway.

    Without this, the management-layer ``store_mcp_oauth_user_credential``
    unit tests prove the token is persisted but nothing pins that the proxy
    actually retrieves and injects it on the MCP path.

    NOTE: refresh through the proxy MCP path lives in DB-backed credentials
    (refresh_user_oauth_token reads/writes prisma). That is hermetically
    covered by ``TestOAuthInteractiveAuthorizationCodeFlow.test_refresh_token_yields_working_access_token``
    which drives the refresh_token grant directly through the proxy /token
    endpoint. A full DB-backed refresh-through-MCP test waits on a
    prisma-enabled e2e harness."""

    @pytest.mark.asyncio
    async def test_stored_user_token_is_injected_into_upstream_mcp_request(
        self, proxy_oauth_url: str, oauth_interactive_server: str
    ) -> None:
        from litellm.constants import LITELLM_PROXY_ADMIN_NAME

        _reset_upstream(oauth_interactive_server)
        server_id = _server_id_for("oauth_interactive")
        user_id = LITELLM_PROXY_ADMIN_NAME  # what sk-1234 resolves to

        outcome = _drive_interactive_oauth(proxy_oauth_url, "oauth_interactive", scope="read call")
        access_token = outcome["token_response"]["access_token"]
        assert access_token.startswith("tk_"), outcome["token_response"]
        issued_before = set(_get_issued_tokens(oauth_interactive_server))
        assert access_token in issued_before

        await _seed_per_user_oauth_token(user_id, server_id, access_token)
        try:
            async with asyncio.timeout(30):
                tools = await _list_tools_through_proxy(proxy_oauth_url, "oauth_interactive")
        finally:
            await _clear_per_user_oauth_token(user_id, server_id)

        assert any(n.endswith("whoami_tool") for n in tools), tools

        issued_after = set(_get_issued_tokens(oauth_interactive_server))
        assert access_token in issued_after, (
            "stored per-user token is gone from the upstream's issued set, "
            "which means the proxy did not present it to the upstream MCP server"
        )
        assert issued_after == issued_before, (
            "proxy minted an extra upstream token instead of reusing the stored "
            f"per-user one. before={issued_before!r} after={issued_after!r}"
        )

    @pytest.mark.asyncio
    async def test_missing_per_user_token_triggers_preemptive_401(
        self, proxy_oauth_url: str, oauth_interactive_server: str
    ) -> None:
        """Counterpart to the success case: if NO per-user token is stored for
        the calling user, the proxy must reject the MCP request with a 401
        + RFC 9728 WWW-Authenticate challenge so a standards-compliant client
        can kick off the PKCE flow. Pins the preemptive-401 guard for
        ``needs_user_oauth_token`` servers."""
        from litellm.constants import LITELLM_PROXY_ADMIN_NAME

        _reset_upstream(oauth_interactive_server)
        server_id = _server_id_for("oauth_interactive")
        await _clear_per_user_oauth_token(LITELLM_PROXY_ADMIN_NAME, server_id)

        async with httpx.AsyncClient(timeout=10) as http:
            r = await http.post(
                f"{proxy_oauth_url}/mcp",
                headers={
                    "x-litellm-api-key": "Bearer sk-1234",
                    "x-mcp-servers": "oauth_interactive",
                    "accept": "application/json, text/event-stream",
                    "content-type": "application/json",
                },
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            )

        assert r.status_code == 401, (r.status_code, r.text)
        challenge = r.headers.get("www-authenticate", "")
        assert "Bearer" in challenge, challenge
        assert "authorization_uri=" in challenge, (
            f"401 missing the RFC 9728 authorization_uri pointer clients need to "
            f"discover the proxy AS metadata; challenge={challenge!r}"
        )


class TestOAuthHealthCheckAndToolFetch:
    """LiteLLM's health check is flow-aware: for client_credentials it can fetch
    a token and reach the upstream, so it reports a concrete healthy/unhealthy
    status and the proxy can list upstream tools. It also re-authenticates once
    the cached token is gone, fetching a fresh token before listing tools. For
    interactive authorization_code servers (per-user tokens) there's no machine
    identity to health-check, so the upstream call is skipped."""

    @pytest.mark.asyncio
    async def test_client_credentials_health_check_is_healthy_and_lists_tools(
        self, proxy_oauth_url: str, oauth_m2m_server: str
    ) -> None:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server = global_mcp_server_manager.get_mcp_server_by_name("oauth_m2m_primary")
        assert server is not None

        async with asyncio.timeout(30):
            result = await global_mcp_server_manager.health_check_server(server.server_id)
        assert result.status == "healthy", (
            f"client_credentials health check should connect upstream and report "
            f"healthy; got status={result.status} error={result.health_check_error}"
        )

        tools = await _list_tools_through_proxy(proxy_oauth_url, "oauth_m2m_primary")
        assert any(n.endswith("whoami_tool") for n in tools), tools

    @pytest.mark.asyncio
    async def test_health_check_reauthenticates_after_token_expiry(
        self, proxy_oauth_url: str, oauth_m2m_server: str
    ) -> None:
        """Once the cached M2M token is gone (TTL expiry, simulated by flushing
        the proxy cache and invalidating it upstream), the health check must mint
        a fresh token and still reach the upstream. A second, distinct minted token
        proves it re-authenticated rather than reusing a dead token, and the proxy
        can still list upstream tools afterward."""
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        _reset_upstream(oauth_m2m_server)
        mcp_oauth2_token_cache.flush_cache()

        server = global_mcp_server_manager.get_mcp_server_by_name("oauth_m2m_primary")
        assert server is not None

        async with asyncio.timeout(30):
            first = await global_mcp_server_manager.health_check_server(server.server_id)
        assert first.status == "healthy", (first.status, first.health_check_error)
        first_tokens = _get_issued_tokens(oauth_m2m_server)
        assert len(first_tokens) == 1, first_tokens

        mcp_oauth2_token_cache.flush_cache()
        _reset_upstream(oauth_m2m_server)

        async with asyncio.timeout(30):
            second = await global_mcp_server_manager.health_check_server(server.server_id)
        assert second.status == "healthy", (second.status, second.health_check_error)
        second_tokens = _get_issued_tokens(oauth_m2m_server)
        assert len(second_tokens) == 1, second_tokens
        assert second_tokens[0] != first_tokens[0], (
            "health check did not re-fetch a fresh token after the cached one was evicted; it reused the dead token"
        )

        tools = await _list_tools_through_proxy(proxy_oauth_url, "oauth_m2m_primary")
        assert any(n.endswith("whoami_tool") for n in tools), tools

    @pytest.mark.asyncio
    async def test_interactive_server_health_check_skips_upstream(
        self, proxy_oauth_url: str, oauth_interactive_server: str
    ) -> None:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        server = global_mcp_server_manager.get_mcp_server_by_name("oauth_interactive")
        assert server is not None
        assert server.requires_per_user_auth, "interactive authorization_code server should require per-user auth"

        async with asyncio.timeout(30):
            result = await global_mcp_server_manager.health_check_server(server.server_id)
        assert result.status == "unknown", (
            f"interactive (per-user) server health check should skip the upstream "
            f"call and report unknown; got status={result.status}"
        )


class TestMCPPermissionManagement:
    """Permission-management surfaces for a loaded MCP server: internal-network
    gating (``available_on_public_internet``) and forwarding of admin static
    headers plus per-request extra headers to the upstream tool fetch."""

    def test_internal_only_server_gates_on_client_ip(self, proxy_oauth_url: str) -> None:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        internal_only = MCPServer(
            server_id="perm-internal",
            name="perm-internal",
            transport="http",
            available_on_public_internet=False,
        )
        public = MCPServer(
            server_id="perm-public",
            name="perm-public",
            transport="http",
            available_on_public_internet=True,
        )

        accessible = global_mcp_server_manager._is_server_accessible_from_ip
        assert accessible(internal_only, "8.8.8.8") is False
        assert accessible(internal_only, "10.0.0.5") is True
        assert accessible(public, "8.8.8.8") is True
        assert accessible(internal_only, None) is True

    def test_internal_only_server_filters_out_of_public_listing(self, proxy_oauth_url: str) -> None:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )
        from litellm.types.mcp_server.mcp_server_manager import MCPServer

        internal_only = MCPServer(
            server_id="perm-internal-filter",
            name="perm-internal-filter",
            transport="http",
            available_on_public_internet=False,
        )
        global_mcp_server_manager.registry[internal_only.server_id] = internal_only
        try:
            kept, blocked = global_mcp_server_manager.filter_server_ids_by_ip_with_info(
                [internal_only.server_id], "8.8.8.8"
            )
            assert internal_only.server_id not in kept
            assert blocked == 1

            kept_internal, blocked_internal = global_mcp_server_manager.filter_server_ids_by_ip_with_info(
                [internal_only.server_id], "10.0.0.5"
            )
            assert kept_internal == [internal_only.server_id]
            assert blocked_internal == 0
        finally:
            global_mcp_server_manager.registry.pop(internal_only.server_id, None)

    @pytest.mark.asyncio
    async def test_static_and_extra_headers_forwarded_upstream(
        self, proxy_oauth_url: str, oauth_m2m_server: str
    ) -> None:
        from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
            global_mcp_server_manager,
        )

        _reset_upstream(oauth_m2m_server)
        server = global_mcp_server_manager.get_mcp_server_by_name("oauth_m2m_primary")
        assert server is not None
        assert server.static_headers == {"x-litellm-tenant": "acme-internal"}

        captured: dict[str, dict[str, str]] = {}
        original = global_mcp_server_manager._create_mcp_client

        async def _capture(*args, **kwargs):
            captured["extra_headers"] = dict(kwargs.get("extra_headers") or {})
            return await original(*args, **kwargs)

        global_mcp_server_manager._create_mcp_client = _capture  # type: ignore[method-assign]
        try:
            async with asyncio.timeout(30):
                tools = await global_mcp_server_manager._get_tools_from_server(
                    server, extra_headers={"x-trace-id": "trace-abc"}
                )
        finally:
            global_mcp_server_manager._create_mcp_client = original  # type: ignore[method-assign]

        assert any(t.name.endswith("whoami_tool") for t in tools), [t.name for t in tools]

        forwarded = captured["extra_headers"]
        assert forwarded.get("x-litellm-tenant") == "acme-internal", forwarded
        assert forwarded.get("x-trace-id") == "trace-abc", forwarded
