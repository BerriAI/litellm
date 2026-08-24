"""Tests for the aggregate gateway DCR flow (register, authorize, complete, token)."""

import hashlib
import json
import re
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.requests import Request

from litellm.caching.caching import DualCache
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
    _AUTH_CODE_DEBUG_KEY,
    CONNECT_FLOW_COOKIE_PREFIX,
    GATEWAY_AUTH_CODE_PREFIX,
    GATEWAY_AUTH_CODE_TTL_SECONDS,
    MANUAL_DELIVERY_AUTH_CODE_TTL_SECONDS,
    ConsentTeam,
    MintedProxyCredential,
    _GatewayAuthCode,
    _open_sealed,
    _seal,
    aggregate_authorize,
    aggregate_token,
    complete_connect_flow,
    is_gateway_dcr_client_id,
    is_proxy_api_resource,
    native_client_auth_contract,
    native_client_authorize,
    open_gateway_dcr_client,
    register_aggregate_client,
    revoke_refresh_token,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    SessionBearerAdmitted,
    SessionRefreshOpened,
    open_session_refresh_bearer,
    resolve_session_bearer,
    session_keys_from_master_key,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_token import SESSION_REFRESH_PREFIX

MASTER_KEY = "sk-gateway-dcr-flow-tests"
REDIRECT_URI = "https://claude.ai/api/mcp/auth_callback"
CODE_VERIFIER = "verifier-" + "v" * 43
CODE_CHALLENGE = urlsafe_b64encode(hashlib.sha256(CODE_VERIFIER.encode("ascii")).digest()).rstrip(b"=").decode("ascii")


@pytest.fixture(autouse=True)
def _salt_key(monkeypatch):
    monkeypatch.setenv("LITELLM_SALT_KEY", MASTER_KEY)


def _request(path="/authorize", query="", cookies=None, method="GET"):
    cookie_header = []
    if cookies:
        cookie = SimpleCookie()
        for name, value in cookies.items():
            cookie[name] = value
        cookie_header = [(b"cookie", cookie.output(header="", sep="; ").strip().encode())]
    return Request(
        {
            "type": "http",
            "method": method,
            "scheme": "https",
            "path": path,
            "query_string": query.encode(),
            "headers": [(b"host", b"llm.example.com"), *cookie_header],
        }
    )


async def _register(redirect_uris) -> dict:
    response = await register_aggregate_client(
        request=_request(path="/register", method="POST"), request_body={"redirect_uris": redirect_uris}
    )
    return json.loads(response.body)


async def _reload_user_active(user_id: str):
    return None


@pytest.mark.asyncio
async def test_register_mints_stateless_public_client():
    body = await _register([REDIRECT_URI])
    assert body["token_endpoint_auth_method"] == "none"
    assert "client_secret" not in body
    assert body["redirect_uris"] == [REDIRECT_URI]
    assert is_gateway_dcr_client_id(body["client_id"])
    record = open_gateway_dcr_client(body["client_id"])
    assert record is not None
    assert record.redirect_uris == (REDIRECT_URI,)


@pytest.mark.asyncio
async def test_register_allows_loopback_http_for_dev_clients():
    body = await _register(["http://localhost:6274/oauth/callback"])
    assert is_gateway_dcr_client_id(body["client_id"])


@pytest.mark.parametrize(
    "code_challenge",
    ["short", "", "p" * 300, "ünïcode-challenge", "AAAA" * 20],
)
def test_pkce_mismatched_challenge_returns_false_never_raises(code_challenge):
    """A wrong-length or non-ASCII code_challenge must VERIFY FALSE, not raise.

    Pins the reason this compares bytes rather than str: hmac.compare_digest raises TypeError on
    two str with non-ASCII content, but on bytes of unequal length it simply returns False. A
    review flagged this as an unhandled 500 on length mismatch; encoding both sides to bytes is
    exactly what makes that impossible, so the claim is pinned here rather than in a comment."""
    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import _pkce_verifier_matches

    assert _pkce_verifier_matches("a" * 43, code_challenge) is False


@pytest.mark.asyncio
async def test_register_allows_allowlisted_native_callback():
    """Native MCP clients register a private-use scheme, not https. Registration shares
    the one redirect-URI shape owner with /authorize, so the callback the allowlist
    already trusts there is registrable here rather than rejected as non-https."""
    body = await _register(["cursor://anysphere.cursor-mcp/oauth/callback"])
    assert is_gateway_dcr_client_id(body["client_id"])
    record = open_gateway_dcr_client(body["client_id"])
    assert record is not None
    assert record.redirect_uris == ("cursor://anysphere.cursor-mcp/oauth/callback",)


@pytest.mark.asyncio
async def test_register_rejects_userinfo_spoofed_origin():
    """``https://claude.ai@attacker.example/cb`` parses with netloc
    ``claude.ai@attacker.example``, so a naive origin display on the consent screen reads
    as claude.ai while the code would be delivered to attacker.example. Rejected at
    registration, which is the only way such a URI could enter a sealed client."""
    response = await register_aggregate_client(
        request=_request(path="/register", method="POST"),
        request_body={"redirect_uris": ["https://claude.ai@attacker.example/callback"]},
    )
    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "invalid_redirect_uri"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "redirect_uris",
    [
        [],
        "not-a-list",
        ["http://evil.example.com/callback"],
        ["https://claude.ai/cb#fragment"],
        ["ftp://claude.ai/cb"],
        ["https://a.example.com/" + "p" * 300],
        ["https://a.example.com/1", "https://a.example.com/2", "https://a.example.com/3", "https://a.example.com/4"],
        [12345],
    ],
)
async def test_register_rejects_bad_redirect_uris(redirect_uris):
    response = await register_aggregate_client(
        request=_request(path="/register", method="POST"), request_body={"redirect_uris": redirect_uris}
    )
    assert response.status_code == 400
    assert json.loads(response.body)["error"] in ("invalid_redirect_uri", "invalid_client_metadata")


@pytest.mark.asyncio
async def test_tampered_client_id_does_not_open():
    body = await _register([REDIRECT_URI])
    tampered = body["client_id"][:-4] + "AAAA"
    assert open_gateway_dcr_client(tampered) is None
    assert open_gateway_dcr_client("llm_dcrc_garbage") is None
    assert open_gateway_dcr_client("other_prefix") is None


def _authorize(
    client_id, session_user_id, redirect_uri=REDIRECT_URI, challenge=CODE_CHALLENGE, method="S256", response_type="code"
):
    return aggregate_authorize(
        request=_request(query=f"client_id={client_id}"),
        client_id=client_id,
        redirect_uri=redirect_uri,
        state="client-state-123",
        code_challenge=challenge,
        code_challenge_method=method,
        response_type=response_type,
        session_user_id=session_user_id,
    )


@pytest.mark.asyncio
async def test_authorize_validation_failures_never_redirect_to_client():
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    for response, expected_error in (
        (_authorize("llm_dcrc_bogus", "u1"), "invalid_client"),
        (_authorize(client_id, "u1", redirect_uri="https://attacker.example.com/cb"), "invalid_request"),
        (_authorize(client_id, "u1", response_type="token"), "unsupported_response_type"),
        (_authorize(client_id, "u1", challenge=None), "invalid_request"),
        (_authorize(client_id, "u1", method="plain"), "invalid_request"),
    ):
        assert response.status_code == 400
        assert json.loads(response.body)["error"] == expected_error


@pytest.mark.asyncio
async def test_authorize_without_session_redirects_to_login_with_return_to():
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    response = _authorize(client_id, session_user_id=None)
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("https://llm.example.com/sso/key/generate?return_to=")
    assert "return_to=%2Fauthorize" in location


@pytest.mark.asyncio
async def test_authorize_with_session_hands_browser_to_connect_page_with_flow_cookie():
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    response = _authorize(client_id, session_user_id="u1")
    assert response.status_code == 303
    location = urlparse(response.headers["location"])
    assert location.path == "/ui/connect"
    params = parse_qs(location.query)
    handle = params["connect_flow"][0]
    assert params["connect_client"] == ["https://claude.ai"]
    set_cookie = response.headers["set-cookie"]
    assert f"{CONNECT_FLOW_COOKIE_PREFIX}{handle}" in set_cookie
    assert "HttpOnly" in set_cookie
    return handle, set_cookie


def _flow_cookie_from(response) -> tuple:
    location = urlparse(response.headers["location"])
    handle = parse_qs(location.query)["connect_flow"][0]
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    name = f"{CONNECT_FLOW_COOKIE_PREFIX}{handle}"
    return handle, {name: cookie[name].value}


@pytest.mark.asyncio
async def test_full_walk_register_authorize_complete_token_and_replay():
    """The whole front door on one deterministic walk: register -> authorize ->
    complete -> token, then the security edges on the same artifacts (user mismatch,
    PKCE mismatch, single-use replay, refresh rotation, cross-client refresh)."""
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    authorize_response = _authorize(client_id, session_user_id="u1")
    handle, cookies = _flow_cookie_from(authorize_response)

    denied = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="attacker",
        cache=DualCache(),
    )
    assert denied.status_code == 403

    anonymous = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id=None,
        cache=DualCache(),
    )
    assert anonymous.status_code == 401

    completed = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="u1",
        cache=DualCache(),
    )
    assert completed.status_code == 303
    redirect = urlparse(completed.headers["location"])
    assert f"{redirect.scheme}://{redirect.netloc}{redirect.path}" == REDIRECT_URI
    params = parse_qs(redirect.query)
    assert params["state"] == ["client-state-123"]
    code = params["code"][0]
    assert code.startswith(GATEWAY_AUTH_CODE_PREFIX)

    cache = DualCache()

    async def _token(**overrides):
        arguments = {
            "request": _request("/token", method="POST"),
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": REDIRECT_URI,
            "client_id": client_id,
            "code_verifier": CODE_VERIFIER,
            "refresh_token": None,
            "master_key": MASTER_KEY,
            "reload_user": _reload_user_active,
            "cache": cache,
        }
        return await aggregate_token(**{**arguments, **overrides})

    wrong_verifier = await _token(code_verifier="wrong-" + "w" * 43)
    assert json.loads(wrong_verifier.body)["error"] == "invalid_grant"

    wrong_client = await _token(client_id=(await _register([REDIRECT_URI]))["client_id"])
    assert json.loads(wrong_client.body)["error"] == "invalid_grant"

    token_response = await _token()
    assert token_response.status_code == 200
    payload = json.loads(token_response.body)
    assert payload["token_type"] == "Bearer"
    assert 0 < payload["expires_in"] <= 3600

    keys = session_keys_from_master_key(MASTER_KEY)
    admitted = resolve_session_bearer(f"Bearer {payload['access_token']}", keys, datetime.now(timezone.utc))
    assert isinstance(admitted, SessionBearerAdmitted)
    assert admitted.principal.user_id == "u1"
    assert admitted.principal.client_id == client_id

    replay = await _token()
    assert json.loads(replay.body)["error"] == "invalid_grant"

    refreshed = await _token(grant_type="refresh_token", code=None, refresh_token=payload["refresh_token"])
    assert refreshed.status_code == 200
    rotated = json.loads(refreshed.body)
    assert rotated["refresh_token"] != payload["refresh_token"]

    # Rotation is single-use: replaying the now-consumed refresh token cannot mint a second pair
    # (a captured token is dead once the legitimate holder has rotated).
    replayed = await _token(grant_type="refresh_token", code=None, refresh_token=payload["refresh_token"])
    assert json.loads(replayed.body)["error"] == "invalid_grant"
    assert "already used" in json.loads(replayed.body).get("error_description", "")

    cross_client = await _token(
        grant_type="refresh_token",
        code=None,
        refresh_token=payload["refresh_token"],
        client_id=(await _register([REDIRECT_URI]))["client_id"],
    )
    assert json.loads(cross_client.body)["error"] == "invalid_grant"


@pytest.mark.asyncio
async def test_complete_rejects_missing_tampered_and_expired_flows():
    missing = await complete_connect_flow(
        request=_request("/authorize/complete", method="POST"),
        flow_handle="nope",
        session_user_id="u1",
        cache=DualCache(),
    )
    assert missing.status_code == 400

    tampered = await complete_connect_flow(
        request=_request("/authorize/complete", cookies={f"{CONNECT_FLOW_COOKIE_PREFIX}h1": "garbage"}, method="POST"),
        flow_handle="h1",
        session_user_id="u1",
        cache=DualCache(),
    )
    assert tampered.status_code == 400


@pytest.mark.asyncio
async def test_token_rejects_expired_code_and_missing_configuration():
    expired_code = _seal(
        GATEWAY_AUTH_CODE_PREFIX,
        _GatewayAuthCode(
            user_id="u1",
            client_id="llm_dcrc_x",
            redirect_uri=REDIRECT_URI,
            code_challenge=CODE_CHALLENGE,
            jti="jti-1",
            iat=int((datetime.now(timezone.utc) - timedelta(seconds=500)).timestamp()),
            exp=int((datetime.now(timezone.utc) - timedelta(seconds=500 - GATEWAY_AUTH_CODE_TTL_SECONDS)).timestamp()),
        ),
    )
    response = await aggregate_token(
        request=_request("/token", method="POST"),
        grant_type="authorization_code",
        code=expired_code,
        redirect_uri=REDIRECT_URI,
        client_id="llm_dcrc_x",
        code_verifier=CODE_VERIFIER,
        refresh_token=None,
        master_key=MASTER_KEY,
        reload_user=_reload_user_active,
        cache=DualCache(),
    )
    assert json.loads(response.body)["error"] == "invalid_grant"

    no_master_key = await aggregate_token(
        request=_request("/token", method="POST"),
        grant_type="authorization_code",
        code="llm_gcode_x",
        redirect_uri=REDIRECT_URI,
        client_id="llm_dcrc_x",
        code_verifier=CODE_VERIFIER,
        refresh_token=None,
        master_key=None,
        reload_user=_reload_user_active,
        cache=DualCache(),
    )
    assert no_master_key.status_code == 500
    assert json.loads(no_master_key.body)["error"] == "server_error"

    unsupported = await aggregate_token(
        request=_request("/token", method="POST"),
        grant_type="password",
        code=None,
        redirect_uri=None,
        client_id="llm_dcrc_x",
        code_verifier=None,
        refresh_token=None,
        master_key=MASTER_KEY,
        reload_user=_reload_user_active,
        cache=DualCache(),
    )
    assert json.loads(unsupported.body)["error"] == "unsupported_grant_type"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,expected_status,expected_error",
    [
        ("no_active_key", 400, "invalid_grant"),
        ("unavailable", 503, "temporarily_unavailable"),
        ("unresolvable", 500, "server_error"),
    ],
)
async def test_token_gates_on_live_user_revalidation(failure, expected_status, expected_error):
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    authorize_response = _authorize(client_id, session_user_id="deactivated-user")
    handle, cookies = _flow_cookie_from(authorize_response)
    completed = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="deactivated-user",
        cache=DualCache(),
    )
    code = parse_qs(urlparse(completed.headers["location"]).query)["code"][0]

    async def _reload_user_failing(user_id: str):
        return failure

    response = await aggregate_token(
        request=_request("/token", method="POST"),
        grant_type="authorization_code",
        code=code,
        redirect_uri=REDIRECT_URI,
        client_id=client_id,
        code_verifier=CODE_VERIFIER,
        refresh_token=None,
        master_key=MASTER_KEY,
        reload_user=_reload_user_failing,
        cache=DualCache(),
    )
    assert response.status_code == expected_status
    assert json.loads(response.body)["error"] == expected_error


@pytest.mark.asyncio
async def test_flow_is_single_use_shared_cache_rejects_second_complete():
    """A double-submit of the finish step mints only ONE code: the second complete over the
    same cache fails invalid_request (atomic flow claim), so one sign-in cannot yield two codes."""
    cache = DualCache()
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    handle, cookies = _flow_cookie_from(_authorize(client_id, session_user_id="u1"))

    first = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="u1",
        cache=cache,
    )
    assert first.status_code == 303
    second = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="u1",
        cache=cache,
    )
    assert second.status_code == 400
    assert json.loads(second.body)["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_token_rejects_out_of_range_code_verifier():
    """RFC 7636: a code_verifier outside 43-128 chars is invalid_request, not a confusing
    invalid_grant PKCE-mismatch."""
    for bad in ["short", "x" * 200]:
        response = await aggregate_token(
            request=_request("/token", method="POST"),
            grant_type="authorization_code",
            code="llm_gcode_whatever",
            redirect_uri=REDIRECT_URI,
            client_id="llm_dcrc_x",
            code_verifier=bad,
            refresh_token=None,
            master_key=MASTER_KEY,
            reload_user=_reload_user_active,
            cache=DualCache(),
        )
        assert response.status_code == 400
        assert json.loads(response.body)["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_authorize_rejects_over_long_state():
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    response = aggregate_authorize(
        request=_request(query=f"client_id={client_id}"),
        client_id=client_id,
        redirect_uri=REDIRECT_URI,
        state="s" * 2000,
        code_challenge=CODE_CHALLENGE,
        code_challenge_method="S256",
        response_type="code",
        session_user_id="u1",
    )
    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "invalid_request"


@pytest.mark.asyncio
async def test_non_ascii_code_challenge_fails_grant_not_500():
    """A non-ASCII code_challenge (unvalidated from the client) must yield a clean
    invalid_grant, never a TypeError-driven 500 (bytes comparison, not str)."""
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    # Seal a code carrying a non-ASCII challenge directly (authorize requires S256 shape,
    # but the challenge charset is not validated there, so this state is reachable).
    from datetime import datetime, timezone

    code = _seal(
        GATEWAY_AUTH_CODE_PREFIX,
        _GatewayAuthCode(
            user_id="u1",
            client_id=client_id,
            redirect_uri=REDIRECT_URI,
            code_challenge="challenge-with-€-non-ascii",
            jti="jti-x",
            iat=int(datetime.now(timezone.utc).timestamp()),
            exp=int(datetime.now(timezone.utc).timestamp()) + 120,
        ),
    )
    response = await aggregate_token(
        request=_request("/token", method="POST"),
        grant_type="authorization_code",
        code=code,
        redirect_uri=REDIRECT_URI,
        client_id=client_id,
        code_verifier=CODE_VERIFIER,
        refresh_token=None,
        master_key=MASTER_KEY,
        reload_user=_reload_user_active,
        cache=DualCache(),
    )
    assert response.status_code == 400
    assert json.loads(response.body)["error"] == "invalid_grant"


@pytest.mark.asyncio
async def test_single_use_guard_in_memory_is_single_use_within_process():
    """No Redis configured (single-replica): the in-memory increment is authoritative — the first claim
    wins, a replay of the same id loses."""
    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import _SingleUseGuard

    guard = _SingleUseGuard(DualCache())  # redis_cache is None
    assert await guard.claim("jti-inmem", 60) == "first"
    assert await guard.claim("jti-inmem", 60) == "replayed"


@pytest.mark.asyncio
async def test_single_use_guard_uses_redis_as_sole_authority_when_configured():
    """With Redis configured it is the SOLE authority: the shared INCR result decides the claim (1 →
    first caller, >1 → replay), and the per-worker in-memory count is never consulted."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import _SingleUseGuard

    cache = DualCache()
    cache.redis_cache = MagicMock()
    cache.redis_cache.async_increment = AsyncMock(return_value=1)
    # in-memory must NOT be consulted when Redis is configured — poison it so any fallback is visible.
    cache.async_increment_cache = AsyncMock(side_effect=AssertionError("must not fall back to in-memory"))

    guard = _SingleUseGuard(cache)
    assert await guard.claim("jti-redis", 60) == "first"
    cache.redis_cache.async_increment = AsyncMock(return_value=2)
    assert await guard.claim("jti-redis", 60) == "replayed"


@pytest.mark.asyncio
async def test_single_use_guard_fails_closed_when_redis_errors():
    """A Redis fault must fail the claim CLOSED (refuse the id) rather than fall back to the per-worker
    in-memory count — which would let each replica observe count==1 and replay the one-time id (the
    Cursor/Veria replay-across-workers finding)."""
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import _SingleUseGuard

    cache = DualCache()
    cache.redis_cache = MagicMock()
    cache.redis_cache.async_increment = AsyncMock(side_effect=ConnectionError("redis down"))
    cache.async_increment_cache = AsyncMock(return_value=1)  # would fail OPEN if the guard fell back

    guard = _SingleUseGuard(cache)
    assert await guard.claim("jti-fault", 60) == "unavailable"  # fail closed, not a fallback count of 1


LOOPBACK_REDIRECT_URI = "http://localhost:3118/callback"


async def _complete(redirect_uri: str, delivery, cookies=None, handle=None, session_user_id="u1"):
    client_id = (await _register([redirect_uri]))["client_id"]
    if cookies is None:
        handle, cookies = _flow_cookie_from(_authorize(client_id, session_user_id="u1", redirect_uri=redirect_uri))
    response = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id=session_user_id,
        cache=DualCache(),
        delivery=delivery,
    )
    return client_id, response


def _callback_url_from_page(response) -> str:
    import html as html_lib
    import re

    match = re.search(r'value="([^"]+)"', response.body.decode())
    assert match is not None
    return html_lib.unescape(match.group(1))


@pytest.mark.asyncio
async def test_manual_delivery_renders_pasteable_callback_url_for_loopback_client():
    """The LIT-4863 headless path: a loopback client on another machine gets the callback
    URL on a page instead of a dead 303, and the code on that page is a full-fidelity
    authorization code (PKCE-bound, single-use, redeemable at /token)."""
    client_id, response = await _complete(LOOPBACK_REDIRECT_URI, delivery="manual")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert f"{CONNECT_FLOW_COOKIE_PREFIX}" in response.headers["set-cookie"]

    callback_url = _callback_url_from_page(response)
    parsed = urlparse(callback_url)
    assert f"{parsed.scheme}://{parsed.netloc}{parsed.path}" == LOOPBACK_REDIRECT_URI
    params = parse_qs(parsed.query)
    assert params["state"] == ["client-state-123"]
    code = params["code"][0]
    assert code.startswith(GATEWAY_AUTH_CODE_PREFIX)

    cache = DualCache()
    token_response = await aggregate_token(
        request=_request("/token", method="POST"),
        grant_type="authorization_code",
        code=code,
        redirect_uri=LOOPBACK_REDIRECT_URI,
        client_id=client_id,
        code_verifier=CODE_VERIFIER,
        refresh_token=None,
        master_key=MASTER_KEY,
        reload_user=_reload_user_active,
        cache=cache,
    )
    assert token_response.status_code == 200

    replay = await aggregate_token(
        request=_request("/token", method="POST"),
        grant_type="authorization_code",
        code=code,
        redirect_uri=LOOPBACK_REDIRECT_URI,
        client_id=client_id,
        code_verifier=CODE_VERIFIER,
        refresh_token=None,
        master_key=MASTER_KEY,
        reload_user=_reload_user_active,
        cache=cache,
    )
    assert json.loads(replay.body)["error"] == "invalid_grant"


@pytest.mark.asyncio
async def test_manual_delivery_code_gets_the_longer_ttl_and_redirect_code_does_not():
    _, manual = await _complete(LOOPBACK_REDIRECT_URI, delivery="manual")
    manual_code = parse_qs(urlparse(_callback_url_from_page(manual)).query)["code"][0]
    opened_manual = _open_sealed(manual_code, GATEWAY_AUTH_CODE_PREFIX, _GatewayAuthCode, _AUTH_CODE_DEBUG_KEY)
    assert opened_manual is not None
    assert opened_manual.exp - opened_manual.iat == MANUAL_DELIVERY_AUTH_CODE_TTL_SECONDS

    _, redirected = await _complete(LOOPBACK_REDIRECT_URI, delivery=None)
    redirect_code = parse_qs(urlparse(redirected.headers["location"]).query)["code"][0]
    opened_redirect = _open_sealed(redirect_code, GATEWAY_AUTH_CODE_PREFIX, _GatewayAuthCode, _AUTH_CODE_DEBUG_KEY)
    assert opened_redirect is not None
    assert opened_redirect.exp - opened_redirect.iat == GATEWAY_AUTH_CODE_TTL_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", [None, "redirect"])
async def test_loopback_client_still_redirects_when_manual_not_requested(delivery):
    _, response = await _complete(LOOPBACK_REDIRECT_URI, delivery=delivery)
    assert response.status_code == 303
    assert response.headers["location"].startswith(LOOPBACK_REDIRECT_URI)


@pytest.mark.asyncio
async def test_manual_delivery_is_ignored_for_routable_redirect_uri():
    """A routable redirect URI works from any browser by construction, so manual is a
    no-op there and the flow keeps its normal shape."""
    _, response = await _complete(REDIRECT_URI, delivery="manual")
    assert response.status_code == 303
    assert response.headers["location"].startswith(REDIRECT_URI)


@pytest.mark.asyncio
async def test_unknown_delivery_value_is_rejected_before_the_flow_is_consumed():
    """A typo'd delivery must not burn the single-use flow: the user fixes the form and
    finishes normally."""
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    handle, cookies = _flow_cookie_from(_authorize(client_id, session_user_id="u1", redirect_uri=LOOPBACK_REDIRECT_URI))

    rejected = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="u1",
        cache=DualCache(),
        delivery="carrier-pigeon",
    )
    assert rejected.status_code == 400
    assert json.loads(rejected.body)["error"] == "invalid_request"

    retried = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="u1",
        cache=DualCache(),
        delivery="manual",
    )
    assert retried.status_code == 200


@pytest.mark.asyncio
async def test_manual_delivery_page_escapes_client_influenced_values():
    """redirect_uri (and everything else on the page) is client-registered input; a quote
    or tag in its path must render inert."""
    hostile_uri = 'http://127.0.0.1:9/cb"><script>alert(1)</script>'
    _, response = await _complete(hostile_uri, delivery="manual")
    assert response.status_code == 200
    body = response.body.decode()
    assert "<script>alert(1)</script>" not in body
    assert "&lt;script&gt;" in body


class _TtlRecordingCache(DualCache):
    """Captures the TTL of every single-use claim recorded through the in-memory arm."""

    def __init__(self):
        super().__init__()
        self.claim_ttls: dict = {}

    async def async_increment_cache(self, key, value, ttl=None, **kwargs):
        self.claim_ttls[key] = ttl
        return await super().async_increment_cache(key, value, ttl=ttl, **kwargs)


@pytest.mark.asyncio
async def test_used_code_marker_outlives_the_manually_delivered_code():
    """Veria review finding on the LIT-4863 change: a manual code lives 300s, but the
    used-code marker was retained for the 120s redirect lifetime plus buffer, so a client
    could redeem, wait out the marker, and redeem the still-valid code again. The marker's
    TTL must cover the code's own remaining lifetime plus the claim buffer."""
    client_id, response = await _complete(LOOPBACK_REDIRECT_URI, delivery="manual")
    code = parse_qs(urlparse(_callback_url_from_page(response)).query)["code"][0]

    cache = _TtlRecordingCache()
    token_response = await aggregate_token(
        request=_request("/token", method="POST"),
        grant_type="authorization_code",
        code=code,
        redirect_uri=LOOPBACK_REDIRECT_URI,
        client_id=client_id,
        code_verifier=CODE_VERIFIER,
        refresh_token=None,
        master_key=MASTER_KEY,
        reload_user=_reload_user_active,
        cache=cache,
    )
    assert token_response.status_code == 200

    marker_ttls = [ttl for key, ttl in cache.claim_ttls.items() if key.startswith("mcp_gateway_dcr_code_used:")]
    assert len(marker_ttls) == 1
    assert marker_ttls[0] >= MANUAL_DELIVERY_AUTH_CODE_TTL_SECONDS


@pytest.mark.asyncio
@pytest.mark.parametrize("redirect_uri", [LOOPBACK_REDIRECT_URI, "http://127.0.0.1:9/cb$(whoami)&calc& rem x"])
async def test_manual_delivery_page_renders_the_url_as_data_never_as_a_shell_command(redirect_uri):
    """Two review rounds proved no single command string is safe across POSIX shells,
    cmd.exe, and PowerShell (single quotes are not quoting in cmd.exe; percent expands
    there even inside double quotes), so the page must render the callback URL as data
    only and never as a ready-to-paste command."""
    _, response = await _complete(redirect_uri, delivery="manual")
    assert response.status_code == 200
    body = response.body.decode()
    assert "<code>" not in body
    assert 'curl "' not in body
    assert "curl '" not in body
    assert 'value="' in body


def _scoped_mcp_server(name="github", **kw):
    from litellm.types.mcp import MCPAuth
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

    return MCPServer(
        server_id=f"{name}-id",
        name=name,
        server_name=name,
        alias=name,
        url="https://upstream.example/mcp",
        transport="http",
        auth_type=MCPAuth.oauth2,
        **kw,
    )


SCOPED_RESOURCE = "https://llm.example.com/mcp/github"
_MANAGER_PATCH = "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager"


def _scoped_authorize(client_id, resource, session_user_id="u1"):
    return aggregate_authorize(
        request=_request(query=f"client_id={client_id}"),
        client_id=client_id,
        redirect_uri=REDIRECT_URI,
        state="client-state-123",
        code_challenge=CODE_CHALLENGE,
        code_challenge_method="S256",
        response_type="code",
        session_user_id=session_user_id,
        resource=resource,
    )


async def _redeem(code, client_id, cache=None, **overrides):
    arguments = {
        "request": _request("/token", method="POST"),
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": CODE_VERIFIER,
        "refresh_token": None,
        "master_key": MASTER_KEY,
        "reload_user": _reload_user_active,
        "cache": cache or DualCache(),
    }
    return await aggregate_token(**{**arguments, **overrides})


def _opened_principal(payload):
    keys = session_keys_from_master_key(MASTER_KEY)
    admitted = resolve_session_bearer(f"Bearer {payload['access_token']}", keys, datetime.now(timezone.utc))
    assert isinstance(admitted, SessionBearerAdmitted)
    return admitted.principal


async def _finish_connect_page(response):
    handle, cookies = _flow_cookie_from(response)
    completed = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="u1",
        cache=DualCache(),
    )
    return parse_qs(urlparse(completed.headers["location"]).query)["code"][0]


def _sealed_wire_json(sealed, prefix, debug_key):
    from litellm.proxy.common_utils.encrypt_decrypt_utils import decrypt_value_helper

    raw = decrypt_value_helper(sealed.removeprefix(prefix), debug_key, return_original_value=False)
    assert isinstance(raw, str)
    return json.loads(raw)


@pytest.mark.asyncio
async def test_scoped_authorize_runs_connect_page_with_sealed_scope():
    """LIT-4917: a per-server RFC 8707 resource naming a gateway-managed oauth2 server
    seals that server into the flow. The connect page interlude runs exactly as before
    (the scope restricts, it never skips consent), and the code minted at the finish step
    and the session pair it redeems for are both scoped."""
    from unittest.mock import patch

    client_id = (await _register([REDIRECT_URI]))["client_id"]
    with patch(_MANAGER_PATCH) as manager:
        manager.get_mcp_server_by_name.return_value = _scoped_mcp_server()
        response = _scoped_authorize(client_id, SCOPED_RESOURCE)
    assert response.status_code == 303
    assert "/ui/connect" in response.headers["location"]
    _, cookies = _flow_cookie_from(response)
    assert (
        _sealed_wire_json(next(iter(cookies.values())), "", "gateway_connect_flow")["resource_server_id"] == "github-id"
    )
    code = await _finish_connect_page(response)
    assert (
        _sealed_wire_json(code, GATEWAY_AUTH_CODE_PREFIX, "gateway_authorization_code")["resource_server_id"]
        == "github-id"
    )
    token_response = await _redeem(code, client_id)
    assert token_response.status_code == 200
    principal = _opened_principal(json.loads(token_response.body))
    assert principal.resource_server_id == "github-id"
    assert principal.user_id == "u1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource, resolves",
    [
        (None, False),
        ("https://llm.example.com/mcp", False),
        ("https://other.example.com/mcp/github", False),
        ("https://llm.example.com/mcp/github,linear", False),
        ("https://llm.example.com/mcp/unknown", None),
        ("not a url", False),
    ],
)
async def test_unscoped_resources_leave_flow_and_token_byte_identical(resource, resolves):
    """Every resource shape outside 'exactly one gateway-managed server' keeps today's flow:
    connect page interlude, and NONE of the minted artifacts carry the scope key on the
    wire, not the flow cookie, not the code, not the session JWT, so an unscoped flow
    started on a new pod completes on a pod whose strict models predate the claim."""
    import base64
    from unittest.mock import patch

    client_id = (await _register([REDIRECT_URI]))["client_id"]
    with patch(_MANAGER_PATCH) as manager:
        manager.get_mcp_server_by_name.return_value = None if resolves is None else _scoped_mcp_server()
        response = _scoped_authorize(client_id, resource)
    assert response.status_code == 303
    assert "/ui/connect" in response.headers["location"]
    _, cookies = _flow_cookie_from(response)
    assert "resource_server_id" not in _sealed_wire_json(next(iter(cookies.values())), "", "gateway_connect_flow")
    code = await _finish_connect_page(response)
    assert "resource_server_id" not in _sealed_wire_json(code, GATEWAY_AUTH_CODE_PREFIX, "gateway_authorization_code")
    token_response = await _redeem(code, client_id)
    payload = json.loads(token_response.body)
    assert _opened_principal(payload).resource_server_id is None
    jwt_payload_segment = payload["access_token"].removeprefix("llm_session_").split(".")[1]
    claims = json.loads(base64.urlsafe_b64decode(jwt_payload_segment + "=" * (-len(jwt_payload_segment) % 4)))
    assert "resource_server_id" not in claims


@pytest.mark.asyncio
async def test_scoped_authorize_delegate_server_stays_unscoped():
    """A delegate-auth oauth2 server is outside the gateway-managed set (its keyless flow is
    upstream PKCE via the relay), so a resource naming it never scopes the gateway flow."""
    from unittest.mock import patch

    client_id = (await _register([REDIRECT_URI]))["client_id"]
    with patch(_MANAGER_PATCH) as manager:
        manager.get_mcp_server_by_name.return_value = _scoped_mcp_server(delegate_auth_to_upstream=True)
        response = _scoped_authorize(client_id, SCOPED_RESOURCE)
    assert "/ui/connect" in response.headers["location"]
    code = await _finish_connect_page(response)
    token_response = await _redeem(code, client_id)
    assert _opened_principal(json.loads(token_response.body)).resource_server_id is None


@pytest.mark.asyncio
async def test_token_rejects_resource_conflicting_with_sealed_scope():
    """RFC 8707 section 2.2: redeeming a scoped code (or rotating a scoped refresh token)
    for a DIFFERENT resource fails with invalid_target; an absent resource redeems fine and
    the sealed scope still binds the minted pair, surviving refresh rotation."""
    from unittest.mock import patch

    client_id = (await _register([REDIRECT_URI]))["client_id"]
    github = _scoped_mcp_server()
    linear = _scoped_mcp_server(name="linear")
    with patch(_MANAGER_PATCH) as manager:
        manager.get_mcp_server_by_name.return_value = github
        response = _scoped_authorize(client_id, SCOPED_RESOURCE)
    code = await _finish_connect_page(response)

    with patch(_MANAGER_PATCH) as manager:
        manager.get_mcp_server_by_name.return_value = linear
        mismatched = await _redeem(code, client_id, resource="https://llm.example.com/mcp/linear")
    assert json.loads(mismatched.body)["error"] == "invalid_target"

    cache = DualCache()
    token_response = await _redeem(code, client_id, cache=cache)
    payload = json.loads(token_response.body)
    assert _opened_principal(payload).resource_server_id == "github-id"

    with patch(_MANAGER_PATCH) as manager:
        manager.get_mcp_server_by_name.return_value = linear
        refresh_mismatch = await _redeem(
            None,
            client_id,
            cache=cache,
            grant_type="refresh_token",
            refresh_token=payload["refresh_token"],
            resource="https://llm.example.com/mcp/linear",
        )
    assert json.loads(refresh_mismatch.body)["error"] == "invalid_target"

    rotated = await _redeem(
        None, client_id, cache=cache, grant_type="refresh_token", refresh_token=payload["refresh_token"]
    )
    assert rotated.status_code == 200
    assert _opened_principal(json.loads(rotated.body)).resource_server_id == "github-id"


@pytest.mark.asyncio
async def test_resolve_scoped_resource_server_matrix():
    """Unit pin of the resource resolver: both per-server URL spellings resolve; the
    aggregate resource, foreign hosts, CSV paths, unknown names, and non-gateway-managed
    modes all return None so nothing outside the served set can enter the scoped flow."""
    from unittest.mock import patch

    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import resolve_scoped_resource_server

    request = _request()
    github = _scoped_mcp_server()
    for resource, resolved_server, expected in [
        ("https://llm.example.com/mcp/github", github, "github-id"),
        ("https://llm.example.com/github/mcp", github, "github-id"),
        ("https://LLM.example.com/mcp/github/", github, "github-id"),
        ("https://llm.example.com/mcp", github, None),
        ("https://other.example.com/mcp/github", github, None),
        ("https://llm.example.com/mcp/a,b", github, None),
        ("https://llm.example.com/mcp/github", None, None),
        ("https://llm.example.com/mcp/github", _scoped_mcp_server(delegate_auth_to_upstream=True), None),
        (None, github, None),
    ]:
        with patch(_MANAGER_PATCH) as manager:
            manager.get_mcp_server_by_name.return_value = resolved_server
            result = resolve_scoped_resource_server(request, resource)
        assert (result.server_id if result is not None else None) == expected, resource


@pytest.mark.asyncio
async def test_resource_resolution_is_identity_not_ip_filtered_access():
    """The resolver decides which server a resource NAMES; per-IP visibility filtering
    belongs to the MCP routes and grant intersection. Filtering here would mint an
    entitlement-wide unscoped bearer exactly when the caller asked to narrow, and IP drift
    between authorize and token would turn a matching redemption into invalid_target."""
    from unittest.mock import patch

    from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import resolve_scoped_resource_server

    with patch(_MANAGER_PATCH) as manager:
        manager.get_mcp_server_by_name.return_value = _scoped_mcp_server()
        result = resolve_scoped_resource_server(_request(), SCOPED_RESOURCE)
    assert result is not None
    manager.get_mcp_server_by_name.assert_called_once_with("github")


LOOPBACK_REDIRECT_URI = "http://127.0.0.1:51234/callback"
PROXY_API_RESOURCE = "https://llm.example.com"
CONSENT_TEAMS = (ConsentTeam(team_id="team-a", team_alias="Team A"), ConsentTeam(team_id="team-b"))


class _Minter:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    async def __call__(self, user_id, team_id):
        self.calls.append((user_id, team_id))
        if self.result is not None:
            return self.result
        return MintedProxyCredential(key=f"sk-cli-{user_id}", expires_in=3600, user_id=user_id, team_id=team_id)


class _ConsentTeams:
    def __init__(self, result=CONSENT_TEAMS):
        self.calls = []
        self.result = result

    async def __call__(self, user_id):
        self.calls.append(user_id)
        return self.result


async def _native_authorize(client_id, session_user_id="u1", lookup=None, **overrides):
    arguments = {
        "request": _request(query=f"resource={PROXY_API_RESOURCE}"),
        "client_id": client_id,
        "redirect_uri": LOOPBACK_REDIRECT_URI,
        "state": "client-state-123",
        "code_challenge": CODE_CHALLENGE,
        "code_challenge_method": "S256",
        "response_type": "code",
        "session_user_id": session_user_id,
        "lookup_consent_teams": lookup if lookup is not None else _ConsentTeams(),
    }
    return await native_client_authorize(**{**arguments, **overrides})


def _consent_cookie_from(response) -> tuple:
    match = re.search(r'name="flow" value="([^"]+)"', response.body.decode())
    assert match is not None
    handle = match.group(1)
    cookie = SimpleCookie()
    cookie.load(response.headers["set-cookie"])
    name = f"{CONNECT_FLOW_COOKIE_PREFIX}{handle}"
    return handle, {name: cookie[name].value}


async def _complete_consent(consent, cache=None, session_user_id="u1", **overrides):
    handle, cookies = _consent_cookie_from(consent)
    return await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id=session_user_id,
        cache=cache or DualCache(),
        **overrides,
    )


def _code_from(response) -> str:
    return parse_qs(urlparse(response.headers["location"]).query)["code"][0]


async def _native_code(client_id, team_id="team-b", cache=None) -> str:
    approved = await _complete_consent(
        await _native_authorize(client_id), cache=cache, decision="approve", team_id=team_id
    )
    assert approved.status_code == 303
    return _code_from(approved)


async def _redeem_native(code, client_id, minter, cache=None, resource=PROXY_API_RESOURCE, **overrides):
    return await _redeem(
        code,
        client_id,
        cache=cache,
        redirect_uri=LOOPBACK_REDIRECT_URI,
        resource=resource,
        mint_proxy_credential=minter,
        **overrides,
    )


async def _refresh_native(refresh_token, client_id, minter, cache, **overrides):
    return await _redeem_native(
        None, client_id, minter, cache=cache, grant_type="refresh_token", refresh_token=refresh_token, **overrides
    )


def _opened_refresh(refresh_token, client_id):
    opened = open_session_refresh_bearer(
        refresh_token,
        session_keys_from_master_key(MASTER_KEY),
        datetime.now(timezone.utc),
        expected_client_id=client_id,
    )
    assert isinstance(opened, SessionRefreshOpened)
    return opened.principal


@pytest.mark.asyncio
async def test_native_authorize_renders_consent_page_and_sets_flow_cookie():
    """A native client (RFC 8707 resource = the proxy itself) gets the server-rendered consent
    page instead of the MCP connect-page redirect: the flow handle rides only in the hidden
    field, the sealed flow in an HttpOnly cookie, and the page can never be framed or cached."""
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    lookup = _ConsentTeams()
    response = await _native_authorize(client_id, lookup=lookup)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert lookup.calls == ["u1"]
    body = response.body.decode()
    assert "http://127.0.0.1:51234" in body
    assert "/callback" not in body
    assert "<strong>u1</strong>" in body
    assert '<option value="team-a">Team A</option>' in body
    assert '<option value="team-b">team-b</option>' in body
    assert 'action="https://llm.example.com/authorize/complete"' in body
    handle, cookies = _consent_cookie_from(response)
    assert "httponly" in response.headers["set-cookie"].lower()
    flow = _sealed_wire_json(next(iter(cookies.values())), "", "gateway_connect_flow")
    assert flow["audience"] == "proxy_api"
    assert flow["client_id"] == client_id
    assert flow["redirect_uri"] == LOOPBACK_REDIRECT_URI
    assert flow["user_id"] == "u1"
    assert "resource_server_id" not in flow
    assert handle not in body.replace(f'value="{handle}"', "")


@pytest.mark.asyncio
async def test_native_authorize_without_session_redirects_to_login_before_any_lookup():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    lookup = _ConsentTeams()
    response = await _native_authorize(client_id, session_user_id=None, lookup=lookup)
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("https://llm.example.com/sso/key/generate?return_to=")
    assert "return_to=%2Fauthorize%3Fresource%3D" in location
    assert lookup.calls == []
    assert "set-cookie" not in response.headers


@pytest.mark.asyncio
async def test_native_authorize_validation_failures_never_reach_consent():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    lookup = _ConsentTeams()
    for presented_client_id, overrides, expected_error in (
        ("llm_dcrc_bogus", {}, "invalid_client"),
        (client_id, {"redirect_uri": "http://127.0.0.1:51235/callback"}, "invalid_request"),
        (client_id, {"response_type": "token"}, "unsupported_response_type"),
        (client_id, {"code_challenge": None}, "invalid_request"),
        (client_id, {"code_challenge_method": "plain"}, "invalid_request"),
    ):
        response = await _native_authorize(presented_client_id, lookup=lookup, **overrides)
        assert response.status_code == 400
        assert json.loads(response.body)["error"] == expected_error
        assert "set-cookie" not in response.headers
    assert lookup.calls == []


@pytest.mark.asyncio
async def test_native_authorize_refuses_a_hosted_redirect_for_the_proxy_api():
    """Registration accepts any https redirect because MCP clients can be hosted, but a
    proxy-API grant hands out the user's personal key, so it only ever goes back to loopback."""
    hosted = "https://evil.example/cb"
    client_id = (await _register([hosted]))["client_id"]
    lookup = _ConsentTeams()
    response = await _native_authorize(client_id, redirect_uri=hosted, lookup=lookup)
    assert response.status_code == 400
    assert json.loads(response.body) == {
        "error": "invalid_request",
        "error_description": "a proxy-API grant may only redirect to a loopback address",
    }
    assert "set-cookie" not in response.headers
    assert lookup.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, status, error",
    [
        ("unavailable", 503, "temporarily_unavailable"),
        ("unresolvable", 500, "server_error"),
        ("no_active_key", 403, "access_denied"),
    ],
)
async def test_native_authorize_consent_lookup_failures_are_oauth_errors_without_a_flow(failure, status, error):
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    response = await _native_authorize(client_id, lookup=_ConsentTeams(failure))
    assert response.status_code == status
    assert json.loads(response.body)["error"] == error
    assert "set-cookie" not in response.headers


@pytest.mark.asyncio
async def test_native_consent_escapes_untrusted_identifiers():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    hostile = (ConsentTeam(team_id='t"><script>', team_alias="<b>Team</b>"), ConsentTeam(team_id="team-b"))
    response = await _native_authorize(
        client_id, session_user_id='<img src=x onerror="x">', lookup=_ConsentTeams(hostile)
    )
    body = response.body.decode()
    assert "<script>" not in body
    assert "<b>Team</b>" not in body
    assert "<img" not in body
    assert "&lt;b&gt;Team&lt;/b&gt;" in body
    assert "&lt;img src=x onerror=&quot;x&quot;&gt;" in body


@pytest.mark.asyncio
async def test_native_deny_redirects_with_access_denied_and_burns_the_flow():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    consent = await _native_authorize(client_id)
    denied = await _complete_consent(consent, cache=cache, decision="deny", team_id="team-a")
    assert denied.status_code == 303
    location = urlparse(denied.headers["location"])
    assert f"{location.scheme}://{location.netloc}{location.path}" == LOOPBACK_REDIRECT_URI
    assert parse_qs(location.query) == {"error": ["access_denied"], "state": ["client-state-123"]}
    assert "max-age=0" in denied.headers["set-cookie"].lower()
    retried = await _complete_consent(consent, cache=cache, decision="approve", team_id="team-a")
    assert retried.status_code == 400


@pytest.mark.asyncio
async def test_native_invalid_decision_is_rejected_before_the_flow_is_consumed():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    consent = await _native_authorize(client_id)
    bad = await _complete_consent(consent, cache=cache, decision="maybe", team_id="team-a")
    assert bad.status_code == 400
    assert json.loads(bad.body)["error"] == "invalid_request"
    assert "set-cookie" not in bad.headers
    approved = await _complete_consent(consent, cache=cache, decision="approve", team_id="team-a")
    assert approved.status_code == 303
    assert "code" in parse_qs(urlparse(approved.headers["location"]).query)


@pytest.mark.asyncio
async def test_native_complete_by_another_session_user_is_refused():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    consent = await _native_authorize(client_id)
    hijacked = await _complete_consent(consent, session_user_id="attacker", decision="approve", team_id="team-a")
    assert hijacked.status_code == 403


@pytest.mark.asyncio
async def test_native_approve_seals_audience_and_chosen_team_into_the_code():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    approved = await _complete_consent(await _native_authorize(client_id), decision="approve", team_id="team-b")
    assert approved.status_code == 303
    assert "max-age=0" in approved.headers["set-cookie"].lower()
    location = urlparse(approved.headers["location"])
    assert location.netloc == "127.0.0.1:51234"
    assert location.path == "/callback"
    query = parse_qs(location.query)
    assert query["state"] == ["client-state-123"]
    wire = _sealed_wire_json(query["code"][0], GATEWAY_AUTH_CODE_PREFIX, _AUTH_CODE_DEBUG_KEY)
    assert wire["audience"] == "proxy_api"
    assert wire["team_id"] == "team-b"
    assert wire["user_id"] == "u1"


@pytest.mark.asyncio
@pytest.mark.parametrize("team_id", [None, ""])
async def test_native_approve_without_a_team_mints_an_unscoped_credential(team_id):
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    consent = await _native_authorize(client_id, lookup=_ConsentTeams(()))
    assert 'name="team_id"' not in consent.body.decode()
    approved = await _complete_consent(consent, decision="approve", team_id=team_id)
    minter = _Minter()
    response = await _redeem_native(_code_from(approved), client_id, minter)
    assert response.status_code == 200
    assert minter.calls == [("u1", None)]
    payload = json.loads(response.body)
    assert payload["team_id"] is None
    assert _opened_refresh(payload["refresh_token"], client_id).team_id is None


@pytest.mark.asyncio
async def test_native_code_redeems_for_the_proxy_credential_and_a_rotating_refresh_token():
    """The whole native walk on one set of artifacts: code -> CLI credential + refresh, code
    replay refused, refresh rotates and keeps the team, the old refresh is dead."""
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    minter = _Minter()
    code = await _native_code(client_id, team_id="team-b", cache=cache)
    response = await _redeem_native(code, client_id, minter, cache=cache)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    payload = json.loads(response.body)
    assert minter.calls == [("u1", "team-b")]
    assert payload["access_token"] == "sk-cli-u1"
    assert payload["token_type"] == "Bearer"
    assert payload["expires_in"] == 3600
    assert payload["user_id"] == "u1"
    assert payload["team_id"] == "team-b"
    assert payload["refresh_token"].startswith(SESSION_REFRESH_PREFIX)
    principal = _opened_refresh(payload["refresh_token"], client_id)
    assert principal.audience == "proxy_api"
    assert principal.team_id == "team-b"
    assert principal.user_id == "u1"
    assert principal.client_id == client_id

    replayed = await _redeem_native(code, client_id, minter, cache=cache)
    assert replayed.status_code == 400
    assert json.loads(replayed.body)["error"] == "invalid_grant"
    assert "already used" in json.loads(replayed.body)["error_description"]

    refreshed = await _refresh_native(payload["refresh_token"], client_id, minter, cache)
    assert refreshed.status_code == 200
    rotated = json.loads(refreshed.body)
    assert minter.calls[-1] == ("u1", "team-b")
    assert rotated["access_token"] == "sk-cli-u1"
    assert rotated["team_id"] == "team-b"
    assert rotated["refresh_token"] != payload["refresh_token"]
    assert _opened_refresh(rotated["refresh_token"], client_id).team_id == "team-b"

    stale = await _refresh_native(payload["refresh_token"], client_id, minter, cache)
    assert stale.status_code == 400
    assert "already used" in json.loads(stale.body)["error_description"]
    assert "access_token" not in json.loads(stale.body)


@pytest.mark.asyncio
async def test_native_refresh_is_bound_to_the_issuing_client():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    other = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    payload = json.loads(
        (await _redeem_native(await _native_code(client_id, cache=cache), client_id, _Minter(), cache=cache)).body
    )
    minter = _Minter()
    stolen = await _refresh_native(payload["refresh_token"], other, minter, cache)
    assert stolen.status_code == 400
    assert json.loads(stolen.body)["error"] == "invalid_grant"
    assert minter.calls == []


@pytest.mark.asyncio
async def test_native_code_without_a_minter_is_refused_server_side():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    code = await _native_code(client_id)
    response = await _redeem(code, client_id, redirect_uri=LOOPBACK_REDIRECT_URI, resource=PROXY_API_RESOURCE)
    assert response.status_code == 500
    assert json.loads(response.body)["error"] == "server_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, status, error",
    [
        ("not_a_member", 400, "invalid_grant"),
        ("team_required", 400, "invalid_grant"),
        ("no_active_key", 400, "invalid_grant"),
        ("unavailable", 503, "temporarily_unavailable"),
        ("unresolvable", 500, "server_error"),
    ],
)
async def test_native_mint_failure_maps_to_an_oauth_error_and_keeps_the_code_usable(failure, status, error):
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    code = await _native_code(client_id, cache=cache)
    failed = await _redeem_native(code, client_id, _Minter(failure), cache=cache)
    assert failed.status_code == status
    assert json.loads(failed.body)["error"] == error
    assert "refresh_token" not in json.loads(failed.body)
    retried = await _redeem_native(code, client_id, _Minter(), cache=cache)
    assert retried.status_code == 200


@pytest.mark.asyncio
async def test_native_refresh_mint_failure_does_not_burn_the_refresh_token():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    payload = json.loads(
        (await _redeem_native(await _native_code(client_id, cache=cache), client_id, _Minter(), cache=cache)).body
    )
    failed = await _refresh_native(payload["refresh_token"], client_id, _Minter("unavailable"), cache)
    assert failed.status_code == 503
    retried = await _refresh_native(payload["refresh_token"], client_id, _Minter(), cache)
    assert retried.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource", [PROXY_API_RESOURCE, "https://llm.example.com/", "HTTPS://LLM.EXAMPLE.COM:443", None]
)
async def test_native_code_accepts_the_proxy_resource_in_any_equivalent_spelling(resource):
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    response = await _redeem_native(await _native_code(client_id), client_id, _Minter(), resource=resource)
    assert response.status_code == 200


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource", ["https://llm.example.com/mcp", "https://other.example.com", "http://llm.example.com", "not a url"]
)
async def test_native_code_refuses_a_foreign_resource_without_burning_it(resource):
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    code = await _native_code(client_id, cache=cache)
    minter = _Minter()
    refused = await _redeem_native(code, client_id, minter, cache=cache, resource=resource)
    assert refused.status_code == 400
    assert json.loads(refused.body)["error"] == "invalid_target"
    assert minter.calls == []
    retried = await _redeem_native(code, client_id, minter, cache=cache)
    assert retried.status_code == 200


@pytest.mark.asyncio
async def test_mcp_code_ignores_the_minter_and_still_yields_a_session_pair():
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    code = await _finish_connect_page(_authorize(client_id, session_user_id="u1"))
    minter = _Minter()
    response = await _redeem(code, client_id, mint_proxy_credential=minter, resource=PROXY_API_RESOURCE)
    assert response.status_code == 200
    payload = json.loads(response.body)
    assert minter.calls == []
    principal = _opened_principal(payload)
    assert principal.audience is None
    assert principal.team_id is None
    assert "user_id" not in payload


@pytest.mark.asyncio
async def test_mcp_wire_formats_carry_no_native_client_fields():
    client_id = (await _register([REDIRECT_URI]))["client_id"]
    handle, cookies = _flow_cookie_from(_authorize(client_id, session_user_id="u1"))
    flow_wire = _sealed_wire_json(next(iter(cookies.values())), "", "gateway_connect_flow")
    assert "audience" not in flow_wire
    assert "team_id" not in flow_wire
    completed = await complete_connect_flow(
        request=_request("/authorize/complete", cookies=cookies, method="POST"),
        flow_handle=handle,
        session_user_id="u1",
        cache=DualCache(),
        team_id="team-a",
        decision="approve",
    )
    code_wire = _sealed_wire_json(_code_from(completed), GATEWAY_AUTH_CODE_PREFIX, _AUTH_CODE_DEBUG_KEY)
    assert "audience" not in code_wire
    assert "team_id" not in code_wire


@pytest.mark.asyncio
async def test_revoke_burns_the_refresh_token_and_answers_200_for_dead_or_unknown_tokens():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    payload = json.loads(
        (await _redeem_native(await _native_code(client_id, cache=cache), client_id, _Minter(), cache=cache)).body
    )
    revoked = await revoke_refresh_token(
        token=payload["refresh_token"], client_id=client_id, master_key=MASTER_KEY, cache=cache
    )
    assert revoked.status_code == 200
    assert json.loads(revoked.body) == {}
    assert revoked.headers["cache-control"] == "no-store"
    refreshed = await _refresh_native(payload["refresh_token"], client_id, _Minter(), cache)
    assert refreshed.status_code == 400
    assert json.loads(refreshed.body)["error"] == "invalid_grant"
    again = await revoke_refresh_token(
        token=payload["refresh_token"], client_id=client_id, master_key=MASTER_KEY, cache=cache
    )
    assert again.status_code == 200
    garbage = await revoke_refresh_token(token="nonsense", client_id=client_id, master_key=MASTER_KEY, cache=cache)
    assert garbage.status_code == 200


def _redis_that(async_increment):
    from unittest.mock import AsyncMock, MagicMock

    cache = DualCache()
    cache.redis_cache = MagicMock()
    cache.redis_cache.async_increment = async_increment
    cache.async_increment_cache = AsyncMock(side_effect=AssertionError("must not fall back to in-memory"))
    return cache


@pytest.mark.asyncio
async def test_revoke_answers_503_while_the_shared_record_cannot_be_written_then_burns_the_token():
    """A revocation whose single-use marker never reached Redis must not report success: the token is
    still redeemable on every worker, so the client has to hear 503 (RFC 7009 2.2.1) and retry."""
    from unittest.mock import AsyncMock

    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    issued = DualCache()
    payload = json.loads(
        (await _redeem_native(await _native_code(client_id, cache=issued), client_id, _Minter(), cache=issued)).body
    )

    redis_down = await revoke_refresh_token(
        token=payload["refresh_token"],
        client_id=client_id,
        master_key=MASTER_KEY,
        cache=_redis_that(AsyncMock(side_effect=ConnectionError("redis down"))),
    )
    assert redis_down.status_code == 503
    assert json.loads(redis_down.body) == {
        "error": "temporarily_unavailable",
        "error_description": "the single-use record is unavailable right now; try again shortly",
    }
    assert redis_down.headers["cache-control"] == "no-store"

    redis_back = await revoke_refresh_token(
        token=payload["refresh_token"],
        client_id=client_id,
        master_key=MASTER_KEY,
        cache=_redis_that(AsyncMock(return_value=1)),
    )
    assert redis_back.status_code == 200
    assert json.loads(redis_back.body) == {}

    already_burned = await revoke_refresh_token(
        token=payload["refresh_token"],
        client_id=client_id,
        master_key=MASTER_KEY,
        cache=_redis_that(AsyncMock(return_value=2)),
    )
    assert already_burned.status_code == 200


@pytest.mark.asyncio
async def test_refresh_answers_503_without_burning_the_token_while_redis_is_down():
    """Fail closed, but say why: a refresh the shared backend could not record is refused with 503
    rather than ``invalid_grant``, so the CLI keeps the key it has and retries instead of telling the
    user the token was already used and sending them back through the browser."""
    from unittest.mock import AsyncMock

    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    issued = DualCache()
    payload = json.loads(
        (await _redeem_native(await _native_code(client_id, cache=issued), client_id, _Minter(), cache=issued)).body
    )

    redis_down = await _refresh_native(
        payload["refresh_token"], client_id, _Minter(), _redis_that(AsyncMock(side_effect=ConnectionError("redis down")))
    )
    assert redis_down.status_code == 503
    assert json.loads(redis_down.body)["error"] == "temporarily_unavailable"
    assert "refresh_token" not in json.loads(redis_down.body)

    redis_back = await _refresh_native(payload["refresh_token"], client_id, _Minter(), _redis_that(AsyncMock(return_value=1)))
    assert redis_back.status_code == 200
    assert json.loads(redis_back.body)["refresh_token"] != payload["refresh_token"]

    replayed = await _refresh_native(payload["refresh_token"], client_id, _Minter(), _redis_that(AsyncMock(return_value=2)))
    assert replayed.status_code == 400
    assert json.loads(replayed.body)["error"] == "invalid_grant"


@pytest.mark.asyncio
async def test_revoke_from_another_client_leaves_the_token_usable():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    other = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    cache = DualCache()
    payload = json.loads(
        (await _redeem_native(await _native_code(client_id, cache=cache), client_id, _Minter(), cache=cache)).body
    )
    revoked = await revoke_refresh_token(
        token=payload["refresh_token"], client_id=other, master_key=MASTER_KEY, cache=cache
    )
    assert revoked.status_code == 200
    refreshed = await _refresh_native(payload["refresh_token"], client_id, _Minter(), cache)
    assert refreshed.status_code == 200


@pytest.mark.asyncio
async def test_revoke_refuses_unknown_clients_and_a_missing_master_key():
    client_id = (await _register([LOOPBACK_REDIRECT_URI]))["client_id"]
    for bogus in ("llm_dcrc_bogus", "anything"):
        response = await revoke_refresh_token(token="x", client_id=bogus, master_key=MASTER_KEY, cache=DualCache())
        assert response.status_code == 401
        assert json.loads(response.body)["error"] == "invalid_client"
    no_key = await revoke_refresh_token(token="x", client_id=client_id, master_key=None, cache=DualCache())
    assert no_key.status_code == 500
    assert json.loads(no_key.body)["error"] == "server_error"


def test_native_client_auth_contract_points_every_endpoint_at_this_proxy():
    assert json.loads(json.dumps(native_client_auth_contract(_request("/.well-known/litellm-cli-auth")))) == {
        "contract_version": 1,
        "issuer": "https://llm.example.com",
        "authorization_endpoint": "https://llm.example.com/authorize",
        "token_endpoint": "https://llm.example.com/token",
        "registration_endpoint": "https://llm.example.com/register",
        "revocation_endpoint": "https://llm.example.com/revoke",
        "resource": "https://llm.example.com",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "revocation_endpoint_auth_methods_supported": ["none"],
    }


@pytest.mark.parametrize(
    "resource, expected",
    [
        ("https://llm.example.com", True),
        ("https://llm.example.com/", True),
        ("HTTPS://LLM.EXAMPLE.COM:443", True),
        ("https://llm.example.com/mcp", False),
        ("http://llm.example.com", False),
        ("https://other.example.com", False),
        ("llm.example.com", False),
        ("", False),
        (None, False),
    ],
)
def test_is_proxy_api_resource_matches_only_this_proxy(resource, expected):
    assert is_proxy_api_resource(_request(), resource) is expected
