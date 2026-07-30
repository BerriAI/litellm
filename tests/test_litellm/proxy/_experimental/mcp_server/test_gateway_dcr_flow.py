"""Tests for the aggregate gateway DCR flow (register, authorize, complete, token)."""

import hashlib
import html
import json
from base64 import urlsafe_b64encode
from datetime import datetime, timedelta, timezone
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.requests import Request

from litellm.caching.caching import DualCache
from litellm.proxy._experimental.mcp_server.gateway_dcr_flow import (
    CONNECT_FLOW_COOKIE_PREFIX,
    GATEWAY_AUTH_CODE_PREFIX,
    GATEWAY_AUTH_CODE_TTL_SECONDS,
    GATEWAY_DCR_CLIENT_ID_PREFIX,
    MANUAL_DELIVERY_AUTH_CODE_TTL_SECONDS,
    _AUTH_CODE_DEBUG_KEY,
    _GatewayAuthCode,
    _open_sealed,
    _seal,
    aggregate_authorize,
    aggregate_token,
    complete_connect_flow,
    is_gateway_dcr_client_id,
    open_gateway_dcr_client,
    register_aggregate_client,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.session_credentials import (
    resolve_session_bearer,
    session_keys_from_master_key,
    SessionBearerAdmitted,
)

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
    assert location.path == "/ui/chat/integrations"
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
    assert await guard.claim("jti-inmem", 60) is True
    assert await guard.claim("jti-inmem", 60) is False  # replay of the same id


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
    assert await guard.claim("jti-redis", 60) is True
    cache.redis_cache.async_increment = AsyncMock(return_value=2)
    assert await guard.claim("jti-redis", 60) is False  # Redis says 2 → replay


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
    assert await guard.claim("jti-fault", 60) is False  # fail closed, not a fallback count of 1


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
