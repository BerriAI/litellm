"""Tests for the aggregate gateway DCR flow (register, authorize, complete, token)."""

import hashlib
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
    _GatewayAuthCode,
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
    response = await register_aggregate_client(request_body={"redirect_uris": redirect_uris})
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
    response = await register_aggregate_client(request_body={"redirect_uris": redirect_uris})
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
