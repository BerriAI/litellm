"""The dashboard session (the `token` cookie and the session key inside it) versus API key auth.

Contract (litellm/proxy/proxy_server.py `login_v2`; litellm/proxy/auth/login_utils.py `authenticate_user`,
`encode_ui_session_jwt`, `_invalid_credentials_message`; litellm/proxy/management_endpoints/ui_sso.py
`set_session_token_cookie`; litellm/proxy/management_endpoints/session_endpoints.py `session_logout`):

* `POST /v2/login` with the admin username and the master key answers 200 with `{"redirect_url", "token"}` and
  sets a `token` cookie (SameSite=lax, not HttpOnly) holding an HS256 JWT signed with the master key. The JWT
  carries an `exp` and a `key` claim: a fresh `sk-` session key stored only by its hash, in team
  `litellm-dashboard`, owned by the proxy admin.
* The cookie is not a credential: API routes read keys from headers only. The JWT itself is not a credential
  either while JWT auth is off. Only the session key, sent as a bearer key, authenticates.
* Wrong password and unknown username get the same 401 (`param` `invalid_credentials`), and nothing echoes the
  submitted password or the master key.
* `POST /session/logout` revokes exactly the presenting session key, refuses ordinary virtual keys with 403, and
  clears the cookie.

The Rust gateway has its own UI session path (`litellm/rust/crates/gateway-auth/src/ui.rs`); these pins are for
the Python proxy the Rust auth core must keep answering the same way.
"""

from __future__ import annotations

import uuid
from typing import Final

import jwt
import pytest
from integration._support.client import Gateway, string_value
from integration.security.dec1_authsec_support import (
    INVALID_VIRTUAL_KEY_PREFIX,
    NO_KEY_MESSAGE,
    assert_auth_error,
    digest,
    login,
    raw_get,
)

ADMIN_USERNAME: Final = "admin"
UI_TEAM: Final = "litellm-dashboard"


def _session(gateway: Gateway) -> tuple[str, dict[str, object]]:
    """Log in as the admin and return the session JWT and its verified claims."""
    response: Final = login(gateway, ADMIN_USERNAME, gateway.key)
    assert response.status_code == 200, response.text
    token: Final = string_value(response.json()["token"])
    claims: Final = jwt.decode(token, gateway.key, algorithms=["HS256"])
    return token, claims


def _logout(gateway: Gateway, session_key: str) -> None:
    gateway.request("POST", "/session/logout", {}, key=session_key)


def test_admin_login_returns_a_redirect_and_a_token_and_sets_the_cookie(gateway: Gateway) -> None:
    """`POST /v2/login` with `admin` and the master key must answer 200 with exactly the keys `redirect_url`
    and `token`, `redirect_url` must end with `/ui?login=success`, and a `token` cookie must be set to the same
    value with `SameSite=lax` and without `HttpOnly` (the dashboard reads it from script)."""
    response: Final = login(gateway, ADMIN_USERNAME, gateway.key)
    assert response.status_code == 200, response.text
    body: Final = response.json()
    assert set(body) == {"redirect_url", "token"}, body
    assert string_value(body["redirect_url"]).endswith("/ui?login=success")
    cookie: Final = response.headers.get("set-cookie", "")
    assert cookie.startswith(f"token={body['token']}"), cookie
    assert "samesite=lax" in cookie.lower() and "httponly" not in cookie.lower(), cookie
    _logout(gateway, string_value(jwt.decode(body["token"], gateway.key, algorithms=["HS256"])["key"]))


def test_the_session_jwt_is_signed_with_the_master_key_and_expires(gateway: Gateway) -> None:
    """The login token must verify as HS256 with the master key as the secret, and must carry an integer `exp`
    claim (`encode_ui_session_jwt`). Verifying it with any other secret must fail."""
    token, claims = _session(gateway)
    assert isinstance(claims["exp"], int), claims
    with pytest.raises(jwt.InvalidSignatureError):
        jwt.decode(token, "dec1-authsec-not-the-master-key", algorithms=["HS256"])
    _logout(gateway, string_value(claims["key"]))


def test_the_session_key_is_stored_only_as_a_hash_in_the_dashboard_team(gateway: Gateway) -> None:
    """The `key` claim of the login token must start with `sk-`, and `LiteLLM_VerificationToken` must hold one
    row whose `token` is its SHA-256 and whose `team_id` is `litellm-dashboard`. No row may hold the raw key."""
    _, claims = _session(gateway)
    session_key: Final = string_value(claims["key"])
    assert session_key.startswith("sk-")
    rows: Final = gateway.rows(
        'SELECT token, team_id FROM "LiteLLM_VerificationToken" WHERE token=%s OR token=%s',
        (digest(session_key), session_key),
    )
    assert rows == [{"token": digest(session_key), "team_id": UI_TEAM}], rows
    _logout(gateway, session_key)


def test_the_session_cookie_alone_does_not_authenticate_an_api_call(gateway: Gateway) -> None:
    """A caller sends `Cookie: token=<session JWT>` and no credential header to `GET /key/list`. The proxy must
    answer 401 with exactly `Authentication Error, No api key passed in.`: cookies are never read as credentials,
    so a cross site request that carries the dashboard cookie cannot act as the admin."""
    token, claims = _session(gateway)
    refused: Final = raw_get(gateway, "/key/list", {"Cookie": f"token={token}"})
    assert_auth_error(refused, 401, NO_KEY_MESSAGE)
    _logout(gateway, string_value(claims["key"]))


def test_the_session_jwt_itself_is_not_a_bearer_credential(gateway: Gateway) -> None:
    """The caller sends `Authorization: Bearer <session JWT>` to `GET /key/list` on a proxy with JWT auth off.
    The proxy must refuse with 401 and a message starting `LiteLLM Virtual Key expected. Received=`: the JWT is
    not an `sk-` key and is not a UI hash key, and the refusal must not contain the full JWT."""
    token, claims = _session(gateway)
    refused: Final = raw_get(gateway, "/key/list", {"Authorization": f"Bearer {token}"})
    message: Final = assert_auth_error(refused, 401)
    assert message.startswith(INVALID_VIRTUAL_KEY_PREFIX), message
    assert token not in refused.text
    _logout(gateway, string_value(claims["key"]))


def test_the_session_key_is_an_admin_on_management_routes(gateway: Gateway) -> None:
    """The `key` claim of an admin login, sent as `Authorization: Bearer <session key>`, must get 200 on the
    admin only route `GET /user/list`: the session key belongs to the proxy admin user."""
    _, claims = _session(gateway)
    session_key: Final = string_value(claims["key"])
    assert gateway.request("GET", "/user/list", key=session_key).status_code == 200
    _logout(gateway, session_key)


def test_a_wrong_password_and_an_unknown_username_get_the_same_401(gateway: Gateway) -> None:
    """`POST /v2/login` once with `admin` and a wrong password and once with an unknown username and the master
    key. Both must answer 401 with `param` `invalid_credentials` and the identical message, so neither the
    username nor the password can be enumerated, and neither body may contain the submitted password or the
    master key."""
    wrong_password: Final = f"dec1-authsec-wrong-{uuid.uuid4().hex}"
    wrong: Final = login(gateway, ADMIN_USERNAME, wrong_password)
    unknown: Final = login(gateway, f"dec1-authsec-nobody-{uuid.uuid4().hex}@example.com", gateway.key)
    for refused in (wrong, unknown):
        assert refused.status_code == 401, refused.text
        assert refused.json()["error"]["param"] == "invalid_credentials", refused.text
        assert gateway.key not in refused.text
    assert wrong_password not in wrong.text
    assert wrong.json()["error"]["message"] == unknown.json()["error"]["message"]


def test_a_virtual_key_is_not_accepted_as_the_admin_password(gateway: Gateway) -> None:
    """`POST /v2/login` with `admin` and a valid virtual key (owned by a proxy admin user) as the password must
    answer 401 `invalid_credentials`: only the master key or `UI_PASSWORD` log in as the admin."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key(user_id=scenario.user(user_role="proxy_admin"))
        refused: Final = login(gateway, ADMIN_USERNAME, key)
        assert refused.status_code == 401, refused.text
        assert refused.json()["error"]["param"] == "invalid_credentials", refused.text


def test_logout_revokes_the_session_key_and_clears_the_cookie(gateway: Gateway) -> None:
    """The session key calls `POST /session/logout`. The answer must be 200 with `message` `Session revoked.` and
    a `Set-Cookie` that clears `token`; the key's row must be gone, and the same session key must then get 401 on
    `GET /user/list`."""
    _, claims = _session(gateway)
    session_key: Final = string_value(claims["key"])
    response: Final = gateway.request("POST", "/session/logout", {}, key=session_key)
    assert response.status_code == 200, response.text
    assert response.json()["message"] == "Session revoked.", response.text
    assert response.headers.get("set-cookie", "").startswith('token=""'), response.headers.get("set-cookie")
    assert gateway.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest(session_key),)) == []
    assert gateway.request("GET", "/user/list", key=session_key).status_code == 401


def test_logout_refuses_an_ordinary_virtual_key_and_keeps_it(gateway: Gateway) -> None:
    """An ordinary virtual key (not in team `litellm-dashboard`) calls `POST /session/logout`. The answer must be
    403 whose text contains `Only UI session tokens can be revoked through this endpoint.`, and the key must
    still exist and still authenticate, so logout can never be used to delete arbitrary keys."""
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        refused: Final = gateway.request("POST", "/session/logout", {}, key=key)
        assert refused.status_code == 403, refused.text
        assert "Only UI session tokens can be revoked through this endpoint." in refused.text
        assert gateway.rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest(key),)) != []


def test_two_logins_mint_two_sessions_and_logout_revokes_only_one(gateway: Gateway) -> None:
    """Two admin logins mint two different session keys. Logging out with the first must leave the second
    working (200 on `GET /user/list`): logout revokes only the presented session."""
    _, first = _session(gateway)
    _, second = _session(gateway)
    first_key: Final = string_value(first["key"])
    second_key: Final = string_value(second["key"])
    assert first_key != second_key
    _logout(gateway, first_key)
    assert gateway.request("GET", "/user/list", key=first_key).status_code == 401
    assert gateway.request("GET", "/user/list", key=second_key).status_code == 200
    _logout(gateway, second_key)
