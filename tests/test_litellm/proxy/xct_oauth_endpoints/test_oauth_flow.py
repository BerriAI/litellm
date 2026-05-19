"""Tests for /oauth/{authorize,token,revoke,introspect} (S4-05/06/07)."""

import base64
import hashlib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy.xct_oauth_endpoints.oauth_endpoints import router


# Test PKCE vectors (RFC 7636 §1.1)
VERIFIER = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk"
CHALLENGE = (
    base64.urlsafe_b64encode(hashlib.sha256(VERIFIER.encode()).digest())
    .decode()
    .rstrip("=")
)


def _client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _app_row(**overrides):
    base = dict(
        app_id="app-1",
        app_name="xct-chat",
        oauth_client_id="xct_abc",
        oauth_client_secret_hash=hashlib.sha256(b"shh").hexdigest(),
        redirect_uris=["https://chat.xct.test/cb"],
        default_team_id="t-1",
        default_scopes=["read", "write"],
        is_active=True,
    )
    base.update(overrides)
    m = MagicMock()
    for k, v in base.items():
        setattr(m, k, v)
    return m


def _code_row(**overrides):
    base = dict(
        code="code-1",
        client_id="xct_abc",
        user_id="u-1",
        redirect_uri="https://chat.xct.test/cb",
        code_challenge=CHALLENGE,
        code_challenge_method="S256",
        scope=["read"],
        expires_at=datetime.utcnow() + timedelta(minutes=5),
        consumed_at=None,
    )
    base.update(overrides)
    m = MagicMock()
    for k, v in base.items():
        setattr(m, k, v)
    return m


def _prisma():
    p = MagicMock()
    p.db.litellm_xctapptable.find_many = AsyncMock(return_value=[])
    p.db.litellm_oauthauthorizationcode.create = AsyncMock()
    p.db.litellm_oauthauthorizationcode.find_unique = AsyncMock(return_value=None)
    p.db.litellm_oauthauthorizationcode.update = AsyncMock()
    p.db.litellm_verificationtoken.create = AsyncMock()
    p.db.litellm_verificationtoken.find_unique = AsyncMock(return_value=None)
    p.db.litellm_verificationtoken.update = AsyncMock()
    return p


# ---------------------------------------------------------------------------
# S4-05  /oauth/authorize
# ---------------------------------------------------------------------------


def test_authorize_redirects_with_code():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().get(
            "/oauth/authorize",
            params={
                "client_id": "xct_abc",
                "redirect_uri": "https://chat.xct.test/cb",
                "response_type": "code",
                "state": "xyz",
                "code_challenge": CHALLENGE,
                "code_challenge_method": "S256",
                "scope": "read",
            },
            headers={"x-xct-user-id": "u-1"},
            follow_redirects=False,
        )
    assert resp.status_code == 302
    loc = resp.headers["location"]
    assert loc.startswith("https://chat.xct.test/cb?code=")
    assert "state=xyz" in loc


def test_authorize_unknown_client_400():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = []
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().get(
            "/oauth/authorize",
            params={
                "client_id": "nope",
                "redirect_uri": "https://chat.xct.test/cb",
                "response_type": "code",
                "code_challenge": CHALLENGE,
                "code_challenge_method": "S256",
            },
            headers={"x-xct-user-id": "u-1"},
            follow_redirects=False,
        )
    assert resp.status_code == 400


def test_authorize_redirect_uri_not_in_whitelist_400():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().get(
            "/oauth/authorize",
            params={
                "client_id": "xct_abc",
                "redirect_uri": "https://evil.example/cb",
                "response_type": "code",
                "code_challenge": CHALLENGE,
                "code_challenge_method": "S256",
            },
            headers={"x-xct-user-id": "u-1"},
            follow_redirects=False,
        )
    assert resp.status_code == 400


def test_authorize_rejects_plain_pkce():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().get(
            "/oauth/authorize",
            params={
                "client_id": "xct_abc",
                "redirect_uri": "https://chat.xct.test/cb",
                "response_type": "code",
                "code_challenge": "anything",
                "code_challenge_method": "plain",  # not allowed
            },
            headers={"x-xct-user-id": "u-1"},
            follow_redirects=False,
        )
    assert resp.status_code == 400


def test_authorize_no_session_401():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().get(
            "/oauth/authorize",
            params={
                "client_id": "xct_abc",
                "redirect_uri": "https://chat.xct.test/cb",
                "response_type": "code",
                "code_challenge": CHALLENGE,
                "code_challenge_method": "S256",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# S4-06  /oauth/token (authorization_code)
# ---------------------------------------------------------------------------


def test_token_authorization_code_happy_path():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    p.db.litellm_oauthauthorizationcode.find_unique.return_value = _code_row()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": "xct_abc",
                "code": "code-1",
                "code_verifier": VERIFIER,
                "redirect_uri": "https://chat.xct.test/cb",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["token_type"] == "Bearer"
    assert body["expires_in"] == 3600
    assert body["access_token"].startswith("sk-xct-")
    assert body["refresh_token"].startswith("sk-xct-")
    # 2 verificationtoken rows created (access + refresh)
    assert p.db.litellm_verificationtoken.create.await_count == 2
    # Auth code marked consumed
    p.db.litellm_oauthauthorizationcode.update.assert_awaited_once()


def test_token_rejects_replayed_code():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    p.db.litellm_oauthauthorizationcode.find_unique.return_value = _code_row(
        consumed_at=datetime.utcnow()
    )
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": "xct_abc",
                "code": "code-1",
                "code_verifier": VERIFIER,
                "redirect_uri": "https://chat.xct.test/cb",
            },
        )
    assert resp.status_code == 400
    assert "already consumed" in resp.json()["detail"]


def test_token_rejects_expired_code():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    p.db.litellm_oauthauthorizationcode.find_unique.return_value = _code_row(
        expires_at=datetime.utcnow() - timedelta(seconds=1)
    )
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": "xct_abc",
                "code": "code-1",
                "code_verifier": VERIFIER,
                "redirect_uri": "https://chat.xct.test/cb",
            },
        )
    assert resp.status_code == 400
    assert "expired" in resp.json()["detail"]


def test_token_rejects_wrong_pkce_verifier():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    p.db.litellm_oauthauthorizationcode.find_unique.return_value = _code_row()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": "xct_abc",
                "code": "code-1",
                "code_verifier": "wrong-verifier",
                "redirect_uri": "https://chat.xct.test/cb",
            },
        )
    assert resp.status_code == 400
    assert "PKCE" in resp.json()["detail"]


def test_token_unsupported_grant_type():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/token",
            data={
                "grant_type": "password",
                "client_id": "xct_abc",
            },
        )
    assert resp.status_code == 400


def test_token_invalid_client_secret_401():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/token",
            data={
                "grant_type": "authorization_code",
                "client_id": "xct_abc",
                "client_secret": "wrong-secret",
                "code": "code-1",
                "code_verifier": VERIFIER,
                "redirect_uri": "https://chat.xct.test/cb",
            },
        )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# S4-07  /oauth/revoke + /oauth/introspect
# ---------------------------------------------------------------------------


def test_revoke_always_returns_200():
    p = _prisma()
    p.db.litellm_verificationtoken.update.side_effect = Exception("not found")
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/revoke",
            data={"token": "sk-xct-unknown"},
        )
    assert resp.status_code == 200
    assert resp.json() == {"revoked": True}


def test_introspect_active_token():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    row = MagicMock()
    row.app_id = "app-1"
    row.expires = datetime.utcnow() + timedelta(hours=1)
    row.user_id = "u-1"
    row.metadata = {"scope": ["read"]}
    row.token_type = "oauth_access"
    p.db.litellm_verificationtoken.find_unique.return_value = row
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/introspect",
            data={
                "token": "sk-xct-something",
                "client_id": "xct_abc",
                "client_secret": "shh",
            },
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is True
    assert body["scope"] == "read"
    assert body["app_id"] == "app-1"
    assert body["sub"] == "u-1"


def test_introspect_unknown_token_returns_active_false():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    p.db.litellm_verificationtoken.find_unique.return_value = None
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/introspect",
            data={
                "token": "sk-xct-unknown",
                "client_id": "xct_abc",
                "client_secret": "shh",
            },
        )
    assert resp.status_code == 200
    assert resp.json() == {"active": False}


def test_introspect_cross_app_token_returns_active_false():
    p = _prisma()
    p.db.litellm_xctapptable.find_many.return_value = [_app_row()]
    row = MagicMock()
    row.app_id = "OTHER-APP"  # different from client_id
    row.expires = datetime.utcnow() + timedelta(hours=1)
    p.db.litellm_verificationtoken.find_unique.return_value = row
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/oauth/introspect",
            data={
                "token": "sk-xct-stolen",
                "client_id": "xct_abc",
                "client_secret": "shh",
            },
        )
    assert resp.json() == {"active": False}
