"""Tests for /v1/xct-apps CRUD (S4-03)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.xct_app_endpoints.endpoints import router


def _client(role=LitellmUserRoles.PROXY_ADMIN):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-x", user_id="admin-1", team_id="t-1", user_role=role
    )
    return TestClient(app)


def _row(**overrides):
    base = dict(
        app_id="app-1",
        app_name="xct-chat",
        display_name="XCT Chat",
        description=None,
        icon_url=None,
        oauth_client_id="xct_abcdef0123456789abcdef01",
        oauth_client_secret_hash="hash",
        redirect_uris=["https://chat.xct.test/oauth/callback"],
        default_team_id="t-1",
        default_scopes=[],
        capability_scope_id=None,
        rpm_limit=None,
        daily_budget=None,
        is_active=True,
    )
    base.update(overrides)
    m = MagicMock()
    for k, v in base.items():
        setattr(m, k, v)
    m.model_dump = MagicMock(return_value=base)
    return m


def _prisma():
    p = MagicMock()
    t = p.db.litellm_xctapptable
    t.create = AsyncMock()
    t.find_many = AsyncMock(return_value=[])
    t.find_unique = AsyncMock(return_value=None)
    t.update = AsyncMock()
    t.delete = AsyncMock()
    return p


def test_create_returns_client_secret_once():
    p = _prisma()
    p.db.litellm_xctapptable.create.return_value = _row()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/v1/xct-apps",
            json={
                "app_name": "xct-chat",
                "display_name": "XCT Chat",
                "redirect_uris": ["https://chat.xct.test/oauth/callback"],
            },
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    body = resp.json()
    # Cleartext secret returned exactly once; stored hash is different.
    assert "client_secret" in body and len(body["client_secret"]) >= 32
    data = p.db.litellm_xctapptable.create.call_args.kwargs["data"]
    assert data["oauth_client_secret_hash"] != body["client_secret"]
    # client_id was auto-generated with xct_ prefix.
    assert data["oauth_client_id"].startswith("xct_")


def test_create_non_admin_403():
    p = _prisma()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client(role=LitellmUserRoles.INTERNAL_USER).post(
            "/v1/xct-apps",
            json={"app_name": "x", "display_name": "X"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 403
    p.db.litellm_xctapptable.create.assert_not_awaited()


def test_create_rejects_redirect_uri_with_wildcard():
    p = _prisma()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/v1/xct-apps",
            json={
                "app_name": "x",
                "display_name": "X",
                "redirect_uris": ["https://*.xct.test/cb"],
            },
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400
    p.db.litellm_xctapptable.create.assert_not_awaited()


def test_create_rejects_non_http_redirect_uri():
    p = _prisma()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/v1/xct-apps",
            json={
                "app_name": "x",
                "display_name": "X",
                "redirect_uris": ["file:///cb"],
            },
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400


def test_get_returns_no_secret_field():
    p = _prisma()
    p.db.litellm_xctapptable.find_unique.return_value = _row()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().get(
            "/v1/xct-apps/app-1", headers={"Authorization": "Bearer k"}
        )
    assert resp.status_code == 200
    body = resp.json()
    # The read shape must not leak the cleartext (we don't have it anyway)
    # nor the hash.
    assert "client_secret" not in body
    assert "oauth_client_secret_hash" not in body
    assert body["oauth_client_id"].startswith("xct_")


def test_patch_owner_admin_updates_partial():
    p = _prisma()
    p.db.litellm_xctapptable.find_unique.return_value = _row()
    p.db.litellm_xctapptable.update.return_value = _row(display_name="New Name")
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().patch(
            "/v1/xct-apps/app-1",
            json={"display_name": "New Name"},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    assert resp.json()["display_name"] == "New Name"


def test_rotate_secret_returns_new_cleartext():
    p = _prisma()
    p.db.litellm_xctapptable.find_unique.return_value = _row()
    p.db.litellm_xctapptable.update.return_value = _row()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().post(
            "/v1/xct-apps/app-1/rotate-secret",
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert "client_secret" in body and len(body["client_secret"]) >= 32
    # update was awaited with a new hash (and only the hash).
    update_data = p.db.litellm_xctapptable.update.call_args.kwargs["data"]
    assert set(update_data.keys()) == {"oauth_client_secret_hash"}


def test_delete_admin_succeeds():
    p = _prisma()
    p.db.litellm_xctapptable.find_unique.return_value = _row()
    with patch("litellm.proxy.proxy_server.prisma_client", p):
        resp = _client().delete(
            "/v1/xct-apps/app-1", headers={"Authorization": "Bearer k"}
        )
    assert resp.status_code == 200
    p.db.litellm_xctapptable.delete.assert_awaited_once()
