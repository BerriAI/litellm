"""Tests for /v1/webhooks CRUD (S6-04)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.webhook_endpoints.endpoints import router


def _client(role=LitellmUserRoles.PROXY_ADMIN, user_id="u-1"):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-x", user_id=user_id, team_id="t-1", user_role=role
    )
    return TestClient(app)


def _mock_row(**overrides):
    base = dict(
        subscription_id="sub-1",
        app_id="xct-chat",
        team_id="t-1",
        user_id="u-1",
        events=["capability.invoked"],
        target_url="https://hooks.example.com/x",
        secret_hash="hash",
        filters=None,
        is_active=True,
        last_success_at=None,
        last_failure_at=None,
        consecutive_failures=0,
    )
    base.update(overrides)
    m = MagicMock()
    for k, v in base.items():
        setattr(m, k, v)
    m.model_dump = MagicMock(return_value=base)
    return m


def _mock_prisma():
    p = MagicMock()
    t = p.db.litellm_webhooksubscriptiontable
    t.create = AsyncMock()
    t.find_many = AsyncMock(return_value=[])
    t.find_unique = AsyncMock(return_value=None)
    t.update = AsyncMock()
    t.delete = AsyncMock()
    return p


def test_create_returns_secret_once():
    prisma = _mock_prisma()
    prisma.db.litellm_webhooksubscriptiontable.create.return_value = _mock_row()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client().post(
            "/v1/webhooks",
            json={
                "target_url": "https://hooks.example.com/x",
                "events": ["capability.invoked"],
                "app_id": "xct-chat",
            },
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["subscription_id"] == "sub-1"
    assert "secret" in body and len(body["secret"]) >= 32
    # The stored hash is NOT the cleartext secret.
    create_data = prisma.db.litellm_webhooksubscriptiontable.create.call_args.kwargs[
        "data"
    ]
    assert create_data["secret_hash"] != body["secret"]


def test_create_rejects_non_http_url():
    prisma = _mock_prisma()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client().post(
            "/v1/webhooks",
            json={
                "target_url": "file:///etc/passwd",
                "events": ["capability.invoked"],
            },
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400
    prisma.db.litellm_webhooksubscriptiontable.create.assert_not_awaited()


def test_create_rejects_localhost():
    prisma = _mock_prisma()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client().post(
            "/v1/webhooks",
            json={
                "target_url": "http://localhost:9999/hook",
                "events": ["capability.invoked"],
            },
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400


def test_create_rejects_cloud_metadata():
    prisma = _mock_prisma()
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client().post(
            "/v1/webhooks",
            json={
                "target_url": "http://169.254.169.254/latest/meta-data/",
                "events": ["capability.invoked"],
            },
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 400


def test_list_scopes_to_caller_for_non_admin():
    prisma = _mock_prisma()
    prisma.db.litellm_webhooksubscriptiontable.find_many.return_value = []
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client(role=LitellmUserRoles.INTERNAL_USER).get(
            "/v1/webhooks", headers={"Authorization": "Bearer k"}
        )
    assert resp.status_code == 200
    where = prisma.db.litellm_webhooksubscriptiontable.find_many.call_args.kwargs[
        "where"
    ]
    assert where["user_id"] == "u-1"


def test_get_403_for_non_owner_non_admin():
    prisma = _mock_prisma()
    prisma.db.litellm_webhooksubscriptiontable.find_unique.return_value = _mock_row(
        user_id="other"
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client(role=LitellmUserRoles.INTERNAL_USER).get(
            "/v1/webhooks/sub-1", headers={"Authorization": "Bearer k"}
        )
    assert resp.status_code == 403


def test_patch_owner_succeeds():
    prisma = _mock_prisma()
    prisma.db.litellm_webhooksubscriptiontable.find_unique.return_value = _mock_row(
        user_id="u-1"
    )
    prisma.db.litellm_webhooksubscriptiontable.update.return_value = _mock_row(
        user_id="u-1", is_active=False
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client(role=LitellmUserRoles.INTERNAL_USER).patch(
            "/v1/webhooks/sub-1",
            json={"is_active": False},
            headers={"Authorization": "Bearer k"},
        )
    assert resp.status_code == 200
    assert resp.json()["is_active"] is False


def test_delete_owner_succeeds():
    prisma = _mock_prisma()
    prisma.db.litellm_webhooksubscriptiontable.find_unique.return_value = _mock_row(
        user_id="u-1"
    )
    with patch("litellm.proxy.proxy_server.prisma_client", prisma):
        resp = _client(role=LitellmUserRoles.INTERNAL_USER).delete(
            "/v1/webhooks/sub-1", headers={"Authorization": "Bearer k"}
        )
    assert resp.status_code == 200
    prisma.db.litellm_webhooksubscriptiontable.delete.assert_awaited_once()
