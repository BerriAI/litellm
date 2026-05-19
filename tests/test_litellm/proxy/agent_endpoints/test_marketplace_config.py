"""Tests for /v1/xct-marketplace/config (S3-07)."""

from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.marketplace_config import (
    DEFAULT_GATEWAY_URL,
    router,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


def _client(role=LitellmUserRoles.PROXY_ADMIN):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        api_key="sk-x", user_id="u-1", user_role=role
    )
    return TestClient(app)


def test_get_returns_default_when_unset(monkeypatch):
    monkeypatch.delenv("XCT_AGENT_GATEWAY_URL", raising=False)
    # Wipe module cache between tests.
    import litellm.proxy.agent_endpoints.marketplace_config as mod

    mod._cache = type(mod._cache)(default_in_memory_ttl=60)
    resp = _client().get(
        "/v1/xct-marketplace/config", headers={"Authorization": "Bearer k"}
    )
    assert resp.status_code == 200
    assert resp.json()["gateway_url"] == DEFAULT_GATEWAY_URL


def test_put_admin_updates_and_get_returns_new_value(monkeypatch):
    monkeypatch.delenv("XCT_AGENT_GATEWAY_URL", raising=False)
    import litellm.proxy.agent_endpoints.marketplace_config as mod

    mod._cache = type(mod._cache)(default_in_memory_ttl=60)
    new_url = "https://xct-agents-staging.up.railway.app"
    client = _client()
    put = client.put(
        "/v1/xct-marketplace/config",
        json={"gateway_url": new_url},
        headers={"Authorization": "Bearer k"},
    )
    assert put.status_code == 200
    assert put.json()["gateway_url"] == new_url
    # Round-trip via GET picks up the cached value.
    got = client.get(
        "/v1/xct-marketplace/config", headers={"Authorization": "Bearer k"}
    )
    assert got.json()["gateway_url"] == new_url


def test_put_non_admin_403():
    client = _client(role=LitellmUserRoles.INTERNAL_USER)
    resp = client.put(
        "/v1/xct-marketplace/config",
        json={"gateway_url": "https://other.example/"},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 403


def test_put_validates_url_scheme():
    resp = _client().put(
        "/v1/xct-marketplace/config",
        json={"gateway_url": "not-a-url"},
        headers={"Authorization": "Bearer k"},
    )
    assert resp.status_code == 400
