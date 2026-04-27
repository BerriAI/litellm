"""
Tests for the /config/update endpoint (litellm.proxy.proxy_server.update_config).

These tests cover the targeted-per-section upsert behavior that replaced the
legacy save_config-based implementation, plus the removal of the
store_model_in_db blanket guard.
"""

import json
import os
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app


class FakeRow:
    def __init__(self, param_name, param_value):
        self.param_name = param_name
        self.param_value = param_value


class FakeLitellmConfig:
    def __init__(self, initial_rows=None):
        self.rows = dict(initial_rows or {})
        self.upsert_calls = []
        self.find_first = AsyncMock(side_effect=self._find_first)
        self.upsert = AsyncMock(side_effect=self._upsert)

    async def _find_first(self, where=None):
        if where and "param_name" in where:
            name = where["param_name"]
            if name in self.rows:
                return FakeRow(name, self.rows[name])
        return None

    async def _upsert(self, where=None, data=None):
        name = where["param_name"]
        # The endpoint calls upsert with json.dumps'd payloads. Some paths in
        # the codebase pass dicts directly, so handle both.
        raw = data["update"]["param_value"]
        value = json.loads(raw) if isinstance(raw, str) else raw
        self.rows[name] = value
        self.upsert_calls.append((name, value))


class FakeDB:
    def __init__(self, initial_rows=None):
        self.litellm_config = FakeLitellmConfig(initial_rows=initial_rows)


class FakePrismaClient:
    def __init__(self, initial_rows=None):
        self.db = FakeDB(initial_rows=initial_rows)
        self.jsonify_object = lambda obj: obj


@pytest.fixture
def admin_auth():
    original = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test_admin",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        api_key="sk-1234",
    )
    yield
    app.dependency_overrides = original


@pytest.fixture
def non_admin_auth():
    original = app.dependency_overrides.copy()
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test_user",
        user_role=LitellmUserRoles.INTERNAL_USER,
        api_key="sk-1234",
    )
    yield
    app.dependency_overrides = original


@pytest.fixture
def patched_proxy(monkeypatch):
    """Returns a callable that installs a FakePrismaClient + no-op add_deployment."""

    def _install(initial_rows=None):
        prisma = FakePrismaClient(initial_rows=initial_rows)
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
        # add_deployment is a coroutine method on proxy_config; replace with no-op
        from litellm.proxy.proxy_server import proxy_config as real_proxy_config

        monkeypatch.setattr(
            real_proxy_config, "add_deployment", AsyncMock(return_value=None)
        )
        # Stub encrypt to be identity so we can assert on plain text
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.encrypt_value_helper",
            lambda value, **_: f"enc:{value}",
        )
        return prisma

    return _install


def test_no_db_returns_500_class_error(admin_auth, monkeypatch):
    """When prisma_client is None, the endpoint surfaces a ProxyException."""
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
    client = TestClient(app)
    resp = client.post(
        "/config/update", json={"general_settings": {"store_model_in_db": True}}
    )
    assert resp.status_code >= 400
    # 'No DB Connected' is the message raised inside the endpoint.
    assert "No DB" in resp.text or "DB Connected" in resp.text


def test_non_admin_rejected(non_admin_auth, patched_proxy):
    patched_proxy()
    client = TestClient(app)
    resp = client.post(
        "/config/update", json={"general_settings": {"store_model_in_db": True}}
    )
    assert resp.status_code in (401, 403)


def test_can_flip_store_model_in_db_when_currently_false(
    admin_auth, patched_proxy, monkeypatch
):
    """
    Regression: previously the endpoint refused all writes when the global
    store_model_in_db flag was False, blocking even the request that would
    flip it to True. After this fix, the request succeeds and the flag is
    persisted to the general_settings row.
    """
    monkeypatch.setattr("litellm.proxy.proxy_server.store_model_in_db", False)
    prisma = patched_proxy()

    client = TestClient(app)
    resp = client.post(
        "/config/update", json={"general_settings": {"store_model_in_db": True}}
    )

    assert resp.status_code == 200
    assert (
        prisma.db.litellm_config.rows["general_settings"]["store_model_in_db"] is True
    )


def test_only_sent_section_is_written(admin_auth, patched_proxy):
    """
    A request that only touches general_settings must not write
    litellm_settings, environment_variables, or router_settings rows.
    """
    prisma = patched_proxy(
        initial_rows={
            "litellm_settings": {"drop_params": True},
            "environment_variables": {"FOO": "enc:bar"},
        }
    )

    client = TestClient(app)
    resp = client.post(
        "/config/update",
        json={"general_settings": {"store_prompts_in_spend_logs": True}},
    )

    assert resp.status_code == 200
    written_param_names = {name for name, _ in prisma.db.litellm_config.upsert_calls}
    assert written_param_names == {"general_settings"}
    # Untouched rows preserved.
    assert prisma.db.litellm_config.rows["litellm_settings"] == {"drop_params": True}
    assert prisma.db.litellm_config.rows["environment_variables"] == {"FOO": "enc:bar"}


def test_environment_variables_encrypted_before_write(admin_auth, patched_proxy):
    prisma = patched_proxy()
    client = TestClient(app)
    resp = client.post(
        "/config/update",
        json={"environment_variables": {"OPENAI_API_KEY": "sk-secret"}},
    )

    assert resp.status_code == 200
    stored = prisma.db.litellm_config.rows["environment_variables"]
    assert stored == {"OPENAI_API_KEY": "enc:sk-secret"}


def test_success_callback_unioned_with_existing(admin_auth, patched_proxy):
    prisma = patched_proxy(
        initial_rows={"litellm_settings": {"success_callback": ["langfuse"]}}
    )

    client = TestClient(app)
    resp = client.post(
        "/config/update",
        json={"litellm_settings": {"success_callback": ["prometheus"]}},
    )

    assert resp.status_code == 200
    stored = prisma.db.litellm_config.rows["litellm_settings"]["success_callback"]
    assert set(stored) == {"langfuse", "prometheus"}


def test_alert_to_webhook_url_enables_slack_alerting(admin_auth, patched_proxy):
    prisma = patched_proxy()
    client = TestClient(app)
    resp = client.post(
        "/config/update",
        json={
            "general_settings": {
                "alert_to_webhook_url": {"spend_reports": "https://hooks/foo"}
            }
        },
    )

    assert resp.status_code == 200
    stored = prisma.db.litellm_config.rows["general_settings"]
    assert stored["alerting"] == ["slack"]
    assert stored["alert_to_webhook_url"] == {"spend_reports": "https://hooks/foo"}


def test_router_settings_merged_with_existing(admin_auth, patched_proxy):
    prisma = patched_proxy(
        initial_rows={"router_settings": {"num_retries": 3, "timeout": 10}}
    )

    client = TestClient(app)
    resp = client.post("/config/update", json={"router_settings": {"num_retries": 5}})

    assert resp.status_code == 200
    stored = prisma.db.litellm_config.rows["router_settings"]
    # New value wins, untouched key preserved.
    assert stored["num_retries"] == 5
    assert stored["timeout"] == 10
