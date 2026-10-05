import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.proxy_server import app

client = TestClient(app)

PUBLISHED_BANNER = {
    "enabled": True,
    "message": "**Scheduled maintenance** tonight at 10 PM UTC. See [status](https://status.example.com).",
    "severity": "warning",
    "revision": "1f2e3d4c5b6a79881f2e3d4c5b6a7988",
}
PUBLISH_BODY = {k: v for k, v in PUBLISHED_BANNER.items() if k != "revision"}
DISABLED_BANNER = {"enabled": False, "message": "", "severity": "info", "revision": ""}


def _auth_override(role: LitellmUserRoles):
    async def override() -> UserAPIKeyAuth:
        return UserAPIKeyAuth(api_key="sk-test", user_id="test-user", user_role=role)

    return override


@pytest.fixture
def admin_auth():
    app.dependency_overrides[user_api_key_auth] = _auth_override(LitellmUserRoles.PROXY_ADMIN)
    yield
    app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.fixture
def internal_user_auth():
    app.dependency_overrides[user_api_key_auth] = _auth_override(LitellmUserRoles.INTERNAL_USER)
    yield
    app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.fixture
def mock_audit_log(monkeypatch):
    audit_mock = AsyncMock(return_value=None)
    monkeypatch.setattr("litellm.proxy.proxy_server.create_config_audit_log", audit_mock)
    return audit_mock


def _mock_prisma(monkeypatch, record=None):
    mock_prisma = MagicMock()
    mock_prisma.db.litellm_uisettings.find_unique = AsyncMock(return_value=record)
    mock_prisma.db.litellm_uisettings.upsert = AsyncMock()
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)
    return mock_prisma


class TestGetUserBanner:
    def test_requires_auth(self, monkeypatch):
        _mock_prisma(monkeypatch)
        monkeypatch.setattr("litellm.proxy.proxy_server.master_key", "sk-1234")
        response = client.get("/get/user_banner")
        assert response.status_code in (401, 403)

    def test_defaults_when_no_record(self, admin_auth, monkeypatch):
        _mock_prisma(monkeypatch, record=None)
        response = client.get("/get/user_banner")
        assert response.status_code == 200
        assert response.json() == DISABLED_BANNER

    def test_returns_persisted_record(self, internal_user_auth, monkeypatch):
        record = SimpleNamespace(ui_settings=json.dumps(PUBLISHED_BANNER))
        _mock_prisma(monkeypatch, record=record)
        response = client.get("/get/user_banner")
        assert response.status_code == 200
        assert response.json() == PUBLISHED_BANNER

    @pytest.mark.parametrize(
        "raw",
        [
            "not valid json",
            json.dumps({"enabled": True, "message": "hi", "severity": "bogus"}),
            json.dumps({"enabled": True, "message": ""}),
        ],
    )
    def test_corrupt_record_falls_back_to_disabled(self, admin_auth, monkeypatch, raw):
        record = SimpleNamespace(ui_settings=raw)
        _mock_prisma(monkeypatch, record=record)
        response = client.get("/get/user_banner")
        assert response.status_code == 200
        assert response.json() == DISABLED_BANNER

    def test_no_database_returns_disabled_banner(self, admin_auth, monkeypatch):
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)
        response = client.get("/get/user_banner")
        assert response.status_code == 200
        assert response.json() == DISABLED_BANNER


class TestUpdateUserBanner:
    def test_rejects_non_admin(self, internal_user_auth, monkeypatch):
        mock_prisma = _mock_prisma(monkeypatch)
        response = client.patch("/update/user_banner", json=PUBLISH_BODY)
        assert response.status_code == 403
        mock_prisma.db.litellm_uisettings.upsert.assert_not_awaited()

    def test_persists_and_round_trips(self, admin_auth, monkeypatch, mock_audit_log):
        mock_prisma = _mock_prisma(monkeypatch, record=None)

        response = client.patch("/update/user_banner", json=PUBLISH_BODY)
        assert response.status_code == 200
        saved = response.json()["banner"]
        assert {k: saved[k] for k in PUBLISH_BODY} == PUBLISH_BODY
        assert saved["revision"] != ""

        upsert_kwargs = mock_prisma.db.litellm_uisettings.upsert.await_args.kwargs
        assert upsert_kwargs["where"] == {"id": "user_banner"}
        persisted_payload = upsert_kwargs["data"]["create"]["ui_settings"]
        assert json.loads(persisted_payload) == saved
        assert json.loads(upsert_kwargs["data"]["update"]["ui_settings"]) == saved

        mock_prisma.db.litellm_uisettings.find_unique = AsyncMock(
            return_value=SimpleNamespace(ui_settings=persisted_payload)
        )
        read_back = client.get("/get/user_banner")
        assert read_back.status_code == 200
        assert read_back.json() == saved

    def test_republish_same_content_gets_fresh_revision(self, admin_auth, monkeypatch, mock_audit_log):
        _mock_prisma(monkeypatch, record=None)

        first = client.patch("/update/user_banner", json=PUBLISH_BODY).json()["banner"]["revision"]
        second = client.patch("/update/user_banner", json=PUBLISH_BODY).json()["banner"]["revision"]
        assert first != ""
        assert second != ""
        assert first != second

    def test_client_supplied_revision_is_ignored(self, admin_auth, monkeypatch, mock_audit_log):
        _mock_prisma(monkeypatch, record=None)
        response = client.patch("/update/user_banner", json={**PUBLISH_BODY, "revision": "spoofed"})
        assert response.status_code == 200
        saved_revision = response.json()["banner"]["revision"]
        assert saved_revision != "spoofed"
        assert saved_revision != ""

    def test_unpublish_with_empty_message_is_allowed(self, admin_auth, monkeypatch, mock_audit_log):
        _mock_prisma(monkeypatch, record=SimpleNamespace(ui_settings=json.dumps(PUBLISHED_BANNER)))
        response = client.patch(
            "/update/user_banner",
            json={"enabled": False, "message": "", "severity": "info"},
        )
        assert response.status_code == 200
        saved = response.json()["banner"]
        assert {k: saved[k] for k in ("enabled", "message", "severity")} == {
            "enabled": False,
            "message": "",
            "severity": "info",
        }
        assert saved["revision"] not in ("", PUBLISHED_BANNER["revision"])

    @pytest.mark.parametrize(
        "payload",
        [
            {"enabled": True, "message": "hi", "severity": "critical"},
            {"enabled": True, "message": "   ", "severity": "info"},
            {"enabled": True, "message": "x" * 4001, "severity": "info"},
        ],
    )
    def test_rejects_invalid_payloads(self, admin_auth, monkeypatch, payload):
        mock_prisma = _mock_prisma(monkeypatch)
        response = client.patch("/update/user_banner", json=payload)
        assert response.status_code == 422
        mock_prisma.db.litellm_uisettings.upsert.assert_not_awaited()
