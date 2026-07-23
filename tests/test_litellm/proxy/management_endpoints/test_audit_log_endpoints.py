from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import app


class _AuditLogRow:
    def __init__(self, **data):
        self._data = data

    def model_dump(self):
        return self._data


@pytest.fixture
def audit_logs_client(monkeypatch):
    original_overrides = app.dependency_overrides.copy()

    mock_table = MagicMock()
    mock_table.count = AsyncMock(return_value=1)
    mock_table.find_many = AsyncMock(
        return_value=[
            _AuditLogRow(
                id="audit-1",
                updated_at=datetime(2026, 7, 23, 10, 0, tzinfo=timezone.utc),
                changed_by="admin",
                changed_by_api_key="hashed-key",
                action="updated",
                table_name="LiteLLM_UserTable",
                object_id="user-1",
                before_value=None,
                updated_values={"user_id": "user-1"},
            )
        ]
    )
    mock_prisma = MagicMock()
    mock_prisma.db = SimpleNamespace(litellm_auditlog=mock_table)

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="admin",
        user_role=LitellmUserRoles.PROXY_ADMIN.value,
    )

    yield TestClient(app), mock_table

    app.dependency_overrides = original_overrides


def test_audit_logs_compat_endpoint_matches_grid_contract(audit_logs_client):
    client, mock_table = audit_logs_client

    response = client.get(
        "/audit/logs",
        params={
            "start_date": "2026-07-22T18:30:00.000Z",
            "end_date": "2026-07-23T10:19:00.000Z",
            "page": "2",
            "page_size": "50",
            "action": "updated",
            "table_name": "LiteLLM_UserTable",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"data", "total", "page", "page_size", "total_pages"}
    assert body["data"][0]["id"] == "audit-1"
    assert body["total"] == 1
    assert body["page"] == 2
    assert body["page_size"] == 50
    assert body["total_pages"] == 1

    expected_where = {
        "action": "updated",
        "table_name": "LiteLLM_UserTable",
        "updated_at": {
            "gte": datetime(2026, 7, 22, 18, 30, tzinfo=timezone.utc),
            "lte": datetime(2026, 7, 23, 10, 19, tzinfo=timezone.utc),
        },
    }
    mock_table.count.assert_awaited_once_with(where=expected_where)
    mock_table.find_many.assert_awaited_once_with(
        where=expected_where,
        skip=50,
        take=50,
        order={"updated_at": "desc"},
    )


def test_audit_logs_compat_endpoint_rejects_non_admin(audit_logs_client):
    client, _ = audit_logs_client
    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="viewer",
        user_role=LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value,
    )

    response = client.get("/audit/logs")

    assert response.status_code == 403
    assert response.json()["detail"]["error"] == "Only admins can view audit logs."


def test_audit_logs_compat_endpoint_rejects_invalid_dates(audit_logs_client):
    client, _ = audit_logs_client

    response = client.get("/audit/logs", params={"start_date": "not-a-date"})

    assert response.status_code == 400
    assert "Invalid start_date format" in response.json()["detail"]["error"]
