"""
Tests for access group management endpoints.
"""

import os
import sys
import types
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import litellm.proxy.proxy_server as ps
from litellm.proxy.proxy_server import app
from litellm.proxy._types import (
    CommonProxyErrors,
    LitellmUserRoles,
    UserAPIKeyAuth,
)

sys.path.insert(0, os.path.abspath("../../../"))


def _make_access_group_record(
    access_group_id: str = "ag-123",
    access_group_name: str = "test-group",
    description: str | None = "Test description",
    access_model_ids: list | None = None,
    access_mcp_server_ids: list | None = None,
    access_agent_ids: list | None = None,
    assigned_team_ids: list | None = None,
    assigned_key_ids: list | None = None,
    created_by: str | None = "admin-user",
    updated_by: str | None = "admin-user",
):
    record = MagicMock()
    record.access_group_id = access_group_id
    record.access_group_name = access_group_name
    record.description = description
    record.access_model_ids = access_model_ids or []
    record.access_mcp_server_ids = access_mcp_server_ids or []
    record.access_agent_ids = access_agent_ids or []
    record.assigned_team_ids = assigned_team_ids or []
    record.assigned_key_ids = assigned_key_ids or []
    record.created_at = datetime.now()
    record.created_by = created_by
    record.updated_at = datetime.now()
    record.updated_by = updated_by
    return record


@pytest.fixture
def client_and_mocks(monkeypatch):
    """Setup mock prisma and admin auth for access group endpoints."""
    mock_access_group_table = MagicMock()
    mock_prisma = MagicMock()

    def _create_side_effect(*, data):
        return _make_access_group_record(
            access_group_id="ag-new",
            access_group_name=data.get("access_group_name", "new"),
            description=data.get("description"),
            access_model_ids=data.get("access_model_ids", []),
            access_mcp_server_ids=data.get("access_mcp_server_ids", []),
            access_agent_ids=data.get("access_agent_ids", []),
            assigned_team_ids=data.get("assigned_team_ids", []),
            assigned_key_ids=data.get("assigned_key_ids", []),
            created_by=data.get("created_by"),
            updated_by=data.get("updated_by"),
        )

    mock_access_group_table.create = AsyncMock(side_effect=_create_side_effect)
    mock_access_group_table.find_unique = AsyncMock(return_value=None)
    mock_access_group_table.find_many = AsyncMock(return_value=[])
    mock_access_group_table.update = AsyncMock(side_effect=lambda *, where, data: _make_access_group_record(
        access_group_id=where.get("access_group_id", "ag-123"),
        access_group_name=data.get("access_group_name", "updated"),
        description=data.get("description"),
        access_model_ids=data.get("access_model_ids", []),
        access_mcp_server_ids=data.get("access_mcp_server_ids", []),
        access_agent_ids=data.get("access_agent_ids", []),
        assigned_team_ids=data.get("assigned_team_ids", []),
        assigned_key_ids=data.get("assigned_key_ids", []),
        updated_by=data.get("updated_by"),
    ))
    mock_access_group_table.delete = AsyncMock(return_value=None)

    mock_prisma.db = types.SimpleNamespace(
        litellm_accessgrouptable=mock_access_group_table,
    )

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    admin_user = UserAPIKeyAuth(
        user_id="admin_user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: admin_user

    client = TestClient(app)

    yield client, mock_prisma, mock_access_group_table

    app.dependency_overrides.clear()
    monkeypatch.setattr(ps, "prisma_client", ps.prisma_client)


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"access_group_name": "group-a"},
        {
            "access_group_name": "group-b",
            "description": "Group B description",
            "access_model_ids": ["model-1"],
            "access_mcp_server_ids": ["mcp-1"],
            "assigned_team_ids": ["team-1"],
        },
    ],
)
def test_create_access_group_success(client_and_mocks, payload):
    """Create access group with various payloads returns 201."""
    client, _, mock_table = client_and_mocks

    resp = client.post("/v1/access_group", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["access_group_name"] == payload["access_group_name"]
    assert body.get("access_group_id") is not None
    mock_table.create.assert_awaited_once()


def test_create_access_group_duplicate_name_conflict(client_and_mocks):
    """Create with duplicate name returns 409."""
    client, _, mock_table = client_and_mocks

    existing = _make_access_group_record(access_group_name="existing-group")
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.post("/v1/access_group", json={"access_group_name": "existing-group"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_create_access_group_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot create access groups."""
    client, _, _ = client_and_mocks

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="regular_user",
        user_role=user_role,
    )

    resp = client.post("/v1/access_group", json={"access_group_name": "forbidden"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == CommonProxyErrors.not_allowed_access.value


# ---------------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------------


def test_list_access_groups_success_empty(client_and_mocks):
    """List access groups returns empty list when none exist."""
    client, _, mock_table = client_and_mocks

    resp = client.get("/v1/access_group")
    assert resp.status_code == 200
    assert resp.json() == []
    mock_table.find_many.assert_awaited_once()


def test_list_access_groups_success_with_items(client_and_mocks):
    """List access groups returns items when they exist."""
    client, _, mock_table = client_and_mocks

    records = [
        _make_access_group_record(access_group_id="ag-1", access_group_name="group-1"),
        _make_access_group_record(access_group_id="ag-2", access_group_name="group-2"),
    ]
    mock_table.find_many = AsyncMock(return_value=records)

    resp = client.get("/v1/access_group")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["access_group_name"] == "group-1"
    assert body[1]["access_group_name"] == "group-2"


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_list_access_groups_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot list access groups."""
    client, _, _ = client_and_mocks

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="regular_user",
        user_role=user_role,
    )

    resp = client.get("/v1/access_group")
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == CommonProxyErrors.not_allowed_access.value


# ---------------------------------------------------------------------------
# GET
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("access_group_id", ["ag-123", "ag-other-id"])
def test_get_access_group_success(client_and_mocks, access_group_id):
    """Get access group by id returns record when found."""
    client, _, mock_table = client_and_mocks

    record = _make_access_group_record(access_group_id=access_group_id)
    mock_table.find_unique = AsyncMock(return_value=record)

    resp = client.get(f"/v1/access_group/{access_group_id}")
    assert resp.status_code == 200
    assert resp.json()["access_group_id"] == access_group_id


def test_get_access_group_not_found(client_and_mocks):
    """Get access group returns 404 when not found."""
    client, _, mock_table = client_and_mocks

    mock_table.find_unique = AsyncMock(return_value=None)

    resp = client.get("/v1/access_group/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_get_access_group_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot get access group."""
    client, _, _ = client_and_mocks

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="regular_user",
        user_role=user_role,
    )

    resp = client.get("/v1/access_group/ag-123")
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == CommonProxyErrors.not_allowed_access.value


# ---------------------------------------------------------------------------
# UPDATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "update_payload",
    [
        {"description": "Updated description"},
        {"access_model_ids": ["model-1", "model-2"]},
        {"assigned_team_ids": [], "assigned_key_ids": ["key-1"]},
    ],
)
def test_update_access_group_success(client_and_mocks, update_payload):
    """Update access group with various payloads returns 200."""
    client, _, mock_table = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-update")
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.put("/v1/access_group/ag-update", json=update_payload)
    assert resp.status_code == 200
    mock_table.update.assert_awaited_once()


def test_update_access_group_not_found(client_and_mocks):
    """Update access group returns 404 when not found."""
    client, _, mock_table = client_and_mocks

    mock_table.find_unique = AsyncMock(return_value=None)

    resp = client.put(
        "/v1/access_group/nonexistent-id",
        json={"description": "Updated"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]
    mock_table.update.assert_not_awaited()


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_update_access_group_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot update access groups."""
    client, _, _ = client_and_mocks

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="regular_user",
        user_role=user_role,
    )

    resp = client.put("/v1/access_group/ag-123", json={"description": "Updated"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == CommonProxyErrors.not_allowed_access.value


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("access_group_id", ["ag-123", "ag-delete-me"])
def test_delete_access_group_success(client_and_mocks, access_group_id):
    """Delete access group returns 204 when found."""
    client, _, mock_table = client_and_mocks

    existing = _make_access_group_record(access_group_id=access_group_id)
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.delete(f"/v1/access_group/{access_group_id}")
    assert resp.status_code == 204
    mock_table.delete.assert_awaited_once()


def test_delete_access_group_not_found(client_and_mocks):
    """Delete access group returns 404 when not found."""
    client, _, mock_table = client_and_mocks

    mock_table.find_unique = AsyncMock(return_value=None)

    resp = client.delete("/v1/access_group/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]
    mock_table.delete.assert_not_awaited()


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_delete_access_group_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot delete access groups."""
    client, _, _ = client_and_mocks

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="regular_user",
        user_role=user_role,
    )

    resp = client.delete("/v1/access_group/ag-123")
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == CommonProxyErrors.not_allowed_access.value


# ---------------------------------------------------------------------------
# DB NOT CONNECTED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,url,factory",
    [
        ("post", "/v1/access_group", lambda: {"json": {"access_group_name": "test"}}),
        ("get", "/v1/access_group", lambda: {}),
        ("get", "/v1/access_group/ag-123", lambda: {}),
        ("put", "/v1/access_group/ag-123", lambda: {"json": {"description": "x"}}),
        ("delete", "/v1/access_group/ag-123", lambda: {}),
    ],
)
def test_access_group_endpoints_db_not_connected(client_and_mocks, monkeypatch, method, url, factory):
    """All endpoints return 500 when DB is not connected."""
    client, _, _ = client_and_mocks

    monkeypatch.setattr(ps, "prisma_client", None)

    resp = getattr(client, method)(url, **factory())
    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == CommonProxyErrors.db_not_connected_error.value
