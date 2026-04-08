"""
Tests for access group management endpoints.
"""

import os
import sys
import types
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient
from prisma.errors import PrismaError

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
    access_model_names: list | None = None,
    access_mcp_server_ids: list | None = None,
    access_agent_ids: list | None = None,
    assigned_team_ids: list | None = None,
    assigned_key_ids: list | None = None,
    created_by: str | None = "admin-user",
    updated_by: str | None = "admin-user",
    created_at: datetime | None = None,
):
    created_at_val = created_at or datetime.now()
    updated_at_val = datetime.now()
    data = {
        "access_group_id": access_group_id,
        "access_group_name": access_group_name,
        "description": description,
        "access_model_names": access_model_names or [],
        "access_mcp_server_ids": access_mcp_server_ids or [],
        "access_agent_ids": access_agent_ids or [],
        "assigned_team_ids": assigned_team_ids or [],
        "assigned_key_ids": assigned_key_ids or [],
        "created_at": created_at_val,
        "created_by": created_by,
        "updated_at": updated_at_val,
        "updated_by": updated_by,
    }
    record = MagicMock()
    for k, v in data.items():
        setattr(record, k, v)
    record.dict = lambda: data
    record.model_dump = lambda: data
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
            access_model_names=data.get("access_model_names", []),
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
        access_model_names=data.get("access_model_names", []),
        access_mcp_server_ids=data.get("access_mcp_server_ids", []),
        access_agent_ids=data.get("access_agent_ids", []),
        assigned_team_ids=data.get("assigned_team_ids", []),
        assigned_key_ids=data.get("assigned_key_ids", []),
        updated_by=data.get("updated_by"),
    ))
    mock_access_group_table.delete = AsyncMock(return_value=None)

    mock_team_table = MagicMock()
    mock_team_table.find_many = AsyncMock(return_value=[])
    mock_team_table.find_unique = AsyncMock(return_value=None)
    mock_team_table.update = AsyncMock(return_value=None)

    mock_key_table = MagicMock()
    mock_key_table.find_many = AsyncMock(return_value=[])
    mock_key_table.find_unique = AsyncMock(return_value=None)
    mock_key_table.update = AsyncMock(return_value=None)

    # Object-permission table: needed by _sync_add/remove helpers
    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=None)
    mock_op_table.upsert = AsyncMock(return_value=MagicMock(object_permission_id="op-new"))
    mock_op_table.update = AsyncMock(return_value=None)

    @asynccontextmanager
    async def mock_tx():
        tx = types.SimpleNamespace(
            litellm_accessgrouptable=mock_access_group_table,
            litellm_teamtable=mock_team_table,
            litellm_verificationtoken=mock_key_table,
            litellm_objectpermissiontable=mock_op_table,
        )
        yield tx

    mock_db = types.SimpleNamespace(
        litellm_accessgrouptable=mock_access_group_table,
        litellm_teamtable=mock_team_table,
        litellm_verificationtoken=mock_key_table,
        litellm_objectpermissiontable=mock_op_table,
        tx=mock_tx,
    )
    mock_prisma.db = mock_db

    monkeypatch.setattr(ps, "prisma_client", mock_prisma)

    # Mock user_api_key_cache and proxy_logging_obj for cache operations (create/update/delete)
    mock_cache = MagicMock()
    mock_cache.async_set_cache = AsyncMock(return_value=None)
    mock_cache.async_get_cache = AsyncMock(return_value=None)
    mock_cache.delete_cache = MagicMock(return_value=None)
    monkeypatch.setattr(ps, "user_api_key_cache", mock_cache)

    mock_proxy_logging = MagicMock()
    mock_proxy_logging.internal_usage_cache = MagicMock()
    mock_proxy_logging.internal_usage_cache.dual_cache = MagicMock()
    mock_proxy_logging.internal_usage_cache.dual_cache.async_delete_cache = AsyncMock(
        return_value=None
    )
    mock_proxy_logging.internal_usage_cache.dual_cache.async_get_cache = AsyncMock(
        return_value=None
    )
    mock_proxy_logging.internal_usage_cache.dual_cache.async_set_cache = AsyncMock(
        return_value=None
    )
    monkeypatch.setattr(ps, "proxy_logging_obj", mock_proxy_logging)

    admin_user = UserAPIKeyAuth(
        user_id="admin_user",
        user_role=LitellmUserRoles.PROXY_ADMIN,
    )
    app.dependency_overrides[ps.user_api_key_auth] = lambda: admin_user

    client = TestClient(app)

    yield client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging

    app.dependency_overrides.clear()
    monkeypatch.setattr(ps, "prisma_client", ps.prisma_client)


# Paths for primary and alias endpoints (alias: /v1/unified_access_group)
ACCESS_GROUP_PATHS = ["/v1/access_group", "/v1/unified_access_group"]


# ---------------------------------------------------------------------------
# CREATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_path", ACCESS_GROUP_PATHS)
@pytest.mark.parametrize(
    "payload",
    [
        {"access_group_name": "group-a"},
        {
            "access_group_name": "group-b",
            "description": "Group B description",
            "access_model_names": ["model-1"],
            "access_mcp_server_ids": ["mcp-1"],
            "assigned_team_ids": ["team-1"],
        },
    ],
)
def test_create_access_group_success(client_and_mocks, base_path, payload):
    """Create access group with various payloads returns 201."""
    client, _, mock_table, *_ = client_and_mocks

    resp = client.post(base_path, json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["access_group_name"] == payload["access_group_name"]
    assert body.get("access_group_id") is not None
    mock_table.create.assert_awaited_once()


def test_create_access_group_duplicate_name_conflict(client_and_mocks):
    """Create with duplicate name returns 409."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_name="existing-group")
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.post("/v1/access_group", json={"access_group_name": "existing-group"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.parametrize(
    "error_message",
    [
        "Unique constraint failed on the fields: (`access_group_name`)",
        "P2002: Unique constraint failed",
        "unique constraint violation",
    ],
)
def test_create_access_group_race_condition_returns_409(client_and_mocks, error_message):
    """Create race condition: Prisma unique constraint surfaces as 409, not 500."""
    client, _, mock_table, *_ = client_and_mocks

    mock_table.find_unique = AsyncMock(return_value=None)
    mock_table.create = AsyncMock(side_effect=Exception(error_message))

    resp = client.post("/v1/access_group", json={"access_group_name": "race-group"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_create_access_group_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot create access groups."""
    client, *_ = client_and_mocks

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="regular_user",
        user_role=user_role,
    )

    resp = client.post("/v1/access_group", json={"access_group_name": "forbidden"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == CommonProxyErrors.not_allowed_access.value


def test_create_access_group_validation_missing_name(client_and_mocks):
    """Create with missing access_group_name returns 422."""
    client, *_ = client_and_mocks

    resp = client.post("/v1/access_group", json={})
    assert resp.status_code == 422


def test_create_access_group_500_on_non_constraint_prisma_error(client_and_mocks):
    """Create with non-unique-constraint Prisma error returns 500."""
    client, _, mock_table, *_ = client_and_mocks

    mock_table.find_unique = AsyncMock(return_value=None)
    mock_table.create = AsyncMock(side_effect=Exception("Some other database error"))

    # Use raise_server_exceptions=False so unhandled exceptions become 500 responses
    test_client = TestClient(app, raise_server_exceptions=False)
    resp = test_client.post("/v1/access_group", json={"access_group_name": "test-group"})
    assert resp.status_code == 500


# ---------------------------------------------------------------------------
# LIST
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_path", ACCESS_GROUP_PATHS)
def test_list_access_groups_success_empty(client_and_mocks, base_path):
    """List access groups returns empty list when none exist."""
    client, _, mock_table, *_ = client_and_mocks

    resp = client.get(base_path)
    assert resp.status_code == 200
    assert resp.json() == []
    mock_table.find_many.assert_awaited_once()


@pytest.mark.parametrize("base_path", ACCESS_GROUP_PATHS)
def test_list_access_groups_success_with_items(client_and_mocks, base_path):
    """List access groups returns items when they exist."""
    client, _, mock_table, *_ = client_and_mocks

    records = [
        _make_access_group_record(access_group_id="ag-1", access_group_name="group-1"),
        _make_access_group_record(access_group_id="ag-2", access_group_name="group-2"),
    ]
    mock_table.find_many = AsyncMock(return_value=records)

    resp = client.get(base_path)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    assert body[0]["access_group_name"] == "group-1"
    assert body[1]["access_group_name"] == "group-2"


@pytest.mark.parametrize("base_path", ACCESS_GROUP_PATHS)
def test_list_access_groups_ordered_by_created_at_desc(client_and_mocks, base_path):
    """List access groups calls find_many with created_at desc order."""
    client, _, mock_table, *_ = client_and_mocks

    older = datetime(2025, 1, 1, 12, 0, 0)
    newer = datetime(2025, 1, 2, 12, 0, 0)
    records = [
        _make_access_group_record(
            access_group_id="ag-newer",
            access_group_name="newer-group",
            created_at=newer,
        ),
        _make_access_group_record(
            access_group_id="ag-older",
            access_group_name="older-group",
            created_at=older,
        ),
    ]
    mock_table.find_many = AsyncMock(return_value=records)

    resp = client.get(base_path)
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 2
    # Mock returns newest first (simulating Prisma order desc)
    assert body[0]["access_group_name"] == "newer-group"
    assert body[1]["access_group_name"] == "older-group"
    mock_table.find_many.assert_awaited_once_with(order={"created_at": "desc"})


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_list_access_groups_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot list access groups."""
    client, *_ = client_and_mocks

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


@pytest.mark.parametrize("base_path", ACCESS_GROUP_PATHS)
@pytest.mark.parametrize("access_group_id", ["ag-123", "ag-other-id"])
def test_get_access_group_success(client_and_mocks, base_path, access_group_id):
    """Get access group by id returns record when found."""
    client, _, mock_table, *_ = client_and_mocks

    record = _make_access_group_record(access_group_id=access_group_id)
    mock_table.find_unique = AsyncMock(return_value=record)

    resp = client.get(f"{base_path}/{access_group_id}")
    assert resp.status_code == 200
    assert resp.json()["access_group_id"] == access_group_id


def test_get_access_group_not_found(client_and_mocks):
    """Get access group returns 404 when not found."""
    client, _, mock_table, *_ = client_and_mocks

    mock_table.find_unique = AsyncMock(return_value=None)

    resp = client.get("/v1/access_group/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_get_access_group_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot get access group."""
    client, *_ = client_and_mocks

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


@pytest.mark.parametrize("base_path", ACCESS_GROUP_PATHS)
@pytest.mark.parametrize(
    "update_payload",
    [
        {"description": "Updated description"},
        {"access_model_names": ["model-1", "model-2"]},
        {"assigned_team_ids": [], "assigned_key_ids": ["key-1"]},
    ],
)
def test_update_access_group_success(client_and_mocks, base_path, update_payload):
    """Update access group with various payloads returns 200."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-update")
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.put(f"{base_path}/ag-update", json=update_payload)
    assert resp.status_code == 200
    mock_table.update.assert_awaited_once()


def test_update_access_group_not_found(client_and_mocks):
    """Update access group returns 404 when not found."""
    client, _, mock_table, *_ = client_and_mocks

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
    client, *_ = client_and_mocks

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="regular_user",
        user_role=user_role,
    )

    resp = client.put("/v1/access_group/ag-123", json={"description": "Updated"})
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == CommonProxyErrors.not_allowed_access.value


def test_update_access_group_empty_body(client_and_mocks):
    """Update with empty body succeeds; only updated_by is set."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-update", access_group_name="unchanged")
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.put("/v1/access_group/ag-update", json={})
    assert resp.status_code == 200
    mock_table.update.assert_awaited_once()
    call_kwargs = mock_table.update.call_args.kwargs
    assert call_kwargs["where"] == {"access_group_id": "ag-update"}
    assert "updated_by" in call_kwargs["data"]
    assert call_kwargs["data"]["updated_by"] == "admin_user"


def test_update_access_group_name_success(client_and_mocks):
    """Update access_group_name succeeds when new name is unique."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-update", access_group_name="old-name")
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.put("/v1/access_group/ag-update", json={"access_group_name": "new-name"})
    assert resp.status_code == 200
    mock_table.update.assert_awaited_once()
    call_kwargs = mock_table.update.call_args.kwargs
    assert call_kwargs["data"]["access_group_name"] == "new-name"


def test_update_access_group_name_duplicate_conflict(client_and_mocks):
    """Update access_group_name to existing name returns 409 (unique constraint)."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-update", access_group_name="old-name")
    mock_table.find_unique = AsyncMock(return_value=existing)
    mock_table.update = AsyncMock(
        side_effect=Exception("Unique constraint failed on the fields: (`access_group_name`)")
    )

    resp = client.put("/v1/access_group/ag-update", json={"access_group_name": "taken-name"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]
    mock_table.update.assert_awaited_once()


@pytest.mark.parametrize(
    "error_message",
    [
        "Unique constraint failed on the fields: (`access_group_name`)",
        "P2002: Unique constraint failed",
        "unique constraint violation",
    ],
)
def test_update_access_group_name_unique_constraint_returns_409(client_and_mocks, error_message):
    """Update access_group_name: Prisma unique constraint surfaces as 409."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-update", access_group_name="old-name")
    mock_table.find_unique = AsyncMock(return_value=existing)
    mock_table.update = AsyncMock(side_effect=Exception(error_message))

    resp = client.put("/v1/access_group/ag-update", json={"access_group_name": "race-name"})
    assert resp.status_code == 409
    assert "already exists" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# DELETE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("base_path", ACCESS_GROUP_PATHS)
@pytest.mark.parametrize("access_group_id", ["ag-123", "ag-delete-me"])
def test_delete_access_group_success(client_and_mocks, base_path, access_group_id):
    """Delete access group returns 204 when found."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id=access_group_id)
    mock_table.find_unique = AsyncMock(return_value=existing)

    resp = client.delete(f"{base_path}/{access_group_id}")
    assert resp.status_code == 204
    mock_table.delete.assert_awaited_once()


def test_delete_access_group_not_found(client_and_mocks):
    """Delete access group returns 404 when not found."""
    client, _, mock_table, *_ = client_and_mocks

    mock_table.find_unique = AsyncMock(return_value=None)

    resp = client.delete("/v1/access_group/nonexistent-id")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]
    mock_table.delete.assert_not_awaited()


@pytest.mark.parametrize("user_role", [LitellmUserRoles.INTERNAL_USER, LitellmUserRoles.INTERNAL_USER_VIEW_ONLY])
def test_delete_access_group_forbidden_non_admin(client_and_mocks, user_role):
    """Non-admin users cannot delete access groups."""
    client, *_ = client_and_mocks

    app.dependency_overrides[ps.user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="regular_user",
        user_role=user_role,
    )

    resp = client.delete("/v1/access_group/ag-123")
    assert resp.status_code == 403
    assert resp.json()["detail"]["error"] == CommonProxyErrors.not_allowed_access.value


def test_delete_access_group_cleans_up_teams_and_keys(client_and_mocks):
    """Delete removes access_group_id and access-group-sourced models/MCP/agents
    from teams and keys before deleting the group."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable
    mock_key_table = mock_prisma.db.litellm_verificationtoken

    existing = _make_access_group_record(
        access_group_id="ag-to-delete",
        access_model_names=["model-from-ag"],
    )
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    team_with_group = MagicMock()
    team_with_group.team_id = "team-1"
    team_with_group.access_group_ids = ["ag-to-delete", "ag-other"]
    team_with_group.models = ["model-from-ag", "direct-model"]
    team_with_group.object_permission_id = None
    mock_team_table.find_many = AsyncMock(return_value=[team_with_group])
    mock_team_table.find_unique = AsyncMock(return_value=team_with_group)

    key_with_group = MagicMock()
    key_with_group.token = "key-token-1"
    key_with_group.access_group_ids = ["ag-to-delete"]
    key_with_group.models = ["model-from-ag"]
    key_with_group.object_permission_id = None
    mock_key_table.find_many = AsyncMock(return_value=[key_with_group])
    mock_key_table.find_unique = AsyncMock(return_value=key_with_group)

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 204

    # Team update: access_group_id removed and model-from-ag removed; direct-model kept
    mock_team_table.update.assert_awaited_once()
    team_update_data = mock_team_table.update.call_args.kwargs["data"]
    assert team_update_data["access_group_ids"] == ["ag-other"]
    assert "model-from-ag" not in team_update_data.get("models", [])
    assert "direct-model" in team_update_data.get("models", [])

    # Key update: access_group_id removed and model-from-ag removed
    mock_key_table.update.assert_awaited_once()
    key_update_data = mock_key_table.update.call_args.kwargs["data"]
    assert key_update_data["access_group_ids"] == []
    assert "model-from-ag" not in key_update_data.get("models", [])

    mock_access_group_table.delete.assert_awaited_once_with(
        where={"access_group_id": "ag-to-delete"}
    )


@pytest.mark.parametrize(
    "team_cache_group_ids,key_cache_group_ids,expected_team_ids_after,expected_key_ids_after",
    [
        # Team and key both cached with the deleted group
        (
            ["ag-to-delete", "ag-keep"],
            ["ag-to-delete", "ag-stay"],
            ["ag-keep"],
            ["ag-stay"],
        ),
        # Only team cached; key not in cache
        (
            ["ag-to-delete"],
            None,
            [],
            None,
        ),
        # Only key cached; team not in cache
        (
            None,
            ["ag-to-delete"],
            None,
            [],
        ),
        # Neither cached — nothing to patch
        (
            None,
            None,
            None,
            None,
        ),
        # Cached team has only the deleted group
        (
            ["ag-to-delete"],
            ["ag-to-delete"],
            [],
            [],
        ),
        # Cached objects have multiple groups, only the deleted one is removed
        (
            ["ag-alpha", "ag-to-delete", "ag-beta"],
            ["ag-to-delete", "ag-gamma"],
            ["ag-alpha", "ag-beta"],
            ["ag-gamma"],
        ),
    ],
    ids=[
        "both_cached",
        "only_team_cached",
        "only_key_cached",
        "neither_cached",
        "single_group_removed",
        "multi_group_partial_removal",
    ],
)
def test_delete_access_group_patches_cached_team_and_key(
    client_and_mocks,
    team_cache_group_ids,
    key_cache_group_ids,
    expected_team_ids_after,
    expected_key_ids_after,
):
    """Delete patches cached team/key objects to remove the deleted access_group_id."""
    from litellm.proxy._types import LiteLLM_TeamTableCachedObj

    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable
    mock_key_table = mock_prisma.db.litellm_verificationtoken

    existing = _make_access_group_record(access_group_id="ag-to-delete")
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    # Set up a team and key in the DB that reference the group
    team_with_group = MagicMock()
    team_with_group.team_id = "team-1"
    team_with_group.access_group_ids = ["ag-to-delete", "ag-keep"]
    team_with_group.models = []
    team_with_group.object_permission_id = None  # no object_permission to clean up
    mock_team_table.find_many = AsyncMock(return_value=[team_with_group])
    mock_team_table.find_unique = AsyncMock(return_value=team_with_group)

    key_with_group = MagicMock()
    key_with_group.token = "hashed-key-1"
    key_with_group.access_group_ids = ["ag-to-delete"]
    key_with_group.models = []
    key_with_group.object_permission_id = None  # no object_permission to clean up
    mock_key_table.find_many = AsyncMock(return_value=[key_with_group])
    mock_key_table.find_unique = AsyncMock(return_value=key_with_group)

    # Build cached team object (returned from proxy_logging dual cache)
    if team_cache_group_ids is not None:
        cached_team = LiteLLM_TeamTableCachedObj(
            team_id="team-1",
            access_group_ids=list(team_cache_group_ids),
        )
        mock_proxy_logging.internal_usage_cache.dual_cache.async_get_cache = AsyncMock(
            return_value=cached_team
        )
    else:
        mock_proxy_logging.internal_usage_cache.dual_cache.async_get_cache = AsyncMock(
            return_value=None
        )

    # Build cached key object (returned from user_api_key_cache)
    if key_cache_group_ids is not None:
        cached_key = UserAPIKeyAuth(
            token="hashed-key-1",
            access_group_ids=list(key_cache_group_ids),
        )
        mock_cache.async_get_cache = AsyncMock(return_value=cached_key)
    else:
        mock_cache.async_get_cache = AsyncMock(return_value=None)

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 204

    # Verify DB cleanup always happens
    mock_team_table.update.assert_awaited_once()
    mock_key_table.update.assert_awaited_once()

    # Verify cache patching
    if expected_team_ids_after is not None:
        # _cache_team_object writes via _cache_management_object -> async_set_cache
        team_set_calls = [
            c for c in mock_cache.async_set_cache.call_args_list
            if c.kwargs.get("key", "") == "team_id:team-1"
            or (len(c.args) >= 1 and c.args[0] == "team_id:team-1")
        ]
        assert len(team_set_calls) >= 1, "Expected team cache to be patched"
        # The cached team object should have the updated access_group_ids
        written_team = team_set_calls[0].kwargs.get("value") or team_set_calls[0].args[1]
        if isinstance(written_team, LiteLLM_TeamTableCachedObj):
            assert written_team.access_group_ids == expected_team_ids_after
    else:
        # No team in cache — async_set_cache should not be called for team_id key
        team_set_calls = [
            c for c in mock_cache.async_set_cache.call_args_list
            if c.kwargs.get("key", "") == "team_id:team-1"
            or (len(c.args) >= 1 and c.args[0] == "team_id:team-1")
        ]
        assert len(team_set_calls) == 0, "Should not patch team cache when not cached"

    if expected_key_ids_after is not None:
        key_set_calls = [
            c for c in mock_cache.async_set_cache.call_args_list
            if c.kwargs.get("key", "") == "hashed-key-1"
            or (len(c.args) >= 1 and c.args[0] == "hashed-key-1")
        ]
        assert len(key_set_calls) >= 1, "Expected key cache to be patched"
        written_key = key_set_calls[0].kwargs.get("value") or key_set_calls[0].args[1]
        if isinstance(written_key, UserAPIKeyAuth):
            assert written_key.access_group_ids == expected_key_ids_after
    else:
        key_set_calls = [
            c for c in mock_cache.async_set_cache.call_args_list
            if c.kwargs.get("key", "") == "hashed-key-1"
            or (len(c.args) >= 1 and c.args[0] == "hashed-key-1")
        ]
        assert len(key_set_calls) == 0, "Should not patch key cache when not cached"


def test_delete_access_group_patches_key_cached_as_dict(client_and_mocks):
    """Delete correctly patches a key cached as a raw dict (not UserAPIKeyAuth)."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable
    mock_key_table = mock_prisma.db.litellm_verificationtoken

    existing = _make_access_group_record(access_group_id="ag-to-delete")
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    mock_team_table.find_many = AsyncMock(return_value=[])

    key_with_group = MagicMock()
    key_with_group.token = "hashed-key-dict"
    key_with_group.access_group_ids = ["ag-to-delete", "ag-other"]
    mock_key_table.find_many = AsyncMock(return_value=[key_with_group])
    mock_key_table.find_unique = AsyncMock(return_value=key_with_group)

    # No team in cache
    mock_proxy_logging.internal_usage_cache.dual_cache.async_get_cache = AsyncMock(
        return_value=None
    )

    # Key cached as a plain dict (as can happen with Redis serialization)
    mock_cache.async_get_cache = AsyncMock(
        return_value={
            "token": "hashed-key-dict",
            "access_group_ids": ["ag-to-delete", "ag-other"],
        }
    )

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 204

    # The key should have been re-cached with the deleted group removed
    key_set_calls = [
        c for c in mock_cache.async_set_cache.call_args_list
        if c.kwargs.get("key", "") == "hashed-key-dict"
        or (len(c.args) >= 1 and c.args[0] == "hashed-key-dict")
    ]
    assert len(key_set_calls) >= 1, "Expected key cache to be patched"
    written_key = key_set_calls[0].kwargs.get("value") or key_set_calls[0].args[1]
    if isinstance(written_key, UserAPIKeyAuth):
        assert written_key.access_group_ids == ["ag-other"]


def test_delete_access_group_503_on_db_connection_error(client_and_mocks):
    """Delete returns 503 when DB connection error occurs during transaction."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-to-delete")
    mock_table.find_unique = AsyncMock(return_value=existing)
    mock_table.delete = AsyncMock(side_effect=PrismaError())

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 503
    assert resp.json()["detail"] == CommonProxyErrors.db_not_connected_error.value


def test_delete_access_group_404_on_p2025_or_record_not_found(client_and_mocks):
    """Delete returns 404 when Prisma raises P2025 or record-not-found error."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-to-delete")
    mock_table.find_unique = AsyncMock(return_value=existing)
    mock_table.delete = AsyncMock(side_effect=Exception("P2025: Record to delete does not exist"))

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_delete_access_group_500_on_generic_exception(client_and_mocks):
    """Delete returns 500 when generic exception occurs during transaction."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(access_group_id="ag-to-delete")
    mock_table.find_unique = AsyncMock(return_value=existing)
    mock_table.delete = AsyncMock(side_effect=RuntimeError("Unexpected error"))

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 500
    assert "Failed to delete access group" in resp.json()["detail"]


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
        # Alias: /v1/unified_access_group
        ("post", "/v1/unified_access_group", lambda: {"json": {"access_group_name": "test"}}),
        ("get", "/v1/unified_access_group", lambda: {}),
        ("get", "/v1/unified_access_group/ag-123", lambda: {}),
        ("put", "/v1/unified_access_group/ag-123", lambda: {"json": {"description": "x"}}),
        ("delete", "/v1/unified_access_group/ag-123", lambda: {}),
    ],
)
def test_access_group_endpoints_db_not_connected(client_and_mocks, monkeypatch, method, url, factory):
    """All endpoints return 500 when DB is not connected."""
    client, *_ = client_and_mocks

    monkeypatch.setattr(ps, "prisma_client", None)

    resp = getattr(client, method)(url, **factory())
    assert resp.status_code == 500
    assert resp.json()["detail"]["error"] == CommonProxyErrors.db_not_connected_error.value


# ---------------------------------------------------------------------------
# Unit tests for cache helpers (_record_to_access_group_table)
# ---------------------------------------------------------------------------


def test_record_to_access_group_table():
    """Test _record_to_access_group_table converts Prisma-like record to LiteLLM_AccessGroupTable."""
    from litellm.proxy.management_endpoints.access_group_endpoints import _record_to_access_group_table

    record = _make_access_group_record(
        access_group_id="ag-unit-test",
        access_group_name="unit-test-group",
        access_model_names=["gpt-4", "claude-3"],
        access_agent_ids=["agent-1"],
    )
    result = _record_to_access_group_table(record)
    assert result.access_group_id == "ag-unit-test"
    assert result.access_group_name == "unit-test-group"
    assert result.access_model_names == ["gpt-4", "claude-3"]
    assert result.access_agent_ids == ["agent-1"]


# ---------------------------------------------------------------------------
# Sync tests: CREATE
# ---------------------------------------------------------------------------


def test_create_access_group_syncs_assigned_teams(client_and_mocks):
    """Create adds access_group_id to each assigned team's access_group_ids in DB."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable

    team_record = MagicMock()
    team_record.team_id = "team-1"
    team_record.access_group_ids = []
    mock_team_table.find_unique = AsyncMock(return_value=team_record)

    resp = client.post(
        "/v1/access_group",
        json={"access_group_name": "new-group", "assigned_team_ids": ["team-1"]},
    )
    assert resp.status_code == 201

    mock_team_table.find_unique.assert_awaited_once_with(where={"team_id": "team-1"})
    mock_team_table.update.assert_awaited_once()
    call_kwargs = mock_team_table.update.call_args.kwargs
    assert call_kwargs["where"] == {"team_id": "team-1"}
    # The newly created access group id ("ag-new") should be in the updated list
    assert "ag-new" in call_kwargs["data"]["access_group_ids"]


def test_create_access_group_syncs_assigned_keys(client_and_mocks):
    """Create adds access_group_id to each assigned key's access_group_ids in DB."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_key_table = mock_prisma.db.litellm_verificationtoken

    key_record = MagicMock()
    key_record.token = "hashed-token-1"
    key_record.access_group_ids = []
    mock_key_table.find_unique = AsyncMock(return_value=key_record)

    resp = client.post(
        "/v1/access_group",
        json={"access_group_name": "new-group", "assigned_key_ids": ["hashed-token-1"]},
    )
    assert resp.status_code == 201

    mock_key_table.find_unique.assert_awaited_once_with(where={"token": "hashed-token-1"})
    mock_key_table.update.assert_awaited_once()
    call_kwargs = mock_key_table.update.call_args.kwargs
    assert call_kwargs["where"] == {"token": "hashed-token-1"}
    assert "ag-new" in call_kwargs["data"]["access_group_ids"]


def test_create_access_group_skips_sync_for_nonexistent_team(client_and_mocks):
    """Create skips updating a team that doesn't exist in DB."""
    client, mock_prisma, _, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable
    mock_team_table.find_unique = AsyncMock(return_value=None)

    resp = client.post(
        "/v1/access_group",
        json={"access_group_name": "new-group", "assigned_team_ids": ["nonexistent-team"]},
    )
    assert resp.status_code == 201
    mock_team_table.update.assert_not_awaited()


def test_create_access_group_idempotent_team_sync(client_and_mocks):
    """Create skips updating a team that already has the access_group_id."""
    client, mock_prisma, _, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable

    team_record = MagicMock()
    team_record.team_id = "team-1"
    team_record.access_group_ids = ["ag-new"]  # already synced
    mock_team_table.find_unique = AsyncMock(return_value=team_record)

    resp = client.post(
        "/v1/access_group",
        json={"access_group_name": "new-group", "assigned_team_ids": ["team-1"]},
    )
    assert resp.status_code == 201
    mock_team_table.update.assert_not_awaited()


# ---------------------------------------------------------------------------
# Sync tests: UPDATE
# ---------------------------------------------------------------------------


def test_update_access_group_syncs_added_teams(client_and_mocks):
    """Update adds access_group_id to newly assigned teams."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable

    existing = _make_access_group_record(
        access_group_id="ag-update", assigned_team_ids=["team-existing"]
    )
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    team_record = MagicMock()
    team_record.team_id = "team-new"
    team_record.access_group_ids = []
    mock_team_table.find_unique = AsyncMock(return_value=team_record)

    resp = client.put(
        "/v1/access_group/ag-update",
        json={"assigned_team_ids": ["team-existing", "team-new"]},
    )
    assert resp.status_code == 200

    mock_team_table.find_unique.assert_awaited_once_with(where={"team_id": "team-new"})
    mock_team_table.update.assert_awaited_once()
    call_kwargs = mock_team_table.update.call_args.kwargs
    assert call_kwargs["where"] == {"team_id": "team-new"}
    assert "ag-update" in call_kwargs["data"]["access_group_ids"]


def test_update_access_group_syncs_removed_teams(client_and_mocks):
    """Update removes access_group_id from de-assigned teams."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable

    existing = _make_access_group_record(
        access_group_id="ag-update", assigned_team_ids=["team-keep", "team-remove"]
    )
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    team_to_remove = MagicMock()
    team_to_remove.team_id = "team-remove"
    team_to_remove.access_group_ids = ["ag-update"]
    mock_team_table.find_unique = AsyncMock(return_value=team_to_remove)

    resp = client.put(
        "/v1/access_group/ag-update",
        json={"assigned_team_ids": ["team-keep"]},
    )
    assert resp.status_code == 200

    mock_team_table.find_unique.assert_awaited_once_with(where={"team_id": "team-remove"})
    mock_team_table.update.assert_awaited_once()
    call_kwargs = mock_team_table.update.call_args.kwargs
    assert call_kwargs["where"] == {"team_id": "team-remove"}
    assert "ag-update" not in call_kwargs["data"]["access_group_ids"]


def test_update_access_group_no_team_sync_when_ids_not_in_payload(client_and_mocks):
    """Update does not sync teams when assigned_team_ids is absent from the payload."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable

    existing = _make_access_group_record(
        access_group_id="ag-update", assigned_team_ids=["team-1"]
    )
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    resp = client.put("/v1/access_group/ag-update", json={"description": "new desc"})
    assert resp.status_code == 200

    mock_team_table.find_unique.assert_not_awaited()
    mock_team_table.update.assert_not_awaited()


def test_update_access_group_syncs_added_keys(client_and_mocks):
    """Update adds access_group_id to newly assigned keys."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_key_table = mock_prisma.db.litellm_verificationtoken

    existing = _make_access_group_record(
        access_group_id="ag-update", assigned_key_ids=["old-token"]
    )
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    key_record = MagicMock()
    key_record.token = "new-token"
    key_record.access_group_ids = []
    mock_key_table.find_unique = AsyncMock(return_value=key_record)

    resp = client.put(
        "/v1/access_group/ag-update",
        json={"assigned_key_ids": ["old-token", "new-token"]},
    )
    assert resp.status_code == 200

    mock_key_table.find_unique.assert_awaited_once_with(where={"token": "new-token"})
    mock_key_table.update.assert_awaited_once()
    call_kwargs = mock_key_table.update.call_args.kwargs
    assert call_kwargs["where"] == {"token": "new-token"}
    assert "ag-update" in call_kwargs["data"]["access_group_ids"]


def test_update_access_group_syncs_removed_keys(client_and_mocks):
    """Update removes access_group_id from de-assigned keys."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_key_table = mock_prisma.db.litellm_verificationtoken

    existing = _make_access_group_record(
        access_group_id="ag-update", assigned_key_ids=["keep-token", "remove-token"]
    )
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    key_to_remove = MagicMock()
    key_to_remove.token = "remove-token"
    key_to_remove.access_group_ids = ["ag-update"]
    mock_key_table.find_unique = AsyncMock(return_value=key_to_remove)

    resp = client.put(
        "/v1/access_group/ag-update",
        json={"assigned_key_ids": ["keep-token"]},
    )
    assert resp.status_code == 200

    mock_key_table.find_unique.assert_awaited_once_with(where={"token": "remove-token"})
    mock_key_table.update.assert_awaited_once()
    call_kwargs = mock_key_table.update.call_args.kwargs
    assert call_kwargs["where"] == {"token": "remove-token"}
    assert "ag-update" not in call_kwargs["data"]["access_group_ids"]


# ---------------------------------------------------------------------------
# Sync tests: DELETE (out-of-sync data handling)
# ---------------------------------------------------------------------------


def test_delete_access_group_handles_out_of_sync_assigned_teams(client_and_mocks):
    """Delete includes teams from assigned_team_ids even when not found by hasSome query."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable

    # Access group has assigned_team_ids but the team's access_group_ids is not synced
    existing = _make_access_group_record(
        access_group_id="ag-to-delete",
        assigned_team_ids=["team-out-of-sync"],
    )
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    # hasSome query finds nothing (team's own access_group_ids is out of sync)
    mock_team_table.find_many = AsyncMock(return_value=[])

    out_of_sync_team = MagicMock()
    out_of_sync_team.team_id = "team-out-of-sync"
    out_of_sync_team.access_group_ids = []  # already clean, no update needed
    mock_team_table.find_unique = AsyncMock(return_value=out_of_sync_team)

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 204

    # find_unique is called for the out-of-sync team (included via union with assigned_team_ids)
    mock_team_table.find_unique.assert_awaited_once_with(where={"team_id": "team-out-of-sync"})
    # No update needed since team's access_group_ids doesn't contain "ag-to-delete"
    mock_team_table.update.assert_not_awaited()


def test_delete_access_group_handles_out_of_sync_assigned_keys(client_and_mocks):
    """Delete includes keys from assigned_key_ids even when not found by hasSome query."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_key_table = mock_prisma.db.litellm_verificationtoken

    existing = _make_access_group_record(
        access_group_id="ag-to-delete",
        assigned_key_ids=["token-out-of-sync"],
    )
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    mock_key_table.find_many = AsyncMock(return_value=[])

    out_of_sync_key = MagicMock()
    out_of_sync_key.token = "token-out-of-sync"
    out_of_sync_key.access_group_ids = []
    mock_key_table.find_unique = AsyncMock(return_value=out_of_sync_key)

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 204

    mock_key_table.find_unique.assert_awaited_once_with(where={"token": "token-out-of-sync"})
    mock_key_table.update.assert_not_awaited()


def test_update_access_group_null_assigned_ids_treated_as_empty(client_and_mocks):
    """Update with explicit null for assigned_*_ids clears the list and writes [] to DB."""
    client, _, mock_table, *_ = client_and_mocks

    existing = _make_access_group_record(
        access_group_id="ag-update",
        assigned_team_ids=["team-1"],
        assigned_key_ids=["key-1"],
    )
    mock_table.find_unique = AsyncMock(return_value=existing)

    # Sending null for assigned_team_ids and assigned_key_ids
    resp = client.put(
        "/v1/access_group/ag-update",
        json={"assigned_team_ids": None, "assigned_key_ids": None},
    )
    assert resp.status_code == 200

    # Verify the DB update was called with [] (not null) for list fields
    update_call_kwargs = mock_table.update.call_args.kwargs
    assert update_call_kwargs["data"]["assigned_team_ids"] == []
    assert update_call_kwargs["data"]["assigned_key_ids"] == []


# ---------------------------------------------------------------------------
# _merge_access_group_resources_into_data_json unit tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_access_group_resources_empty_ids():
    """Empty access_group_ids returns data_json unchanged."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _merge_access_group_resources_into_data_json,
    )

    mock_prisma = MagicMock()
    data_json = {"models": ["existing-model"]}
    result = await _merge_access_group_resources_into_data_json(
        data_json=data_json, access_group_ids=[], prisma_client=mock_prisma
    )
    assert result == {"models": ["existing-model"]}
    mock_prisma.db.litellm_accessgrouptable.find_many.assert_not_called()


@pytest.mark.asyncio
async def test_merge_access_group_resources_merges_models():
    """Models from access groups are merged into data_json['models']."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _merge_access_group_resources_into_data_json,
    )

    ag_record = MagicMock()
    ag_record.access_model_names = ["gpt-4", "gpt-3.5-turbo"]
    ag_record.access_mcp_server_ids = []
    ag_record.access_agent_ids = []

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
        return_value=[ag_record]
    )

    data_json = {"models": ["claude-3"]}
    result = await _merge_access_group_resources_into_data_json(
        data_json=data_json,
        access_group_ids=["ag-1"],
        prisma_client=mock_prisma,
    )

    assert set(result["models"]) == {"claude-3", "gpt-4", "gpt-3.5-turbo"}
    mock_prisma.db.litellm_accessgrouptable.find_many.assert_awaited_once_with(
        where={"access_group_id": {"in": ["ag-1"]}}
    )


@pytest.mark.asyncio
async def test_merge_access_group_resources_merges_mcp_servers():
    """MCP server IDs from access groups are merged into object_permission.mcp_servers."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _merge_access_group_resources_into_data_json,
    )

    ag_record = MagicMock()
    ag_record.access_model_names = []
    ag_record.access_mcp_server_ids = ["mcp-server-1", "mcp-server-2"]
    ag_record.access_agent_ids = []

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
        return_value=[ag_record]
    )

    data_json: dict = {}
    result = await _merge_access_group_resources_into_data_json(
        data_json=data_json,
        access_group_ids=["ag-1"],
        prisma_client=mock_prisma,
    )

    assert "object_permission" in result
    assert set(result["object_permission"]["mcp_servers"]) == {
        "mcp-server-1",
        "mcp-server-2",
    }


@pytest.mark.asyncio
async def test_merge_access_group_resources_merges_agents():
    """Agent IDs from access groups are merged into object_permission.agents."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _merge_access_group_resources_into_data_json,
    )

    ag_record = MagicMock()
    ag_record.access_model_names = []
    ag_record.access_mcp_server_ids = []
    ag_record.access_agent_ids = ["agent-1"]

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
        return_value=[ag_record]
    )

    data_json: dict = {}
    result = await _merge_access_group_resources_into_data_json(
        data_json=data_json,
        access_group_ids=["ag-1"],
        prisma_client=mock_prisma,
    )

    assert "object_permission" in result
    assert result["object_permission"]["agents"] == ["agent-1"]


@pytest.mark.asyncio
async def test_merge_access_group_resources_preserves_existing():
    """Existing models/MCP servers in data_json are preserved when merging."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _merge_access_group_resources_into_data_json,
    )

    ag_record = MagicMock()
    ag_record.access_model_names = ["gpt-4"]
    ag_record.access_mcp_server_ids = ["new-mcp"]
    ag_record.access_agent_ids = []

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
        return_value=[ag_record]
    )

    data_json = {
        "models": ["existing-model"],
        "object_permission": {"mcp_servers": ["existing-mcp"]},
    }
    result = await _merge_access_group_resources_into_data_json(
        data_json=data_json,
        access_group_ids=["ag-1"],
        prisma_client=mock_prisma,
    )

    assert "existing-model" in result["models"]
    assert "gpt-4" in result["models"]
    assert "existing-mcp" in result["object_permission"]["mcp_servers"]
    assert "new-mcp" in result["object_permission"]["mcp_servers"]


@pytest.mark.asyncio
async def test_merge_access_group_resources_deduplicates():
    """Models/MCP servers that appear in multiple access groups are deduplicated."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _merge_access_group_resources_into_data_json,
    )

    ag1 = MagicMock()
    ag1.access_model_names = ["gpt-4", "gpt-3.5"]
    ag1.access_mcp_server_ids = ["mcp-1"]
    ag1.access_agent_ids = []

    ag2 = MagicMock()
    ag2.access_model_names = ["gpt-4", "claude-3"]  # gpt-4 duplicated
    ag2.access_mcp_server_ids = ["mcp-1", "mcp-2"]  # mcp-1 duplicated
    ag2.access_agent_ids = []

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_accessgrouptable.find_many = AsyncMock(
        return_value=[ag1, ag2]
    )

    data_json: dict = {}
    result = await _merge_access_group_resources_into_data_json(
        data_json=data_json,
        access_group_ids=["ag-1", "ag-2"],
        prisma_client=mock_prisma,
    )

    assert result["models"].count("gpt-4") == 1
    assert result["object_permission"]["mcp_servers"].count("mcp-1") == 1


# ---------------------------------------------------------------------------
# _sync_add_access_group_to_teams resource propagation tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_add_access_group_to_teams_merges_models():
    """Adding an access group to a team also merges the group's models into team.models."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_add_access_group_to_teams,
    )

    team_record = MagicMock()
    team_record.access_group_ids = []
    team_record.models = ["existing-model"]

    mock_team_table = MagicMock()
    mock_team_table.find_unique = AsyncMock(return_value=team_record)
    mock_team_table.update = AsyncMock(return_value=None)

    tx = types.SimpleNamespace(litellm_teamtable=mock_team_table)

    ag_record = MagicMock()
    ag_record.access_model_names = ["gpt-4", "claude-3"]
    ag_record.access_mcp_server_ids = []
    ag_record.access_agent_ids = []

    await _sync_add_access_group_to_teams(
        tx=tx,
        team_ids=["team-1"],
        access_group_id="ag-1",
        access_group_record=ag_record,
    )

    mock_team_table.update.assert_awaited_once()
    call_data = mock_team_table.update.call_args.kwargs["data"]
    assert "ag-1" in call_data["access_group_ids"]
    assert set(call_data["models"]) == {"existing-model", "gpt-4", "claude-3"}


@pytest.mark.asyncio
async def test_sync_add_access_group_to_teams_no_models():
    """Adding an access group with no models does not set models key."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_add_access_group_to_teams,
    )

    team_record = MagicMock()
    team_record.access_group_ids = []
    team_record.models = []

    mock_team_table = MagicMock()
    mock_team_table.find_unique = AsyncMock(return_value=team_record)
    mock_team_table.update = AsyncMock(return_value=None)

    tx = types.SimpleNamespace(litellm_teamtable=mock_team_table)

    ag_record = MagicMock()
    ag_record.access_model_names = []
    ag_record.access_mcp_server_ids = []
    ag_record.access_agent_ids = []

    await _sync_add_access_group_to_teams(
        tx=tx,
        team_ids=["team-1"],
        access_group_id="ag-1",
        access_group_record=ag_record,
    )

    call_data = mock_team_table.update.call_args.kwargs["data"]
    assert "models" not in call_data


@pytest.mark.asyncio
async def test_sync_remove_access_group_from_teams_recomputes_models():
    """Removing an access group from a team removes models exclusive to that group.

    - ``gpt-4`` is only in ag-1 (being removed) → removed from team
    - ``claude-3`` is in ag-2 (remaining) → kept
    - ``direct-model`` is not in any AG → preserved (was directly assigned)
    """
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_remove_access_group_from_teams,
    )

    team_record = MagicMock()
    team_record.access_group_ids = ["ag-1", "ag-2"]
    # direct-model is a model directly assigned to the team (not from any AG)
    team_record.models = ["gpt-4", "claude-3", "direct-model"]
    team_record.object_permission_id = None

    removed_ag = MagicMock()
    removed_ag.access_model_names = ["gpt-4"]   # ag-1 contributes gpt-4
    removed_ag.access_mcp_server_ids = []
    removed_ag.access_agent_ids = []

    remaining_ag = MagicMock()
    remaining_ag.access_model_names = ["claude-3"]  # ag-2 only has claude-3
    remaining_ag.access_mcp_server_ids = []
    remaining_ag.access_agent_ids = []

    mock_team_table = MagicMock()
    mock_team_table.find_unique = AsyncMock(return_value=team_record)
    mock_team_table.update = AsyncMock(return_value=None)

    mock_ag_table = MagicMock()
    mock_ag_table.find_many = AsyncMock(return_value=[remaining_ag])

    tx = types.SimpleNamespace(
        litellm_teamtable=mock_team_table,
        litellm_accessgrouptable=mock_ag_table,
    )

    await _sync_remove_access_group_from_teams(
        tx=tx,
        team_ids=["team-1"],
        access_group_id="ag-1",  # removing ag-1
        removed_access_group_record=removed_ag,
    )

    mock_team_table.update.assert_awaited_once()
    call_data = mock_team_table.update.call_args.kwargs["data"]
    assert call_data["access_group_ids"] == ["ag-2"]
    # gpt-4 (exclusive to removed ag-1) gone; claude-3 and direct-model preserved
    assert set(call_data["models"]) == {"claude-3", "direct-model"}


@pytest.mark.asyncio
async def test_sync_add_access_group_to_keys_merges_models():
    """Adding an access group to a key also merges the group's models into key.models."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_add_access_group_to_keys,
    )

    key_record = MagicMock()
    key_record.access_group_ids = []
    key_record.models = ["existing-model"]

    mock_key_table = MagicMock()
    mock_key_table.find_unique = AsyncMock(return_value=key_record)
    mock_key_table.update = AsyncMock(return_value=None)

    tx = types.SimpleNamespace(litellm_verificationtoken=mock_key_table)

    ag_record = MagicMock()
    ag_record.access_model_names = ["gpt-4"]
    ag_record.access_mcp_server_ids = []
    ag_record.access_agent_ids = []

    await _sync_add_access_group_to_keys(
        tx=tx,
        key_tokens=["sk-token-1"],
        access_group_id="ag-1",
        access_group_record=ag_record,
    )

    mock_key_table.update.assert_awaited_once()
    call_data = mock_key_table.update.call_args.kwargs["data"]
    assert "ag-1" in call_data["access_group_ids"]
    assert set(call_data["models"]) == {"existing-model", "gpt-4"}


@pytest.mark.asyncio
async def test_sync_remove_access_group_from_keys_recomputes_models():
    """Removing an access group from a key removes models exclusive to that group.

    - ``gpt-4`` is only in ag-1 (being removed) → removed from key
    - ``claude-3`` is in ag-2 (remaining) → kept
    - ``direct-model`` is not in any AG → preserved (was directly assigned)
    """
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_remove_access_group_from_keys,
    )

    key_record = MagicMock()
    key_record.access_group_ids = ["ag-1", "ag-2"]
    key_record.models = ["gpt-4", "claude-3", "direct-model"]
    key_record.object_permission_id = None

    removed_ag = MagicMock()
    removed_ag.access_model_names = ["gpt-4"]
    removed_ag.access_mcp_server_ids = []
    removed_ag.access_agent_ids = []

    remaining_ag = MagicMock()
    remaining_ag.access_model_names = ["claude-3"]
    remaining_ag.access_mcp_server_ids = []
    remaining_ag.access_agent_ids = []

    mock_key_table = MagicMock()
    mock_key_table.find_unique = AsyncMock(return_value=key_record)
    mock_key_table.update = AsyncMock(return_value=None)

    mock_ag_table = MagicMock()
    mock_ag_table.find_many = AsyncMock(return_value=[remaining_ag])

    tx = types.SimpleNamespace(
        litellm_verificationtoken=mock_key_table,
        litellm_accessgrouptable=mock_ag_table,
    )

    await _sync_remove_access_group_from_keys(
        tx=tx,
        key_tokens=["sk-token-1"],
        access_group_id="ag-1",
        removed_access_group_record=removed_ag,
    )

    mock_key_table.update.assert_awaited_once()
    call_data = mock_key_table.update.call_args.kwargs["data"]
    assert call_data["access_group_ids"] == ["ag-2"]
    # gpt-4 (exclusive to removed ag-1) gone; claude-3 and direct-model preserved
    assert set(call_data["models"]) == {"claude-3", "direct-model"}


# ---------------------------------------------------------------------------
# New tests: MCP/agent propagation and direct-model preservation
# ---------------------------------------------------------------------------


def _make_op_record(
    op_id: str = "op-123",
    mcp_servers: list | None = None,
    agents: list | None = None,
):
    """Create a minimal mock LiteLLM_ObjectPermissionTable record."""
    record = MagicMock()
    record.object_permission_id = op_id
    record.mcp_servers = mcp_servers or []
    record.agents = agents or []
    record.mcp_access_groups = []
    record.mcp_tool_permissions = {}
    record.vector_stores = []
    record.agent_access_groups = []
    try:
        record.model_dump = lambda exclude_none=False: {
            "object_permission_id": op_id,
            "mcp_servers": mcp_servers or [],
            "agents": agents or [],
        }
    except Exception:
        pass
    return record


# ---------------------------------------------------------------------------
# _upsert_mcp_agents_in_object_permission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_mcp_agents_creates_new_record_when_no_existing():
    """When no existing object_permission exists, a new row is created and its
    id is returned."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _upsert_mcp_agents_in_object_permission,
    )

    new_op_record = _make_op_record(op_id="new-op-id")
    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=None)
    mock_op_table.upsert = AsyncMock(return_value=new_op_record)

    tx = types.SimpleNamespace(litellm_objectpermissiontable=mock_op_table)

    result_id = await _upsert_mcp_agents_in_object_permission(
        tx=tx,
        existing_op_id=None,
        ag_mcp_servers=["mcp-server-1"],
        ag_agents=["agent-1"],
    )

    assert result_id == "new-op-id"
    mock_op_table.upsert.assert_awaited_once()
    upsert_call = mock_op_table.upsert.call_args
    create_data = upsert_call.kwargs["data"]["create"]
    assert "mcp-server-1" in create_data["mcp_servers"]
    assert "agent-1" in create_data["agents"]


@pytest.mark.asyncio
async def test_upsert_mcp_agents_merges_into_existing_record():
    """When an existing object_permission exists, the new MCP/agents are merged."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _upsert_mcp_agents_in_object_permission,
    )

    existing_op = _make_op_record(
        op_id="existing-op",
        mcp_servers=["already-there"],
        agents=["existing-agent"],
    )
    updated_op = _make_op_record(op_id="existing-op")
    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=existing_op)
    mock_op_table.upsert = AsyncMock(return_value=updated_op)

    tx = types.SimpleNamespace(litellm_objectpermissiontable=mock_op_table)

    result_id = await _upsert_mcp_agents_in_object_permission(
        tx=tx,
        existing_op_id="existing-op",
        ag_mcp_servers=["new-mcp"],
        ag_agents=["new-agent"],
    )

    assert result_id == "existing-op"
    upsert_call = mock_op_table.upsert.call_args
    update_data = upsert_call.kwargs["data"]["update"]
    assert set(update_data["mcp_servers"]) == {"already-there", "new-mcp"}
    assert set(update_data["agents"]) == {"existing-agent", "new-agent"}


@pytest.mark.asyncio
async def test_upsert_mcp_agents_returns_none_when_both_lists_empty():
    """No DB call is made when both MCP and agent lists are empty."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _upsert_mcp_agents_in_object_permission,
    )

    mock_op_table = MagicMock()
    tx = types.SimpleNamespace(litellm_objectpermissiontable=mock_op_table)

    result = await _upsert_mcp_agents_in_object_permission(
        tx=tx,
        existing_op_id=None,
        ag_mcp_servers=[],
        ag_agents=[],
    )

    assert result is None
    mock_op_table.upsert.assert_not_called()


# ---------------------------------------------------------------------------
# _remove_mcp_agents_from_object_permission
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remove_mcp_agents_removes_specified_servers_and_agents():
    """Specified MCP servers and agents are removed from the object_permission."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _remove_mcp_agents_from_object_permission,
    )

    existing_op = _make_op_record(
        op_id="op-1",
        mcp_servers=["keep-mcp", "remove-mcp"],
        agents=["keep-agent", "remove-agent"],
    )
    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=existing_op)
    mock_op_table.update = AsyncMock(return_value=None)

    tx = types.SimpleNamespace(litellm_objectpermissiontable=mock_op_table)

    await _remove_mcp_agents_from_object_permission(
        tx=tx,
        existing_op_id="op-1",
        mcp_servers_to_remove=["remove-mcp"],
        agents_to_remove=["remove-agent"],
    )

    mock_op_table.update.assert_awaited_once()
    update_data = mock_op_table.update.call_args.kwargs["data"]
    assert update_data["mcp_servers"] == ["keep-mcp"]
    assert update_data["agents"] == ["keep-agent"]


@pytest.mark.asyncio
async def test_remove_mcp_agents_noop_when_no_existing_op_id():
    """No DB call when existing_op_id is None."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _remove_mcp_agents_from_object_permission,
    )

    mock_op_table = MagicMock()
    tx = types.SimpleNamespace(litellm_objectpermissiontable=mock_op_table)

    await _remove_mcp_agents_from_object_permission(
        tx=tx,
        existing_op_id=None,
        mcp_servers_to_remove=["mcp-1"],
        agents_to_remove=[],
    )

    mock_op_table.find_unique.assert_not_called()
    mock_op_table.update.assert_not_called()


# ---------------------------------------------------------------------------
# _sync_add_access_group_to_teams — MCP/agent merging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_add_access_group_to_teams_creates_object_permission_for_mcp():
    """When access group has MCP servers, a new object_permission row is created
    and linked to the team."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_add_access_group_to_teams,
    )

    team_record = MagicMock()
    team_record.access_group_ids = []
    team_record.models = []
    team_record.object_permission_id = None  # no existing object_permission

    new_op = _make_op_record(op_id="created-op-id")
    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=None)
    mock_op_table.upsert = AsyncMock(return_value=new_op)

    mock_team_table = MagicMock()
    mock_team_table.find_unique = AsyncMock(return_value=team_record)
    mock_team_table.update = AsyncMock(return_value=None)

    tx = types.SimpleNamespace(
        litellm_teamtable=mock_team_table,
        litellm_objectpermissiontable=mock_op_table,
    )

    ag_record = MagicMock()
    ag_record.access_model_names = []
    ag_record.access_mcp_server_ids = ["mcp-server-1", "mcp-server-2"]
    ag_record.access_agent_ids = []

    await _sync_add_access_group_to_teams(
        tx=tx,
        team_ids=["team-1"],
        access_group_id="ag-1",
        access_group_record=ag_record,
    )

    # object_permission upsert was called
    mock_op_table.upsert.assert_awaited_once()
    # team update links the new op id
    mock_team_table.update.assert_awaited_once()
    team_update_data = mock_team_table.update.call_args.kwargs["data"]
    assert team_update_data["object_permission_id"] == "created-op-id"


@pytest.mark.asyncio
async def test_sync_add_access_group_to_teams_merges_agents_into_existing_op():
    """When team already has an object_permission, agents are merged in."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_add_access_group_to_teams,
    )

    existing_op = _make_op_record(
        op_id="team-op-id",
        mcp_servers=["existing-mcp"],
        agents=["existing-agent"],
    )
    updated_op = _make_op_record(op_id="team-op-id")

    team_record = MagicMock()
    team_record.access_group_ids = []
    team_record.models = []
    team_record.object_permission_id = "team-op-id"

    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=existing_op)
    mock_op_table.upsert = AsyncMock(return_value=updated_op)

    mock_team_table = MagicMock()
    mock_team_table.find_unique = AsyncMock(return_value=team_record)
    mock_team_table.update = AsyncMock(return_value=None)

    tx = types.SimpleNamespace(
        litellm_teamtable=mock_team_table,
        litellm_objectpermissiontable=mock_op_table,
    )

    ag_record = MagicMock()
    ag_record.access_model_names = []
    ag_record.access_mcp_server_ids = []
    ag_record.access_agent_ids = ["new-agent"]

    await _sync_add_access_group_to_teams(
        tx=tx,
        team_ids=["team-1"],
        access_group_id="ag-1",
        access_group_record=ag_record,
    )

    upsert_call = mock_op_table.upsert.call_args
    update_data = upsert_call.kwargs["data"]["update"]
    assert "new-agent" in update_data["agents"]
    # Existing agent preserved
    assert "existing-agent" in update_data["agents"]

    # No new op id link needed (existing op id unchanged)
    team_update_data = mock_team_table.update.call_args.kwargs["data"]
    assert "object_permission_id" not in team_update_data


# ---------------------------------------------------------------------------
# _sync_add_access_group_to_keys — MCP/agent merging
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_add_access_group_to_keys_creates_object_permission_for_mcp():
    """When access group has MCP servers, a new object_permission row is created
    and linked to the key."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_add_access_group_to_keys,
    )

    key_record = MagicMock()
    key_record.access_group_ids = []
    key_record.models = []
    key_record.object_permission_id = None

    new_op = _make_op_record(op_id="key-op-id")
    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=None)
    mock_op_table.upsert = AsyncMock(return_value=new_op)

    mock_key_table = MagicMock()
    mock_key_table.find_unique = AsyncMock(return_value=key_record)
    mock_key_table.update = AsyncMock(return_value=None)

    tx = types.SimpleNamespace(
        litellm_verificationtoken=mock_key_table,
        litellm_objectpermissiontable=mock_op_table,
    )

    ag_record = MagicMock()
    ag_record.access_model_names = ["model-x"]
    ag_record.access_mcp_server_ids = ["mcp-1"]
    ag_record.access_agent_ids = ["agent-1"]

    await _sync_add_access_group_to_keys(
        tx=tx,
        key_tokens=["sk-token-1"],
        access_group_id="ag-1",
        access_group_record=ag_record,
    )

    # MCP upsert was called
    mock_op_table.upsert.assert_awaited_once()
    # Key update includes both model merge and new op id
    mock_key_table.update.assert_awaited_once()
    key_update_data = mock_key_table.update.call_args.kwargs["data"]
    assert "model-x" in key_update_data["models"]
    assert key_update_data["object_permission_id"] == "key-op-id"


# ---------------------------------------------------------------------------
# _sync_remove_access_group_from_teams — MCP cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_remove_access_group_from_teams_cleans_up_exclusive_mcp():
    """MCP servers exclusively from the removed AG are cleaned up from object_permission."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_remove_access_group_from_teams,
    )

    existing_op = _make_op_record(
        op_id="team-op",
        mcp_servers=["shared-mcp", "exclusive-mcp"],
        agents=[],
    )

    team_record = MagicMock()
    team_record.access_group_ids = ["ag-1", "ag-2"]
    team_record.models = []
    team_record.object_permission_id = "team-op"

    removed_ag = MagicMock()
    removed_ag.access_model_names = []
    removed_ag.access_mcp_server_ids = ["exclusive-mcp", "shared-mcp"]
    removed_ag.access_agent_ids = []

    remaining_ag = MagicMock()
    remaining_ag.access_model_names = []
    remaining_ag.access_mcp_server_ids = ["shared-mcp"]  # still in ag-2
    remaining_ag.access_agent_ids = []

    mock_team_table = MagicMock()
    mock_team_table.find_unique = AsyncMock(return_value=team_record)
    mock_team_table.update = AsyncMock(return_value=None)

    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=existing_op)
    mock_op_table.update = AsyncMock(return_value=None)

    mock_ag_table = MagicMock()
    mock_ag_table.find_many = AsyncMock(return_value=[remaining_ag])

    tx = types.SimpleNamespace(
        litellm_teamtable=mock_team_table,
        litellm_accessgrouptable=mock_ag_table,
        litellm_objectpermissiontable=mock_op_table,
    )

    await _sync_remove_access_group_from_teams(
        tx=tx,
        team_ids=["team-1"],
        access_group_id="ag-1",
        removed_access_group_record=removed_ag,
    )

    # object_permission updated to remove exclusive-mcp but keep shared-mcp
    mock_op_table.update.assert_awaited_once()
    op_update_data = mock_op_table.update.call_args.kwargs["data"]
    assert op_update_data["mcp_servers"] == ["shared-mcp"]


@pytest.mark.asyncio
async def test_sync_remove_access_group_from_teams_preserves_direct_models():
    """Models assigned directly to the team (not via any AG) are preserved
    when an access group is removed."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_remove_access_group_from_teams,
    )

    team_record = MagicMock()
    team_record.access_group_ids = ["ag-1"]
    # "direct-model" is not in any AG; "ag-model" is in ag-1
    team_record.models = ["ag-model", "direct-model"]
    team_record.object_permission_id = None

    removed_ag = MagicMock()
    removed_ag.access_model_names = ["ag-model"]
    removed_ag.access_mcp_server_ids = []
    removed_ag.access_agent_ids = []

    mock_team_table = MagicMock()
    mock_team_table.find_unique = AsyncMock(return_value=team_record)
    mock_team_table.update = AsyncMock(return_value=None)

    mock_ag_table = MagicMock()
    mock_ag_table.find_many = AsyncMock(return_value=[])  # no remaining AGs

    tx = types.SimpleNamespace(
        litellm_teamtable=mock_team_table,
        litellm_accessgrouptable=mock_ag_table,
    )

    await _sync_remove_access_group_from_teams(
        tx=tx,
        team_ids=["team-1"],
        access_group_id="ag-1",
        removed_access_group_record=removed_ag,
    )

    mock_team_table.update.assert_awaited_once()
    call_data = mock_team_table.update.call_args.kwargs["data"]
    # ag-model removed (was exclusively from ag-1); direct-model preserved
    assert call_data["models"] == ["direct-model"]
    assert call_data["access_group_ids"] == []


@pytest.mark.asyncio
async def test_sync_remove_access_group_from_teams_keeps_model_shared_with_remaining_ag():
    """A model that appears in both the removed AG and a remaining AG is NOT removed."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_remove_access_group_from_teams,
    )

    team_record = MagicMock()
    team_record.access_group_ids = ["ag-1", "ag-2"]
    team_record.models = ["shared-model"]
    team_record.object_permission_id = None

    removed_ag = MagicMock()
    removed_ag.access_model_names = ["shared-model"]
    removed_ag.access_mcp_server_ids = []
    removed_ag.access_agent_ids = []

    remaining_ag = MagicMock()
    remaining_ag.access_model_names = ["shared-model"]  # also in ag-2
    remaining_ag.access_mcp_server_ids = []
    remaining_ag.access_agent_ids = []

    mock_team_table = MagicMock()
    mock_team_table.find_unique = AsyncMock(return_value=team_record)
    mock_team_table.update = AsyncMock(return_value=None)

    mock_ag_table = MagicMock()
    mock_ag_table.find_many = AsyncMock(return_value=[remaining_ag])

    tx = types.SimpleNamespace(
        litellm_teamtable=mock_team_table,
        litellm_accessgrouptable=mock_ag_table,
    )

    await _sync_remove_access_group_from_teams(
        tx=tx,
        team_ids=["team-1"],
        access_group_id="ag-1",
        removed_access_group_record=removed_ag,
    )

    call_data = mock_team_table.update.call_args.kwargs["data"]
    # shared-model still in ag-2, so it's not removed
    assert "shared-model" in call_data["models"]


# ---------------------------------------------------------------------------
# _sync_remove_access_group_from_keys — MCP cleanup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_remove_access_group_from_keys_cleans_up_exclusive_mcp():
    """MCP servers exclusively from the removed AG are removed from object_permission."""
    from litellm.proxy.management_endpoints.access_group_endpoints import (
        _sync_remove_access_group_from_keys,
    )

    existing_op = _make_op_record(
        op_id="key-op",
        mcp_servers=["exclusive-mcp", "shared-mcp"],
        agents=["exclusive-agent"],
    )

    key_record = MagicMock()
    key_record.access_group_ids = ["ag-1", "ag-2"]
    key_record.models = []
    key_record.object_permission_id = "key-op"

    removed_ag = MagicMock()
    removed_ag.access_model_names = []
    removed_ag.access_mcp_server_ids = ["exclusive-mcp", "shared-mcp"]
    removed_ag.access_agent_ids = ["exclusive-agent"]

    remaining_ag = MagicMock()
    remaining_ag.access_model_names = []
    remaining_ag.access_mcp_server_ids = ["shared-mcp"]
    remaining_ag.access_agent_ids = []

    mock_key_table = MagicMock()
    mock_key_table.find_unique = AsyncMock(return_value=key_record)
    mock_key_table.update = AsyncMock(return_value=None)

    mock_op_table = MagicMock()
    mock_op_table.find_unique = AsyncMock(return_value=existing_op)
    mock_op_table.update = AsyncMock(return_value=None)

    mock_ag_table = MagicMock()
    mock_ag_table.find_many = AsyncMock(return_value=[remaining_ag])

    tx = types.SimpleNamespace(
        litellm_verificationtoken=mock_key_table,
        litellm_accessgrouptable=mock_ag_table,
        litellm_objectpermissiontable=mock_op_table,
    )

    await _sync_remove_access_group_from_keys(
        tx=tx,
        key_tokens=["sk-token-1"],
        access_group_id="ag-1",
        removed_access_group_record=removed_ag,
    )

    mock_op_table.update.assert_awaited_once()
    op_update = mock_op_table.update.call_args.kwargs["data"]
    # exclusive-mcp removed; shared-mcp kept; exclusive-agent removed
    assert op_update["mcp_servers"] == ["shared-mcp"]
    assert op_update["agents"] == []
