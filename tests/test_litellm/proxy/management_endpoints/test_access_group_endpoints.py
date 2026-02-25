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
    mock_team_table.update = AsyncMock(return_value=None)

    mock_key_table = MagicMock()
    mock_key_table.find_many = AsyncMock(return_value=[])
    mock_key_table.update = AsyncMock(return_value=None)

    @asynccontextmanager
    async def mock_tx():
        tx = types.SimpleNamespace(
            litellm_accessgrouptable=mock_access_group_table,
            litellm_teamtable=mock_team_table,
            litellm_verificationtoken=mock_key_table,
        )
        yield tx

    mock_db = types.SimpleNamespace(
        litellm_accessgrouptable=mock_access_group_table,
        litellm_teamtable=mock_team_table,
        litellm_verificationtoken=mock_key_table,
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
    """Delete removes access_group_id from teams and keys before deleting the group."""
    client, mock_prisma, mock_access_group_table, mock_cache, mock_proxy_logging = client_and_mocks
    mock_team_table = mock_prisma.db.litellm_teamtable
    mock_key_table = mock_prisma.db.litellm_verificationtoken

    existing = _make_access_group_record(access_group_id="ag-to-delete")
    mock_access_group_table.find_unique = AsyncMock(return_value=existing)

    team_with_group = MagicMock()
    team_with_group.team_id = "team-1"
    team_with_group.access_group_ids = ["ag-to-delete", "ag-other"]
    mock_team_table.find_many = AsyncMock(return_value=[team_with_group])

    key_with_group = MagicMock()
    key_with_group.token = "key-token-1"
    key_with_group.access_group_ids = ["ag-to-delete"]
    mock_key_table.find_many = AsyncMock(return_value=[key_with_group])

    resp = client.delete("/v1/access_group/ag-to-delete")
    assert resp.status_code == 204

    mock_team_table.update.assert_awaited_once_with(
        where={"team_id": "team-1"},
        data={"access_group_ids": ["ag-other"]},
    )
    mock_key_table.update.assert_awaited_once_with(
        where={"token": "key-token-1"},
        data={"access_group_ids": []},
    )
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
    mock_team_table.find_many = AsyncMock(return_value=[team_with_group])

    key_with_group = MagicMock()
    key_with_group.token = "hashed-key-1"
    key_with_group.access_group_ids = ["ag-to-delete"]
    mock_key_table.find_many = AsyncMock(return_value=[key_with_group])

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
