import json
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(
    0, os.path.abspath("../../../..")
)

from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionBase,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTableCachedObj,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.management_helpers.object_permission_utils import (
    _set_object_permission,
    validate_key_mcp_servers_against_team,
)


@pytest.mark.asyncio
async def test_set_object_permission():
    """
    Test that _set_object_permission correctly:
    1. Creates an object permission record in the database
    2. Excludes None values from the data
    3. Excludes object_permission_id from the data sent to create
    4. Serializes mcp_tool_permissions to JSON string
    5. Returns data_json with object_permission_id set and object_permission removed
    """
    # Mock prisma client
    mock_prisma_client = MagicMock()
    mock_created_permission = MagicMock()
    mock_created_permission.object_permission_id = "test_perm_id_123"

    mock_prisma_client.db.litellm_objectpermissiontable.create = AsyncMock(
        return_value=mock_created_permission
    )

    # Test data with object_permission
    data_json = {
        "user_id": "test_user",
        "models": ["gpt-4"],
        "object_permission": {
            "vector_stores": ["store_1", "store_2"],
            "mcp_servers": ["server_a"],
            "mcp_tool_permissions": {
                "server_a": ["tool1", "tool2"]
            },
            "object_permission_id": "should_be_excluded",
            "mcp_access_groups": None,  # This should be excluded
        }
    }

    # Call the function
    result = await _set_object_permission(
        data_json=data_json,
        prisma_client=mock_prisma_client
    )

    # Verify object_permission_id was added to result
    assert result["object_permission_id"] == "test_perm_id_123"

    # Verify object_permission was removed from result
    assert "object_permission" not in result

    # Verify create was called
    mock_prisma_client.db.litellm_objectpermissiontable.create.assert_called_once()

    # Verify the data passed to create excludes None values and object_permission_id
    call_args = mock_prisma_client.db.litellm_objectpermissiontable.create.call_args
    created_data = call_args.kwargs["data"]

    assert "object_permission_id" not in created_data
    assert "mcp_access_groups" not in created_data  # None value should be excluded
    assert created_data["vector_stores"] == ["store_1", "store_2"]
    assert created_data["mcp_servers"] == ["server_a"]

    # Verify mcp_tool_permissions was serialized to JSON string
    assert isinstance(created_data["mcp_tool_permissions"], str)
    mcp_tools_parsed = json.loads(created_data["mcp_tool_permissions"])
    assert mcp_tools_parsed == {"server_a": ["tool1", "tool2"]}

    # Verify other fields remain in result
    assert result["user_id"] == "test_user"
    assert result["models"] == ["gpt-4"]


def _make_team_obj(
    team_id: str = "team-1",
    mcp_servers: list = None,
    mcp_access_groups: list = None,
    mcp_tool_permissions: dict = None,
) -> LiteLLM_TeamTableCachedObj:
    """Helper to create a team object with object_permission."""
    obj_perm = None
    if mcp_servers is not None or mcp_access_groups is not None:
        obj_perm = LiteLLM_ObjectPermissionTable(
            object_permission_id="team-perm-1",
            mcp_servers=mcp_servers or [],
            mcp_access_groups=mcp_access_groups or [],
            mcp_tool_permissions=mcp_tool_permissions or {},
            vector_stores=[],
            agents=[],
            agent_access_groups=[],
        )
    return LiteLLM_TeamTableCachedObj(
        team_id=team_id,
        object_permission=obj_perm,
    )


def _make_user_auth(role=LitellmUserRoles.INTERNAL_USER) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        user_role=role,
        api_key="sk-test",
        user_id="user-1",
    )


def _mcp_patches(
    allow_all_keys_ids=None,
    access_group_resolver=None,
):
    """Return patch objects for the lazy-imported MCP dependencies."""
    mock_handler = MagicMock()
    mock_handler._get_mcp_servers_from_access_groups = AsyncMock(
        side_effect=access_group_resolver or (lambda groups: [])
    )

    mock_manager = MagicMock()
    mock_manager.get_allow_all_keys_server_ids.return_value = allow_all_keys_ids or []

    return (
        patch(
            "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler",
            mock_handler,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            mock_manager,
        ),
    )


@pytest.mark.asyncio
async def test_validate_mcp_servers_allowed_by_team():
    """Key creation allowed when MCP server is in team's list."""
    team_obj = _make_team_obj(mcp_servers=["server-1", "server-2"])
    obj_perm = LiteLLM_ObjectPermissionBase(mcp_servers=["server-1"])

    p1, p2 = _mcp_patches()
    with p1, p2:
        # Should not raise
        await validate_key_mcp_servers_against_team(
            object_permission=obj_perm,
            team_obj=team_obj,
            user_api_key_dict=_make_user_auth(),
        )


@pytest.mark.asyncio
async def test_validate_mcp_servers_rejected_when_not_in_team():
    """Key creation rejected when MCP server not in team's list."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    obj_perm = LiteLLM_ObjectPermissionBase(mcp_servers=["server-1", "server-999"])

    p1, p2 = _mcp_patches()
    with p1, p2:
        with pytest.raises(HTTPException) as exc:
            await validate_key_mcp_servers_against_team(
                object_permission=obj_perm,
                team_obj=team_obj,
                user_api_key_dict=_make_user_auth(),
            )
        assert exc.value.status_code == 403
        assert "server-999" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_validate_mcp_servers_admin_bypasses():
    """Admin bypasses validation."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    obj_perm = LiteLLM_ObjectPermissionBase(mcp_servers=["server-999"])

    # Admin should not raise even with disallowed server
    await validate_key_mcp_servers_against_team(
        object_permission=obj_perm,
        team_obj=team_obj,
        user_api_key_dict=_make_user_auth(role=LitellmUserRoles.PROXY_ADMIN),
    )


@pytest.mark.asyncio
async def test_validate_mcp_servers_deny_by_default_no_team_config():
    """Team with no MCP config -> deny-by-default (non-allow_all_keys blocked)."""
    team_obj = _make_team_obj()  # No object_permission
    obj_perm = LiteLLM_ObjectPermissionBase(mcp_servers=["server-1"])

    p1, p2 = _mcp_patches()
    with p1, p2:
        with pytest.raises(HTTPException) as exc:
            await validate_key_mcp_servers_against_team(
                object_permission=obj_perm,
                team_obj=team_obj,
                user_api_key_dict=_make_user_auth(),
            )
        assert exc.value.status_code == 403
        assert "no MCP configuration" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_validate_mcp_servers_allow_all_keys_always_passes():
    """allow_all_keys server always passes even without team config."""
    team_obj = _make_team_obj()  # No object_permission
    obj_perm = LiteLLM_ObjectPermissionBase(mcp_servers=["public-server"])

    p1, p2 = _mcp_patches(allow_all_keys_ids=["public-server"])
    with p1, p2:
        # Should not raise
        await validate_key_mcp_servers_against_team(
            object_permission=obj_perm,
            team_obj=team_obj,
            user_api_key_dict=_make_user_auth(),
        )


@pytest.mark.asyncio
async def test_validate_mcp_access_groups_resolve_to_unauthorized_servers():
    """Access groups resolved to unauthorized servers -> rejected."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    obj_perm = LiteLLM_ObjectPermissionBase(mcp_access_groups=["group-evil"])

    p1, p2 = _mcp_patches(
        access_group_resolver=lambda groups: (
            ["server-unauthorized"] if "group-evil" in groups else []
        ),
    )
    with p1, p2:
        with pytest.raises(HTTPException) as exc:
            await validate_key_mcp_servers_against_team(
                object_permission=obj_perm,
                team_obj=team_obj,
                user_api_key_dict=_make_user_auth(),
            )
        assert exc.value.status_code == 403
        assert "server-unauthorized" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_validate_mcp_tool_permissions_unauthorized_server():
    """mcp_tool_permissions with unauthorized server key -> rejected."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    obj_perm = LiteLLM_ObjectPermissionBase(
        mcp_tool_permissions={"server-1": ["tool1"], "server-999": ["tool2"]}
    )

    p1, p2 = _mcp_patches()
    with p1, p2:
        with pytest.raises(HTTPException) as exc:
            await validate_key_mcp_servers_against_team(
                object_permission=obj_perm,
                team_obj=team_obj,
                user_api_key_dict=_make_user_auth(),
            )
        assert exc.value.status_code == 403
        assert "server-999" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_validate_mcp_none_object_permission_passes():
    """No object_permission -> no validation needed."""
    await validate_key_mcp_servers_against_team(
        object_permission=None,
        team_obj=_make_team_obj(),
        user_api_key_dict=_make_user_auth(),
    )
