import json
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(
    0, os.path.abspath("../../../..")
)

from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import LiteLLM_ObjectPermissionTable
from litellm.proxy.management_helpers.object_permission_utils import (
    _extract_requested_mcp_access_groups,
    _extract_requested_mcp_server_ids,
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


# ---- Tests for _extract_requested_mcp_server_ids ----


def test_extract_requested_mcp_server_ids_from_mcp_servers():
    obj_perm = {"mcp_servers": ["server-1", "server-2"]}
    assert _extract_requested_mcp_server_ids(obj_perm) == {"server-1", "server-2"}


def test_extract_requested_mcp_server_ids_from_tool_permissions():
    obj_perm = {"mcp_tool_permissions": {"server-a": ["tool1"], "server-b": ["tool2"]}}
    assert _extract_requested_mcp_server_ids(obj_perm) == {"server-a", "server-b"}


def test_extract_requested_mcp_server_ids_combined():
    obj_perm = {
        "mcp_servers": ["server-1"],
        "mcp_tool_permissions": {"server-2": ["tool1"]},
    }
    assert _extract_requested_mcp_server_ids(obj_perm) == {"server-1", "server-2"}


def test_extract_requested_mcp_server_ids_none():
    assert _extract_requested_mcp_server_ids(None) == set()
    assert _extract_requested_mcp_server_ids({}) == set()


# ---- Tests for _extract_requested_mcp_access_groups ----


def test_extract_requested_mcp_access_groups():
    obj_perm = {"mcp_access_groups": ["group-a", "group-b"]}
    assert _extract_requested_mcp_access_groups(obj_perm) == {"group-a", "group-b"}


def test_extract_requested_mcp_access_groups_none():
    assert _extract_requested_mcp_access_groups(None) == set()
    assert _extract_requested_mcp_access_groups({}) == set()


# ---- Tests for validate_key_mcp_servers_against_team ----


def _make_team_obj(
    team_id="team-1",
    mcp_servers=None,
    mcp_access_groups=None,
    mcp_tool_permissions=None,
):
    """Create a mock team object with the given MCP permissions."""
    mock_team = MagicMock()
    mock_team.team_id = team_id

    if mcp_servers is not None or mcp_access_groups is not None or mcp_tool_permissions is not None:
        mock_team.object_permission = MagicMock(spec=LiteLLM_ObjectPermissionTable)
        mock_team.object_permission.mcp_servers = mcp_servers or []
        mock_team.object_permission.mcp_access_groups = mcp_access_groups or []
        mock_team.object_permission.mcp_tool_permissions = mcp_tool_permissions or {}
    else:
        mock_team.object_permission = None

    return mock_team


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_no_object_permission(mock_access_groups, mock_allow_all):
    """No object_permission on key — should pass without error."""
    await validate_key_mcp_servers_against_team(
        object_permission=None,
        team_obj=_make_team_obj(mcp_servers=["server-1"]),
    )


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_key_servers_within_team_scope(mock_access_groups, mock_allow_all):
    """Key requests servers that are in the team's scope — should pass."""
    team_obj = _make_team_obj(mcp_servers=["server-1", "server-2", "server-3"])
    await validate_key_mcp_servers_against_team(
        object_permission={"mcp_servers": ["server-1", "server-2"]},
        team_obj=team_obj,
    )


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_key_servers_outside_team_scope_raises(mock_access_groups, mock_allow_all):
    """Key requests servers NOT in the team's scope — should raise 403."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["server-1", "server-outside"]},
            team_obj=team_obj,
        )
    assert exc_info.value.status_code == 403
    assert "server-outside" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value={"global-server"},
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_allow_all_keys_servers_always_allowed(mock_access_groups, mock_allow_all):
    """allow_all_keys servers should be accessible even if not in team scope."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    await validate_key_mcp_servers_against_team(
        object_permission={"mcp_servers": ["server-1", "global-server"]},
        team_obj=team_obj,
    )


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value={"global-server"},
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_no_team_only_allow_all_keys(mock_access_groups, mock_allow_all):
    """Key without a team can only use allow_all_keys servers."""
    # This should pass — requesting a global server without a team
    await validate_key_mcp_servers_against_team(
        object_permission={"mcp_servers": ["global-server"]},
        team_obj=None,
    )


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value={"global-server"},
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_no_team_non_global_server_raises(mock_access_groups, mock_allow_all):
    """Key without a team requesting a non-global server — should raise 403."""
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["private-server"]},
            team_obj=None,
        )
    assert exc_info.value.status_code == 403
    assert "not in a team" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_team_no_mcp_config_blocks_all(mock_access_groups, mock_allow_all):
    """Team with no object_permission — key can't use any non-global MCP servers."""
    team_obj = _make_team_obj()  # No object_permission
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["some-server"]},
            team_obj=team_obj,
        )
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_tool_permissions_validated_against_team(mock_access_groups, mock_allow_all):
    """Server IDs in mcp_tool_permissions should also be validated."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={
                "mcp_tool_permissions": {"server-outside": ["tool1"]}
            },
            team_obj=team_obj,
        )
    assert exc_info.value.status_code == 403
    assert "server-outside" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_access_groups_within_team_scope(mock_access_groups, mock_allow_all):
    """Key requests access groups that are in the team's scope — should pass."""
    team_obj = _make_team_obj(mcp_access_groups=["group-a", "group-b"])
    await validate_key_mcp_servers_against_team(
        object_permission={"mcp_access_groups": ["group-a"]},
        team_obj=team_obj,
    )


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_access_groups_outside_team_scope_raises(mock_access_groups, mock_allow_all):
    """Key requests access groups NOT in the team's scope — should raise 403."""
    team_obj = _make_team_obj(mcp_access_groups=["group-a"])
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_access_groups": ["group-outside"]},
            team_obj=team_obj,
        )
    assert exc_info.value.status_code == 403
    assert "group-outside" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_access_groups_no_team_raises(mock_access_groups, mock_allow_all):
    """Key without a team requesting access groups — should raise 403."""
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_access_groups": ["group-a"]},
            team_obj=None,
        )
    assert exc_info.value.status_code == 403
    assert "not in a team" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=["server-from-group"],
)
async def test_validate_team_access_groups_resolve_to_servers(mock_access_groups, mock_allow_all):
    """Team access groups should resolve to server IDs and be included in allowed set."""
    team_obj = _make_team_obj(mcp_access_groups=["group-a"])
    # Key requests a server that comes from the team's access group
    await validate_key_mcp_servers_against_team(
        object_permission={"mcp_servers": ["server-from-group"]},
        team_obj=team_obj,
    )

