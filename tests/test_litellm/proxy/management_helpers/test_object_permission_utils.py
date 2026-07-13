import json
import os
import sys

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath("../../../.."))

from unittest.mock import AsyncMock, MagicMock, patch

from litellm.proxy._types import (
    LiteLLM_ObjectPermissionBase,
    LiteLLM_ObjectPermissionTable,
    ObjectPermissionDict,
    SpecialMCPServerName,
)
from litellm.proxy.management_helpers.object_permission_utils import (
    _extract_requested_mcp_access_groups,
    _extract_requested_mcp_server_ids,
    _resolve_team_allowed_mcp_servers,
    _rewrite_object_permission_mcp_servers,
    _set_object_permission,
    enforce_all_proxy_mcp_servers_grant_is_admin_only,
    validate_key_allowed_skills_against_team,
    validate_key_mcp_servers_against_team,
    validate_key_search_tools_against_team,
    validate_key_vector_stores_against_team,
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

    mock_prisma_client.db.litellm_objectpermissiontable.create = AsyncMock(return_value=mock_created_permission)

    # Test data with object_permission
    data_json = {
        "user_id": "test_user",
        "models": ["gpt-4"],
        "object_permission": {
            "vector_stores": ["store_1", "store_2"],
            "mcp_servers": ["server_a"],
            "mcp_tool_permissions": {"server_a": ["tool1", "tool2"]},
            "object_permission_id": "should_be_excluded",
            "mcp_access_groups": None,  # This should be excluded
        },
    }

    # Call the function
    result = await _set_object_permission(data_json=data_json, prisma_client=mock_prisma_client)

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


@pytest.mark.asyncio
async def test_set_object_permission_persists_mcp_tool_search_enabled():
    """
    Regression: mcp_tool_search_enabled must be carried into the Prisma create
    payload so it persists to LiteLLM_ObjectPermissionTable. The field was
    present on the Pydantic models but missing from the create path, so keys
    generated with mcp_tool_search_enabled=True silently lost the flag.
    """
    mock_prisma_client = MagicMock()
    mock_created_permission = MagicMock()
    mock_created_permission.object_permission_id = "perm_id"
    mock_prisma_client.db.litellm_objectpermissiontable.create = AsyncMock(return_value=mock_created_permission)

    data_json = {
        "object_permission": {
            "mcp_servers": ["server_a"],
            "mcp_tool_search_enabled": True,
        },
    }

    await _set_object_permission(data_json=data_json, prisma_client=mock_prisma_client)

    created_data = mock_prisma_client.db.litellm_objectpermissiontable.create.call_args.kwargs["data"]
    assert created_data["mcp_tool_search_enabled"] is True


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


def test_extract_requested_mcp_server_ids_excludes_no_mcp_servers_sentinel():
    obj_perm = {"mcp_servers": ["no-mcp-servers", "server-1"]}
    assert _extract_requested_mcp_server_ids(obj_perm) == {"server-1"}


def test_rewrite_object_permission_mcp_servers_preserves_sentinel():
    obj_perm = {"mcp_servers": ["no-mcp-servers", "alias-1"]}
    _rewrite_object_permission_mcp_servers(obj_perm, {"alias-1": {"server-1"}})
    assert obj_perm["mcp_servers"] == ["no-mcp-servers", "server-1"]


@pytest.mark.asyncio
async def test_validate_no_mcp_servers_sentinel_passes_and_preserved():
    """A key scoped to no-mcp-servers passes team validation untouched, keeping the
    sentinel so it is not mistaken for an unknown server and rejected."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    obj_perm = {"mcp_servers": ["no-mcp-servers"]}
    result = await validate_key_mcp_servers_against_team(
        object_permission=obj_perm,
        team_obj=team_obj,
    )
    assert result == obj_perm
    assert obj_perm["mcp_servers"] == ["no-mcp-servers"]


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


def _make_mock_mcp_server(
    server_id: str,
    alias=None,
    server_name=None,
    name=None,
):
    mock_server = MagicMock()
    mock_server.server_id = server_id
    mock_server.alias = alias
    mock_server.server_name = server_name
    mock_server.name = name or server_name or alias or server_id
    return mock_server


def _make_mock_mcp_manager(*existing_ids: str, servers=None):
    """
    Return a mock global_mcp_server_manager with a registry containing every
    explicit server plus simple server objects for every ID in *existing_ids.
    """
    mock_mgr = MagicMock()
    server_objs = {server.server_id: server for server in (servers or [])}
    for server_id in existing_ids:
        server_objs.setdefault(server_id, _make_mock_mcp_server(server_id))
    mock_mgr.get_registry.return_value = server_objs
    mock_mgr.get_mcp_server_by_id.side_effect = lambda sid: server_objs.get(sid)
    return mock_mgr


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
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("server-1", "server-2"),
)
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
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("server-1", "server-outside"),
)
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
    """Key requests a server that exists but is NOT in the team's scope — should raise 403."""
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
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("server-1", "global-server"),
)
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
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("global-server"),
)
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
    await validate_key_mcp_servers_against_team(
        object_permission={"mcp_servers": ["global-server"]},
        team_obj=None,
    )


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("private-server"),
)
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
    """Key without a team requesting an existing non-global server — should raise 403."""
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["private-server"]},
            team_obj=None,
        )
    assert exc_info.value.status_code == 403
    assert "not in a team" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("private-server"),
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_no_team_proxy_admin_can_assign_private_server(mock_access_groups, mock_allow_all):
    """Proxy admin assigning a non-global server to a teamless key — should pass (LIT-3815)."""
    result = await validate_key_mcp_servers_against_team(
        object_permission={"mcp_servers": ["private-server"]},
        team_obj=None,
        is_proxy_admin=True,
    )
    assert result["mcp_servers"] == ["private-server"]


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("private-server"),
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_no_team_non_admin_private_server_still_raises(mock_access_groups, mock_allow_all):
    """The teamless override is gated on proxy admin — a non-admin still gets 403."""
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["private-server"]},
            team_obj=None,
            is_proxy_admin=False,
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
async def test_validate_no_team_proxy_admin_can_assign_access_group(mock_access_groups, mock_allow_all):
    """Proxy admin assigning an access group to a teamless key — should pass (LIT-3815)."""
    result = await validate_key_mcp_servers_against_team(
        object_permission={"mcp_access_groups": ["group-1"]},
        team_obj=None,
        is_proxy_admin=True,
    )
    assert result["mcp_access_groups"] == ["group-1"]


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("server-1", "server-outside"),
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_proxy_admin_still_bounded_by_team_scope(mock_access_groups, mock_allow_all):
    """The override is scoped to teamless keys — an admin assigning beyond a team's scope still raises."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["server-1", "server-outside"]},
            team_obj=team_obj,
            is_proxy_admin=True,
        )
    assert exc_info.value.status_code == 403
    assert "server-outside" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("some-server"),
)
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
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("server-outside"),
)
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
    """Server IDs in mcp_tool_permissions should also be validated when they exist."""
    team_obj = _make_team_obj(mcp_servers=["server-1"])
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_tool_permissions": {"server-outside": ["tool1"]}},
            team_obj=team_obj,
        )
    assert exc_info.value.status_code == 403
    assert "server-outside" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager(),  # empty registry — all IDs are stale
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_stale_mcp_server_ids_are_silently_dropped(mock_access_groups, mock_allow_all):
    """
    Stale MCP server IDs (servers deleted and no longer in the registry) must not
    block a key save with a 403. They are silently stripped instead.

    Scenario: key/team were configured with S1+S2, those servers were deleted and
    replaced with S3+S4. The UI form still holds S1+S2 in its local state. Saving
    should succeed, not raise a 403.
    """
    team_obj = _make_team_obj(mcp_servers=["s3", "s4"])
    await validate_key_mcp_servers_against_team(
        object_permission={"mcp_servers": ["s1-stale", "s2-stale"]},
        team_obj=team_obj,
    )  # Must not raise


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager(),  # empty registry — all IDs are stale
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_stale_ids_in_mcp_tool_permissions_silently_dropped(mock_access_groups, mock_allow_all):
    """
    Stale server IDs referenced only as keys in mcp_tool_permissions (not in
    mcp_servers) must also be silently stripped rather than raising a 403.
    """
    team_obj = _make_team_obj(mcp_servers=["s3", "s4"])
    object_permission = {"mcp_tool_permissions": {"s1-stale": ["tool1"]}}
    await validate_key_mcp_servers_against_team(
        object_permission=object_permission,
        team_obj=team_obj,
    )  # Must not raise
    assert object_permission["mcp_tool_permissions"] == {}


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager(),  # empty registry — all IDs are stale
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_stale_mcp_server_ids_are_removed_from_object_permission(mock_access_groups, mock_allow_all):
    team_obj = _make_team_obj(mcp_servers=["s3", "s4"])
    object_permission = {"mcp_servers": ["s1-stale", "s2-stale"]}
    await validate_key_mcp_servers_against_team(
        object_permission=object_permission,
        team_obj=team_obj,
    )
    assert object_permission["mcp_servers"] == []


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager(
        "team-server",
        servers=[
            _make_mock_mcp_server(
                "private-server-id",
                alias="private-alias",
                server_name="Private Server",
            )
        ],
    ),
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_mcp_server_alias_outside_team_scope_raises(mock_access_groups, mock_allow_all):
    team_obj = _make_team_obj(mcp_servers=["team-server"])
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["private-alias"]},
            team_obj=team_obj,
        )
    assert exc_info.value.status_code == 403
    assert "private-server-id" in str(exc_info.value.detail)


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager(
        servers=[
            _make_mock_mcp_server(
                "allowed-server-id",
                alias="allowed-alias",
                server_name="Allowed Server",
            )
        ],
    ),
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_mcp_server_alias_is_normalized_before_save(mock_access_groups, mock_allow_all):
    team_obj = _make_team_obj(mcp_servers=["allowed-server-id"])
    object_permission = {
        "mcp_servers": ["allowed-alias"],
        "mcp_tool_permissions": {"Allowed Server": ["tool1"], "stale-id": ["tool2"]},
    }

    await validate_key_mcp_servers_against_team(
        object_permission=object_permission,
        team_obj=team_obj,
    )

    assert object_permission["mcp_servers"] == ["allowed-server-id"]
    assert object_permission["mcp_tool_permissions"] == {"allowed-server-id": ["tool1"]}


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager(),
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_db_mcp_server_alias_outside_team_scope_raises_when_registry_empty(
    mock_access_groups, mock_allow_all
):
    mock_prisma_client = MagicMock()
    mock_db_server = MagicMock()
    mock_db_server.server_id = "private-server-id"
    mock_db_server.alias = "private-alias"
    mock_db_server.server_name = "Private Server"
    mock_prisma_client.db.litellm_mcpservertable.find_many = AsyncMock(return_value=[mock_db_server])

    team_obj = _make_team_obj(mcp_servers=[])
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["private-alias"]},
            team_obj=team_obj,
            prisma_client=mock_prisma_client,
        )

    assert exc_info.value.status_code == 403
    assert "private-server-id" in str(exc_info.value.detail)


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


# ---- Tests for _resolve_team_allowed_mcp_servers with JSON string mcp_tool_permissions ----


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_resolve_team_allowed_mcp_servers_string_tool_permissions(
    mock_access_groups,
):
    """mcp_tool_permissions stored as a JSON string (via safe_dumps) should be deserialized correctly."""
    mock_perm = MagicMock(spec=LiteLLM_ObjectPermissionTable)
    mock_perm.mcp_servers = ["server-1"]
    mock_perm.mcp_access_groups = []
    mock_perm.mcp_tool_permissions = json.dumps({"server-2": ["tool1"]})

    result = await _resolve_team_allowed_mcp_servers(mock_perm)
    assert result == {"server-1", "server-2"}


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_resolve_team_allowed_mcp_servers_dict_tool_permissions(
    mock_access_groups,
):
    """mcp_tool_permissions as a dict should work without deserialization."""
    mock_perm = MagicMock(spec=LiteLLM_ObjectPermissionTable)
    mock_perm.mcp_servers = []
    mock_perm.mcp_access_groups = []
    mock_perm.mcp_tool_permissions = {"server-a": ["tool1"]}

    result = await _resolve_team_allowed_mcp_servers(mock_perm)
    assert result == {"server-a"}


# ---- Tests for the all-proxy-mcpservers sentinel (team scoped to every server) ----


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_resolve_team_all_proxy_sentinel_resolves_dynamically(mock_access_groups):
    """A team whose object_permission.mcp_servers holds the all-proxy sentinel
    resolves to every registered server id, and picks up a server registered
    later without any change to the team's stored permission (this kills the
    early-return that maps the sentinel to the live registry)."""
    registry = {
        "srv-x": _make_mock_mcp_server("srv-x"),
        "srv-y": _make_mock_mcp_server("srv-y"),
    }
    mock_mgr = MagicMock()
    mock_mgr.get_registry.return_value = registry

    team_perm = MagicMock(spec=LiteLLM_ObjectPermissionTable)
    team_perm.mcp_servers = [SpecialMCPServerName.all_proxy_servers.value]
    team_perm.mcp_access_groups = []
    team_perm.mcp_tool_permissions = {}

    with patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        mock_mgr,
    ):
        assert await _resolve_team_allowed_mcp_servers(team_perm) == {"srv-x", "srv-y"}

        registry["srv-z"] = _make_mock_mcp_server("srv-z")
        assert await _resolve_team_allowed_mcp_servers(team_perm) == {
            "srv-x",
            "srv-y",
            "srv-z",
        }


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("srv-x", "srv-y", "srv-z"),
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_key_scoped_to_server_added_after_team_all_proxy(mock_access_groups, mock_allow_all):
    """The exact user scenario: a team scoped to the all-proxy sentinel, a server
    (srv-z) registered afterwards, and a key scoped to just srv-z. Because the
    team ceiling resolves to every registered server, the key passes validation
    and keeps srv-z in its normalized permission."""
    team_obj = _make_team_obj(mcp_servers=[SpecialMCPServerName.all_proxy_servers.value])
    object_permission = {"mcp_servers": ["srv-z"]}
    result = await validate_key_mcp_servers_against_team(
        object_permission=object_permission,
        team_obj=team_obj,
    )
    assert result is not None
    assert result["mcp_servers"] == ["srv-z"]


@pytest.mark.asyncio
@patch(
    "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
    new=_make_mock_mcp_manager("srv-x", "srv-z"),
)
@patch(
    "litellm.proxy.management_helpers.object_permission_utils._get_allow_all_keys_server_ids",
    return_value=set(),
)
@patch(
    "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
    new_callable=AsyncMock,
    return_value=[],
)
async def test_validate_key_scoped_to_server_rejected_when_team_not_all_proxy(mock_access_groups, mock_allow_all):
    """Contrast with the sentinel case: a team scoped to a concrete server list
    (srv-x, not the sentinel) does NOT unlock srv-z for a key. It is the sentinel
    specifically, not a blanket allow, that widens the team ceiling."""
    team_obj = _make_team_obj(mcp_servers=["srv-x"])
    with pytest.raises(HTTPException) as exc_info:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_servers": ["srv-z"]},
            team_obj=team_obj,
        )
    assert exc_info.value.status_code == 403
    assert "srv-z" in str(exc_info.value.detail)


# ---- Tests for the proxy-admin gate on granting a team the all-proxy sentinel ----


@pytest.mark.asyncio
async def test_enforce_all_proxy_mcp_grant_blocks_non_admin_adding_sentinel():
    """A non-proxy-admin (e.g. a team admin) cannot newly grant a team the all-proxy
    MCP sentinel. Without this gate a team admin could self-escalate their team to
    every MCP server on the proxy via team create/update."""
    with pytest.raises(HTTPException) as exc_info:
        await enforce_all_proxy_mcp_servers_grant_is_admin_only(
            requested_mcp_servers=[SpecialMCPServerName.all_proxy_servers.value],
            existing_object_permission_id=None,
            is_proxy_admin=False,
            prisma_client=None,
        )
    assert exc_info.value.status_code == 403
    assert "all-proxy-mcpservers" in str(exc_info.value.detail)


@pytest.mark.asyncio
async def test_enforce_all_proxy_mcp_grant_allows_proxy_admin():
    """A proxy admin may grant the sentinel — the intended way to scope a team to all
    proxy MCP servers."""
    await enforce_all_proxy_mcp_servers_grant_is_admin_only(
        requested_mcp_servers=[SpecialMCPServerName.all_proxy_servers.value],
        existing_object_permission_id=None,
        is_proxy_admin=True,
        prisma_client=None,
    )


@pytest.mark.asyncio
async def test_enforce_all_proxy_mcp_grant_allows_non_admin_without_sentinel():
    """A non-admin scoping a team to concrete servers is unaffected by the gate."""
    await enforce_all_proxy_mcp_servers_grant_is_admin_only(
        requested_mcp_servers=["srv-x", "srv-y"],
        existing_object_permission_id=None,
        is_proxy_admin=False,
        prisma_client=None,
    )


@pytest.mark.asyncio
async def test_enforce_all_proxy_mcp_grant_allows_non_admin_when_sentinel_already_set():
    """The gate blocks only NEW grants: a non-admin editing a team a proxy admin
    already scoped to all-proxy is not forced to strip the sentinel, so unrelated
    edits still succeed. The existing permission is read from the DB by id."""
    existing_row = MagicMock()
    existing_row.mcp_servers = [SpecialMCPServerName.all_proxy_servers.value]
    mock_repo = MagicMock()
    mock_repo.table.find_unique = AsyncMock(return_value=existing_row)

    with patch(
        "litellm.proxy.management_helpers.object_permission_utils.ObjectPermissionRepository",
        return_value=mock_repo,
    ):
        await enforce_all_proxy_mcp_servers_grant_is_admin_only(
            requested_mcp_servers=[SpecialMCPServerName.all_proxy_servers.value],
            existing_object_permission_id="op-1",
            is_proxy_admin=False,
            prisma_client=MagicMock(),
        )
    mock_repo.table.find_unique.assert_awaited_once()


# ---- Tests for validate_key_search_tools_against_team ----


def _make_team_obj_search(team_id="team-1", search_tools=None):
    mock_team = MagicMock()
    mock_team.team_id = team_id
    if search_tools is not None:
        mock_team.object_permission = MagicMock(spec=LiteLLM_ObjectPermissionTable)
        mock_team.object_permission.search_tools = search_tools
    else:
        mock_team.object_permission = None
    return mock_team


@pytest.mark.asyncio
async def test_validate_search_tools_no_key_request():
    await validate_key_search_tools_against_team(
        object_permission=None,
        team_obj=_make_team_obj_search(search_tools=["t1"]),
    )


@pytest.mark.asyncio
async def test_validate_search_tools_team_unrestricted():
    """Empty team search allowlist means unrestricted — key subset check skipped."""
    await validate_key_search_tools_against_team(
        object_permission={"search_tools": ["any-tool"]},
        team_obj=_make_team_obj_search(search_tools=[]),
    )


@pytest.mark.asyncio
async def test_validate_search_tools_subset_ok():
    await validate_key_search_tools_against_team(
        object_permission={"search_tools": ["t1"]},
        team_obj=_make_team_obj_search(search_tools=["t1", "t2"]),
    )


@pytest.mark.asyncio
async def test_validate_search_tools_raises_when_not_subset():
    with pytest.raises(HTTPException) as exc:
        await validate_key_search_tools_against_team(
            object_permission={"search_tools": ["bad"]},
            team_obj=_make_team_obj_search(search_tools=["t1"]),
        )
    assert exc.value.status_code == 403


# ---- Personal-key non-admin gates on toolsets / vector_stores / search_tools ----


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
async def test_personal_non_admin_cannot_assign_mcp_toolsets(mock_access_groups, mock_allow_all):
    with pytest.raises(HTTPException) as exc:
        await validate_key_mcp_servers_against_team(
            object_permission={"mcp_toolsets": ["ts-private"]},
            team_obj=None,
            is_proxy_admin=False,
        )
    assert exc.value.status_code == 403
    assert "ts-private" in str(exc.value.detail)


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
async def test_personal_admin_can_assign_mcp_toolsets(mock_access_groups, mock_allow_all):
    await validate_key_mcp_servers_against_team(
        object_permission={"mcp_toolsets": ["ts-private"]},
        team_obj=None,
        is_proxy_admin=True,
    )


@pytest.mark.asyncio
async def test_personal_non_admin_cannot_assign_vector_stores():
    with pytest.raises(HTTPException) as exc:
        await validate_key_vector_stores_against_team(
            object_permission={"vector_stores": ["vs-private"]},
            team_obj=None,
            is_proxy_admin=False,
        )
    assert exc.value.status_code == 403
    assert "vs-private" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_personal_admin_can_assign_vector_stores():
    await validate_key_vector_stores_against_team(
        object_permission={"vector_stores": ["vs-private"]},
        team_obj=None,
        is_proxy_admin=True,
    )


@pytest.mark.asyncio
async def test_team_key_vector_stores_unrestricted_at_create():
    """Team-scoped keys retain their existing trust model at create time."""
    team_obj = _make_team_obj_search()
    await validate_key_vector_stores_against_team(
        object_permission={"vector_stores": ["vs-anything"]},
        team_obj=team_obj,
        is_proxy_admin=False,
    )


@pytest.mark.asyncio
async def test_personal_non_admin_cannot_assign_search_tools():
    with pytest.raises(HTTPException) as exc:
        await validate_key_search_tools_against_team(
            object_permission={"search_tools": ["st-private"]},
            team_obj=None,
            is_proxy_admin=False,
        )
    assert exc.value.status_code == 403
    assert "st-private" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_personal_admin_can_assign_search_tools():
    await validate_key_search_tools_against_team(
        object_permission={"search_tools": ["st-private"]},
        team_obj=None,
        is_proxy_admin=True,
    )


@pytest.mark.asyncio
async def test_empty_object_permission_passes_for_personal_non_admin():
    """An empty / absent object_permission must not be blocked."""
    await validate_key_vector_stores_against_team(
        object_permission=None,
        team_obj=None,
        is_proxy_admin=False,
    )
    await validate_key_vector_stores_against_team(
        object_permission={"vector_stores": []},
        team_obj=None,
        is_proxy_admin=False,
    )
    await validate_key_search_tools_against_team(
        object_permission=None,
        team_obj=None,
        is_proxy_admin=False,
    )


@pytest.mark.asyncio
async def test_personal_non_admin_cannot_assign_allowed_skills():
    """
    Regression test: a non-admin caller with no team must not be able to
    self-assign Claude Code skill access on their own key - allowed_skills
    is the authorization boundary get_allowed_skills() reads from directly,
    with no team/org ceiling to fall back on for a personal key.
    """
    with pytest.raises(HTTPException) as exc:
        await validate_key_allowed_skills_against_team(
            object_permission={"allowed_skills": ["anthropic-agent-skills--private-skill"]},
            team_obj=None,
            is_proxy_admin=False,
        )
    assert exc.value.status_code == 403
    assert "anthropic-agent-skills--private-skill" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_personal_admin_can_assign_allowed_skills():
    await validate_key_allowed_skills_against_team(
        object_permission={"allowed_skills": ["anthropic-agent-skills--private-skill"]},
        team_obj=None,
        is_proxy_admin=True,
    )


@pytest.mark.asyncio
async def test_team_key_allowed_skills_unrestricted_at_create():
    """Team-scoped keys retain their existing trust model at create time."""
    team_obj = _make_team_obj_search()
    await validate_key_allowed_skills_against_team(
        object_permission={"allowed_skills": ["anthropic-agent-skills--anything"]},
        team_obj=team_obj,
        is_proxy_admin=False,
    )


@pytest.mark.asyncio
async def test_empty_allowed_skills_passes_for_personal_non_admin():
    await validate_key_allowed_skills_against_team(
        object_permission=None,
        team_obj=None,
        is_proxy_admin=False,
    )
    await validate_key_allowed_skills_against_team(
        object_permission={"allowed_skills": []},
        team_obj=None,
        is_proxy_admin=False,
    )


def _make_team_obj_allowed_skills(team_id="team-1", allowed_skills=None):
    mock_team = MagicMock()
    mock_team.team_id = team_id
    if allowed_skills is not None:
        mock_team.object_permission = MagicMock(spec=LiteLLM_ObjectPermissionTable)
        mock_team.object_permission.allowed_skills = allowed_skills
    else:
        mock_team.object_permission = None
    return mock_team


@pytest.mark.asyncio
async def test_validate_allowed_skills_subset_ok():
    await validate_key_allowed_skills_against_team(
        object_permission={"allowed_skills": ["marketplace--skill-a"]},
        team_obj=_make_team_obj_allowed_skills(allowed_skills=["marketplace--skill-a", "marketplace--skill-b"]),
        is_proxy_admin=False,
    )


@pytest.mark.asyncio
async def test_validate_allowed_skills_raises_when_not_subset():
    """
    Regression test: once a team has an explicit allowed_skills allowlist,
    a member of that team must not be able to self-assign a skill outside
    it - get_allowed_skills() reads allowed_skills directly with no
    downstream id-resolution step to catch an over-broad grant later.
    """
    with pytest.raises(HTTPException) as exc:
        await validate_key_allowed_skills_against_team(
            object_permission={"allowed_skills": ["marketplace--private-skill"]},
            team_obj=_make_team_obj_allowed_skills(allowed_skills=["marketplace--skill-a"]),
            is_proxy_admin=False,
        )
    assert exc.value.status_code == 403
    assert "marketplace--private-skill" in str(exc.value.detail)


@pytest.mark.asyncio
async def test_validate_allowed_skills_team_unrestricted_allows_any():
    """Empty team allowed_skills allowlist means unrestricted at the team
    layer, matching get_allowed_skills' own intersection semantics - the
    subset check is skipped, not treated as deny-all."""
    await validate_key_allowed_skills_against_team(
        object_permission={"allowed_skills": ["marketplace--anything"]},
        team_obj=_make_team_obj_allowed_skills(allowed_skills=[]),
        is_proxy_admin=False,
    )


def test_object_permission_dict_mirrors_pydantic_model():
    """ObjectPermissionDict must stay field-for-field aligned with
    LiteLLM_ObjectPermissionBase. If a new field is added to the Pydantic
    model, this test fails until the TypedDict is updated to match."""
    from typing import get_type_hints

    pydantic_fields = set(LiteLLM_ObjectPermissionBase.model_fields.keys())
    typeddict_fields = set(get_type_hints(ObjectPermissionDict).keys())
    assert pydantic_fields == typeddict_fields, (
        f"ObjectPermissionDict drifted from LiteLLM_ObjectPermissionBase.\n"
        f"Only in Pydantic model: {sorted(pydantic_fields - typeddict_fields)}\n"
        f"Only in TypedDict:      {sorted(typeddict_fields - pydantic_fields)}"
    )
