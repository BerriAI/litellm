from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._experimental.mcp_server.tool_permission_backfill import (
    ConvertedRow,
    Unavailable,
    convert_row,
    resolve_granted_server_ids,
    run_mcp_tool_permission_backfill,
)

INVENTORY = {"list_items": "list things", "search_notes": "find notes", "delete_item": "remove one"}


def _row(**overrides):
    fields = {
        "object_permission_id": "perm-1",
        "mcp_servers": ["server-a"],
        "mcp_access_groups": [],
        "mcp_tool_permissions": None,
        "mcp_toolsets": None,
        "mcp_permission_version": 0,
    }
    fields.update(overrides)
    return LiteLLM_ObjectPermissionTable(**fields)


def test_convert_selected_delete_tool_goes_to_allow_and_unselected_nondelete_to_deny():
    row = _row(mcp_tool_permissions={"server-a": ["list_items", "delete_item"]})
    result = convert_row(row, {"server-a": INVENTORY})
    assert isinstance(result, ConvertedRow)
    assert result.mcp_tool_overrides["server-a"] == {
        "allow": ["delete_item"],
        "deny": ["search_notes"],
    }
    assert result.mcp_tool_permissions == {}
    assert result.mcp_tool_permissions_archive == {"server-a": ["list_items", "delete_item"]}
    assert result.mcp_permission_version == 1


def test_convert_preserves_undiscovered_stored_allows():
    row = _row(mcp_tool_permissions={"server-a": ["ghost_tool"]})
    result = convert_row(row, {"server-a": INVENTORY})
    assert isinstance(result, ConvertedRow)
    assert result.mcp_tool_overrides["server-a"]["allow"] == ["ghost_tool"]


def test_convert_empty_legacy_list_stays_deny_all_untouched():
    row = _row(mcp_tool_permissions={"server-a": []})
    result = convert_row(row, {"server-a": INVENTORY})
    assert isinstance(result, ConvertedRow)
    assert "server-a" not in result.mcp_tool_overrides
    assert result.mcp_tool_permissions == {"server-a": []}


def test_convert_unrestricted_grant_keeps_only_current_deletes():
    row = _row()
    result = convert_row(row, {"server-a": INVENTORY})
    assert isinstance(result, ConvertedRow)
    assert result.mcp_tool_overrides["server-a"] == {"allow": ["delete_item"], "deny": []}
    assert result.mcp_tool_permissions == {}


@pytest.mark.asyncio
async def test_converted_row_denies_new_delete_allows_new_nondelete():
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        level_allowed_tools,
    )

    row = _row()
    conversion = convert_row(row, {"server-a": INVENTORY})
    assert isinstance(conversion, ConvertedRow)
    converted = _row(
        mcp_tool_overrides=dict(conversion.mcp_tool_overrides),
        mcp_permission_version=1,
    )
    manager = MagicMock()
    manager.expand_permission_list = MagicMock(side_effect=lambda servers: servers)
    manager.expand_tool_permissions = MagicMock(side_effect=lambda perms: perms or {})
    manager.expand_tool_overrides = MagicMock(side_effect=lambda overrides: overrides or {})
    with patch(
        "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
        manager,
    ):
        allowed = level_allowed_tools(
            row=converted,
            server_id="server-a",
            grants_server=True,
            toolset_tools=None,
            inventory={**INVENTORY, "delete_link": "unlink", "create_item": "make one"},
        )
    assert allowed is not None
    assert "create_item" in allowed
    assert "delete_item" in allowed
    assert "delete_link" not in allowed


def test_convert_inventory_unavailable_marks_row_without_writes():
    row = _row()
    result = convert_row(row, {"server-a": None})
    assert isinstance(result, Unavailable)
    assert result.server_ids == {"server-a"}


def _manager(inventories=None, registry=None):
    manager = MagicMock()
    manager.expand_permission_list = MagicMock(side_effect=lambda servers: servers)
    manager.expand_tool_permissions = MagicMock(side_effect=lambda perms: perms or {})
    manager.expand_tool_overrides = MagicMock(side_effect=lambda overrides: overrides or {})
    manager.get_registry = MagicMock(return_value=registry or {"server-a": MagicMock(server_id="server-a")})
    manager.fetch_unfiltered_inventory = AsyncMock(side_effect=lambda server_id: (inventories or {}).get(server_id))
    return manager


def _prisma(rows, update_count=1):
    prisma = MagicMock()
    table = prisma.db.litellm_objectpermissiontable
    table.find_many = AsyncMock(return_value=rows)
    table.update_many = AsyncMock(return_value=update_count)
    return prisma


@pytest.mark.asyncio
async def test_runner_converts_row_with_cas_update():
    row = _row()
    prisma = _prisma([row])
    manager = _manager(inventories={"server-a": INVENTORY})
    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
        AsyncMock(return_value=[]),
    ):
        report = await run_mcp_tool_permission_backfill(prisma, manager)
    assert report.converted == {"perm-1"}
    import json

    data = prisma.db.litellm_objectpermissiontable.update_many.await_args.kwargs["data"]
    assert json.loads(data["mcp_tool_overrides"])["server-a"]["allow"] == ["delete_item"]
    assert data["mcp_permission_version"] == 1
    where = prisma.db.litellm_objectpermissiontable.update_many.await_args.kwargs["where"]
    assert where["object_permission_id"] == "perm-1"
    assert where["mcp_permission_version"] == {"in": [0, None]}


@pytest.mark.asyncio
async def test_runner_unavailable_server_skips_row_no_write():
    prisma = _prisma([_row()])
    manager = _manager(inventories={"server-a": None})
    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
        AsyncMock(return_value=[]),
    ):
        report = await run_mcp_tool_permission_backfill(prisma, manager)
    assert report.unavailable == {"perm-1": frozenset({"server-a"})}
    prisma.db.litellm_objectpermissiontable.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_cas_miss_is_counted_without_retry():
    prisma = _prisma([_row()], update_count=0)
    manager = _manager(inventories={"server-a": INVENTORY})
    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
        AsyncMock(return_value=[]),
    ):
        report = await run_mcp_tool_permission_backfill(prisma, manager)
    assert report.cas_missed == {"perm-1"}
    assert prisma.db.litellm_objectpermissiontable.update_many.await_count == 1


@pytest.mark.asyncio
async def test_runner_second_run_is_noop():
    prisma = _prisma([])
    manager = _manager()
    report = await run_mcp_tool_permission_backfill(prisma, _manager())
    assert report.converted == frozenset()
    assert report.cas_missed == frozenset()
    prisma.db.litellm_objectpermissiontable.update_many.assert_not_awaited()
    manager.fetch_unfiltered_inventory.assert_not_awaited()


@pytest.mark.asyncio
async def test_runner_skips_row_with_no_grants():
    row = _row(mcp_servers=[], mcp_tool_permissions=None)
    prisma = _prisma([row])
    manager = _manager()
    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
        AsyncMock(return_value=[]),
    ):
        report = await run_mcp_tool_permission_backfill(prisma, manager)
    assert report.skipped_no_grants == {"perm-1"}
    prisma.db.litellm_objectpermissiontable.update_many.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_granted_server_ids_expands_all_proxy_sentinel():
    row = _row(mcp_servers=["all-proxy-mcpservers"])
    registry = {
        "server-a": MagicMock(server_id="server-a"),
        "server-b": MagicMock(server_id="server-b"),
    }
    manager = _manager(registry=registry)
    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
        AsyncMock(return_value=[]),
    ):
        granted = await resolve_granted_server_ids(row, manager)
    assert granted == frozenset({"server-a", "server-b"})


@pytest.mark.asyncio
async def test_resolve_granted_server_ids_excludes_toolset_only_servers():
    row = _row(mcp_servers=[], mcp_toolsets=["toolset-1"])
    manager = _manager()
    with patch(
        "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler._get_mcp_servers_from_access_groups",
        AsyncMock(return_value=[]),
    ):
        granted = await resolve_granted_server_ids(row, manager)
    assert granted == frozenset()
