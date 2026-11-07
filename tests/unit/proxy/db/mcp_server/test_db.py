from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy.mcp_server.db import (
    approve_mcp_server,
    get_disabled_config_server_ids,
    get_mcp_servers_by_team,
    get_runtime_mcp_server_rows,
    reject_mcp_server,
    set_config_mcp_server_override,
    set_mcp_server_enabled,
)


def _prisma_client_returning(team_record: object) -> MagicMock:
    prisma_client = MagicMock()
    prisma_client.db.litellm_teamtable.find_unique = AsyncMock(return_value=team_record)
    return prisma_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "team_record, expected",
    [
        (None, []),
        (SimpleNamespace(object_permission=None), []),
        (SimpleNamespace(object_permission=SimpleNamespace(mcp_servers=None)), []),
        (SimpleNamespace(object_permission=SimpleNamespace(mcp_servers=[])), []),
        (
            SimpleNamespace(
                object_permission=SimpleNamespace(mcp_servers=["server_a", "server_b"])
            ),
            ["server_a", "server_b"],
        ),
    ],
)
async def test_fetch_mcp_servers_by_team(team_record, expected):
    prisma_client = _prisma_client_returning(team_record)

    assert await get_mcp_servers_by_team(prisma_client, "team-123") == expected

    prisma_client.db.litellm_teamtable.find_unique.assert_awaited_once_with(
        where={"team_id": "team-123"},
        include={"object_permission": True},
    )


def _prisma_client_with_missing_mcp_server_row() -> MagicMock:
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.update = AsyncMock(return_value=None)
    return prisma_client


@pytest.mark.asyncio
async def test_approve_mcp_server_raises_value_error_when_row_missing():
    prisma_client = _prisma_client_with_missing_mcp_server_row()

    with pytest.raises(ValueError, match=r"^MCP server not found, passed server_id=server-gone$"):
        await approve_mcp_server(prisma_client, "server-gone", touched_by="admin")


@pytest.mark.asyncio
async def test_reject_mcp_server_raises_value_error_when_row_missing():
    prisma_client = _prisma_client_with_missing_mcp_server_row()

    with pytest.raises(ValueError, match=r"^MCP server not found, passed server_id=server-gone$"):
        await reject_mcp_server(
            prisma_client,
            "server-gone",
            touched_by="admin",
            review_notes="spam",
        )


def _prisma_client_for_runtime_rows() -> MagicMock:
    prisma_client = MagicMock()
    prisma_client.writer_db.litellm_mcpservertable.find_many = AsyncMock(return_value=[])
    return prisma_client


@pytest.mark.asyncio
async def test_get_runtime_rows_excludes_disabled():
    prisma_client = _prisma_client_for_runtime_rows()

    await get_runtime_mcp_server_rows(prisma_client)

    where = prisma_client.writer_db.litellm_mcpservertable.find_many.call_args.kwargs["where"]
    assert where == {
        "AND": [
            {"OR": [{"approval_status": None}, {"approval_status": {"in": ["active", "approved"]}}]},
            {"enabled": True},
        ]
    }


@pytest.mark.asyncio
async def test_get_runtime_rows_enabled_gate_is_valid_non_null_filter():
    # ``enabled`` is a NOT NULL column, so the gate must be a plain ``{"enabled": True}``.
    # A ``{"enabled": None}`` arm is invalid Prisma (null filter on a non-nullable field)
    # and raised MissingRequiredValueError at runtime against a real client.
    prisma_client = _prisma_client_for_runtime_rows()

    await get_runtime_mcp_server_rows(prisma_client)

    where = prisma_client.writer_db.litellm_mcpservertable.find_many.call_args.kwargs["where"]
    enabled_gate = where["AND"][1]
    assert enabled_gate == {"enabled": True}
    assert None not in enabled_gate.values()


@pytest.mark.asyncio
async def test_set_mcp_server_enabled_roundtrip():
    prisma_client = MagicMock()
    existing = SimpleNamespace()
    row = MagicMock()
    row.model_dump.return_value = {"server_id": "s1", "transport": "http", "enabled": False}
    prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=existing)
    prisma_client.db.litellm_mcpservertable.update = AsyncMock(return_value=row)

    result = await set_mcp_server_enabled(prisma_client, "s1", False, touched_by="admin")

    assert result is not None
    assert result.server_id == "s1"
    assert result.enabled is False
    kwargs = prisma_client.db.litellm_mcpservertable.update.call_args.kwargs
    assert kwargs["where"] == {"server_id": "s1"}
    assert kwargs["data"] == {"enabled": False, "updated_by": "admin"}


@pytest.mark.asyncio
async def test_set_mcp_server_enabled_returns_none_when_row_missing():
    prisma_client = MagicMock()
    prisma_client.db.litellm_mcpservertable.find_unique = AsyncMock(return_value=None)
    prisma_client.db.litellm_mcpservertable.update = AsyncMock(return_value=None)

    assert await set_mcp_server_enabled(prisma_client, "gone", True, touched_by="admin") is None
    prisma_client.db.litellm_mcpservertable.update.assert_not_awaited()


@pytest.mark.asyncio
async def test_config_override_upsert_and_query():
    prisma_client = MagicMock()
    table = prisma_client.writer_db.litellm_mcpserverstateoverride
    table.upsert = AsyncMock()
    table.find_many = AsyncMock(return_value=[SimpleNamespace(server_id="a"), SimpleNamespace(server_id="b")])

    await set_config_mcp_server_override(prisma_client, "a", False, touched_by="admin")

    table.upsert.assert_awaited_once_with(
        where={"server_id": "a"},
        data={
            "create": {"server_id": "a", "enabled": False, "updated_by": "admin"},
            "update": {"enabled": False, "updated_by": "admin"},
        },
    )

    assert await get_disabled_config_server_ids(prisma_client) == {"a", "b"}
    table.find_many.assert_awaited_once_with(where={"enabled": False})
