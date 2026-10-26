"""
Drop a deleted agent or MCP server from every unified access group that still lists it.

Access groups store agent and MCP server ids in plain arrays with no foreign key, so a delete that
leaves them alone keeps the stale id in every group forever.
"""

from typing import Final

from litellm.proxy.management_helpers.access_group_model_sync import raw_executor
from litellm.proxy.management_helpers.access_group_team_sync import invalidate_access_group_caches
from litellm.types.llms.base import LiteLLMBaseModel


class _TouchedGroupRow(LiteLLMBaseModel):
    access_group_id: str


_REMOVE_AGENT_ID_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_agent_ids" = array_remove("access_agent_ids", $1) '
    'WHERE $1 = ANY("access_agent_ids") '
    'RETURNING "access_group_id"'
)

_REMOVE_MCP_SERVER_ID_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_mcp_server_ids" = array_remove("access_mcp_server_ids", $1), '
    '"mcp_tool_permissions" = "mcp_tool_permissions" - $1 '
    'WHERE $1 = ANY("access_mcp_server_ids") OR jsonb_exists("mcp_tool_permissions", $1) '
    'RETURNING "access_group_id"'
)


async def _remove_from_groups(prisma_client: object, sql: str, resource_id: str) -> None:
    touched_rows: Final = await raw_executor(prisma_client).query_raw(sql, resource_id)
    await invalidate_access_group_caches(
        tuple(_TouchedGroupRow.model_validate(row).access_group_id for row in touched_rows)
    )


async def remove_deleted_agent_from_access_groups(prisma_client: object, agent_id: str) -> None:
    await _remove_from_groups(prisma_client, _REMOVE_AGENT_ID_SQL, agent_id)


async def remove_deleted_mcp_server_from_access_groups(prisma_client: object, server_id: str) -> None:
    await _remove_from_groups(prisma_client, _REMOVE_MCP_SERVER_ID_SQL, server_id)
