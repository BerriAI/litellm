"""
Drop a deleted agent or MCP server from every unified access group that lists it.

`access_agent_ids`, `access_mcp_server_ids` and the keys of `mcp_tool_permissions` hold ids
with no foreign key to the agent or MCP server row, so a delete that leaves them alone
strands the id in every group that granted it, and a resource later created under the same
id would inherit those grants.
"""

from typing import Final

from pydantic import ConfigDict

from litellm.proxy.management_helpers.access_group_model_sync import RawExecutor, raw_executor
from litellm.proxy.management_helpers.access_group_team_sync import invalidate_access_group_caches
from litellm.types.llms.base import LiteLLMBaseModel


class _TouchedGroupRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    access_group_id: str


_REMOVE_AGENT_ID_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_agent_ids" = array_remove("access_agent_ids", $1) '
    'WHERE $1 = ANY("access_agent_ids") '
    'RETURNING "access_group_id"'
)

# `mcp_tool_permissions` is written as a JSON null when a group has none, and `jsonb - text`
# raises on a scalar, so the key is only removed from an object.
_REMOVE_MCP_SERVER_ID_SQL: Final = (
    'UPDATE "LiteLLM_AccessGroupTable" '
    'SET "access_mcp_server_ids" = array_remove("access_mcp_server_ids", $1), '
    '"mcp_tool_permissions" = CASE '
    'WHEN jsonb_typeof("mcp_tool_permissions") = \'object\' THEN "mcp_tool_permissions" - $1::text '
    'ELSE "mcp_tool_permissions" END '
    'WHERE $1 = ANY("access_mcp_server_ids") '
    'OR (jsonb_typeof("mcp_tool_permissions") = \'object\' AND jsonb_exists("mcp_tool_permissions", $1::text)) '
    'RETURNING "access_group_id"'
)


async def _remove_member(executor: RawExecutor, sql: str, member_id: str) -> None:
    touched_rows: Final = await executor.query_raw(sql, member_id)
    await invalidate_access_group_caches(
        tuple(_TouchedGroupRow.model_validate(row).access_group_id for row in touched_rows)
    )


async def sync_access_groups_for_deleted_agent(prisma_client: object, *, agent_id: str) -> None:
    await _remove_member(raw_executor(prisma_client), _REMOVE_AGENT_ID_SQL, agent_id)


async def sync_access_groups_for_deleted_mcp_server(prisma_client: object, *, server_id: str) -> None:
    await _remove_member(raw_executor(prisma_client), _REMOVE_MCP_SERVER_ID_SQL, server_id)
