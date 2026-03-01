"""
DB helpers for LiteLLM_ToolTable — the global tool registry.

Tools are auto-discovered from LLM responses and upserted here.
Admins use the management endpoints to read and update call_policy.

NOTE: Uses raw SQL (query_raw / execute_raw) instead of Prisma model methods
because the generated Prisma Python client may not have LiteLLM_ToolTable
when running against an older generated schema.
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Dict, List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ToolDiscoveryQueueItem
from litellm.types.tool_management import LiteLLM_ToolTableRow, ToolCallPolicy

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def _row_to_model(row: dict) -> LiteLLM_ToolTableRow:
    return LiteLLM_ToolTableRow(
        tool_id=row.get("tool_id", ""),
        tool_name=row.get("tool_name", ""),
        origin=row.get("origin"),
        call_policy=row.get("call_policy", "untrusted"),
        call_count=int(row.get("call_count") or 0),
        assignments=row.get("assignments"),
        key_hash=row.get("key_hash"),
        team_id=row.get("team_id"),
        key_alias=row.get("key_alias"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        created_by=row.get("created_by"),
        updated_by=row.get("updated_by"),
    )


async def batch_upsert_tools(
    prisma_client: "PrismaClient",
    items: List[ToolDiscoveryQueueItem],
) -> None:
    """
    Batch-upsert tool registry rows via raw SQL.

    On first insert: sets call_policy = "untrusted" (schema default), call_count = 1.
    On conflict: increments call_count; preserves existing call_policy.
    """
    if not items:
        return
    try:
        data = [item for item in items if item.get("tool_name")]
        if not data:
            return
        for item in data:
            tool_name = item.get("tool_name", "")
            origin = item.get("origin") or "user_defined"
            created_by = item.get("created_by") or "system"
            key_hash = item.get("key_hash")
            team_id = item.get("team_id")
            key_alias = item.get("key_alias")
            now = datetime.now(timezone.utc).isoformat()
            await prisma_client.db.execute_raw(
                'INSERT INTO "LiteLLM_ToolTable" '
                "(tool_id, tool_name, origin, call_policy, call_count, created_by, updated_by, key_hash, team_id, key_alias, created_at, updated_at) "
                "VALUES ($7, $1, $2, 'untrusted', 1, $3, $3, $4, $5, $6, $8, $8) "
                "ON CONFLICT (tool_name) DO UPDATE SET "
                "call_count = \"LiteLLM_ToolTable\".call_count + 1, "
                "updated_at = $8",
                tool_name,
                origin,
                created_by,
                key_hash,
                team_id,
                key_alias,
                str(uuid.uuid4()),
                now,
            )
        verbose_proxy_logger.debug(
            "tool_registry_writer: upserted %d tool(s)", len(data)
        )
    except Exception as e:
        verbose_proxy_logger.error("tool_registry_writer batch_upsert_tools error: %s", e)


async def list_tools(
    prisma_client: "PrismaClient",
    call_policy: Optional[ToolCallPolicy] = None,
) -> List[LiteLLM_ToolTableRow]:
    """Return all tools, optionally filtered by call_policy."""
    try:
        if call_policy is not None:
            rows = await prisma_client.db.query_raw(
                'SELECT tool_id, tool_name, origin, call_policy, call_count, assignments, '
                'key_hash, team_id, key_alias, created_at, updated_at, created_by, updated_by '
                'FROM "LiteLLM_ToolTable" WHERE call_policy = $1 ORDER BY created_at DESC',
                call_policy,
            )
        else:
            rows = await prisma_client.db.query_raw(
                'SELECT tool_id, tool_name, origin, call_policy, call_count, assignments, '
                'key_hash, team_id, key_alias, created_at, updated_at, created_by, updated_by '
                'FROM "LiteLLM_ToolTable" ORDER BY created_at DESC',
            )
        return [_row_to_model(row) for row in rows]
    except Exception as e:
        verbose_proxy_logger.error("tool_registry_writer list_tools error: %s", e)
        return []


async def get_tool(
    prisma_client: "PrismaClient",
    tool_name: str,
) -> Optional[LiteLLM_ToolTableRow]:
    """Return a single tool row by tool_name."""
    try:
        rows = await prisma_client.db.query_raw(
            'SELECT tool_id, tool_name, origin, call_policy, call_count, assignments, '
            'key_hash, team_id, key_alias, created_at, updated_at, created_by, updated_by '
            'FROM "LiteLLM_ToolTable" WHERE tool_name = $1',
            tool_name,
        )
        if not rows:
            return None
        return _row_to_model(rows[0])
    except Exception as e:
        verbose_proxy_logger.error("tool_registry_writer get_tool error: %s", e)
        return None


async def update_tool_policy(
    prisma_client: "PrismaClient",
    tool_name: str,
    call_policy: ToolCallPolicy,
    updated_by: Optional[str],
) -> Optional[LiteLLM_ToolTableRow]:
    """Update the call_policy for a tool. Upserts the row if it does not exist yet."""
    try:
        _updated_by = updated_by or "system"
        now = datetime.now(timezone.utc).isoformat()
        await prisma_client.db.execute_raw(
            'INSERT INTO "LiteLLM_ToolTable" (tool_id, tool_name, call_policy, created_by, updated_by, created_at, updated_at) '
            "VALUES ($4, $1, $2, $3, $3, $5, $5) "
            "ON CONFLICT (tool_name) DO UPDATE SET call_policy = $2, updated_by = $3, updated_at = $5",
            tool_name,
            call_policy,
            _updated_by,
            str(uuid.uuid4()),
            now,
        )
        return await get_tool(prisma_client, tool_name)
    except Exception as e:
        verbose_proxy_logger.error("tool_registry_writer update_tool_policy error: %s", e)
        return None


async def get_tools_by_names(
    prisma_client: "PrismaClient",
    tool_names: List[str],
) -> Dict[str, str]:
    """
    Return a {tool_name: call_policy} map for the given tool names.
    Used by the policy enforcement guardrail — single batch query, never N+1.
    """
    if not tool_names:
        return {}
    try:
        placeholders = ", ".join(f"${i+1}" for i in range(len(tool_names)))
        rows = await prisma_client.db.query_raw(
            f'SELECT tool_name, call_policy FROM "LiteLLM_ToolTable" WHERE tool_name IN ({placeholders})',
            *tool_names,
        )
        return {row["tool_name"]: row["call_policy"] for row in rows}
    except Exception as e:
        verbose_proxy_logger.error("tool_registry_writer get_tools_by_names error: %s", e)
        return {}
