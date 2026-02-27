"""
DB helpers for LiteLLM_ToolTable — the global tool registry.

Tools are auto-discovered from LLM responses and upserted here.
Admins use the management endpoints to read and update call_policy.
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import ToolDiscoveryQueueItem
from litellm.types.tool_management import (
    LiteLLM_ToolTableRow,
    ToolCallPolicy,
    ToolPolicyOverrideRow,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def _row_to_model(row: Union[dict, Any]) -> LiteLLM_ToolTableRow:
    """Convert a Prisma model instance or dict to LiteLLM_ToolTableRow."""
    model_dump = getattr(row, "model_dump", None)
    if callable(model_dump):
        row = model_dump()
    elif not isinstance(row, dict):
        row = {
            k: getattr(row, k, None)
            for k in (
                "tool_id",
                "tool_name",
                "origin",
                "call_policy",
                "call_count",
                "assignments",
                "key_hash",
                "team_id",
                "key_alias",
                "created_at",
                "updated_at",
                "created_by",
                "updated_by",
            )
        }
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
    Batch-upsert tool registry rows via Prisma.

    On first insert: sets call_policy = "untrusted" (schema default), call_count = 1.
    On conflict: increments call_count; preserves existing call_policy.
    """
    if not items:
        return
    try:
        data = [item for item in items if item.get("tool_name")]
        if not data:
            return
        now = datetime.now(timezone.utc)
        table = prisma_client.db.litellm_tooltable
        for item in data:
            tool_name = item.get("tool_name", "")
            origin = item.get("origin") or "user_defined"
            created_by = item.get("created_by") or "system"
            key_hash = item.get("key_hash")
            team_id = item.get("team_id")
            key_alias = item.get("key_alias")
            await table.upsert(
                where={"tool_name": tool_name},
                data={
                    "create": {
                        "tool_id": str(uuid.uuid4()),
                        "tool_name": tool_name,
                        "origin": origin,
                        "call_policy": "untrusted",
                        "call_count": 1,
                        "created_by": created_by,
                        "updated_by": created_by,
                        "key_hash": key_hash,
                        "team_id": team_id,
                        "key_alias": key_alias,
                    },
                    "update": {
                        "call_count": {"increment": 1},
                        "updated_at": now,
                    },
                },
            )
        verbose_proxy_logger.debug(
            "tool_registry_writer: upserted %d tool(s)", len(data)
        )
    except Exception as e:
        verbose_proxy_logger.error(
            "tool_registry_writer batch_upsert_tools error: %s", e
        )


async def list_tools(
    prisma_client: "PrismaClient",
    call_policy: Optional[ToolCallPolicy] = None,
) -> List[LiteLLM_ToolTableRow]:
    """Return all tools, optionally filtered by call_policy."""
    try:
        where = {"call_policy": call_policy} if call_policy is not None else {}
        rows = await prisma_client.db.litellm_tooltable.find_many(
            where=where,
            order={"created_at": "desc"},
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
        row = await prisma_client.db.litellm_tooltable.find_unique(
            where={"tool_name": tool_name},
        )
        if row is None:
            return None
        return _row_to_model(row)
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
        now = datetime.now(timezone.utc)
        await prisma_client.db.litellm_tooltable.upsert(
            where={"tool_name": tool_name},
            data={
                "create": {
                    "tool_id": str(uuid.uuid4()),
                    "tool_name": tool_name,
                    "call_policy": call_policy,
                    "created_by": _updated_by,
                    "updated_by": _updated_by,
                    "created_at": now,
                    "updated_at": now,
                },
                "update": {
                    "call_policy": call_policy,
                    "updated_by": _updated_by,
                    "updated_at": now,
                },
            },
        )
        return await get_tool(prisma_client, tool_name)
    except Exception as e:
        verbose_proxy_logger.error(
            "tool_registry_writer update_tool_policy error: %s", e
        )
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
        rows = await prisma_client.db.litellm_tooltable.find_many(
            where={"tool_name": {"in": tool_names}},
        )
        return {row.tool_name: row.call_policy for row in rows}
    except Exception as e:
        verbose_proxy_logger.error(
            "tool_registry_writer get_tools_by_names error: %s", e
        )
        return {}


async def list_overrides_for_tool(
    prisma_client: "PrismaClient",
    tool_name: str,
) -> List[ToolPolicyOverrideRow]:
    """
    Return override-like rows for a tool by finding object permissions that have
    this tool in blocked_tools, then resolving each permission to key/team scope for display.
    """
    out: List[ToolPolicyOverrideRow] = []
    try:
        perms = await prisma_client.db.litellm_objectpermissiontable.find_many(
            where={"blocked_tools": {"has": tool_name}},
            include={
                "verification_tokens": True,
                "teams": True,
            },
        )
        for perm in perms:
            op_id = getattr(perm, "object_permission_id", None) or ""
            tokens = getattr(perm, "verification_tokens", []) or []
            teams = getattr(perm, "teams", []) or []
            for t in tokens:
                out.append(
                    ToolPolicyOverrideRow(
                        override_id=op_id,
                        tool_name=tool_name,
                        team_id=None,
                        key_hash=getattr(t, "token", None),
                        call_policy="blocked",
                        key_alias=getattr(t, "key_alias", None),
                        created_at=None,
                        updated_at=None,
                    )
                )
            for team in teams:
                out.append(
                    ToolPolicyOverrideRow(
                        override_id=op_id,
                        tool_name=tool_name,
                        team_id=getattr(team, "team_id", None),
                        key_hash=None,
                        call_policy="blocked",
                        key_alias=getattr(team, "team_alias", None),
                        created_at=None,
                        updated_at=None,
                    )
                )
        return out
    except Exception as e:
        verbose_proxy_logger.error(
            "tool_registry_writer list_overrides_for_tool error: %s", e
        )
        return []


class ToolPolicyRegistry:
    """
    In-memory registry of tool policies synced from DB.
    Synced in _init_tool_policy_in_db (from add_deployment / _init_non_llm_objects_in_db).
    Hot path uses get_effective_policies only — no DB, no cache.
    """

    def __init__(self) -> None:
        self._global_tool_policies: Dict[str, str] = {}
        self._blocked_tools_by_op_id: Dict[str, List[str]] = {}
        self._initialized: bool = False

    def is_initialized(self) -> bool:
        return self._initialized

    async def sync_tool_policy_from_db(self, prisma_client: "PrismaClient") -> None:
        """Load all tool policies and object-permission blocked_tools from DB; replace in-memory state."""
        try:
            tools = await prisma_client.db.litellm_tooltable.find_many()
            self._global_tool_policies = {row.tool_name: row.call_policy for row in tools}

            perms = await prisma_client.db.litellm_objectpermissiontable.find_many()
            self._blocked_tools_by_op_id = {}
            for row in perms:
                op_id = getattr(row, "object_permission_id", None)
                blocked = getattr(row, "blocked_tools", None) or []
                if op_id:
                    self._blocked_tools_by_op_id[op_id] = list(blocked)

            self._initialized = True
            verbose_proxy_logger.info(
                "ToolPolicyRegistry: synced %d global tool policies and %d object permissions from DB",
                len(self._global_tool_policies),
                len(self._blocked_tools_by_op_id),
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                "ToolPolicyRegistry sync_tool_policy_from_db error: %s", e
            )
            raise

    def get_effective_policies(
        self,
        tool_names: List[str],
        object_permission_id: Optional[str] = None,
        team_object_permission_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """
        Return effective call_policy per tool from in-memory state.
        If tool is in key or team blocked_tools -> "blocked", else global policy or "untrusted".
        """
        if not tool_names:
            return {}
        blocked: set = set()
        for op_id in (object_permission_id, team_object_permission_id):
            if op_id and op_id.strip():
                blocked.update(
                    self._blocked_tools_by_op_id.get(op_id.strip(), [])
                )
        result: Dict[str, str] = {}
        for name in tool_names:
            if name in blocked:
                result[name] = "blocked"
            else:
                result[name] = self._global_tool_policies.get(name, "untrusted")
        return result


_tool_policy_registry: Optional[ToolPolicyRegistry] = None


def get_tool_policy_registry() -> ToolPolicyRegistry:
    """Return the global ToolPolicyRegistry singleton."""
    global _tool_policy_registry
    if _tool_policy_registry is None:
        _tool_policy_registry = ToolPolicyRegistry()
    return _tool_policy_registry


async def add_tool_to_object_permission_blocked(
    prisma_client: "PrismaClient",
    object_permission_id: str,
    tool_name: str,
) -> bool:
    """Add tool_name to the permission's blocked_tools if not already present."""
    if not object_permission_id or not tool_name:
        return False
    try:
        row = await prisma_client.db.litellm_objectpermissiontable.find_unique(
            where={"object_permission_id": object_permission_id},
        )
        if row is None:
            return False
        current = list(getattr(row, "blocked_tools", []) or [])
        if tool_name in current:
            return True
        current.append(tool_name)
        await prisma_client.db.litellm_objectpermissiontable.update(
            where={"object_permission_id": object_permission_id},
            data={"blocked_tools": current},
        )
        return True
    except Exception as e:
        verbose_proxy_logger.error(
            "tool_registry_writer add_tool_to_object_permission_blocked error: %s", e
        )
        return False


async def remove_tool_from_object_permission_blocked(
    prisma_client: "PrismaClient",
    object_permission_id: str,
    tool_name: str,
) -> bool:
    """Remove tool_name from the permission's blocked_tools. Returns False if tool was not in list."""
    if not object_permission_id or not tool_name:
        return False
    try:
        row = await prisma_client.db.litellm_objectpermissiontable.find_unique(
            where={"object_permission_id": object_permission_id},
        )
        if row is None:
            return False
        current = list(getattr(row, "blocked_tools", []) or [])
        if tool_name not in current:
            return False
        current = [t for t in current if t != tool_name]
        await prisma_client.db.litellm_objectpermissiontable.update(
            where={"object_permission_id": object_permission_id},
            data={"blocked_tools": current},
        )
        return True
    except Exception as e:
        verbose_proxy_logger.error(
            "tool_registry_writer remove_tool_from_object_permission_blocked error: %s",
            e,
        )
        return False
