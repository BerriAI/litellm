"""
DB helpers for LiteLLM_ToolTable — the global tool registry.

Tools are auto-discovered from LLM responses and upserted here.
Admins use the management endpoints to read and update call_policy.
"""

import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from litellm._logging import verbose_proxy_logger
from litellm.caching.dual_cache import DualCache
from litellm.constants import TOOL_POLICY_CACHE_TTL_SECONDS
from litellm.proxy._types import ToolDiscoveryQueueItem
from litellm.types.tool_management import (LiteLLM_ToolTableRow,
                                           ToolCallPolicy,
                                           ToolPolicyOverrideRow)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

TOOL_POLICY_CACHE_KEY_PREFIX = "tool_policy:"


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


async def _get_merged_blocked_tools(
    prisma_client: "PrismaClient",
    object_permission_id: Optional[str],
    team_object_permission_id: Optional[str],
) -> set:
    """Return union of blocked_tools from key and team object permissions."""
    blocked: set = set()
    for op_id in (object_permission_id, team_object_permission_id):
        if not op_id or not op_id.strip():
            continue
        try:
            row = await prisma_client.db.litellm_objectpermissiontable.find_unique(
                where={"object_permission_id": op_id.strip()},
                select={"blocked_tools": True},
            )
            if row is not None and getattr(row, "blocked_tools", None):
                blocked.update(row.blocked_tools)
        except Exception as e:
            verbose_proxy_logger.debug(
                "tool_registry_writer _get_merged_blocked_tools error for %s: %s",
                op_id,
                e,
            )
    return blocked


async def get_effective_policies(
    prisma_client: "PrismaClient",
    tool_names: List[str],
    object_permission_id: Optional[str] = None,
    team_object_permission_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Return effective call_policy per tool: if tool is in key/team object permission
    blocked_tools then "blocked", otherwise global policy from LiteLLM_ToolTable.
    """
    if not tool_names:
        return {}
    try:
        blocked = await _get_merged_blocked_tools(
            prisma_client=prisma_client,
            object_permission_id=object_permission_id,
            team_object_permission_id=team_object_permission_id,
        )
        result: Dict[str, str] = {}
        for name in tool_names:
            if name in blocked:
                result[name] = "blocked"
        missing = [n for n in tool_names if n not in result]
        if missing:
            global_map = await get_tools_by_names(
                prisma_client=prisma_client, tool_names=missing
            )
            result.update(global_map)
        return result
    except Exception as e:
        verbose_proxy_logger.error(
            "tool_registry_writer get_effective_policies error: %s", e
        )
        return {}


def _effective_cache_suffix(
    object_permission_id: Optional[str],
    team_object_permission_id: Optional[str],
) -> str:
    """Cache key suffix so different request contexts get correct policies."""
    return f":{object_permission_id or ''}:{team_object_permission_id or ''}"


async def get_tool_policies_cached(
    tool_names: List[str],
    cache: DualCache,
    prisma_client: Optional["PrismaClient"],
    object_permission_id: Optional[str] = None,
    team_object_permission_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Return effective call_policy per tool (blocked if in object permission blocked_tools,
    else global). Cache-first; cache key includes object_permission_id(s).
    """
    if not tool_names:
        return {}
    suffix = _effective_cache_suffix(object_permission_id, team_object_permission_id)
    result: Dict[str, str] = {}
    cache_misses: List[str] = []
    for name in tool_names:
        key = f"{TOOL_POLICY_CACHE_KEY_PREFIX}{name}{suffix}"
        cached = await cache.async_get_cache(key=key)
        if cached is not None and isinstance(cached, str):
            result[name] = cached
        else:
            cache_misses.append(name)
    if cache_misses and prisma_client is not None:
        try:
            if object_permission_id or team_object_permission_id:
                fetched = await get_effective_policies(
                    prisma_client=prisma_client,
                    tool_names=cache_misses,
                    object_permission_id=object_permission_id,
                    team_object_permission_id=team_object_permission_id,
                )
            else:
                fetched = await get_tools_by_names(
                    prisma_client=prisma_client, tool_names=cache_misses
                )
            for name, policy in fetched.items():
                result[name] = policy
                await cache.async_set_cache(
                    key=f"{TOOL_POLICY_CACHE_KEY_PREFIX}{name}{suffix}",
                    value=policy,
                    ttl=TOOL_POLICY_CACHE_TTL_SECONDS,
                )
            verbose_proxy_logger.debug(
                "get_tool_policies_cached: fetched %d from DB (hits: %d)",
                len(cache_misses),
                len(tool_names) - len(cache_misses),
            )
        except Exception as e:
            verbose_proxy_logger.error(
                "tool_registry_writer get_tool_policies_cached error: %s", e
            )
    return result


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
            select={"blocked_tools": True},
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
            select={"blocked_tools": True},
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
