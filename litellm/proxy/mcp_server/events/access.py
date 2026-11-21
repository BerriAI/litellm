import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._types import UserAPIKeyAuth

EventGrant = frozenset[str] | None


def principal_of(caller: UserAPIKeyAuth | None) -> str | None:
    if caller is None:
        return None
    if caller.api_key:
        return "key:" + hashlib.sha256(caller.api_key.encode()).hexdigest()
    if caller.user_id:
        return "user:" + hashlib.sha256(caller.user_id.encode()).hexdigest()
    return None


async def _caller_permissions(caller: UserAPIKeyAuth) -> Sequence[LiteLLM_ObjectPermissionTable | None]:
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler

    return (
        await MCPRequestHandler.key_object_permission_hydrated(caller),
        await MCPRequestHandler.team_object_permission(caller),
        await MCPRequestHandler._get_user_object_permission(caller),  # pyright: ignore[reportPrivateUsage]  # same user ceiling the tool path reads
        await MCPRequestHandler._get_org_object_permission(caller),  # pyright: ignore[reportPrivateUsage]  # same org ceiling the tool path reads
    )


def _expand(permissions: Mapping[str, list[str]]) -> Mapping[str, list[str]]:
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    return global_mcp_server_manager.expand_tool_permissions(dict(permissions))


def intersect_grants(
    levels: Sequence[LiteLLM_ObjectPermissionTable | None],
    server_id: str,
    expand: Callable[[Mapping[str, list[str]]], Mapping[str, list[str]]] = _expand,
) -> EventGrant:
    restrictions: Final = tuple(
        frozenset(entries)
        for entries in (
            expand(level.mcp_event_permissions).get(server_id)
            for level in levels
            if level is not None and level.mcp_event_permissions
        )
        if entries is not None
    )
    return frozenset.intersection(*restrictions) if restrictions else None


async def event_grant(
    server_id: str,
    caller: UserAPIKeyAuth | None,
    source: Callable[[UserAPIKeyAuth], Awaitable[Sequence[LiteLLM_ObjectPermissionTable | None]]] = _caller_permissions,
) -> EventGrant:
    if caller is None:
        return None
    return intersect_grants(await source(caller), server_id)


def grant_allows(grant: EventGrant, upstream_name: str) -> bool:
    return grant is None or upstream_name in grant
