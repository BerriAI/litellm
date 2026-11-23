import fnmatch
import hashlib
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm.constants import LITELLM_PROXY_MASTER_KEY_ALIAS
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

SourceGrant: TypeAlias = tuple[frozenset[str], ...]


@dataclass(frozen=True, slots=True)
class AnyOf:
    """A caller reaching a server through several grant sources may use an event any one of them allows."""

    sources: tuple[SourceGrant, ...]


EventGrant: TypeAlias = SourceGrant | AnyOf | None


_TOKEN_HASH: Final = re.compile(r"[0-9a-f]{64}")
_JWT_TOKEN_PREFIX: Final = "hashed-jwt-"


def _master_key_hash() -> str | None:
    from litellm.proxy.proxy_server import master_key
    from litellm.proxy.utils import hash_token

    return hash_token(master_key) if isinstance(master_key, str) else None


def stored_key_hash(
    caller: UserAPIKeyAuth | None, master_key_hash: Callable[[], str | None] = _master_key_hash
) -> str | None:
    token: Final = caller.token if caller is not None and caller.via_virtual_key else None
    if token is None:
        return None
    if (
        token == LITELLM_PROXY_MASTER_KEY_ALIAS
        and caller is not None
        and caller.user_role == LitellmUserRoles.PROXY_ADMIN
    ):
        return master_key_hash()
    return token if _TOKEN_HASH.fullmatch(token) else None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def principal_of(caller: UserAPIKeyAuth | None) -> str | None:
    if caller is None:
        return None
    key: Final = stored_key_hash(caller)
    if key is not None:
        return "key:" + key
    if caller.user_id:
        return "user:" + _digest(caller.user_id)
    if caller.team_id:
        return "team:" + _digest(caller.team_id)
    token: Final = caller.token
    return "token:" + _digest(token) if token and not token.startswith(_JWT_TOKEN_PREFIX) else None


def holds_webhooks(caller: UserAPIKeyAuth | None) -> bool:
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import is_mcp_admitted_user_subject

    if caller is None or caller.end_user_id or caller.agent_id:
        return False
    return stored_key_hash(caller) is not None or (is_mcp_admitted_user_subject(caller) and bool(caller.user_id))


async def _caller_permissions(caller: UserAPIKeyAuth) -> Sequence[LiteLLM_ObjectPermissionTable | None]:
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
    from litellm.proxy.proxy_server import prisma_client

    return (
        await MCPRequestHandler.key_object_permission_hydrated(caller),
        await MCPRequestHandler.team_object_permission(caller),
        await MCPRequestHandler._get_user_object_permission(caller),  # pyright: ignore[reportPrivateUsage]  # same user ceiling the tool path reads
        await MCPRequestHandler._get_org_object_permission(caller),  # pyright: ignore[reportPrivateUsage]  # same org ceiling the tool path reads
        await MCPRequestHandler._get_end_user_object_permission(caller, prisma_client)  # pyright: ignore[reportPrivateUsage]  # same end-user ceiling the tool path reads
        if prisma_client is not None
        else None,
    )


async def _granting_sources(caller: UserAPIKeyAuth, server_id: str) -> Sequence[UserAPIKeyAuth] | None:
    """The sources that grant ``server_id`` to a caller pinned to no single team, as the tool path resolves them;
    None for a single credential, whose own levels decide."""
    from litellm.proxy.auth.auth_checks import personal_key_inherits_team_grants
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler, is_mcp_admitted_user_subject
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    if is_mcp_admitted_user_subject(caller):
        open_channel: Final = server_id in await global_mcp_server_manager.operator_open_server_ids(caller)
        return tuple(
            source
            for source, granted in await MCPRequestHandler.admitted_source_grants(caller)
            if server_id in granted or (open_channel and source.team_id is None)
        )
    if personal_key_inherits_team_grants(caller):
        granting: Final = tuple(
            source
            for source, access in await MCPRequestHandler.personal_key_team_server_grants(caller)
            if server_id in access.server_ids
        )
        return granting or (caller,)
    return None


def _expand(permissions: Mapping[str, Sequence[str]]) -> Mapping[str, Sequence[str]]:
    from litellm.proxy.mcp_server.mcp_server_manager import global_mcp_server_manager

    return global_mcp_server_manager.expand_tool_permissions({key: list(value) for key, value in permissions.items()})


def intersect_grants(
    levels: Sequence[LiteLLM_ObjectPermissionTable | None],
    server_id: str,
    expand: Callable[[Mapping[str, Sequence[str]]], Mapping[str, Sequence[str]]] = _expand,
) -> SourceGrant | None:
    restrictions: Final[tuple[frozenset[str], ...]] = tuple(
        frozenset(entries)
        for entries in (
            expand(level.mcp_event_permissions).get(server_id)
            for level in levels
            if level is not None and level.mcp_event_permissions
        )
        if entries is not None
    )
    return restrictions or None


async def event_grant(
    server_id: str,
    caller: UserAPIKeyAuth | None,
    source: Callable[[UserAPIKeyAuth], Awaitable[Sequence[LiteLLM_ObjectPermissionTable | None]]] = _caller_permissions,
    granting: Callable[[UserAPIKeyAuth, str], Awaitable[Sequence[UserAPIKeyAuth] | None]] = _granting_sources,
) -> EventGrant:
    """A single credential's levels intersect; a multi-source caller may use what any granting source allows,
    and a granting source with no event rule allows every event."""
    if caller is None:
        return None
    sources: Final = await granting(caller, server_id)
    if sources is None:
        return intersect_grants(await source(caller), server_id)
    per_source: Final = [intersect_grants(await source(item), server_id) for item in sources]
    if any(grant is None for grant in per_source):
        return None
    return AnyOf(tuple(grant for grant in per_source if isinstance(grant, tuple)))


def grant_allows(grant: EventGrant, upstream_name: str) -> bool:
    """Every restricting level must list the event, by exact name or glob like a server's allowed_events."""
    if grant is None:
        return True
    if isinstance(grant, AnyOf):
        return any(_source_allows(source, upstream_name) for source in grant.sources)
    return _source_allows(grant, upstream_name)


def _source_allows(levels: SourceGrant, upstream_name: str) -> bool:
    return all(any(fnmatch.fnmatchcase(upstream_name, pattern) for pattern in level) for level in levels)
