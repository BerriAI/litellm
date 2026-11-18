from typing import Final

from litellm.proxy._types import UI_TEAM_ID, LiteLLM_ObjectPermissionTable, LiteLLM_TeamTableCachedObj, UserAPIKeyAuth


def grants_in_precedence(
    agent: LiteLLM_ObjectPermissionTable | None,
    key: LiteLLM_ObjectPermissionTable | None,
    team: LiteLLM_ObjectPermissionTable | None,
) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(carrier.mcp_auth_profiles or ()) for carrier in (agent, key, team) if carrier is not None)


async def caller_auth_profile_grants(user_api_key_auth: UserAPIKeyAuth | None) -> tuple[tuple[str, ...], ...]:
    if user_api_key_auth is None:
        return ()
    context: Final = user_api_key_auth.managed_agent_context
    if context is not None and context.mode == "delegated":
        return ()
    from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler, is_mcp_admitted_user_subject

    admitted_subject: Final = is_mcp_admitted_user_subject(user_api_key_auth)
    single_team: Final = bool(user_api_key_auth.team_id) and not admitted_subject
    return grants_in_precedence(
        agent=await MCPRequestHandler.agent_object_permission(user_api_key_auth) if context is not None else None,
        key=None if admitted_subject else await _key_object_permission(user_api_key_auth),
        team=await _team_object_permission(user_api_key_auth) if single_team else None,
    )


async def _team_object_permission(user_api_key_auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
    team: Final = await _caller_team(user_api_key_auth)
    if team is None or team.object_permission is not None or not team.object_permission_id:
        return None if team is None else team.object_permission
    return await _read_object_permission(team.object_permission_id, user_api_key_auth)


async def _caller_team(user_api_key_auth: UserAPIKeyAuth) -> LiteLLM_TeamTableCachedObj | None:
    from litellm.proxy.auth.auth_checks import get_team_object, get_team_object_from_cache
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    team_id: Final = user_api_key_auth.team_id
    if not team_id or team_id == UI_TEAM_ID or prisma_client is None:
        return None
    fresh: Final = user_api_key_auth.requires_fresh_policy
    cached: Final = (
        await get_team_object_from_cache(f"team_id:{team_id}", user_api_key_cache, user_api_key_auth.parent_otel_span)
        if fresh
        else None
    )
    return cached or await get_team_object(
        team_id=team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_auth.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
        check_db_only=fresh,
    )


async def _key_object_permission(user_api_key_auth: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
    loaded: Final = user_api_key_auth.object_permission
    if not user_api_key_auth.object_permission_id or (
        loaded is not None and loaded.object_permission_id == user_api_key_auth.object_permission_id
    ):
        return loaded
    return await _read_object_permission(user_api_key_auth.object_permission_id, user_api_key_auth)


async def _read_object_permission(
    object_permission_id: str, user_api_key_auth: UserAPIKeyAuth
) -> LiteLLM_ObjectPermissionTable | None:
    from litellm.proxy.auth.auth_checks import get_object_permission
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        return None
    return await get_object_permission(
        object_permission_id=object_permission_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_auth.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
        check_db_only=True,
    )
