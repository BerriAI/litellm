from litellm._logging import verbose_logger
from litellm.proxy._types import UI_TEAM_ID, UserAPIKeyAuth
from litellm.proxy.utils import PrismaClient


async def _get_allowed_skills_for_key(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
) -> frozenset[str]:
    """Key's own allowed_skills ceiling from its object_permission.

    object_permission is normally already loaded onto the auth dict by
    get_key_object() in the main auth flow; fall back to a DB lookup by
    object_permission_id for the rare case it wasn't.
    """
    from litellm.proxy.auth.auth_checks import get_object_permission
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    key_object_permission = user_api_key_dict.object_permission
    if key_object_permission is None and user_api_key_dict.object_permission_id and prisma_client is not None:
        key_object_permission = await get_object_permission(
            object_permission_id=user_api_key_dict.object_permission_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )

    if key_object_permission is None:
        return frozenset()

    return frozenset(key_object_permission.allowed_skills or [])


async def _get_allowed_skills_for_team(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
) -> frozenset[str]:
    """Team's allowed_skills ceiling from team.object_permission."""
    from litellm.proxy.auth.auth_checks import get_team_object
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    if not user_api_key_dict.team_id or prisma_client is None:
        return frozenset()

    if user_api_key_dict.team_id == UI_TEAM_ID:
        return frozenset()

    team_obj = await get_team_object(
        team_id=user_api_key_dict.team_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_dict.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    if team_obj.object_permission is None:
        return frozenset()

    return frozenset(team_obj.object_permission.allowed_skills or [])


async def _get_allowed_skills_for_org(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
) -> frozenset[str]:
    """Org's allowed_skills ceiling from org.object_permission."""
    from litellm.proxy.auth.auth_checks import get_object_permission, get_org_object
    from litellm.proxy.proxy_server import proxy_logging_obj, user_api_key_cache

    if not user_api_key_dict.org_id or prisma_client is None:
        return frozenset()

    org_obj = await get_org_object(
        org_id=user_api_key_dict.org_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_dict.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    if org_obj is None or not org_obj.object_permission_id:
        return frozenset()

    org_object_permission = await get_object_permission(
        object_permission_id=org_obj.object_permission_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        parent_otel_span=user_api_key_dict.parent_otel_span,
        proxy_logging_obj=proxy_logging_obj,
    )
    if org_object_permission is None:
        return frozenset()

    return frozenset(org_object_permission.allowed_skills or [])


async def get_allowed_skills(
    user_api_key_dict: UserAPIKeyAuth,
    prisma_client: PrismaClient | None,
) -> frozenset[str]:
    """
    Resolve the set of Claude Code skill names (already-namespaced, e.g.
    "anthropic-agent-skills--document-skills") this key is allowed to see,
    beyond whatever the default/managed marketplace already exposes.

    Permission hierarchy (all rules are intersections, mirrors
    MCPRequestHandler.get_allowed_mcp_servers for key/team/org):
    - An empty/missing allowed_skills list at a given level means that level
      places no restriction and is skipped from the intersection.
    - key/team: if both restrict, intersect; if only one restricts, use it.
    - org: acts as a ceiling. If the org has an explicit list, the key/team
      result is capped to it; if there's no lower-level restriction, the org
      list becomes the result outright.
    """
    try:
        key_skills = await _get_allowed_skills_for_key(user_api_key_dict, prisma_client)
        team_skills = await _get_allowed_skills_for_team(user_api_key_dict, prisma_client)

        if not team_skills:
            allowed_skills = key_skills
        elif not key_skills:
            allowed_skills = team_skills
        else:
            allowed_skills = key_skills & team_skills

        has_lower_level_restrictions = bool(key_skills or team_skills)

        if user_api_key_dict.org_id:
            org_skills = await _get_allowed_skills_for_org(user_api_key_dict, prisma_client)
            if org_skills:
                allowed_skills = allowed_skills & org_skills if has_lower_level_restrictions else org_skills
    except Exception as e:  # noqa: BLE001  # never let an authz lookup crash the caller
        verbose_logger.warning(f"Failed to get allowed skills: {e!s}")
        return frozenset()
    else:
        return allowed_skills
