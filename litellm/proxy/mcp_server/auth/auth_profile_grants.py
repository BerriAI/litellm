from types import MappingProxyType
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable, StoredProfileGrantError
from litellm.proxy._types import UI_TEAM_ID, LiteLLM_TeamTable, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.managed_authorization import managed_agent_policy
from litellm.proxy.auth.auth_checks import get_object_permission
from litellm.proxy.auth.jwt import ClaimReading
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup
from litellm.proxy.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler, is_mcp_admitted_user_subject
from litellm.repositories.table_repositories import AgentsRepository
from litellm.types.mcp_server.auth_profile_grants import ProfileGrants, UnreadableGrants


def _is_unreadable(link: str | None, permission: LiteLLM_ObjectPermissionTable | None) -> bool:
    return bool(link) and (permission is None or link != permission.object_permission_id)


async def _agent_permission(caller: UserAPIKeyAuth) -> LiteLLM_ObjectPermissionTable | None:
    if caller.agent_id is None:
        return None
    managed: Final = managed_agent_policy(caller)
    if managed is not None:
        if managed.object_permission is None:
            return None
        # Read the row's current grants, since an SSE session keeps the policy it connected with.
        admitted: Final = LiteLLM_ObjectPermissionTable.model_validate(managed.object_permission)
        return LiteLLM_ObjectPermissionTable(object_permission_id=admitted.object_permission_id)
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        return None
    row: Final = await bounded_db_lookup(
        AgentsRepository(prisma_client, use_writer=True).table.find_unique(where={"agent_id": caller.agent_id}),
        name="agent_link",
    )
    # A config.yaml agent has no row and links no permission. Admission refuses a deleted agent.
    if row is None or not row.object_permission_id:
        return None
    return await get_object_permission(row.object_permission_id, prisma_client, user_api_key_cache, check_db_only=True)


def _inherits_one_team(caller: UserAPIKeyAuth) -> bool:
    if caller.team_id in (None, "", UI_TEAM_ID):
        return False
    if not caller.jwt_claims:
        return True
    from litellm.proxy.proxy_server import litellm_jwtauth

    teams: Final = ClaimReading(litellm_jwtauth).get_all_jwt_team_ids(caller.jwt_claims)
    return len(frozenset(teams)) <= 1


def _team_permission(team: LiteLLM_TeamTable) -> LiteLLM_ObjectPermissionTable | None:
    if team.object_permission is not None or not team.object_permission_id:
        return team.object_permission
    return LiteLLM_ObjectPermissionTable(object_permission_id=team.object_permission_id)


async def _with_grant_column(permission_id: str, *, fresh: bool) -> LiteLLM_ObjectPermissionTable | None:
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        return None
    return await get_object_permission(
        permission_id, prisma_client, user_api_key_cache, check_db_only=fresh, require_profile_grants=True
    )


async def _stored_grants(permission: LiteLLM_ObjectPermissionTable, *, fresh: bool) -> ProfileGrants | UnreadableGrants:
    read: Final = (
        permission
        if "mcp_auth_profiles" in permission.model_fields_set
        else await _with_grant_column(permission.object_permission_id, fresh=fresh)
    )
    if read is None or isinstance(read.mcp_auth_profiles, StoredProfileGrantError):
        return UnreadableGrants()
    return MappingProxyType(dict(read.mcp_auth_profiles or {}))


async def caller_auth_profile_grants(caller: UserAPIKeyAuth | None) -> tuple[ProfileGrants, ...] | UnreadableGrants:
    if caller is None or (
        caller.managed_agent_context is not None and caller.managed_agent_context.mode == "delegated"
    ):
        return ()
    if caller.catalog_key_agent_changed:
        return UnreadableGrants()
    try:
        admitted_subject: Final = is_mcp_admitted_user_subject(caller)
        key_id: Final = None if admitted_subject or caller.is_session_token else caller.object_permission_id
        key: Final = (
            await MCPRequestHandler.key_object_permission_hydrated(caller.take_catalog_key_permission())
            if key_id
            else None
        )
        if _is_unreadable(key_id, key):
            return UnreadableGrants()
        team: Final = (
            await MCPRequestHandler.permission_team(caller)
            if not admitted_subject and _inherits_one_team(caller)
            else None
        )
        team_permission: Final = _team_permission(team) if team is not None else None
        if team is not None and _is_unreadable(team.object_permission_id, team_permission):
            return UnreadableGrants()
        carriers: Final = tuple(found for found in (await _agent_permission(caller), key, team_permission) if found)
        grants: Final = [await _stored_grants(carrier, fresh=caller.requires_fresh_policy) for carrier in carriers]
        readable: Final = tuple(grant for grant in grants if not isinstance(grant, UnreadableGrants))
        return readable if len(readable) == len(grants) else UnreadableGrants()
    except Exception as exc:  # noqa: BLE001  # store and cache backends can raise different read failures
        verbose_proxy_logger.warning("Could not read MCP auth profile grants (%s)", type(exc).__name__)
        verbose_proxy_logger.debug("MCP auth profile grant read failure", exc_info=True)
        return UnreadableGrants()
