"""
Agent Permission Handler for LiteLLM Proxy.

Handles agent permission checking for keys and teams using object_permission_id.
Follows the same pattern as MCP permission handling.
"""

from dataclasses import dataclass
from typing import assert_never

from litellm._logging import verbose_logger
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.repositories.table_repositories import AgentsRepository


@dataclass(frozen=True, slots=True)
class UnrestrictedAgents:
    """The level declared no agent restriction, so it does not narrow the caller."""


@dataclass(frozen=True, slots=True)
class RestrictedAgents:
    """The exact agents the caller may reach. An empty set reaches none of them."""

    agent_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class UnresolvableAgents:
    """A restriction is declared but could not be read, so it cannot be enforced."""

    reason: str


AgentScope = UnrestrictedAgents | RestrictedAgents | UnresolvableAgents


def _combine_key_and_team_scope(key_scope: AgentScope, team_scope: AgentScope) -> AgentScope:
    """Resolve the key and team levels into the scope a request is authorized against.

    A level that declares nothing does not narrow the other one. Two levels that both
    declare narrow each other, and a pair that shares no agent leaves the caller with
    none rather than with all: "restricted to nothing" and "not restricted" are
    different answers, and only the second one may widen access.
    """
    match (key_scope, team_scope):
        case (UnresolvableAgents(), _):
            return key_scope
        case (_, UnresolvableAgents()):
            return team_scope
        case (UnrestrictedAgents(), _):
            return team_scope
        case (RestrictedAgents(), UnrestrictedAgents()):
            return key_scope
        case (RestrictedAgents(key_ids), RestrictedAgents(team_ids)):
            return RestrictedAgents(key_ids & team_ids)
        case _:
            assert_never(key_scope)


class AgentRequestHandler:
    """
    Class to handle agent permission checking, including:
    1. Key-level agent permissions
    2. Team-level agent permissions
    3. Agent access group resolution

    Follows the same inheritance logic as MCP:
    - If team has restrictions and key has restrictions: use intersection
    - If team has restrictions and key has none: inherit from team
    - If team has no restrictions: use key restrictions
    - If no restrictions: allow all agents
    - If the intersection is empty, or a declared restriction cannot be read: allow none
    """

    @staticmethod
    async def resolve_agent_scope(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> AgentScope:
        """
        Resolve the agents the given key/team may reach.

        Returns:
            AgentScope: whether the caller is unrestricted, restricted to a specific set
            (possibly empty), or carries a restriction that could not be read.
        """
        key_scope = await AgentRequestHandler._resolve_agents_for_key(user_api_key_auth)
        team_scope = await AgentRequestHandler._resolve_agents_for_team(user_api_key_auth)
        return _combine_key_and_team_scope(key_scope, team_scope)

    @staticmethod
    async def is_agent_allowed(
        agent_id: str,
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> bool:
        """
        Check if a specific agent is allowed for the given user/key.

        Args:
            agent_id: The agent ID to check
            user_api_key_auth: User authentication info

        Returns:
            bool: True if agent is allowed, False otherwise
        """
        scope = await AgentRequestHandler.resolve_agent_scope(user_api_key_auth)
        match scope:
            case UnrestrictedAgents():
                return True
            case RestrictedAgents(agent_ids):
                return agent_id in agent_ids
            case UnresolvableAgents(reason):
                verbose_logger.warning(f"Denying agent access, permissions unreadable: {reason}")
                return False
            case _:
                assert_never(scope)

    @staticmethod
    def _get_key_object_permission(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> LiteLLM_ObjectPermissionTable | None:
        """
        Get key object_permission - already loaded by get_key_object() in main auth flow.

        Note: object_permission is automatically populated when the key is fetched via
        get_key_object() in litellm/proxy/auth/auth_checks.py
        """
        if not user_api_key_auth:
            return None

        return user_api_key_auth.object_permission

    @staticmethod
    async def _get_team_object_permission(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> LiteLLM_ObjectPermissionTable | None:
        """
        Get team object_permission - automatically loaded by get_team_object() in main auth flow.

        Note: object_permission is automatically populated when the team is fetched via
        get_team_object() in litellm/proxy/auth/auth_checks.py
        """
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if not user_api_key_auth or not user_api_key_auth.team_id or not prisma_client:
            return None

        # Get the team object (which has object_permission already loaded)
        team_obj: LiteLLM_TeamTable | None = await get_team_object(
            team_id=user_api_key_auth.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_auth.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )

        if not team_obj:
            return None

        return team_obj.object_permission

    @staticmethod
    async def _resolve_agents_for_key(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> AgentScope:
        """
        Resolve the key level.

        1. First checks native key-level agent permissions (object_permission)
        2. Also includes agents from key's access_group_ids (unified access groups)

        Note: object_permission is already loaded by get_key_object() in main auth flow.
        A key that names agents or access groups is restricted even when those names
        resolve to nothing, so a group that is empty (or whose agents were deleted)
        cannot hand the key every agent.
        """
        if user_api_key_auth is None:
            return UnrestrictedAgents()

        try:
            key_object_permission = AgentRequestHandler._get_key_object_permission(user_api_key_auth)
            direct_agents = tuple(key_object_permission.agents or ()) if key_object_permission else ()
            agent_access_groups = (
                tuple(key_object_permission.agent_access_groups or ()) if key_object_permission else ()
            )
            key_access_group_ids = tuple(user_api_key_auth.access_group_ids or ())

            if not (direct_agents or agent_access_groups or key_access_group_ids):
                return UnrestrictedAgents()

            access_group_agents = await AgentRequestHandler._get_agents_from_access_groups(list(agent_access_groups))

            unified_agents: tuple[str, ...] = ()
            if key_access_group_ids:
                from litellm.proxy.auth.auth_checks import (
                    _get_agent_ids_from_access_groups,
                )

                unified_agents = tuple(
                    await _get_agent_ids_from_access_groups(access_group_ids=list(key_access_group_ids))
                )

            return RestrictedAgents(frozenset(direct_agents + tuple(access_group_agents) + unified_agents))
        except Exception as e:
            return UnresolvableAgents(f"key agent permissions: {e!s}")

    @staticmethod
    async def _resolve_agents_for_team(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> AgentScope:
        """
        Resolve the team level.

        1. First checks native team-level agent permissions (object_permission)
        2. Also includes agents from team's access_group_ids (unified access groups)

        Fetches the team object once and reuses it for both permission sources.
        """
        if user_api_key_auth is None or user_api_key_auth.team_id is None:
            return UnrestrictedAgents()

        try:
            from litellm.proxy.auth.auth_checks import get_team_object
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            if not prisma_client:
                return UnrestrictedAgents()

            # Fetch the team object once for both permission sources
            team_obj = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )

            if team_obj is None:
                return UnrestrictedAgents()

            object_permissions = team_obj.object_permission
            direct_agents = tuple(object_permissions.agents or ()) if object_permissions else ()
            agent_access_groups = tuple(object_permissions.agent_access_groups or ()) if object_permissions else ()
            team_access_group_ids = tuple(team_obj.access_group_ids or ())

            if not (direct_agents or agent_access_groups or team_access_group_ids):
                return UnrestrictedAgents()

            access_group_agents = await AgentRequestHandler._get_agents_from_access_groups(list(agent_access_groups))

            unified_agents: tuple[str, ...] = ()
            if team_access_group_ids:
                from litellm.proxy.auth.auth_checks import (
                    _get_agent_ids_from_access_groups,
                )

                unified_agents = tuple(
                    await _get_agent_ids_from_access_groups(access_group_ids=list(team_access_group_ids))
                )

            return RestrictedAgents(frozenset(direct_agents + tuple(access_group_agents) + unified_agents))
        except Exception as e:
            # litellm-dashboard is the default UI team and will never have agents;
            # skip noisy warnings for it.
            if user_api_key_auth.team_id != UI_TEAM_ID:
                verbose_logger.warning(f"Failed to get allowed agents for team: {e!s}")
            return UnresolvableAgents(f"team agent permissions: {e!s}")

    @staticmethod
    def _get_config_agent_ids_for_access_groups(config_agents: list, access_groups: list[str]) -> set[str]:
        """
        Helper to get agent_ids from config-loaded agents that match any of the given access groups.
        """
        server_ids: set[str] = set()
        for agent in config_agents:
            agent_access_groups = getattr(agent, "agent_access_groups", None)
            if agent_access_groups:
                if any(group in agent_access_groups for group in access_groups):
                    server_ids.add(agent.agent_id)
        return server_ids

    @staticmethod
    async def _get_db_agent_ids_for_access_groups(prisma_client, access_groups: list[str]) -> set[str]:
        """
        Helper to get agent_ids from DB agents that match any of the given access groups.

        A failed query raises: an unreadable grant is not an empty grant, and swallowing
        it here would report a restricted caller as carrying no restriction at all.
        """
        if not access_groups or prisma_client is None:
            return set()
        agents = await AgentsRepository(prisma_client).table.find_many(
            where={"agent_access_groups": {"hasSome": access_groups}}
        )
        return {agent.agent_id for agent in agents}

    @staticmethod
    async def _get_agents_from_access_groups(
        access_groups: list[str],
    ) -> list[str]:
        """
        Resolve agent access groups to agent IDs by querying BOTH the agent table (DB) AND config-loaded agents.
        """
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.proxy.proxy_server import prisma_client

        # Use the helper for config-loaded agents
        agent_ids = AgentRequestHandler._get_config_agent_ids_for_access_groups(
            global_agent_registry.agent_list, access_groups
        )

        # Use the helper for DB agents
        db_agent_ids = await AgentRequestHandler._get_db_agent_ids_for_access_groups(prisma_client, access_groups)

        return list(agent_ids | db_agent_ids)

    @staticmethod
    async def get_agent_access_groups(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> list[str]:
        """
        Get list of agent access groups for the given user/key based on permissions.
        """
        access_groups: list[str] = []
        access_groups_for_key = await AgentRequestHandler._get_agent_access_groups_for_key(user_api_key_auth)
        access_groups_for_team = await AgentRequestHandler._get_agent_access_groups_for_team(user_api_key_auth)

        # If team has access groups, then key must have a subset of the team's access groups
        if len(access_groups_for_team) > 0:
            for access_group in access_groups_for_key:
                if access_group in access_groups_for_team:
                    access_groups.append(access_group)
        else:
            access_groups = access_groups_for_key

        return list(set(access_groups))

    @staticmethod
    async def _get_agent_access_groups_for_key(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> list[str]:
        """Get agent access groups for the key."""
        from litellm.proxy.auth.auth_checks import get_object_permission
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.object_permission_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            key_object_permission = await get_object_permission(
                object_permission_id=user_api_key_auth.object_permission_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if key_object_permission is None:
                return []

            return key_object_permission.agent_access_groups or []
        except Exception as e:
            verbose_logger.warning(f"Failed to get agent access groups for key: {e!s}")
            return []

    @staticmethod
    async def _get_agent_access_groups_for_team(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> list[str]:
        """Get agent access groups for the team."""
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if user_api_key_auth is None:
            return []

        if user_api_key_auth.team_id is None:
            return []

        if prisma_client is None:
            verbose_logger.debug("prisma_client is None")
            return []

        try:
            team_obj: LiteLLM_TeamTable | None = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_obj is None:
                verbose_logger.debug("team_obj is None")
                return []

            object_permissions = team_obj.object_permission
            if object_permissions is None:
                return []

            return object_permissions.agent_access_groups or []
        except Exception as e:
            verbose_logger.warning(f"Failed to get agent access groups for team: {e!s}")
            return []
