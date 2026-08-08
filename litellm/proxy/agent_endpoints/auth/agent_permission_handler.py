"""
Agent Permission Handler for LiteLLM Proxy.

Handles agent permission checking for keys and teams using object_permission_id.
Follows the same pattern as MCP permission handling.
"""

from dataclasses import dataclass
from typing import Final, TypeAlias

from litellm._logging import verbose_logger
from litellm.proxy._types import (
    UI_TEAM_ID,
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.repositories.table_repositories import AgentsRepository


@dataclass(frozen=True, slots=True)
class UnrestrictedAgentAccess:
    """No agent grant exists on the key or its team, so every agent is reachable."""


@dataclass(frozen=True, slots=True)
class RestrictedAgentAccess:
    """Only ``agent_ids`` are reachable. An empty set denies every agent."""

    agent_ids: frozenset[str]


AgentAccess: TypeAlias = UnrestrictedAgentAccess | RestrictedAgentAccess


def _to_stable_ids(agent_ids: frozenset[str]) -> frozenset[str]:
    from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

    return frozenset(global_agent_registry.stable_agent_id(agent_id) for agent_id in agent_ids)


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
    """

    @staticmethod
    async def resolve_agent_access(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> AgentAccess:
        """
        Resolve the agents the given user/key may reach.

        ``UnrestrictedAgentAccess`` is only returned when neither the key nor its team
        carries any grant. Grants that intersect to nothing stay restricted, so
        narrowing a caller can never widen what it reaches.
        """
        try:
            key_access: Final = await AgentRequestHandler._get_allowed_agents_for_key(user_api_key_auth)
            team_access: Final = await AgentRequestHandler._get_allowed_agents_for_team(user_api_key_auth)

            match (key_access, team_access):
                case (UnrestrictedAgentAccess(), UnrestrictedAgentAccess()):
                    return UnrestrictedAgentAccess()
                case (UnrestrictedAgentAccess(), RestrictedAgentAccess(team_ids)):
                    return RestrictedAgentAccess(_to_stable_ids(team_ids))
                case (RestrictedAgentAccess(key_ids), UnrestrictedAgentAccess()):
                    return RestrictedAgentAccess(_to_stable_ids(key_ids))
                case (RestrictedAgentAccess(key_ids), RestrictedAgentAccess(team_ids)):
                    return RestrictedAgentAccess(_to_stable_ids(key_ids) & _to_stable_ids(team_ids))
        except Exception as e:
            verbose_logger.warning("Failed to get allowed agents: %s", e)
            return UnrestrictedAgentAccess()

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
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        match await AgentRequestHandler.resolve_agent_access(user_api_key_auth):
            case UnrestrictedAgentAccess():
                return True
            case RestrictedAgentAccess(allowed_agent_ids):
                stable_id: Final = global_agent_registry.stable_agent_id(agent_id)
                return not global_agent_registry.ids_for_agent(stable_id).isdisjoint(allowed_agent_ids)

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
        team_obj: Final[LiteLLM_TeamTable | None] = await get_team_object(
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
    async def _get_allowed_agents_for_key(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> AgentAccess:
        """
        Get allowed agents for a key.

        1. First checks native key-level agent permissions (object_permission)
        2. Also includes agents from key's access_group_ids (unified access groups)

        A key that declares agents or access groups is restricted even when those
        declarations resolve to nothing, so an emptied or deleted access group denies
        rather than opening the key up. Lookup failures still propagate to the caller,
        which keeps them fail-open.

        Note: object_permission is already loaded by get_key_object() in main auth flow.
        """
        if user_api_key_auth is None:
            return UnrestrictedAgentAccess()

        try:
            # 1. Get agents from object_permission (native permissions)
            key_object_permission: Final = AgentRequestHandler._get_key_object_permission(user_api_key_auth)
            direct_agents: Final = tuple(
                key_object_permission.agents or () if key_object_permission is not None else ()
            )
            declared_access_groups: Final = tuple(
                key_object_permission.agent_access_groups or () if key_object_permission is not None else ()
            )
            # 2. Fallback: get agent IDs from key's access_group_ids (unified access groups)
            key_access_group_ids: Final = tuple(user_api_key_auth.access_group_ids or ())

            if not direct_agents and not declared_access_groups and not key_access_group_ids:
                return UnrestrictedAgentAccess()

            access_group_agents: Final = (
                tuple(await AgentRequestHandler._get_agents_from_access_groups(list(declared_access_groups)))
                if declared_access_groups
                else ()
            )
            unified_agents: Final = (
                tuple(await AgentRequestHandler._get_unified_access_group_agents(list(key_access_group_ids)))
                if key_access_group_ids
                else ()
            )

            return RestrictedAgentAccess(frozenset(direct_agents + access_group_agents + unified_agents))
        except Exception as e:
            verbose_logger.warning("Failed to get allowed agents for key: %s", e)
            return UnrestrictedAgentAccess()

    @staticmethod
    async def _get_allowed_agents_for_team(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> AgentAccess:
        """
        Get allowed agents for a team.

        1. First checks native team-level agent permissions (object_permission)
        2. Also includes agents from team's access_group_ids (unified access groups)

        Fetches the team object once and reuses it for both permission sources.
        Declared-but-empty grants stay restricted; see `_get_allowed_agents_for_key`.
        """
        if user_api_key_auth is None:
            return UnrestrictedAgentAccess()

        if user_api_key_auth.team_id is None:
            return UnrestrictedAgentAccess()

        try:
            from litellm.proxy.auth.auth_checks import get_team_object
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            if not prisma_client:
                return UnrestrictedAgentAccess()

            # Fetch the team object once for both permission sources
            team_obj: Final = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )

            if team_obj is None:
                return UnrestrictedAgentAccess()

            # 1. Get agents from object_permission (native permissions)
            object_permissions: Final = team_obj.object_permission
            direct_agents: Final = tuple(object_permissions.agents or () if object_permissions is not None else ())
            declared_access_groups: Final = tuple(
                object_permissions.agent_access_groups or () if object_permissions is not None else ()
            )
            # 2. Also include agents from team's access_group_ids (unified access groups)
            team_access_group_ids: Final = tuple(team_obj.access_group_ids or ())

            if not direct_agents and not declared_access_groups and not team_access_group_ids:
                return UnrestrictedAgentAccess()

            access_group_agents: Final = (
                tuple(await AgentRequestHandler._get_agents_from_access_groups(list(declared_access_groups)))
                if declared_access_groups
                else ()
            )
            unified_agents: Final = (
                tuple(await AgentRequestHandler._get_unified_access_group_agents(list(team_access_group_ids)))
                if team_access_group_ids
                else ()
            )

            return RestrictedAgentAccess(frozenset(direct_agents + access_group_agents + unified_agents))
        except Exception as e:
            # litellm-dashboard is the default UI team and will never have agents;
            # skip noisy warnings for it.
            if user_api_key_auth.team_id != UI_TEAM_ID:
                verbose_logger.warning("Failed to get allowed agents for team: %s", e)
            return UnrestrictedAgentAccess()

    @staticmethod
    def _get_config_agent_ids_for_access_groups(config_agents: list, access_groups: list[str]) -> set[str]:
        """
        Helper to get agent_ids from config-loaded agents that match any of the given access groups.
        """
        server_ids: Final[set[str]] = set()
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

        Query failures propagate so the caller can tell "this group is empty" (deny)
        apart from "the lookup failed" (fail-open).
        """
        if not access_groups or prisma_client is None:
            return set()

        agents: Final = await AgentsRepository(prisma_client).table.find_many(
            where={"agent_access_groups": {"hasSome": access_groups}}
        )
        return {agent.agent_id for agent in agents}

    @staticmethod
    async def _get_unified_access_group_agents(access_group_ids: list[str]) -> list[str]:
        """
        Resolve unified access group ids to agent IDs.
        """
        from litellm.proxy.auth.auth_checks import _get_agent_ids_from_access_groups

        return await _get_agent_ids_from_access_groups(access_group_ids=access_group_ids)

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
        config_agent_ids: Final = AgentRequestHandler._get_config_agent_ids_for_access_groups(
            global_agent_registry.agent_list, access_groups
        )

        # Use the helper for DB agents
        db_agent_ids: Final = await AgentRequestHandler._get_db_agent_ids_for_access_groups(
            prisma_client, access_groups
        )

        return list(config_agent_ids | db_agent_ids)

    @staticmethod
    async def get_agent_access_groups(
        user_api_key_auth: UserAPIKeyAuth | None = None,
    ) -> list[str]:
        """
        Get list of agent access groups for the given user/key based on permissions.
        """
        access_groups: list[str] = []
        access_groups_for_key: Final = await AgentRequestHandler._get_agent_access_groups_for_key(user_api_key_auth)
        access_groups_for_team: Final = await AgentRequestHandler._get_agent_access_groups_for_team(user_api_key_auth)

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
            key_object_permission: Final = await get_object_permission(
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
            verbose_logger.warning("Failed to get agent access groups for key: %s", e)
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
            team_obj: Final[LiteLLM_TeamTable | None] = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )
            if team_obj is None:
                verbose_logger.debug("team_obj is None")
                return []

            object_permissions: Final = team_obj.object_permission
            if object_permissions is None:
                return []

            return object_permissions.agent_access_groups or []
        except Exception as e:
            verbose_logger.warning("Failed to get agent access groups for team: %s", e)
            return []
