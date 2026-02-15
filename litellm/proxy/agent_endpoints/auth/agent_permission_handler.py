"""
Agent Permission Handler for LiteLLM Proxy.

Handles agent permission checking for keys and teams using object_permission_id.
Follows the same pattern as MCP permission handling.
"""

from typing import List, Optional, Set

from litellm._logging import verbose_logger
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)


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
    async def get_allowed_agents(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get list of allowed agent IDs for the given user/key based on permissions.

        Returns:
            List[str]: List of allowed agent IDs. Empty list means no restrictions (allow all).
        """
        try:
            allowed_agents: List[str] = []
            allowed_agents_for_key = (
                await AgentRequestHandler._get_allowed_agents_for_key(user_api_key_auth)
            )
            allowed_agents_for_team = (
                await AgentRequestHandler._get_allowed_agents_for_team(
                    user_api_key_auth
                )
            )

            # If team has agent restrictions, handle inheritance and intersection logic
            if len(allowed_agents_for_team) > 0:
                if len(allowed_agents_for_key) > 0:
                    # Key has its own agent permissions - use intersection with team permissions
                    for agent_id in allowed_agents_for_key:
                        if agent_id in allowed_agents_for_team:
                            allowed_agents.append(agent_id)
                else:
                    # Key has no agent permissions - inherit from team
                    allowed_agents = allowed_agents_for_team
            else:
                allowed_agents = allowed_agents_for_key

            return list(set(allowed_agents))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed agents: {str(e)}")
            return []

    @staticmethod
    async def is_agent_allowed(
        agent_id: str,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> bool:
        """
        Check if a specific agent is allowed for the given user/key.

        Args:
            agent_id: The agent ID to check
            user_api_key_auth: User authentication info

        Returns:
            bool: True if agent is allowed, False otherwise
        """
        allowed_agents = await AgentRequestHandler.get_allowed_agents(user_api_key_auth)

        # Empty list means no restrictions - allow all
        if len(allowed_agents) == 0:
            return True

        return agent_id in allowed_agents

    @staticmethod
    def _get_key_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> Optional[LiteLLM_ObjectPermissionTable]:
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
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> Optional[LiteLLM_ObjectPermissionTable]:
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
        team_obj: Optional[LiteLLM_TeamTable] = await get_team_object(
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
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get allowed agents for a key.

        1. First checks native key-level agent permissions (object_permission)
        2. Also includes agents from key's access_group_ids (unified access groups)

        Note: object_permission is already loaded by get_key_object() in main auth flow.
        """
        if user_api_key_auth is None:
            return []

        try:
            all_agents: List[str] = []

            # 1. Get agents from object_permission (native permissions)
            key_object_permission = AgentRequestHandler._get_key_object_permission(
                user_api_key_auth
            )
            if key_object_permission is not None:
                # Get direct agents
                direct_agents = key_object_permission.agents or []

                # Get agents from access groups
                access_group_agents = (
                    await AgentRequestHandler._get_agents_from_access_groups(
                        key_object_permission.agent_access_groups or []
                    )
                )

                all_agents = direct_agents + access_group_agents

            # 2. Fallback: get agent IDs from key's access_group_ids (unified access groups)
            key_access_group_ids = user_api_key_auth.access_group_ids or []
            if key_access_group_ids:
                from litellm.proxy.auth.auth_checks import (
                    _get_agent_ids_from_access_groups,
                )

                unified_agents = await _get_agent_ids_from_access_groups(
                    access_group_ids=key_access_group_ids,
                )
                all_agents.extend(unified_agents)

            return list(set(all_agents))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed agents for key: {str(e)}")
            return []

    @staticmethod
    async def _get_allowed_agents_for_team(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get allowed agents for a team.

        1. First checks native team-level agent permissions (object_permission)
        2. Also includes agents from team's access_group_ids (unified access groups)

        Fetches the team object once and reuses it for both permission sources.
        """
        if user_api_key_auth is None:
            return []

        if user_api_key_auth.team_id is None:
            return []

        try:
            from litellm.proxy.auth.auth_checks import get_team_object
            from litellm.proxy.proxy_server import (
                prisma_client,
                proxy_logging_obj,
                user_api_key_cache,
            )

            if not prisma_client:
                return []

            # Fetch the team object once for both permission sources
            team_obj = await get_team_object(
                team_id=user_api_key_auth.team_id,
                prisma_client=prisma_client,
                user_api_key_cache=user_api_key_cache,
                parent_otel_span=user_api_key_auth.parent_otel_span,
                proxy_logging_obj=proxy_logging_obj,
            )

            if team_obj is None:
                return []

            all_agents: List[str] = []

            # 1. Get agents from object_permission (native permissions)
            object_permissions = team_obj.object_permission
            if object_permissions is not None:
                # Get direct agents
                direct_agents = object_permissions.agents or []

                # Get agents from access groups
                access_group_agents = (
                    await AgentRequestHandler._get_agents_from_access_groups(
                        object_permissions.agent_access_groups or []
                    )
                )

                all_agents = direct_agents + access_group_agents

            # 2. Also include agents from team's access_group_ids (unified access groups)
            team_access_group_ids = team_obj.access_group_ids or []
            if team_access_group_ids:
                from litellm.proxy.auth.auth_checks import (
                    _get_agent_ids_from_access_groups,
                )

                unified_agents = await _get_agent_ids_from_access_groups(
                    access_group_ids=team_access_group_ids,
                )
                all_agents.extend(unified_agents)

            return list(set(all_agents))
        except Exception as e:
            verbose_logger.warning(f"Failed to get allowed agents for team: {str(e)}")
            return []

    @staticmethod
    def _get_config_agent_ids_for_access_groups(
        config_agents: List, access_groups: List[str]
    ) -> Set[str]:
        """
        Helper to get agent_ids from config-loaded agents that match any of the given access groups.
        """
        server_ids: Set[str] = set()
        for agent in config_agents:
            agent_access_groups = getattr(agent, "agent_access_groups", None)
            if agent_access_groups:
                if any(group in agent_access_groups for group in access_groups):
                    server_ids.add(agent.agent_id)
        return server_ids

    @staticmethod
    async def _get_db_agent_ids_for_access_groups(
        prisma_client, access_groups: List[str]
    ) -> Set[str]:
        """
        Helper to get agent_ids from DB agents that match any of the given access groups.
        """
        agent_ids: Set[str] = set()
        if access_groups and prisma_client is not None:
            try:
                agents = await prisma_client.db.litellm_agentstable.find_many(
                    where={"agent_access_groups": {"hasSome": access_groups}}
                )
                for agent in agents:
                    agent_ids.add(agent.agent_id)
            except Exception as e:
                verbose_logger.debug(f"Error getting agents from access groups: {e}")
        return agent_ids

    @staticmethod
    async def _get_agents_from_access_groups(
        access_groups: List[str],
    ) -> List[str]:
        """
        Resolve agent access groups to agent IDs by querying BOTH the agent table (DB) AND config-loaded agents.
        """
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry
        from litellm.proxy.proxy_server import prisma_client

        try:
            # Use the helper for config-loaded agents
            agent_ids = AgentRequestHandler._get_config_agent_ids_for_access_groups(
                global_agent_registry.agent_list, access_groups
            )

            # Use the helper for DB agents
            db_agent_ids = (
                await AgentRequestHandler._get_db_agent_ids_for_access_groups(
                    prisma_client, access_groups
                )
            )
            agent_ids.update(db_agent_ids)

            return list(agent_ids)
        except Exception as e:
            verbose_logger.warning(f"Failed to get agents from access groups: {str(e)}")
            return []

    @staticmethod
    async def get_agent_access_groups(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get list of agent access groups for the given user/key based on permissions.
        """
        access_groups: List[str] = []
        access_groups_for_key = (
            await AgentRequestHandler._get_agent_access_groups_for_key(
                user_api_key_auth
            )
        )
        access_groups_for_team = (
            await AgentRequestHandler._get_agent_access_groups_for_team(
                user_api_key_auth
            )
        )

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
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
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
            verbose_logger.warning(
                f"Failed to get agent access groups for key: {str(e)}"
            )
            return []

    @staticmethod
    async def _get_agent_access_groups_for_team(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
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
            team_obj: Optional[LiteLLM_TeamTable] = await get_team_object(
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
            verbose_logger.warning(
                f"Failed to get agent access groups for team: {str(e)}"
            )
            return []
