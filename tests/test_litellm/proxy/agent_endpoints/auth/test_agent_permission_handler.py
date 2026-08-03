"""
Unit tests for AgentRequestHandler - Agent permission management for keys and teams.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
    AgentRequestHandler,
    RestrictedAgents,
    UnresolvableAgents,
    UnrestrictedAgents,
)


@pytest.mark.asyncio
class TestAgentRequestHandler:
    """
    Test suite for AgentRequestHandler permission logic.
    """

    async def test_resolve_agent_scope_intersection_logic(self):
        """
        Test key/team intersection: when both have restrictions, only common agents are allowed.
        When team has restrictions but key has none, key inherits from team.
        When neither has restrictions, the caller is unrestricted.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
        )

        with patch.object(AgentRequestHandler, "_resolve_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_resolve_agents_for_team") as mock_team:
                mock_key.return_value = RestrictedAgents(frozenset({"agent1", "agent2", "agent3"}))
                mock_team.return_value = RestrictedAgents(frozenset({"agent2", "agent4"}))

                result = await AgentRequestHandler.resolve_agent_scope(user_api_key_auth=mock_user_auth)
                assert result == RestrictedAgents(frozenset({"agent2"}))

        with patch.object(AgentRequestHandler, "_resolve_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_resolve_agents_for_team") as mock_team:
                mock_key.return_value = UnrestrictedAgents()
                mock_team.return_value = RestrictedAgents(frozenset({"team_agent1", "team_agent2"}))

                result = await AgentRequestHandler.resolve_agent_scope(user_api_key_auth=mock_user_auth)
                assert result == RestrictedAgents(frozenset({"team_agent1", "team_agent2"}))

        with patch.object(AgentRequestHandler, "_resolve_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_resolve_agents_for_team") as mock_team:
                mock_key.return_value = UnrestrictedAgents()
                mock_team.return_value = UnrestrictedAgents()

                result = await AgentRequestHandler.resolve_agent_scope(user_api_key_auth=mock_user_auth)
                assert result == UnrestrictedAgents()

    async def test_disjoint_key_and_team_permissions_deny_every_agent(self):
        """
        Regression: a key restricted to agent1 inside a team restricted to agent2 shares no
        agent with its team, so it must reach neither of them (and certainly not a third
        agent nobody granted it). The intersection used to collapse to an empty list, which
        the caller read as "no restrictions" and turned into access to every agent.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
        )

        with patch.object(AgentRequestHandler, "_resolve_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_resolve_agents_for_team") as mock_team:
                mock_key.return_value = RestrictedAgents(frozenset({"agent1"}))
                mock_team.return_value = RestrictedAgents(frozenset({"agent2"}))

                assert await AgentRequestHandler.resolve_agent_scope(
                    user_api_key_auth=mock_user_auth
                ) == RestrictedAgents(frozenset())

                for agent_id in ("agent1", "agent2", "agent-nobody-granted"):
                    assert (
                        await AgentRequestHandler.is_agent_allowed(
                            agent_id=agent_id, user_api_key_auth=mock_user_auth
                        )
                        is False
                    )

    async def test_is_agent_allowed_respects_permissions(self):
        """
        Test is_agent_allowed: returns True if agent in allowed set or if no restrictions.
        Returns False if agent not in allowed set.
        """
        mock_user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

        with patch.object(AgentRequestHandler, "resolve_agent_scope") as mock_scope:
            mock_scope.return_value = RestrictedAgents(frozenset({"agent1", "agent2"}))
            assert (
                await AgentRequestHandler.is_agent_allowed(agent_id="agent1", user_api_key_auth=mock_user_auth) is True
            )

        with patch.object(AgentRequestHandler, "resolve_agent_scope") as mock_scope:
            mock_scope.return_value = RestrictedAgents(frozenset({"agent1", "agent2"}))
            assert (
                await AgentRequestHandler.is_agent_allowed(agent_id="agent3", user_api_key_auth=mock_user_auth) is False
            )

        with patch.object(AgentRequestHandler, "resolve_agent_scope") as mock_scope:
            mock_scope.return_value = UnrestrictedAgents()
            assert (
                await AgentRequestHandler.is_agent_allowed(agent_id="any_agent", user_api_key_auth=mock_user_auth)
                is True
            )

    async def test_no_auth_allows_all_agents(self):
        """
        Test that when user_api_key_auth is None, all agents are allowed (no restrictions).
        """
        assert await AgentRequestHandler.resolve_agent_scope(user_api_key_auth=None) == UnrestrictedAgents()

        assert await AgentRequestHandler.is_agent_allowed(agent_id="any_agent", user_api_key_auth=None) is True

    async def test_unreadable_permissions_deny_instead_of_allowing_everything(self):
        """
        Regression: a lookup that raises (a database outage, say) leaves the caller's
        restriction unknown. That used to surface as an empty list, i.e. "unrestricted",
        so an outage handed every restricted key every agent. It must deny instead.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            object_permission_id="test-permission",
        )

        with patch.object(AgentRequestHandler, "_resolve_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_resolve_agents_for_team") as mock_team:
                mock_key.return_value = UnresolvableAgents("DB Error")
                mock_team.return_value = UnrestrictedAgents()

                assert await AgentRequestHandler.resolve_agent_scope(
                    user_api_key_auth=mock_user_auth
                ) == UnresolvableAgents("DB Error")

                assert (
                    await AgentRequestHandler.is_agent_allowed(agent_id="agent1", user_api_key_auth=mock_user_auth)
                    is False
                )

    async def test_key_access_group_lookup_failure_is_unresolvable(self):
        """
        Regression: the access group query used to swallow its own exception and return no
        agents, which made a restricted key look unrestricted. The failure has to reach the
        caller as UnresolvableAgents so the request is denied.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            access_group_ids=["ag-1"],
        )

        with patch(
            "litellm.proxy.auth.auth_checks._get_agent_ids_from_access_groups",
            new_callable=AsyncMock,
            side_effect=Exception("connection refused"),
        ):
            scope = await AgentRequestHandler._resolve_agents_for_key(user_api_key_auth=mock_user_auth)

        assert isinstance(scope, UnresolvableAgents)
        assert "connection refused" in scope.reason

    async def test_access_group_that_resolves_to_nothing_still_restricts(self):
        """
        Regression: a key whose only grant is an access group with no agents left is
        restricted to nothing, not unrestricted.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            access_group_ids=["ag-empty"],
        )

        with patch(
            "litellm.proxy.auth.auth_checks._get_agent_ids_from_access_groups",
            new_callable=AsyncMock,
            return_value=[],
        ):
            scope = await AgentRequestHandler._resolve_agents_for_key(user_api_key_auth=mock_user_auth)

        assert scope == RestrictedAgents(frozenset())

    async def test_resolve_agents_for_key_via_access_group_ids(self):
        """
        Test that _resolve_agents_for_key includes agents from key's access_group_ids
        (unified access groups) when key has no native object_permission.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            access_group_ids=["ag-with-agents"],
        )

        with patch.object(AgentRequestHandler, "_get_key_object_permission", return_value=None):
            with patch(
                "litellm.proxy.auth.auth_checks._get_agent_ids_from_access_groups",
                new_callable=AsyncMock,
                return_value=["agent-from-ag-1", "agent-from-ag-2"],
            ):
                result = await AgentRequestHandler._resolve_agents_for_key(user_api_key_auth=mock_user_auth)
                assert result == RestrictedAgents(frozenset({"agent-from-ag-1", "agent-from-ag-2"}))

    async def test_resolve_agents_for_key_combines_native_and_access_groups(self):
        """
        Test that _resolve_agents_for_key combines agents from native object_permission
        and key's access_group_ids (unified access groups).
        """
        mock_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="obj-1",
            agents=["native-agent-1"],
            agent_access_groups=[],
        )
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            access_group_ids=["ag-1"],
        )
        mock_user_auth.object_permission = mock_permission

        with patch(
            "litellm.proxy.auth.auth_checks._get_agent_ids_from_access_groups",
            new_callable=AsyncMock,
            return_value=["agent-from-ag"],
        ):
            result = await AgentRequestHandler._resolve_agents_for_key(user_api_key_auth=mock_user_auth)
            assert result == RestrictedAgents(frozenset({"agent-from-ag", "native-agent-1"}))
