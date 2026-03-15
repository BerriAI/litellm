"""
Unit tests for AgentRequestHandler - Agent permission management for keys and teams.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.agent_permission_handler import \
    AgentRequestHandler


@pytest.mark.asyncio
class TestAgentRequestHandler:
    """
    Test suite for AgentRequestHandler permission logic.
    """

    async def test_get_allowed_agents_intersection_logic(self):
        """
        Test key/team intersection: when both have restrictions, only common agents are allowed.
        When team has restrictions but key has none, key inherits from team.
        When neither has restrictions, returns empty list (meaning allow all).
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
        )

        # Case 1: Both key and team have agents - intersection
        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                mock_key.return_value = ["agent1", "agent2", "agent3"]
                mock_team.return_value = ["agent2", "agent4"]

                result = await AgentRequestHandler.get_allowed_agents(user_api_key_auth=mock_user_auth)
                assert sorted(result) == ["agent2"]

        # Case 2: Team has agents, key has none - inherit from team
        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                mock_key.return_value = []
                mock_team.return_value = ["team_agent1", "team_agent2"]

                result = await AgentRequestHandler.get_allowed_agents(user_api_key_auth=mock_user_auth)
                assert sorted(result) == ["team_agent1", "team_agent2"]

        # Case 3: No restrictions - returns empty list (allow all)
        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                mock_key.return_value = []
                mock_team.return_value = []

                result = await AgentRequestHandler.get_allowed_agents(user_api_key_auth=mock_user_auth)
                assert result == []

    async def test_is_agent_allowed_respects_permissions(self):
        """
        Test is_agent_allowed: returns True if agent in allowed list or if no restrictions.
        Returns False if agent not in allowed list.
        """
        mock_user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

        # Agent in allowed list - should be allowed
        with patch.object(AgentRequestHandler, "get_allowed_agents") as mock_get_allowed:
            mock_get_allowed.return_value = ["agent1", "agent2"]
            assert await AgentRequestHandler.is_agent_allowed(agent_id="agent1", user_api_key_auth=mock_user_auth) is True

        # Agent not in allowed list - should be denied
        with patch.object(AgentRequestHandler, "get_allowed_agents") as mock_get_allowed:
            mock_get_allowed.return_value = ["agent1", "agent2"]
            assert await AgentRequestHandler.is_agent_allowed(agent_id="agent3", user_api_key_auth=mock_user_auth) is False

        # Empty list means no restrictions - should allow any agent
        with patch.object(AgentRequestHandler, "get_allowed_agents") as mock_get_allowed:
            mock_get_allowed.return_value = []
            assert await AgentRequestHandler.is_agent_allowed(agent_id="any_agent", user_api_key_auth=mock_user_auth) is True

    async def test_no_auth_allows_all_agents(self):
        """
        Test that when user_api_key_auth is None, all agents are allowed (no restrictions).
        """
        result = await AgentRequestHandler.get_allowed_agents(user_api_key_auth=None)
        assert result == []

        is_allowed = await AgentRequestHandler.is_agent_allowed(agent_id="any_agent", user_api_key_auth=None)
        assert is_allowed is True

    async def test_get_allowed_agents_handles_errors_gracefully(self):
        """
        Test that errors during permission lookup are handled gracefully (returns empty list).
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            object_permission_id="test-permission",
        )

        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                mock_key.side_effect = Exception("DB Error")
                mock_team.return_value = []

                result = await AgentRequestHandler.get_allowed_agents(user_api_key_auth=mock_user_auth)
                assert result == []

    async def test_get_allowed_agents_for_key_via_access_group_ids(self):
        """
        Test that _get_allowed_agents_for_key includes agents from key's access_group_ids
        (unified access groups) when key has no native object_permission.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            access_group_ids=["ag-with-agents"],
        )

        with patch.object(
            AgentRequestHandler, "_get_key_object_permission", return_value=None
        ):
            with patch(
                "litellm.proxy.auth.auth_checks._get_agent_ids_from_access_groups",
                new_callable=AsyncMock,
                return_value=["agent-from-ag-1", "agent-from-ag-2"],
            ):
                result = await AgentRequestHandler._get_allowed_agents_for_key(
                    user_api_key_auth=mock_user_auth
                )
                assert sorted(result) == ["agent-from-ag-1", "agent-from-ag-2"]

    async def test_get_allowed_agents_for_key_combines_native_and_access_groups(self):
        """
        Test that _get_allowed_agents_for_key combines agents from native object_permission
        and key's access_group_ids (unified access groups).
        """
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

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
        # Attach object_permission so _get_key_object_permission returns it
        mock_user_auth.object_permission = mock_permission

        with patch(
            "litellm.proxy.auth.auth_checks._get_agent_ids_from_access_groups",
            new_callable=AsyncMock,
            return_value=["agent-from-ag"],
        ):
            result = await AgentRequestHandler._get_allowed_agents_for_key(
                user_api_key_auth=mock_user_auth
            )
            assert sorted(result) == ["agent-from-ag", "native-agent-1"]

    async def test_calling_agent_restricts_allowed_sub_agents(self):
        """
        When a key is scoped to an agent that has sub-agent restrictions,
        the result is intersected with key/team permissions.
        """
        from litellm.types.agents import AgentResponse

        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="parent-agent",
        )

        parent_agent = AgentResponse(
            agent_id="parent-agent",
            agent_name="parent",
            agent_card_params={"name": "parent"},
            object_permission={
                "agents": ["sub-agent-1", "sub-agent-2"],
            },
        )

        mock_registry = MagicMock()
        mock_registry.get_agent_by_id.return_value = parent_agent

        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                with patch(
                    "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
                    mock_registry,
                ):
                    mock_key.return_value = []
                    mock_team.return_value = []

                    result = await AgentRequestHandler.get_allowed_agents(
                        user_api_key_auth=mock_user_auth
                    )
                    assert sorted(result) == ["sub-agent-1", "sub-agent-2"]

    async def test_calling_agent_intersects_with_key_team(self):
        """
        When both key/team and calling agent have restrictions,
        the final result is the intersection.
        """
        from litellm.types.agents import AgentResponse

        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="parent-agent",
        )

        parent_agent = AgentResponse(
            agent_id="parent-agent",
            agent_name="parent",
            agent_card_params={"name": "parent"},
            object_permission={
                "agents": ["sub-agent-1", "sub-agent-2", "sub-agent-3"],
            },
        )

        mock_registry = MagicMock()
        mock_registry.get_agent_by_id.return_value = parent_agent

        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                with patch(
                    "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
                    mock_registry,
                ):
                    mock_key.return_value = ["sub-agent-1", "sub-agent-4"]
                    mock_team.return_value = []

                    result = await AgentRequestHandler.get_allowed_agents(
                        user_api_key_auth=mock_user_auth
                    )
                    assert result == ["sub-agent-1"]

    async def test_calling_agent_no_restrictions_passes_through(self):
        """
        When the calling agent has no object_permission (no sub-agent restrictions),
        key/team permissions pass through unchanged.
        """
        from litellm.types.agents import AgentResponse

        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="parent-agent",
        )

        parent_agent = AgentResponse(
            agent_id="parent-agent",
            agent_name="parent",
            agent_card_params={"name": "parent"},
            object_permission=None,
        )

        mock_registry = MagicMock()
        mock_registry.get_agent_by_id.return_value = parent_agent

        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                with patch(
                    "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
                    mock_registry,
                ):
                    mock_key.return_value = ["agent-a", "agent-b"]
                    mock_team.return_value = []

                    result = await AgentRequestHandler.get_allowed_agents(
                        user_api_key_auth=mock_user_auth
                    )
                    assert sorted(result) == ["agent-a", "agent-b"]

    async def test_calling_agent_with_access_groups(self):
        """
        When the calling agent's object_permission uses agent_access_groups,
        those are resolved to agent IDs and used for intersection.
        """
        from litellm.types.agents import AgentResponse

        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id="parent-agent",
        )

        parent_agent = AgentResponse(
            agent_id="parent-agent",
            agent_name="parent",
            agent_card_params={"name": "parent"},
            object_permission={
                "agents": ["sub-agent-1"],
                "agent_access_groups": ["group-a"],
            },
        )

        mock_registry = MagicMock()
        mock_registry.get_agent_by_id.return_value = parent_agent

        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                with patch(
                    "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
                    mock_registry,
                ):
                    with patch.object(
                        AgentRequestHandler,
                        "_get_agents_from_access_groups",
                        new_callable=AsyncMock,
                        return_value=["sub-agent-from-group"],
                    ):
                        mock_key.return_value = []
                        mock_team.return_value = []

                        result = await AgentRequestHandler.get_allowed_agents(
                            user_api_key_auth=mock_user_auth
                        )
                        assert sorted(result) == [
                            "sub-agent-1",
                            "sub-agent-from-group",
                        ]

    async def test_no_agent_id_on_key_skips_agent_level_check(self):
        """
        When the key has no agent_id, the agent-level check is skipped entirely.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
        )

        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                mock_key.return_value = ["agent-x"]
                mock_team.return_value = []

                result = await AgentRequestHandler.get_allowed_agents(
                    user_api_key_auth=mock_user_auth
                )
                assert result == ["agent-x"]
