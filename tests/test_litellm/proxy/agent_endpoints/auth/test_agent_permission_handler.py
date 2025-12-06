"""
Unit tests for AgentRequestHandler - Agent permission management for keys and teams.
"""

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
    AgentRequestHandler,
)


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
