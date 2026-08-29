"""
Unit tests for AgentRequestHandler - Agent permission management for keys and teams.
"""

import hashlib
import json
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest


from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
    AgentRequestHandler,
    RestrictedAgentAccess,
    UnrestrictedAgentAccess,
)


@pytest.mark.asyncio
class TestAgentRequestHandler:
    """
    Test suite for AgentRequestHandler permission logic.
    """

    async def test_resolve_agent_access_intersection_logic(self):
        """
        Test key/team intersection: when both have restrictions, only common agents are allowed.
        When team has restrictions but key has none, key inherits from team.
        Only a caller with no grant anywhere is unrestricted.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
        )

        # Case 1: Both key and team have agents - intersection
        with patch.object(
            AgentRequestHandler, "_get_allowed_agents_for_key"
        ) as mock_key:
            with patch.object(
                AgentRequestHandler, "_get_allowed_agents_for_team"
            ) as mock_team:
                mock_key.return_value = RestrictedAgentAccess(
                    frozenset({"agent1", "agent2", "agent3"})
                )
                mock_team.return_value = RestrictedAgentAccess(
                    frozenset({"agent2", "agent4"})
                )

                result = await AgentRequestHandler.resolve_agent_access(
                    user_api_key_auth=mock_user_auth
                )
                assert result == RestrictedAgentAccess(frozenset({"agent2"}))

        # Case 2: Team has agents, key has none - inherit from team
        with patch.object(
            AgentRequestHandler, "_get_allowed_agents_for_key"
        ) as mock_key:
            with patch.object(
                AgentRequestHandler, "_get_allowed_agents_for_team"
            ) as mock_team:
                mock_key.return_value = UnrestrictedAgentAccess()
                mock_team.return_value = RestrictedAgentAccess(
                    frozenset({"team_agent1", "team_agent2"})
                )

                result = await AgentRequestHandler.resolve_agent_access(
                    user_api_key_auth=mock_user_auth
                )
                assert result == RestrictedAgentAccess(
                    frozenset({"team_agent1", "team_agent2"})
                )

        # Case 3: Key has agents, team has none - key restrictions stand
        with patch.object(
            AgentRequestHandler, "_get_allowed_agents_for_key"
        ) as mock_key:
            with patch.object(
                AgentRequestHandler, "_get_allowed_agents_for_team"
            ) as mock_team:
                mock_key.return_value = RestrictedAgentAccess(frozenset({"key_agent1"}))
                mock_team.return_value = UnrestrictedAgentAccess()

                result = await AgentRequestHandler.resolve_agent_access(
                    user_api_key_auth=mock_user_auth
                )
                assert result == RestrictedAgentAccess(frozenset({"key_agent1"}))

        # Case 4: No grant anywhere - unrestricted (documented open-by-default)
        with patch.object(
            AgentRequestHandler, "_get_allowed_agents_for_key"
        ) as mock_key:
            with patch.object(
                AgentRequestHandler, "_get_allowed_agents_for_team"
            ) as mock_team:
                mock_key.return_value = UnrestrictedAgentAccess()
                mock_team.return_value = UnrestrictedAgentAccess()

                result = await AgentRequestHandler.resolve_agent_access(
                    user_api_key_auth=mock_user_auth
                )
                assert result == UnrestrictedAgentAccess()

    async def test_disjoint_key_and_team_grants_deny_every_agent(self):
        """LIT-5143: a key restricted to one agent inside a team restricted to another
        must reach nothing. The empty intersection used to read as "no restrictions",
        so adding the team grant handed the key every agent on the proxy."""
        mock_user_auth: Final = UserAPIKeyAuth(
            api_key="test-key", user_id="test-user", team_id="test-team"
        )

        with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                mock_key.return_value = RestrictedAgentAccess(frozenset({"agent-alpha"}))
                mock_team.return_value = RestrictedAgentAccess(frozenset({"agent-beta"}))

                assert await AgentRequestHandler.resolve_agent_access(
                    user_api_key_auth=mock_user_auth
                ) == RestrictedAgentAccess(frozenset())

                for agent_id in ("agent-alpha", "agent-beta", "agent-secret"):
                    assert (
                        await AgentRequestHandler.is_agent_allowed(
                            agent_id=agent_id, user_api_key_auth=mock_user_auth
                        )
                        is False
                    ), agent_id

    async def test_empty_access_group_denies_every_agent(self):
        """LIT-5143: a key restricted to an access group that resolves to no agents is
        restricted to nothing, not unrestricted. A failed group lookup still fails open."""
        from litellm.proxy._types import LiteLLM_ObjectPermissionTable

        mock_user_auth: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        mock_user_auth.object_permission = LiteLLM_ObjectPermissionTable(
            object_permission_id="obj-1",
            agents=[],
            agent_access_groups=["group-with-no-agents"],
        )

        with patch.object(
            AgentRequestHandler, "_get_agents_from_access_groups", new_callable=AsyncMock
        ) as mock_groups:
            mock_groups.return_value = []

            assert await AgentRequestHandler._get_allowed_agents_for_key(
                user_api_key_auth=mock_user_auth
            ) == RestrictedAgentAccess(frozenset())

            assert (
                await AgentRequestHandler.is_agent_allowed(
                    agent_id="agent-secret", user_api_key_auth=mock_user_auth
                )
                is False
            )

        with patch.object(
            AgentRequestHandler, "_get_agents_from_access_groups", new_callable=AsyncMock
        ) as mock_groups:
            mock_groups.side_effect = Exception("DB Error")

            assert await AgentRequestHandler._get_allowed_agents_for_key(
                user_api_key_auth=mock_user_auth
            ) == UnrestrictedAgentAccess()

    async def test_is_agent_allowed_respects_permissions(self):
        """
        Test is_agent_allowed: returns True if agent in allowed list or if unrestricted.
        Returns False if agent not in allowed list.
        """
        mock_user_auth = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

        # Agent in allowed list - should be allowed
        with patch.object(
            AgentRequestHandler, "resolve_agent_access"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = RestrictedAgentAccess(
                frozenset({"agent1", "agent2"})
            )
            assert (
                await AgentRequestHandler.is_agent_allowed(
                    agent_id="agent1", user_api_key_auth=mock_user_auth
                )
                is True
            )

        # Agent not in allowed list - should be denied
        with patch.object(
            AgentRequestHandler, "resolve_agent_access"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = RestrictedAgentAccess(
                frozenset({"agent1", "agent2"})
            )
            assert (
                await AgentRequestHandler.is_agent_allowed(
                    agent_id="agent3", user_api_key_auth=mock_user_auth
                )
                is False
            )

        # Restricted to nothing - should deny every agent
        with patch.object(
            AgentRequestHandler, "resolve_agent_access"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = RestrictedAgentAccess(frozenset())
            assert (
                await AgentRequestHandler.is_agent_allowed(
                    agent_id="any_agent", user_api_key_auth=mock_user_auth
                )
                is False
            )

        # Unrestricted - should allow any agent
        with patch.object(
            AgentRequestHandler, "resolve_agent_access"
        ) as mock_get_allowed:
            mock_get_allowed.return_value = UnrestrictedAgentAccess()
            assert (
                await AgentRequestHandler.is_agent_allowed(
                    agent_id="any_agent", user_api_key_auth=mock_user_auth
                )
                is True
            )

    async def test_no_auth_allows_all_agents(self):
        """
        Test that when user_api_key_auth is None, all agents are allowed (no restrictions).
        """
        result = await AgentRequestHandler.resolve_agent_access(user_api_key_auth=None)
        assert result == UnrestrictedAgentAccess()

        is_allowed = await AgentRequestHandler.is_agent_allowed(
            agent_id="any_agent", user_api_key_auth=None
        )
        assert is_allowed is True

    async def test_resolve_agent_access_handles_errors_gracefully(self):
        """
        Test that errors during permission lookup are handled gracefully. This stays
        fail-open for now to preserve existing availability behavior; fail-closed is
        tracked separately.
        """
        mock_user_auth = UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            team_id="test-team",
            object_permission_id="test-permission",
        )

        with patch.object(
            AgentRequestHandler, "_get_allowed_agents_for_key"
        ) as mock_key:
            with patch.object(
                AgentRequestHandler, "_get_allowed_agents_for_team"
            ) as mock_team:
                mock_key.side_effect = Exception("DB Error")
                mock_team.return_value = UnrestrictedAgentAccess()

                result = await AgentRequestHandler.resolve_agent_access(
                    user_api_key_auth=mock_user_auth
                )
                assert result == UnrestrictedAgentAccess()

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
                assert result == RestrictedAgentAccess(
                    frozenset({"agent-from-ag-1", "agent-from-ag-2"})
                )

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
            assert result == RestrictedAgentAccess(
                frozenset({"agent-from-ag", "native-agent-1"})
            )

    async def test_is_agent_allowed_accepts_legacy_config_agent_id_grants(self):
        """LIT-5144: object_permission grants stored under the pre-fix full-entry hash
        must keep authorizing the agent after its id became name-based."""
        entry: Final = {
            "agent_name": "granted-agent",
            "agent_card_params": {
                "name": "Granted Agent",
                "url": "http://localhost",
                "version": "1.0.0",
            },
            "static_headers": {"x-upstream-token": "token-v1"},
        }
        registry: Final = AgentRegistry()
        registry.load_agents_from_config([entry])
        agent: Final = registry.get_agent_by_name("granted-agent")
        assert agent is not None
        legacy_id: Final = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        assert legacy_id != agent.agent_id
        mock_user_auth: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user")

        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            registry,
        ):
            with patch.object(AgentRequestHandler, "resolve_agent_access") as mock_get_allowed:
                for grant, expected in (
                    (RestrictedAgentAccess(frozenset({legacy_id})), True),
                    (RestrictedAgentAccess(frozenset({agent.agent_id})), True),
                    (RestrictedAgentAccess(frozenset({"unrelated-agent-id"})), False),
                    (RestrictedAgentAccess(frozenset()), False),
                    (UnrestrictedAgentAccess(), True),
                ):
                    mock_get_allowed.return_value = grant
                    assert (
                        await AgentRequestHandler.is_agent_allowed(
                            agent_id=agent.agent_id,
                            user_api_key_auth=mock_user_auth,
                        )
                        is expected
                    ), grant

    async def test_resolve_agent_access_intersects_legacy_team_grant_with_stable_key_grant(self):
        """LIT-5144: a team grant stored under the pre-fix full-entry hash and a key grant
        stored under the name-based id name the same agent; the intersection must resolve
        to that agent instead of collapsing to an empty set."""
        entry: Final = {
            "agent_name": "shared-agent",
            "agent_card_params": {
                "name": "Shared Agent",
                "url": "http://localhost",
                "version": "1.0.0",
            },
        }
        registry: Final = AgentRegistry()
        registry.load_agents_from_config([entry])
        agent: Final = registry.get_agent_by_name("shared-agent")
        assert agent is not None
        legacy_id: Final = hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()
        mock_user_auth: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user", team_id="test-team")

        with patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            registry,
        ):
            with patch.object(AgentRequestHandler, "_get_allowed_agents_for_key") as mock_key:
                with patch.object(AgentRequestHandler, "_get_allowed_agents_for_team") as mock_team:
                    for key_grant, team_grant in (
                        (
                            RestrictedAgentAccess(frozenset({agent.agent_id})),
                            RestrictedAgentAccess(frozenset({legacy_id})),
                        ),
                        (
                            RestrictedAgentAccess(frozenset({legacy_id})),
                            RestrictedAgentAccess(frozenset({agent.agent_id})),
                        ),
                        (
                            RestrictedAgentAccess(frozenset({legacy_id})),
                            UnrestrictedAgentAccess(),
                        ),
                    ):
                        mock_key.return_value = key_grant
                        mock_team.return_value = team_grant
                        assert await AgentRequestHandler.resolve_agent_access(
                            user_api_key_auth=mock_user_auth
                        ) == RestrictedAgentAccess(frozenset({agent.agent_id})), (key_grant, team_grant)
