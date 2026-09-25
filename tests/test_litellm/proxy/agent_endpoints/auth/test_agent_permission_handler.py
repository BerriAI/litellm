"""
Unit tests for AgentRequestHandler - Agent permission management for keys and teams.
"""

import hashlib
import json
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.agent_registry import AgentRegistry
from litellm.proxy.agent_endpoints.auth.agent_access_groups import AgentAccessGroupCeiling, CeilingResolver
from litellm.proxy.agent_endpoints.auth.agent_permission_handler import (
    AgentAccess,
    AgentRequestHandler,
    RestrictedAgentAccess,
    UnrestrictedAgentAccess,
    accessible_agents,
)
from litellm.types.agents import AgentCaller


def _registry_with(*agent_names: str) -> AgentRegistry:
    registry: Final = AgentRegistry()
    registry.load_agents_from_config(
        [
            {
                "agent_name": name,
                "agent_card_params": {"name": name, "url": "http://localhost", "version": "1.0.0"},
            }
            for name in agent_names
        ]
    )
    return registry


def _agent_id(registry: AgentRegistry, agent_name: str) -> str:
    agent: Final = registry.get_agent_by_name(agent_name)
    assert agent is not None
    return agent.agent_id


async def _single_context(user_api_key_auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
    return [user_api_key_auth]


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

    @staticmethod
    def _ceiling_resolver(agent_ids: frozenset[str] | None) -> tuple[CeilingResolver, list[str]]:
        """A resolver that records the agent ids it was asked about and answers with a fixed
        ceiling, or None when the agent has no access groups attached."""
        asked: Final[list[str]] = []

        async def resolve(agent_id: str) -> AgentAccessGroupCeiling | None:
            asked.append(agent_id)
            if agent_ids is None:
                return None
            return AgentAccessGroupCeiling(
                access_group_ids=("ag-1",), models=frozenset(), mcp_server_ids=frozenset(), agent_ids=agent_ids
            )

        return resolve, asked

    @staticmethod
    def _key_granting(agent_ids: list[str], agent_id: str | None) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key="test-key",
            user_id="test-user",
            agent_id=agent_id,
            object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="obj-1", agents=agent_ids),
        )

    async def test_agent_access_groups_cap_an_otherwise_unrestricted_key(self):
        """A key with no agent grant of its own may still only reach the agents its
        agent's attached access groups name."""
        agent_key: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user", agent_id="caller-agent")
        resolve, asked = self._ceiling_resolver(frozenset({"agent-beta"}))

        assert await AgentRequestHandler.resolve_agent_access(agent_key, resolve) == RestrictedAgentAccess(
            frozenset({"agent-beta"})
        )
        assert await AgentRequestHandler.is_agent_allowed("agent-beta", agent_key, resolve) is True
        assert await AgentRequestHandler.is_agent_allowed("agent-alpha", agent_key, resolve) is False
        assert asked == ["caller-agent"] * 3

    @staticmethod
    def _team_grants(grants: dict[str, AgentAccess]) -> AsyncMock:
        async def by_team(user_api_key_auth: UserAPIKeyAuth | None = None, *, strict: bool = False) -> AgentAccess:
            assert user_api_key_auth is not None
            return grants.get(user_api_key_auth.team_id or "", UnrestrictedAgentAccess())

        return AsyncMock(side_effect=by_team)

    async def test_agent_key_acting_for_a_user_is_capped_at_the_invoking_teams_agents(self):
        """LIT-8014: the agent's key and access groups reach alpha and beta, but the human who
        invoked it belongs to a team granted only beta, so on their behalf the agent reaches only beta."""
        agent_key: Final = self._key_granting(["agent-alpha", "agent-beta"], agent_id="caller-agent")
        agent_key.agent_caller = AgentCaller(user_id="alice", team_id="callers")
        resolve, _ = self._ceiling_resolver(frozenset({"agent-alpha", "agent-beta", "agent-gamma"}))

        with patch.object(  # test-quality-ok: the team resolver reads proxy_server globals with no injection seam
            AgentRequestHandler,
            "_get_allowed_agents_for_team",
            self._team_grants({"callers": RestrictedAgentAccess(frozenset({"agent-beta", "agent-gamma"}))}),
        ) as mock_team:
            assert await AgentRequestHandler.resolve_agent_access(agent_key, resolve) == RestrictedAgentAccess(
                frozenset({"agent-beta"})
            )
            assert await AgentRequestHandler.is_agent_allowed("agent-alpha", agent_key, resolve) is False

        assert {call.args[0].team_id for call in mock_team.call_args_list} == {None, "callers"}

    async def test_agent_key_acting_for_a_user_whose_team_grants_no_agent_reaches_none(self):
        agent_key: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user", agent_id="caller-agent")
        agent_key.agent_caller = AgentCaller(user_id="alice", team_id="callers")
        resolve, _ = self._ceiling_resolver(None)

        with patch.object(  # test-quality-ok: the team resolver reads proxy_server globals with no injection seam
            AgentRequestHandler,
            "_get_allowed_agents_for_team",
            self._team_grants({"callers": RestrictedAgentAccess(frozenset())}),
        ):
            assert await AgentRequestHandler.resolve_agent_access(agent_key, resolve) == RestrictedAgentAccess(
                frozenset()
            )

    async def test_agent_key_acting_for_an_ungranted_caller_keeps_its_own_agents(self):
        agent_key: Final = self._key_granting(["agent-alpha"], agent_id="caller-agent")
        agent_key.agent_caller = AgentCaller(user_id="alice", team_id="callers")
        resolve, _ = self._ceiling_resolver(None)

        with patch.object(  # test-quality-ok: the team resolver reads proxy_server globals with no injection seam
            AgentRequestHandler, "_get_allowed_agents_for_team", self._team_grants({})
        ):
            assert await AgentRequestHandler.resolve_agent_access(agent_key, resolve) == RestrictedAgentAccess(
                frozenset({"agent-alpha"})
            )

    async def test_agent_access_groups_intersect_with_key_grants(self):
        agent_key: Final = self._key_granting(["agent-alpha", "agent-beta"], agent_id="caller-agent")
        resolve, _ = self._ceiling_resolver(frozenset({"agent-beta", "agent-gamma"}))

        assert await AgentRequestHandler.resolve_agent_access(agent_key, resolve) == RestrictedAgentAccess(
            frozenset({"agent-beta"})
        )
        assert await AgentRequestHandler.is_agent_allowed("agent-gamma", agent_key, resolve) is False

    async def test_agent_access_groups_naming_no_agent_deny_every_agent(self):
        agent_key: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user", agent_id="caller-agent")
        resolve, _ = self._ceiling_resolver(frozenset())

        assert await AgentRequestHandler.resolve_agent_access(agent_key, resolve) == RestrictedAgentAccess(frozenset())
        assert await AgentRequestHandler.is_agent_allowed("agent-alpha", agent_key, resolve) is False

    async def test_agent_without_access_groups_keeps_key_grants(self):
        agent_key: Final = self._key_granting(["agent-alpha"], agent_id="caller-agent")
        resolve, asked = self._ceiling_resolver(None)

        assert await AgentRequestHandler.resolve_agent_access(agent_key, resolve) == RestrictedAgentAccess(
            frozenset({"agent-alpha"})
        )
        assert asked == ["caller-agent"]

    async def test_key_without_agent_never_consults_agent_access_groups(self):
        plain_key: Final = UserAPIKeyAuth(api_key="test-key", user_id="test-user")
        resolve, asked = self._ceiling_resolver(frozenset())

        assert await AgentRequestHandler.resolve_agent_access(plain_key, resolve) == UnrestrictedAgentAccess()
        assert asked == []

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

    async def test_accessible_agents_hides_ungranted_agents_from_non_admins(self):
        """LIT-6862: a key with no agent grant on itself or its team must list nothing,
        while a proxy admin with the same lack of grants still lists every agent."""
        registry: Final = _registry_with("alpha", "beta")
        internal_user: Final = UserAPIKeyAuth(
            api_key="test-key", user_id="alice", team_id="team-no-perms", user_role=LitellmUserRoles.INTERNAL_USER
        )
        proxy_admin: Final = UserAPIKeyAuth(
            api_key="admin-key", user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN
        )

        async def no_grant_anywhere(user_api_key_auth: UserAPIKeyAuth) -> AgentAccess:
            return UnrestrictedAgentAccess()

        assert (
            await accessible_agents(internal_user, registry.get_agent_list(), no_grant_anywhere, _single_context) == ()
        )
        assert {
            agent.agent_name
            for agent in await accessible_agents(
                proxy_admin, registry.get_agent_list(), no_grant_anywhere, _single_context
            )
        } == {"alpha", "beta"}

    async def test_accessible_agents_lists_only_granted_agents(self):
        """A grant for one agent lists that agent and hides the ungranted one."""
        registry: Final = _registry_with("alpha", "beta")
        granted_user: Final = UserAPIKeyAuth(
            api_key="test-key", user_id="bob", team_id="team-granted", user_role=LitellmUserRoles.INTERNAL_USER
        )

        async def alpha_only(user_api_key_auth: UserAPIKeyAuth) -> AgentAccess:
            return RestrictedAgentAccess(frozenset({_agent_id(registry, "alpha")}))

        listed: Final = await accessible_agents(granted_user, registry.get_agent_list(), alpha_only, _single_context)
        assert [agent.agent_name for agent in listed] == ["alpha"]

    async def test_accessible_agents_resolves_dashboard_session_through_real_teams_and_user(self):
        """LIT-6862: a dashboard session carries the shared litellm-dashboard team id, which holds no
        grants. Listing must union the grants of the user's real teams and of the user row instead
        of treating the session as ungranted or as unrestricted."""
        registry: Final = _registry_with("alpha", "beta", "gamma")
        session: Final = UserAPIKeyAuth(
            api_key="session-key",
            user_id="alice",
            team_id=UI_SESSION_TOKEN_TEAM_ID,
            user_role=LitellmUserRoles.INTERNAL_USER,
        )
        admitted_user: Final = UserAPIKeyAuth(user_id="alice", user_role=LitellmUserRoles.INTERNAL_USER)
        grants: Final = {
            "team-granted": RestrictedAgentAccess(frozenset({_agent_id(registry, "alpha")})),
            "team-no-perms": UnrestrictedAgentAccess(),
            UI_SESSION_TOKEN_TEAM_ID: UnrestrictedAgentAccess(),
        }

        async def effective_contexts(user_api_key_auth: UserAPIKeyAuth) -> list[UserAPIKeyAuth]:
            assert user_api_key_auth is session
            return [
                session.model_copy(update={"team_id": "team-granted"}),
                session.model_copy(update={"team_id": "team-no-perms"}),
                admitted_user,
            ]

        async def resolve_access(user_api_key_auth: UserAPIKeyAuth) -> AgentAccess:
            if user_api_key_auth is admitted_user:
                return RestrictedAgentAccess(frozenset({_agent_id(registry, "beta")}))
            assert user_api_key_auth.team_id is not None
            return grants[user_api_key_auth.team_id]

        listed: Final = await accessible_agents(session, registry.get_agent_list(), resolve_access, effective_contexts)
        assert {agent.agent_name for agent in listed} == {"alpha", "beta"}

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "state,allowed",
    [
        ({}, True),
        ({"enabled": False}, False),
        ({"directory_active": False}, False),
        ({"directory_access_group_ids": ()}, False),
    ],
)
async def test_managed_invocation_requires_local_and_directory_admission(
    monkeypatch: pytest.MonkeyPatch, state: dict[str, object], allowed: bool
) -> None:
    from unittest.mock import MagicMock

    from litellm.proxy import proxy_server
    from litellm.types.agents import AgentResponse
    from litellm.types.proxy.agent_identity import AgentIdentityBinding

    binding: Final = AgentIdentityBinding(
        agent_id="target",
        provider="microsoft_entra",
        tenant_id="tenant",
        client_id="client",
        issuer="issuer",
        revision="revision",
    )
    target: Final = AgentResponse(
        agent_id="target", agent_name="Target", agent_card_params={}, identity=binding, identity_managed=True
    ).model_copy(update=state)
    client: Final = MagicMock()
    client.writer_db.litellm_agentstable.find_unique = AsyncMock(return_value=target)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    permission: Final = LiteLLM_ObjectPermissionTable(object_permission_id="human-grant", agents=["target"])
    auth: Final = UserAPIKeyAuth(user_id="human", object_permission=permission)
    assert await AgentRequestHandler.is_agent_allowed("target", auth) is allowed


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [True, False])
async def test_managed_agent_invocation_grants_intersect_verified_user_grants(
    monkeypatch: pytest.MonkeyPatch, delegated: bool
) -> None:
    from unittest.mock import MagicMock

    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_UserTable
    from litellm.proxy.auth import auth_checks
    from litellm.types.agents import AgentResponse
    from litellm.types.proxy.agent_identity import ManagedAgentContext

    database: Final = MagicMock()
    monkeypatch.setattr(proxy_server, "prisma_client", database)
    own: Final = LiteLLM_ObjectPermissionTable(object_permission_id="own", agents=["shared", "agent-only"])
    human_grants: Final = LiteLLM_ObjectPermissionTable(object_permission_id="human", agents=["shared", "human-only"])
    human: Final = LiteLLM_UserTable(user_id="human", teams=[], object_permission=human_grants)
    monkeypatch.setattr(auth_checks, "get_user_object", AsyncMock(return_value=human))
    auth: Final = UserAPIKeyAuth(agent_id="actor")
    auth.managed_agent_policy = AgentResponse(
        agent_id="actor", agent_name="Actor", agent_card_params={}, object_permission=own.model_dump()
    )
    auth.managed_agent_context = ManagedAgentContext(
        agent_id="actor", mode="delegated" if delegated else "autonomous", user_id="human" if delegated else None
    )
    access: Final = await AgentRequestHandler.resolve_agent_access(auth)
    assert access == RestrictedAgentAccess(frozenset({"shared"} if delegated else {"shared", "agent-only"}))


@pytest.mark.asyncio
@pytest.mark.parametrize("revoked", ["user", "team-member", "team-grant", "direct-grant", "access-group"])
async def test_delegated_grants_revoke_with_warm_user_team_and_permission_caches(
    monkeypatch: pytest.MonkeyPatch, revoked: str
) -> None:
    from unittest.mock import MagicMock

    from litellm.proxy import proxy_server
    from litellm.proxy._types import LiteLLM_AccessGroupTable, LiteLLM_TeamTable, LiteLLM_UserTable
    from litellm.proxy.agent_endpoints.auth.agent_permission_handler import verified_human_agent_grants
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, object_permission_cache_key

    direct: Final = revoked == "direct-grant"
    grouped: Final = revoked == "access-group"
    permission: Final = LiteLLM_ObjectPermissionTable(object_permission_id="grant", agents=["target"])
    human: Final = LiteLLM_UserTable(
        user_id="human",
        teams=[] if direct else ["team"],
        organization_memberships=[],
        object_permission_id="grant" if direct else None,
    )
    team: Final = LiteLLM_TeamTable(
        team_id="team",
        models=[],
        members_with_roles=[{"user_id": "human", "role": "user"}],
        object_permission=None if grouped else permission,
        access_group_ids=["group"] if grouped else [],
    )
    group: Final = LiteLLM_AccessGroupTable(
        access_group_id="group", access_group_name="Group", access_agent_ids=["target"]
    )
    cache: Final = UserApiKeyCache()
    cache.set_cache("human", human)
    cache.set_cache("team_id:team", team)
    cache.set_cache(object_permission_cache_key("grant"), permission)
    cache.set_cache("access_group_id:group", group)
    client: Final = MagicMock()
    client.writer_db.litellm_usertable.find_unique = AsyncMock(return_value=human)
    client.writer_db.litellm_teamtable.find_unique = AsyncMock(return_value=team)
    client.writer_db.litellm_objectpermissiontable.find_unique = AsyncMock(return_value=permission)
    client.writer_db.litellm_accessgrouptable.find_unique = AsyncMock(return_value=group)
    monkeypatch.setattr(proxy_server, "prisma_client", client)
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    assert await verified_human_agent_grants("human") == frozenset({"target"})
    client.writer_db.litellm_usertable.find_unique.return_value = (
        human.model_copy(update={"teams": []}) if revoked == "user" else human
    )
    client.writer_db.litellm_teamtable.find_unique.return_value = (
        team.model_copy(update={"members_with_roles": []})
        if revoked == "team-member"
        else team.model_copy(update={"object_permission": None})
        if revoked == "team-grant"
        else team
    )
    client.writer_db.litellm_objectpermissiontable.find_unique.return_value = (
        permission.model_copy(update={"agents": []}) if direct else permission
    )
    client.writer_db.litellm_accessgrouptable.find_unique.return_value = (
        group.model_copy(update={"access_agent_ids": []}) if grouped else group
    )
    assert await verified_human_agent_grants("human") == frozenset()
    client.db.litellm_usertable.find_unique.assert_not_called()
    client.db.litellm_teamtable.find_unique.assert_not_called()
    client.db.litellm_objectpermissiontable.find_unique.assert_not_called()
    client.db.litellm_accessgrouptable.find_unique.assert_not_called()
