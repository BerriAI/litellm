from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints import endpoints as agent_endpoints
from litellm.proxy.agent_endpoints.endpoints import (
    _attach_keys_to_agents,
    _check_agent_management_permission,
    get_agent_daily_activity,
    router,
    user_api_key_auth,
)
from litellm.types.agents import AgentResponse


def _sample_agent_card_params() -> dict:
    return {
        "protocolVersion": "1.0",
        "name": "Test Agent",
        "description": "desc",
        "url": "http://localhost",
        "version": "1.0.0",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }


def _sample_agent_config() -> dict:
    return {
        "agent_name": "Test Agent",
        "agent_card_params": _sample_agent_card_params(),
        "litellm_params": {"make_public": False},
    }


def _sample_agent_response(agent_id: str = "agent-123", agent_name: str = "Test Agent") -> AgentResponse:
    return AgentResponse(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_card_params=_sample_agent_card_params(),
        litellm_params={"make_public": False},
    )


def _make_app_with_role(role: LitellmUserRoles) -> TestClient:
    """Create a TestClient where the auth dependency returns the given role."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(user_id="test-user", user_role=role)
    return TestClient(test_app)


app = FastAPI()
app.include_router(router)
app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
    user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN
)
client = TestClient(app)


@pytest.fixture
def mock_prisma_client():
    with patch("litellm.proxy.proxy_server.prisma_client") as mock:
        yield mock


@pytest.fixture
def mock_user_api_key_auth():
    with patch("litellm.proxy.agent_endpoints.endpoints.user_api_key_auth") as mock:
        mock.return_value = UserAPIKeyAuth(user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN)
        yield mock


def test_update_agent_success(mock_prisma_client, mock_user_api_key_auth, monkeypatch):
    existing_agent = {
        "agent_id": "agent-123",
        "agent_name": "Existing Agent",
        "agent_card_params": _sample_agent_card_params(),
    }
    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=existing_agent)

    mock_registry = MagicMock()
    mock_registry.update_agent_in_db = AsyncMock(return_value=_sample_agent_response(agent_id="agent-123"))
    mock_registry.deregister_agent = MagicMock()
    mock_registry.register_agent = MagicMock()
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)

    response = client.put(
        "/v1/agents/agent-123",
        json=_sample_agent_config(),
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 200
    assert response.json()["agent_id"] == "agent-123"
    assert response.json()["agent_name"] == "Test Agent"


def test_update_agent_not_found(mock_prisma_client, mock_user_api_key_auth, monkeypatch):
    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

    mock_registry = MagicMock()
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)

    response = client.put(
        "/v1/agents/missing-agent",
        json=_sample_agent_config(),
        headers={"Authorization": "Bearer test-key"},
    )

    assert response.status_code == 404
    assert "Agent with ID missing-agent not found" in response.json()["detail"]


def test_get_agent_by_id_not_found(mock_prisma_client, mock_user_api_key_auth, monkeypatch):
    mock_registry = MagicMock()
    mock_registry.get_agent_by_id = MagicMock(return_value=None)
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)
    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

    response = client.get("/v1/agents/missing-agent", headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 404
    assert "Agent with ID missing-agent not found" in response.json()["detail"]


def test_delete_agent_not_found(mock_prisma_client, mock_user_api_key_auth, monkeypatch):
    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
    mock_registry = MagicMock()
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)

    response = client.delete("/v1/agents/missing-agent", headers={"Authorization": "Bearer test-key"})

    assert response.status_code == 404
    assert "Agent with ID missing-agent not found in DB." in response.json()["detail"]


def test_agent_error_schema_consistency(mock_prisma_client, mock_user_api_key_auth, monkeypatch):
    mock_registry = MagicMock()
    mock_registry.get_agent_by_id = MagicMock(return_value=None)
    mock_registry.update_agent_in_db = AsyncMock(side_effect=Exception("should not run"))
    mock_registry.delete_agent_from_db = AsyncMock(side_effect=Exception("should not run"))
    monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", mock_registry)

    mock_prisma_client.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)

    missing_agent_id = "missing-agent"
    responses = [
        client.get(
            f"/v1/agents/{missing_agent_id}",
            headers={"Authorization": "Bearer test-key"},
        ),
        client.put(
            f"/v1/agents/{missing_agent_id}",
            json=_sample_agent_config(),
            headers={"Authorization": "Bearer test-key"},
        ),
        client.delete(
            f"/v1/agents/{missing_agent_id}",
            headers={"Authorization": "Bearer test-key"},
        ),
    ]

    for resp in responses:
        assert resp.status_code == 404
        detail = resp.json()["detail"]
        assert isinstance(detail, str)
        assert missing_agent_id in detail


@pytest.mark.asyncio
async def test_get_agent_daily_activity_admin_param_passing(monkeypatch):
    mock_prisma = AsyncMock()
    mock_prisma.db.litellm_agentstable.find_many = AsyncMock(return_value=[])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(agent_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    result = await get_agent_daily_activity(
        agent_ids="agent-1,agent-2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model="gpt-4",
        api_key="test-key",
        page=2,
        page_size=5,
        exclude_agent_ids="agent-3",
        user_api_key_dict=auth,
    )

    get_daily_activity_mock.assert_awaited_once()
    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["table_name"] == "litellm_dailyagentspend"
    assert kwargs["entity_id_field"] == "agent_id"
    assert kwargs["entity_id"] == ["agent-1", "agent-2"]
    assert kwargs["exclude_entity_ids"] == ["agent-3"]
    assert kwargs["start_date"] == "2024-01-01"
    assert kwargs["end_date"] == "2024-01-31"
    assert kwargs["model"] == "gpt-4"
    assert kwargs["api_key"] == "test-key"
    assert kwargs["page"] == 2
    assert kwargs["page_size"] == 5
    assert result is mocked_response


@pytest.mark.asyncio
async def test_get_agent_daily_activity_with_agent_names(monkeypatch):
    mock_prisma = AsyncMock()
    mock_agent1 = MagicMock()
    mock_agent1.agent_id = "agent-1"
    mock_agent1.agent_name = "First Agent"
    mock_agent2 = MagicMock()
    mock_agent2.agent_id = "agent-2"
    mock_agent2.agent_name = "Second Agent"

    mock_prisma.db.litellm_agentstable.find_many = AsyncMock(return_value=[mock_agent1, mock_agent2])
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", mock_prisma)

    mocked_response = MagicMock(name="SpendAnalyticsPaginatedResponse")
    get_daily_activity_mock = AsyncMock(return_value=mocked_response)
    monkeypatch.setattr(agent_endpoints, "get_daily_activity", get_daily_activity_mock)

    auth = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN, user_id="admin1")
    await get_agent_daily_activity(
        agent_ids="agent-1,agent-2",
        start_date="2024-01-01",
        end_date="2024-01-31",
        model=None,
        api_key=None,
        page=1,
        page_size=10,
        exclude_agent_ids=None,
        user_api_key_dict=auth,
    )

    kwargs = get_daily_activity_mock.call_args.kwargs
    assert kwargs["entity_metadata_field"] == {
        "agent-1": {"agent_name": "First Agent"},
        "agent-2": {"agent_name": "Second Agent"},
    }


@pytest.mark.asyncio
async def test_attach_keys_to_agents_groups_by_agent_and_omits_secret():
    """
    The agents response must carry each agent's attached virtual keys (derived
    from the key table's agent_id FK), grouped per agent, exposing only
    non-secret summary fields. Agents with no key get None so the UI renders
    "Needs Setup" rather than a stale badge.
    """

    class _Row:
        def __init__(self, token, agent_id, key_alias, key_name):
            self.token = token
            self.agent_id = agent_id
            self.key_alias = key_alias
            self.key_name = key_name
            self.user_id = "secret-owner"  # extra field that must NOT leak

    agent_with_keys = _sample_agent_response(agent_id="agent-1")
    agent_without_keys = _sample_agent_response(agent_id="agent-2")

    mock_prisma = MagicMock()
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(
        return_value=[
            _Row("hash-aaa", "agent-1", "primary", "sk-...aaa"),
            _Row("hash-bbb", "agent-1", "backup", "sk-...bbb"),
        ]
    )

    await _attach_keys_to_agents([agent_with_keys, agent_without_keys], mock_prisma)

    # Query is scoped to the agents being returned, not the whole key table.
    where = mock_prisma.db.litellm_verificationtoken.find_many.call_args.kwargs["where"]
    assert where == {"agent_id": {"in": ["agent-1", "agent-2"]}}

    # agent-1 gets both of its keys; agent-2 gets None.
    assert agent_without_keys.keys is None
    assert agent_with_keys.keys is not None
    assert {k.token for k in agent_with_keys.keys} == {"hash-aaa", "hash-bbb"}
    assert {k.key_alias for k in agent_with_keys.keys} == {"primary", "backup"}

    # Only summary fields are exposed; the row's user_id must not be carried.
    summary = agent_with_keys.keys[0]
    assert set(summary.model_dump().keys()) == {"token", "key_alias", "key_name"}


class TestAgentByIdKeyRedaction:
    """GET /v1/agents/{id} surfaces attached keys to admins but never to
    non-admins, even when the agent has keys attached."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.mock_registry = MagicMock()
        self.mock_registry.get_agent_by_id = MagicMock(return_value=_sample_agent_response())
        monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", self.mock_registry)
        # Redaction, not ACL, is under test here: make the agent visible to a
        # non-admin (public) so the read is allowed and we exercise key redaction.
        monkeypatch.setattr(litellm, "public_agent_groups", ["agent-123"])

    def _get_as(self, role: LitellmUserRoles):
        key_row = MagicMock()
        key_row.token = "hash-aaa"
        key_row.agent_id = "agent-123"
        key_row.key_alias = "primary"
        key_row.key_name = "sk-...aaa"

        test_client = _make_app_with_role(role)
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
            mock_prisma.db.litellm_agentstable.find_many = AsyncMock(return_value=[])
            mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[key_row])
            return test_client.get("/v1/agents/agent-123", headers={"Authorization": "Bearer k"})

    def test_admin_sees_attached_keys(self):
        resp = self._get_as(LitellmUserRoles.PROXY_ADMIN)
        assert resp.status_code == 200
        keys = resp.json()["keys"]
        assert keys is not None
        assert keys[0] == {
            "token": "hash-aaa",
            "key_alias": "primary",
            "key_name": "sk-...aaa",
        }

    def test_non_admin_never_sees_keys(self):
        resp = self._get_as(LitellmUserRoles.INTERNAL_USER)
        assert resp.status_code == 200
        assert resp.json()["keys"] is None


# ---------- RBAC enforcement tests ----------


class TestAgentRBACInternalUser:
    """Internal users should be able to read agents but not create/update/delete."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.internal_client = _make_app_with_role(LitellmUserRoles.INTERNAL_USER)
        self.mock_registry = MagicMock()
        monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", self.mock_registry)

    def test_should_allow_internal_user_to_list_agents(self, monkeypatch):
        self.mock_registry.get_agent_list = MagicMock(return_value=[])
        resp = self.internal_client.get("/v1/agents", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200

    def test_should_allow_internal_user_to_get_agent_by_id(self, monkeypatch):
        """Internal user can read an agent that has been granted to them.

        Note: post-S3-01, simply being authenticated is no longer enough;
        the agent must be in explicit grants, public, or owned. Here we mark
        agent-123 as public so the test asserts the RBAC-allowed path.
        """
        monkeypatch.setattr("litellm.public_agent_groups", ["agent-123"])
        self.mock_registry.get_agent_by_id = MagicMock(return_value=_sample_agent_response())
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
            mock_prisma.db.litellm_agentstable.find_many = AsyncMock(return_value=[])
            mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
            resp = self.internal_client.get("/v1/agents/agent-123", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200

    def test_should_block_internal_user_from_creating_agent(self):
        resp = self.internal_client.post(
            "/v1/agents",
            json=_sample_agent_config(),
            headers={"Authorization": "Bearer k"},
        )
        assert resp.status_code == 403
        assert "Only proxy admins" in resp.json()["detail"]["error"]

    def test_should_block_internal_user_from_updating_agent(self):
        resp = self.internal_client.put(
            "/v1/agents/agent-123",
            json=_sample_agent_config(),
            headers={"Authorization": "Bearer k"},
        )
        assert resp.status_code == 403

    def test_should_block_internal_user_from_patching_agent(self):
        resp = self.internal_client.patch(
            "/v1/agents/agent-123",
            json={"agent_name": "new-name"},
            headers={"Authorization": "Bearer k"},
        )
        assert resp.status_code == 403

    def test_should_block_internal_user_from_deleting_agent(self):
        resp = self.internal_client.delete("/v1/agents/agent-123", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 403


class TestAgentRBACInternalUserViewOnly:
    """View-only internal users should only be able to read agents."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.viewer_client = _make_app_with_role(LitellmUserRoles.INTERNAL_USER_VIEW_ONLY)
        self.mock_registry = MagicMock()
        monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", self.mock_registry)

    def test_should_allow_view_only_user_to_list_agents(self):
        self.mock_registry.get_agent_list = MagicMock(return_value=[])
        resp = self.viewer_client.get("/v1/agents", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200

    def test_should_block_view_only_user_from_creating_agent(self):
        resp = self.viewer_client.post(
            "/v1/agents",
            json=_sample_agent_config(),
            headers={"Authorization": "Bearer k"},
        )
        assert resp.status_code == 403

    def test_should_block_view_only_user_from_deleting_agent(self):
        resp = self.viewer_client.delete("/v1/agents/agent-123", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 403


class TestAgentReadACLNonAdmin:
    """
    Read-ACL enforcement on GET /v1/agents and GET /v1/agents/{id}.

    Regression for S3-01: non-admin callers with no explicit agent grants must
    NOT silently see every agent in the registry. They should see only:
    explicit grants ∪ public agents ∪ agents they created.
    """

    AGENT_GRANTED = "agent-granted"
    AGENT_PUBLIC = "agent-public"
    AGENT_OTHER = "agent-other"
    AGENT_OWNED = "agent-owned"

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.client = _make_app_with_role(LitellmUserRoles.INTERNAL_USER)
        self.mock_registry = MagicMock()
        # Registry holds four agents; the ACL decides which are visible.
        self.mock_registry.get_agent_list = MagicMock(
            return_value=[
                _sample_agent_response(agent_id=self.AGENT_GRANTED, agent_name="Granted"),
                _sample_agent_response(agent_id=self.AGENT_PUBLIC, agent_name="Public"),
                _sample_agent_response(agent_id=self.AGENT_OTHER, agent_name="Other"),
                _sample_agent_response(agent_id=self.AGENT_OWNED, agent_name="Owned"),
            ]
        )
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            self.mock_registry,
        )
        monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", self.mock_registry)
        # Mark exactly one agent public.
        monkeypatch.setattr("litellm.public_agent_groups", [self.AGENT_PUBLIC])

    def _patch_owned(self, owned_ids):
        owned_records = [MagicMock(agent_id=a) for a in owned_ids]
        prisma = MagicMock()
        prisma.db.litellm_agentstable.find_many = AsyncMock(return_value=owned_records)
        prisma.db.litellm_agentstable.find_unique = AsyncMock(return_value=None)
        prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
        return patch("litellm.proxy.proxy_server.prisma_client", prisma)

    # -- list endpoint -------------------------------------------------------

    def test_list_returns_only_explicit_grants_plus_public_plus_owned(self, monkeypatch):
        """The happy path: explicit grants resolve, plus public, plus owned."""
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
            AsyncMock(return_value=[self.AGENT_GRANTED]),
        )
        with self._patch_owned([self.AGENT_OWNED]):
            resp = self.client.get("/v1/agents", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200
        ids = {a["agent_id"] for a in resp.json()}
        assert ids == {self.AGENT_GRANTED, self.AGENT_PUBLIC, self.AGENT_OWNED}
        assert self.AGENT_OTHER not in ids

    def test_list_with_no_grants_returns_only_public_and_owned(self, monkeypatch):
        """
        Branch 3 from CLAUDE.md: name does not resolve at all.
        Empty allowed_agent_ids must NOT silently fall back to "all agents".
        """
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
            AsyncMock(return_value=[]),
        )
        with self._patch_owned([self.AGENT_OWNED]):
            resp = self.client.get("/v1/agents", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200
        ids = {a["agent_id"] for a in resp.json()}
        # Critically: AGENT_OTHER and AGENT_GRANTED must NOT leak through.
        assert ids == {self.AGENT_PUBLIC, self.AGENT_OWNED}

    def test_list_with_no_grants_and_no_owned_returns_only_public(self, monkeypatch):
        """Unscoped caller with no owned agents sees only public agents."""
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
            AsyncMock(return_value=[]),
        )
        with self._patch_owned([]):
            resp = self.client.get("/v1/agents", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200
        ids = {a["agent_id"] for a in resp.json()}
        assert ids == {self.AGENT_PUBLIC}

    # -- detail endpoint -----------------------------------------------------

    def test_get_by_id_allowed_when_explicit_grant(self, monkeypatch):
        """Branch 1: name resolves and UUID is allowed -> 200."""
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
            AsyncMock(return_value=[self.AGENT_GRANTED]),
        )
        self.mock_registry.get_agent_by_id = MagicMock(return_value=_sample_agent_response(agent_id=self.AGENT_GRANTED))
        with self._patch_owned([]):
            resp = self.client.get(
                f"/v1/agents/{self.AGENT_GRANTED}",
                headers={"Authorization": "Bearer k"},
            )
        assert resp.status_code == 200

    def test_get_by_id_blocked_when_grant_does_not_cover_target(self, monkeypatch):
        """
        Branch 2: name resolves but UUID is not allowed -> 403.
        Even with a non-empty allowed_agent_ids, an agent outside the set is blocked.
        """
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
            AsyncMock(return_value=[self.AGENT_GRANTED]),
        )
        with self._patch_owned([]):
            resp = self.client.get(
                f"/v1/agents/{self.AGENT_OTHER}",
                headers={"Authorization": "Bearer k"},
            )
        assert resp.status_code == 403

    def test_get_by_id_blocked_when_no_grants_at_all(self, monkeypatch):
        """
        Branch 3 (the silent-fallback bug): empty allowed list + agent not
        public/owned -> 403. Pre-fix, this returned 200 because is_agent_allowed
        defaulted to True on empty list.
        """
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
            AsyncMock(return_value=[]),
        )
        with self._patch_owned([]):
            resp = self.client.get(
                f"/v1/agents/{self.AGENT_OTHER}",
                headers={"Authorization": "Bearer k"},
            )
        assert resp.status_code == 403

    def test_get_by_id_allowed_for_public_agent_without_grants(self, monkeypatch):
        """Public agents are reachable even without explicit grants."""
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
            AsyncMock(return_value=[]),
        )
        self.mock_registry.get_agent_by_id = MagicMock(return_value=_sample_agent_response(agent_id=self.AGENT_PUBLIC))
        with self._patch_owned([]):
            resp = self.client.get(
                f"/v1/agents/{self.AGENT_PUBLIC}",
                headers={"Authorization": "Bearer k"},
            )
        assert resp.status_code == 200


class TestAgentRBACProxyAdmin:
    """Proxy admins should have full CRUD access to agents."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        self.admin_client = _make_app_with_role(LitellmUserRoles.PROXY_ADMIN)
        self.mock_registry = MagicMock()
        monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", self.mock_registry)

    def test_should_allow_admin_to_create_agent(self, monkeypatch):
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            self.mock_registry.get_agent_by_name = MagicMock(return_value=None)
            self.mock_registry.add_agent_to_db = AsyncMock(return_value=_sample_agent_response())
            self.mock_registry.register_agent = MagicMock()
            resp = self.admin_client.post(
                "/v1/agents",
                json=_sample_agent_config(),
                headers={"Authorization": "Bearer k"},
            )
            assert resp.status_code == 200

    def test_create_agent_applies_litellm_merge_to_stored_card(self):
        """The card stored in the DB must reflect the LiteLLM-fronting merge."""
        with patch("litellm.proxy.proxy_server.prisma_client"):
            self.mock_registry.get_agent_by_name = MagicMock(return_value=None)
            self.mock_registry.add_agent_to_db = AsyncMock(return_value=_sample_agent_response())
            self.mock_registry.register_agent = MagicMock()

            self.admin_client.post(
                "/v1/agents",
                json=_sample_agent_config(),
                headers={"Authorization": "Bearer k"},
            )

            call_kwargs = self.mock_registry.add_agent_to_db.await_args.kwargs
            stored_card = call_kwargs["agent"]["agent_card_params"]
            new_agent_id = call_kwargs["agent_id"]

            # Top-level url is retained for runtime A2A invocation (the public
            # well-known endpoint rewrites it before exposing to clients);
            # supportedInterfaces points at the proxy.
            assert stored_card["url"] == "http://localhost"
            assert stored_card["supportedInterfaces"][0]["protocolBinding"] == "JSONRPC"
            assert stored_card["supportedInterfaces"][0]["url"].endswith(f"/a2a/{new_agent_id}")
            # Security scheme is the LiteLLM scheme.
            assert "LiteLLMKey" in stored_card["securitySchemes"]

    def test_should_allow_admin_to_delete_agent(self):
        existing = {
            "agent_id": "agent-123",
            "agent_name": "Existing Agent",
            "agent_card_params": _sample_agent_card_params(),
        }
        with patch("litellm.proxy.proxy_server.prisma_client") as mock_prisma:
            mock_prisma.db.litellm_agentstable.find_unique = AsyncMock(return_value=existing)
            self.mock_registry.delete_agent_from_db = AsyncMock()
            self.mock_registry.deregister_agent = MagicMock()
            resp = self.admin_client.delete("/v1/agents/agent-123", headers={"Authorization": "Bearer k"})
            assert resp.status_code == 200


class TestCheckAgentManagementPermission:
    """Unit tests for the _check_agent_management_permission helper."""

    def test_should_allow_proxy_admin(self):
        auth = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
        _check_agent_management_permission(auth)

    @pytest.mark.parametrize(
        "role",
        [
            LitellmUserRoles.INTERNAL_USER,
            LitellmUserRoles.INTERNAL_USER_VIEW_ONLY,
            LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY,
        ],
    )
    def test_should_block_non_admin_roles(self, role):
        from fastapi import HTTPException

        auth = UserAPIKeyAuth(user_id="user", user_role=role)
        with pytest.raises(HTTPException) as exc_info:
            _check_agent_management_permission(auth)
        assert exc_info.value.status_code == 403


class TestAgentRoutesIncludesAgentIdPattern:
    """Verify that agent_routes includes the {agent_id} pattern for route access."""

    def test_should_include_agent_id_pattern(self):
        from litellm.proxy._types import LiteLLMRoutes

        assert "/v1/agents/{agent_id}" in LiteLLMRoutes.agent_routes.value


class TestAgentHealthCheck:
    """Tests for the health_check query parameter on GET /v1/agents."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        from litellm.proxy.agent_endpoints import agent_registry as ar_mod

        self.admin_client = _make_app_with_role(LitellmUserRoles.PROXY_ADMIN)
        self.mock_registry = MagicMock()
        monkeypatch.setattr(ar_mod, "global_agent_registry", self.mock_registry)
        # Ensure prisma_client is None so the endpoint skips DB queries.
        # In CI with parallel workers, a MagicMock can leak from other test
        # scopes, causing "object MagicMock can't be used in 'await'" errors.
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", None)

    def _make_agent(self, agent_id: str, url: str | None = None) -> AgentResponse:
        card = _sample_agent_card_params()
        if url is not None:
            card["url"] = url
        else:
            card.pop("url", None)
        return AgentResponse(
            agent_id=agent_id,
            agent_name=f"Agent {agent_id}",
            agent_card_params=card,
            litellm_params={},
        )

    def test_should_return_all_agents_when_health_check_disabled(self):
        agents = [
            self._make_agent("a1", "http://reachable"),
            self._make_agent("a2", "http://unreachable"),
        ]
        self.mock_registry.get_agent_list = MagicMock(return_value=agents)

        resp = self.admin_client.get("/v1/agents", headers={"Authorization": "Bearer k"})
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_should_filter_unhealthy_agents_when_health_check_enabled(self, monkeypatch):
        agents = [
            self._make_agent("a1", "http://reachable"),
            self._make_agent("a2", "http://unreachable"),
        ]
        self.mock_registry.get_agent_list = MagicMock(return_value=agents)

        results = iter(
            [
                {"agent_id": "a1", "healthy": True},
                {"agent_id": "a2", "healthy": False, "error": "Connection refused"},
            ]
        )
        monkeypatch.setattr(
            agent_endpoints,
            "_check_agent_url_health",
            AsyncMock(side_effect=lambda agent: next(results)),
        )

        resp = self.admin_client.get(
            "/v1/agents?health_check=true",
            headers={"Authorization": "Bearer k"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["agent_id"] == "a1"

    def test_should_return_empty_list_when_all_agents_unhealthy(self, monkeypatch):
        agents = [self._make_agent("a1", "http://down")]
        self.mock_registry.get_agent_list = MagicMock(return_value=agents)
        monkeypatch.setattr(
            agent_endpoints,
            "_check_agent_url_health",
            AsyncMock(return_value={"agent_id": "a1", "healthy": False, "error": "timeout"}),
        )

        resp = self.admin_client.get(
            "/v1/agents?health_check=true",
            headers={"Authorization": "Bearer k"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 0

    def test_should_return_all_agents_when_all_healthy(self, monkeypatch):
        agents = [
            self._make_agent("a1", "http://ok1"),
            self._make_agent("a2", "http://ok2"),
        ]
        self.mock_registry.get_agent_list = MagicMock(return_value=agents)

        results = iter(
            [
                {"agent_id": "a1", "healthy": True},
                {"agent_id": "a2", "healthy": True},
            ]
        )
        monkeypatch.setattr(
            agent_endpoints,
            "_check_agent_url_health",
            AsyncMock(side_effect=lambda agent: next(results)),
        )

        resp = self.admin_client.get(
            "/v1/agents?health_check=true",
            headers={"Authorization": "Bearer k"},
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2


class TestCheckAgentUrlHealth:
    """Unit tests for the _check_agent_url_health helper."""

    @pytest.mark.asyncio
    async def test_should_return_healthy_when_no_url(self):
        from litellm.proxy.agent_endpoints.endpoints import _check_agent_url_health

        agent = AgentResponse(
            agent_id="no-url",
            agent_name="No URL Agent",
            agent_card_params={"name": "test"},
            litellm_params={},
        )
        result = await _check_agent_url_health(agent)
        assert result["healthy"] is True
        assert "error" not in result

    @pytest.mark.asyncio
    @patch("litellm.proxy.agent_endpoints.endpoints.get_async_httpx_client")
    async def test_should_return_healthy_for_200(self, mock_get_client):
        from litellm.proxy.agent_endpoints.endpoints import _check_agent_url_health

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        agent = AgentResponse(
            agent_id="ok",
            agent_name="OK Agent",
            agent_card_params={"url": "http://example.com"},
            litellm_params={},
        )
        result = await _check_agent_url_health(agent)
        assert result["healthy"] is True

    @pytest.mark.asyncio
    @patch("litellm.proxy.agent_endpoints.endpoints.get_async_httpx_client")
    async def test_should_return_unhealthy_for_500(self, mock_get_client):
        from litellm.proxy.agent_endpoints.endpoints import _check_agent_url_health

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        agent = AgentResponse(
            agent_id="err",
            agent_name="Error Agent",
            agent_card_params={"url": "http://failing.com"},
            litellm_params={},
        )
        result = await _check_agent_url_health(agent)
        assert result["healthy"] is False
        assert "HTTP 500" in result["error"]

    @pytest.mark.asyncio
    @patch("litellm.proxy.agent_endpoints.endpoints.get_async_httpx_client")
    async def test_should_return_unhealthy_on_connection_error(self, mock_get_client):
        from litellm.proxy.agent_endpoints.endpoints import _check_agent_url_health

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
        mock_get_client.return_value = mock_client

        agent = AgentResponse(
            agent_id="down",
            agent_name="Down Agent",
            agent_card_params={"url": "http://down.com"},
            litellm_params={},
        )
        result = await _check_agent_url_health(agent)
        assert result["healthy"] is False
        assert "Connection refused" in result["error"]

    @pytest.mark.asyncio
    @patch("litellm.proxy.agent_endpoints.endpoints.get_async_httpx_client")
    async def test_should_treat_404_as_healthy(self, mock_get_client):
        """A 404 means the server is reachable, just not the specific path."""
        from litellm.proxy.agent_endpoints.endpoints import _check_agent_url_health

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_get_client.return_value = mock_client

        agent = AgentResponse(
            agent_id="notfound",
            agent_name="NotFound Agent",
            agent_card_params={"url": "http://example.com/missing"},
            litellm_params={},
        )
        result = await _check_agent_url_health(agent)
        assert result["healthy"] is True


# ============================================================================
# S3-02: list filters (q / category / tag / supports_streaming / is_public)
# ============================================================================


class TestListAgentsFilters:
    """Each filter narrows the registry as expected; cursor pagination works."""

    def _make_agent(self, agent_id, **card):
        return _sample_agent_response(agent_id=agent_id, agent_name=card.get("name", agent_id))

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        # Build a small registry where each card has distinct attributes.
        def _a(aid, name, description="", category=None, tag=None, streaming=False):
            base = AgentResponse(
                agent_id=aid,
                agent_name=name,
                agent_card_params={
                    "description": description,
                    "categories": [category] if category else [],
                    "tags": [tag] if tag else [],
                    "capabilities": {"streaming": streaming},
                },
                litellm_params={},
            )
            return base

        self.registry = MagicMock()
        self.registry.get_agent_list.return_value = [
            _a(
                "a-1",
                "alpha",
                description="research helper",
                category="research",
                tag="science",
            ),
            _a(
                "a-2",
                "beta",
                description="writing assistant",
                category="writing",
                streaming=True,
            ),
            _a(
                "a-3",
                "gamma",
                description="another research thing",
                category="research",
            ),
        ]
        monkeypatch.setattr(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            self.registry,
        )
        monkeypatch.setattr(agent_endpoints, "AGENT_REGISTRY", self.registry)
        monkeypatch.setattr(litellm, "public_agent_groups", [])
        # Prisma stub for spend lookup + visibility helpers.
        prisma = MagicMock()
        prisma.db.litellm_agentstable.find_many = AsyncMock(return_value=[])
        prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
        monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", prisma)
        self.client = _make_app_with_role(LitellmUserRoles.PROXY_ADMIN)

    def test_q_filter_matches_name_and_description(self):
        resp = self.client.get("/v1/agents?q=research", headers={"Authorization": "Bearer k"})
        ids = {a["agent_id"] for a in resp.json()}
        assert ids == {"a-1", "a-3"}

    def test_category_filter(self):
        resp = self.client.get("/v1/agents?category=writing", headers={"Authorization": "Bearer k"})
        ids = {a["agent_id"] for a in resp.json()}
        assert ids == {"a-2"}

    def test_supports_streaming_filter(self):
        resp = self.client.get(
            "/v1/agents?supports_streaming=true",
            headers={"Authorization": "Bearer k"},
        )
        ids = {a["agent_id"] for a in resp.json()}
        assert ids == {"a-2"}

    def test_cursor_pagination(self):
        # First page of size 1 -> a-1 (sorted by agent_id).
        r1 = self.client.get("/v1/agents?limit=1", headers={"Authorization": "Bearer k"})
        body1 = r1.json()
        assert [a["agent_id"] for a in body1] == ["a-1"]
        # Next page using a-1 as cursor -> a-2.
        r2 = self.client.get("/v1/agents?limit=1&cursor=a-1", headers={"Authorization": "Bearer k"})
        assert [a["agent_id"] for a in r2.json()] == ["a-2"]


# ============================================================================
# S3-05: per-agent health check config
# ============================================================================


@pytest.mark.asyncio
async def test_health_check_disabled_agent_is_healthy_without_network():
    """An agent with health_check_enabled=False should not be hit at all."""
    from litellm.proxy.agent_endpoints.endpoints import _check_agent_url_health

    agent = _sample_agent_response()
    agent.agent_card_params = {
        "url": "http://this-should-not-be-called/",
        "health_check_enabled": False,
    }
    with patch("litellm.proxy.agent_endpoints.endpoints.get_async_httpx_client") as get_client:
        result = await _check_agent_url_health(agent)
    assert result["healthy"] is True
    assert result.get("skipped") is True
    get_client.assert_not_called()


@pytest.mark.asyncio
async def test_health_check_uses_per_agent_timeout(monkeypatch):
    """health_check_timeout_ms overrides the module default."""
    from litellm.proxy.agent_endpoints.endpoints import _check_agent_url_health

    agent = _sample_agent_response()
    agent.agent_card_params = {
        "url": "http://example/",
        "health_check_timeout_ms": 1500,
    }

    captured = {}

    def _fake_client(*, llm_provider, params):
        captured["timeout"] = params["timeout"]
        client = MagicMock()
        client.get = AsyncMock(return_value=MagicMock(status_code=200))
        return client

    monkeypatch.setattr(
        "litellm.proxy.agent_endpoints.endpoints.get_async_httpx_client",
        _fake_client,
    )
    result = await _check_agent_url_health(agent)
    assert result["healthy"] is True
    # 1500ms → 1.5s
    assert captured["timeout"] == pytest.approx(1.5)


@pytest.mark.parametrize(
    "base_url",
    ["http://0.0.0.0:4000/", "http://localhost:4000/", "https://api.example.com/"],
)
def test_merged_agent_card_url_has_no_double_slash_without_proxy_base_url(monkeypatch, base_url):
    """Without PROXY_BASE_URL, request.base_url carries a trailing slash; the merged
    card's supportedInterfaces URL must still join cleanly (no `//a2a`)."""
    from litellm.proxy.agent_endpoints.endpoints import _build_merged_agent_card

    monkeypatch.delenv("PROXY_BASE_URL", raising=False)
    monkeypatch.delenv("SERVER_ROOT_PATH", raising=False)

    http_request = MagicMock()
    http_request.base_url = base_url

    merged = _build_merged_agent_card(
        _sample_agent_card_params(),
        agent_id="agent-xyz",
        http_request=http_request,
        agent_name="Test Agent",
    )

    interface_url = merged["supportedInterfaces"][0]["url"]
    assert interface_url == f"{base_url.rstrip('/')}/a2a/agent-xyz"
    assert "//a2a" not in interface_url
