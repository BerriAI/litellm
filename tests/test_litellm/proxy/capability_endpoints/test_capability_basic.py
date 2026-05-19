"""
Tests for ``GET /v1/capabilities`` (S1-02 + S1-03).

S1-02 verified the envelope shape with admin role (sees everything).
S1-03 adds scoping for non-admin callers — see TestCapabilitiesScoping.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.capability_endpoints.capability_endpoints import router
from litellm.types.capabilities import CAPABILITIES_SCHEMA_VERSION


def _make_client(role: LitellmUserRoles = LitellmUserRoles.PROXY_ADMIN) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="u-1", team_id="t-1", api_key="sk-test", user_role=role
    )
    return TestClient(app)


def test_capabilities_returns_envelope_shape(monkeypatch):
    """Empty registries: response carries caller + 5 empty lists + schema_version."""
    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(litellm, "public_agent_groups", [])
    fake_registry = MagicMock()
    fake_registry.get_agent_list.return_value = []
    fake_manager = MagicMock()
    fake_manager.get_registered_mcp_servers.return_value = []

    with (
        patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            fake_registry,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            fake_manager,
        ),
    ):
        resp = _make_client().get(
            "/v1/capabilities", headers={"Authorization": "Bearer k"}
        )

    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == CAPABILITIES_SCHEMA_VERSION
    assert body["caller"]["user_id"] == "u-1"
    assert body["caller"]["team_id"] == "t-1"
    assert body["caller"]["is_admin"] is True
    for key in ("models", "agents", "mcps", "skills", "access_groups"):
        assert body[key] == []


def test_capabilities_populates_models_from_model_cost(monkeypatch):
    """A configured model surfaces as a ModelSummary with provider + cost."""
    monkeypatch.setattr(
        litellm,
        "model_cost",
        {
            "deepseek-v3.2": {
                "litellm_provider": "tencent_cloud",
                "mode": "chat",
                "max_input_tokens": 128000,
                "max_output_tokens": 4096,
                "input_cost_per_token": 0.00000027,
                "output_cost_per_token": 0.00000110,
            },
        },
    )
    monkeypatch.setattr(litellm, "public_agent_groups", [])
    fake_registry = MagicMock()
    fake_registry.get_agent_list.return_value = []
    fake_manager = MagicMock()
    fake_manager.get_registered_mcp_servers.return_value = []

    with (
        patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            fake_registry,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            fake_manager,
        ),
    ):
        resp = _make_client().get(
            "/v1/capabilities", headers={"Authorization": "Bearer k"}
        )

    assert resp.status_code == 200
    models = resp.json()["models"]
    assert len(models) == 1
    m = models[0]
    assert m["id"] == "deepseek-v3.2"
    assert m["provider"] == "tencent_cloud"
    assert m["context_window"] == 128000


def test_capabilities_populates_agents_from_registry(monkeypatch):
    """An agent in the registry surfaces with agent_card_url and public flag."""
    monkeypatch.setattr(litellm, "model_cost", {})
    monkeypatch.setattr(litellm, "public_agent_groups", ["a-public"])

    agent = MagicMock()
    agent.agent_id = "a-public"
    agent.agent_name = "PublicAgent"
    agent.agent_card_params = {
        "description": "demo",
        "version": "1.0.0",
        "capabilities": {"streaming": True},
    }
    fake_registry = MagicMock()
    fake_registry.get_agent_list.return_value = [agent]
    fake_manager = MagicMock()
    fake_manager.get_registered_mcp_servers.return_value = []

    with (
        patch(
            "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
            fake_registry,
        ),
        patch(
            "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
            fake_manager,
        ),
    ):
        resp = _make_client().get(
            "/v1/capabilities", headers={"Authorization": "Bearer k"}
        )

    assert resp.status_code == 200
    agents = resp.json()["agents"]
    assert len(agents) == 1
    assert agents[0]["agent_id"] == "a-public"
    assert agents[0]["is_public"] is True
    assert agents[0]["supports_streaming"] is True
    assert agents[0]["agent_card_url"].endswith("/agent-card.json")


# ============================================================================
# S1-03: scoping by caller role
# ============================================================================


class TestCapabilitiesScoping:
    """The 6-level permission filter must drop entities the caller can't use.

    Branches under test (per CLAUDE.md test-all-branches discipline):
      1. Admin: sees everything regardless of grants.
      2. Internal user with explicit grant -> sees granted entities only.
      3. Internal user with no grant -> empty model list, empty mcp list,
         only public/owned agents (NEVER falls back to "all").
    """

    @pytest.fixture(autouse=True)
    def _wire(self, monkeypatch):
        # Three models in model_cost.
        monkeypatch.setattr(
            litellm,
            "model_cost",
            {
                "model-allowed": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "max_input_tokens": 8000,
                },
                "model-forbidden": {
                    "litellm_provider": "openai",
                    "mode": "chat",
                    "max_input_tokens": 8000,
                },
                "model-also-allowed": {
                    "litellm_provider": "anthropic",
                    "mode": "chat",
                    "max_input_tokens": 200000,
                },
            },
        )
        # Public agent set.
        monkeypatch.setattr(litellm, "public_agent_groups", ["agent-public"])

        # Three agents in the registry.
        def _mk_agent(aid, name="x"):
            m = MagicMock()
            m.agent_id = aid
            m.agent_name = name
            m.agent_card_params = {"capabilities": {"streaming": False}}
            return m

        self.fake_registry = MagicMock()
        self.fake_registry.get_agent_list.return_value = [
            _mk_agent("agent-granted"),
            _mk_agent("agent-public"),
            _mk_agent("agent-other"),
        ]

        # Two MCP servers.
        def _mk_server(sid):
            s = MagicMock()
            s.server_id = sid
            s.name = sid
            s.alias = None
            s.transport = "http"
            s.auth_type = "api_key"
            s.mcp_access_groups = []
            return s

        self.fake_manager = MagicMock()
        self.fake_manager.get_registered_mcp_servers.return_value = [
            _mk_server("mcp-allowed"),
            _mk_server("mcp-forbidden"),
        ]

        # Empty proxy model list so non-admin doesn't get the proxy fallback.
        monkeypatch.setattr(
            "litellm.proxy.proxy_server.llm_model_list", [], raising=False
        )

    def _run(self, role, *, key_models, allowed_agents, allowed_mcps, owned_agents):
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_id="u-1",
            team_id="t-1",
            api_key="sk-test",
            user_role=role,
            models=key_models,
        )
        prisma = MagicMock()
        prisma.db.litellm_agentstable.find_many = MagicMock(
            return_value=[MagicMock(agent_id=a) for a in owned_agents]
        )
        # The agent helper uses AsyncMock semantics — wrap with AsyncMock.
        from unittest.mock import AsyncMock as _AsyncMock

        prisma.db.litellm_agentstable.find_many = _AsyncMock(
            return_value=[MagicMock(agent_id=a) for a in owned_agents]
        )

        with (
            patch(
                "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
                self.fake_registry,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                self.fake_manager,
            ),
            patch("litellm.proxy.proxy_server.prisma_client", prisma),
            patch(
                "litellm.proxy.agent_endpoints.auth.agent_permission_handler.AgentRequestHandler.get_allowed_agents",
                _AsyncMock(return_value=allowed_agents),
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.MCPRequestHandler.get_allowed_mcp_servers",
                _AsyncMock(return_value=allowed_mcps),
            ),
        ):
            return TestClient(app).get(
                "/v1/capabilities", headers={"Authorization": "Bearer k"}
            )

    # ---- Branch 1: admin sees everything --------------------------------

    def test_admin_sees_full_registries(self):
        resp = self._run(
            LitellmUserRoles.PROXY_ADMIN,
            key_models=[],
            allowed_agents=[],
            allowed_mcps=[],
            owned_agents=[],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert {m["id"] for m in body["models"]} == {
            "model-allowed",
            "model-forbidden",
            "model-also-allowed",
        }
        assert {a["agent_id"] for a in body["agents"]} == {
            "agent-granted",
            "agent-public",
            "agent-other",
        }
        assert {m["server_id"] for m in body["mcps"]} == {
            "mcp-allowed",
            "mcp-forbidden",
        }

    # ---- Branch 2: explicit grants for non-admin ------------------------

    def test_internal_user_with_explicit_grants_sees_only_granted(self):
        resp = self._run(
            LitellmUserRoles.INTERNAL_USER,
            key_models=["model-allowed", "model-also-allowed"],
            allowed_agents=["agent-granted"],
            allowed_mcps=["mcp-allowed"],
            owned_agents=[],
        )
        assert resp.status_code == 200
        body = resp.json()
        assert {m["id"] for m in body["models"]} == {
            "model-allowed",
            "model-also-allowed",
        }
        # agent-granted plus the public one (and only those).
        assert {a["agent_id"] for a in body["agents"]} == {
            "agent-granted",
            "agent-public",
        }
        assert {m["server_id"] for m in body["mcps"]} == {"mcp-allowed"}

    # ---- Branch 3: no grants — must NOT leak everything -----------------

    def test_internal_user_with_no_grants_sees_only_public_and_owned(self):
        resp = self._run(
            LitellmUserRoles.INTERNAL_USER,
            key_models=[],
            allowed_agents=[],
            allowed_mcps=[],
            owned_agents=["agent-other"],  # 'owned' here
        )
        assert resp.status_code == 200
        body = resp.json()
        # No key/team grants and empty proxy_model_list -> no models.
        assert body["models"] == []
        # Public + owned agents, NOT 'agent-granted'.
        assert {a["agent_id"] for a in body["agents"]} == {
            "agent-public",
            "agent-other",
        }
        # MCPs without explicit grants -> none.
        assert body["mcps"] == []


# ============================================================================
# S1-04: caching
# ============================================================================


class TestCapabilitiesCache:
    """The result is cached per (token, app_id). Same caller hitting back-to-
    back should not re-walk the registry; different caller key gets its own."""

    @pytest.fixture(autouse=True)
    def _wire(self, monkeypatch):
        monkeypatch.setattr(litellm, "model_cost", {})
        monkeypatch.setattr(litellm, "public_agent_groups", [])

        self.fake_registry = MagicMock()
        self.fake_registry.get_agent_list = MagicMock(return_value=[])
        self.fake_manager = MagicMock()
        self.fake_manager.get_registered_mcp_servers = MagicMock(return_value=[])

        # Reset the module-level cache so each test starts cold.
        import litellm.proxy.capability_endpoints.capability_endpoints as mod

        mod._capabilities_cache = None

    def _client(self, token="sk-test"):
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
            user_id="u-1",
            team_id="t-1",
            api_key=token,
            user_role=LitellmUserRoles.PROXY_ADMIN,
        )
        return TestClient(app)

    def test_same_caller_hits_cache_on_second_call(self):
        with (
            patch(
                "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
                self.fake_registry,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                self.fake_manager,
            ),
        ):
            client = self._client()
            r1 = client.get("/v1/capabilities", headers={"Authorization": "Bearer k"})
            r2 = client.get("/v1/capabilities", headers={"Authorization": "Bearer k"})
            assert r1.status_code == 200
            assert r2.status_code == 200
            # Registry walked exactly once between the two requests.
            assert self.fake_registry.get_agent_list.call_count == 1

    def test_different_caller_keys_have_separate_cache_entries(self):
        with (
            patch(
                "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry",
                self.fake_registry,
            ),
            patch(
                "litellm.proxy._experimental.mcp_server.mcp_server_manager.global_mcp_server_manager",
                self.fake_manager,
            ),
        ):
            # Each caller flips a brand-new client (different token).
            c1 = self._client(token="sk-caller-A")
            c2 = self._client(token="sk-caller-B")
            c1.get("/v1/capabilities", headers={"Authorization": "Bearer kA"})
            c2.get("/v1/capabilities", headers={"Authorization": "Bearer kB"})
            assert self.fake_registry.get_agent_list.call_count == 2
