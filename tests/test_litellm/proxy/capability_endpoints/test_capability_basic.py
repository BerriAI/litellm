"""
Tests for ``GET /v1/capabilities`` skeleton (S1-02).

S1-02 does not yet enforce caller-side scoping — it returns the full
registries. The tests here verify:
  1. Auth still required (no token -> 401).
  2. Shape matches CapabilitiesResponse (caller block + 5 lists).
  3. Empty registries -> empty lists, not nulls.
  4. Populated registries surface entities with the expected ids.

Scoping by caller is tightened (and tested) in S1-03.
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
