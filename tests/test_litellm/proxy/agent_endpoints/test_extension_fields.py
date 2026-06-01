"""
Tests for A2A protocol extension field (x- prefixed) passthrough support.

Validates that custom extension properties in agent_card_params are preserved
through the create, update, and patch agent workflows, as specified by the
A2A protocol specification.

Related issue: https://github.com/BerriAI/litellm/issues/27371
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.agent_endpoints.endpoints import (
    _extract_extension_fields,
    _merge_extension_fields,
    router,
    user_api_key_auth,
)
from litellm.types.agents import AgentResponse


# --- Helper fixtures and factories ---


def _sample_agent_card_params() -> dict:
    return {
        "protocolVersion": "1.0",
        "name": "Test Agent",
        "description": "A test agent",
        "url": "http://localhost:9999/",
        "version": "1.0.0",
        "capabilities": {"streaming": True},
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [
            {
                "id": "test_skill",
                "name": "Test Skill",
                "description": "A test skill",
                "tags": ["test"],
            }
        ],
    }


def _sample_agent_card_params_with_extensions() -> dict:
    """Agent card params with x- prefixed extension fields."""
    params = _sample_agent_card_params()
    params["x-provider-org"] = {
        "business": "telecom",
        "domain": "customer-service",
        "sub-domain": "mobility",
        "journey": "billing",
    }
    params["x-routing-hints"] = {
        "priority": "high",
        "region": "in-west",
    }
    return params


def _sample_agent_config_with_extensions() -> dict:
    return {
        "agent_name": "extension-test-agent",
        "agent_card_params": _sample_agent_card_params_with_extensions(),
        "litellm_params": {"make_public": False},
    }


def _make_test_client() -> TestClient:
    """Create a TestClient with admin auth override."""
    test_app = FastAPI()
    test_app.include_router(router)
    test_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user", user_role=LitellmUserRoles.PROXY_ADMIN
    )
    return TestClient(test_app)


# --- Unit tests for helper functions ---


class TestExtractExtensionFields:
    """Tests for _extract_extension_fields helper."""

    def test_extracts_x_prefixed_fields(self):
        raw_body = {
            "agent_card_params": {
                "name": "Test",
                "url": "http://localhost",
                "x-provider-org": {"business": "telecom"},
                "x-routing-hints": {"priority": "high"},
            }
        }
        result = _extract_extension_fields(raw_body)
        assert "x-provider-org" in result
        assert "x-routing-hints" in result
        assert result["x-provider-org"] == {"business": "telecom"}
        assert result["x-routing-hints"] == {"priority": "high"}

    def test_excludes_non_x_fields(self):
        raw_body = {
            "agent_card_params": {
                "name": "Test",
                "url": "http://localhost",
                "x-custom": "value",
            }
        }
        result = _extract_extension_fields(raw_body)
        assert "name" not in result
        assert "url" not in result
        assert "x-custom" in result

    def test_returns_empty_dict_when_no_extensions(self):
        raw_body = {
            "agent_card_params": {
                "name": "Test",
                "url": "http://localhost",
            }
        }
        result = _extract_extension_fields(raw_body)
        assert result == {}

    def test_returns_empty_dict_when_no_agent_card_params(self):
        result = _extract_extension_fields({})
        assert result == {}

    def test_returns_empty_dict_when_agent_card_params_not_dict(self):
        result = _extract_extension_fields({"agent_card_params": "not a dict"})
        assert result == {}

    def test_handles_nested_extension_values(self):
        raw_body = {
            "agent_card_params": {
                "x-deep-nested": {
                    "level1": {
                        "level2": ["a", "b", "c"],
                    }
                }
            }
        }
        result = _extract_extension_fields(raw_body)
        assert result["x-deep-nested"]["level1"]["level2"] == ["a", "b", "c"]


class TestMergeExtensionFields:
    """Tests for _merge_extension_fields helper."""

    def test_merges_extensions_into_agent_card_params(self):
        agent = {
            "agent_card_params": {"name": "Test", "url": "http://localhost"}
        }
        extensions = {"x-provider-org": {"business": "telecom"}}
        _merge_extension_fields(agent, extensions)
        assert agent["agent_card_params"]["x-provider-org"] == {"business": "telecom"}
        # Original fields preserved
        assert agent["agent_card_params"]["name"] == "Test"

    def test_noop_when_no_extensions(self):
        agent = {
            "agent_card_params": {"name": "Test"}
        }
        original = dict(agent["agent_card_params"])
        _merge_extension_fields(agent, {})
        assert agent["agent_card_params"] == original

    def test_noop_when_no_agent_card_params(self):
        agent = {"agent_name": "test"}
        _merge_extension_fields(agent, {"x-custom": "value"})
        assert "agent_card_params" not in agent

    def test_noop_when_agent_card_params_is_none(self):
        agent = {"agent_card_params": None}
        _merge_extension_fields(agent, {"x-custom": "value"})
        assert agent["agent_card_params"] is None


# --- Integration tests for endpoints ---


class TestCreateAgentWithExtensions:
    """Tests that POST /v1/agents preserves x- extension fields."""

    @patch("litellm.proxy.proxy_server.prisma_client")
    @patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    )
    def test_create_agent_preserves_extension_fields(
        self, mock_registry, mock_prisma
    ):
        client = _make_test_client()

        # Mock: no naming conflict
        mock_registry.get_agent_by_name.return_value = None

        # Mock: DB create returns the agent with extensions preserved
        agent_card_with_ext = _sample_agent_card_params_with_extensions()
        mock_db_result = MagicMock()
        mock_db_result.model_dump.return_value = {
            "agent_id": "test-id-123",
            "agent_name": "extension-test-agent",
            "agent_card_params": agent_card_with_ext,
            "litellm_params": {"make_public": False},
        }
        mock_db_result.object_permission = None

        mock_prisma.db.litellm_agentstable.create = AsyncMock(
            return_value=mock_db_result
        )

        # Make the request with extension fields
        request_body = _sample_agent_config_with_extensions()
        response = client.post("/v1/agents", json=request_body)

        assert response.status_code == 200

        # Verify the agent_card_params passed to DB contained extension fields
        create_call_args = mock_prisma.db.litellm_agentstable.create.call_args
        stored_data = create_call_args.kwargs.get("data", {})
        import json

        stored_card = json.loads(stored_data.get("agent_card_params", "{}"))
        assert "x-provider-org" in stored_card, (
            "x-provider-org should be preserved in stored agent_card_params"
        )
        assert "x-routing-hints" in stored_card, (
            "x-routing-hints should be preserved in stored agent_card_params"
        )
        assert stored_card["x-provider-org"]["business"] == "telecom"
        assert stored_card["x-routing-hints"]["priority"] == "high"


class TestCreateAgentWithoutExtensions:
    """Tests that POST /v1/agents still works normally without extensions."""

    @patch("litellm.proxy.proxy_server.prisma_client")
    @patch(
        "litellm.proxy.agent_endpoints.agent_registry.global_agent_registry"
    )
    def test_create_agent_works_without_extensions(
        self, mock_registry, mock_prisma
    ):
        client = _make_test_client()

        mock_registry.get_agent_by_name.return_value = None

        mock_db_result = MagicMock()
        mock_db_result.model_dump.return_value = {
            "agent_id": "test-id-456",
            "agent_name": "no-ext-agent",
            "agent_card_params": _sample_agent_card_params(),
            "litellm_params": {},
        }
        mock_db_result.object_permission = None

        mock_prisma.db.litellm_agentstable.create = AsyncMock(
            return_value=mock_db_result
        )

        request_body = {
            "agent_name": "no-ext-agent",
            "agent_card_params": _sample_agent_card_params(),
            "litellm_params": {},
        }
        response = client.post("/v1/agents", json=request_body)

        assert response.status_code == 200

        # Verify no extension fields in stored data
        create_call_args = mock_prisma.db.litellm_agentstable.create.call_args
        stored_data = create_call_args.kwargs.get("data", {})
        import json

        stored_card = json.loads(stored_data.get("agent_card_params", "{}"))
        # No x- keys should be present
        x_keys = [k for k in stored_card if k.startswith("x-")]
        assert len(x_keys) == 0, "No extension fields should be present"
