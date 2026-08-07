"""
Tests for router settings management endpoints.

Tests the GET endpoints for router settings and router fields.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.router_settings_endpoints import (
    get_router_settings,
)
from litellm.proxy.proxy_server import app
from litellm.router import Router

client = TestClient(app)


class TestRouterSettingsEndpoints:
    """Test suite for router settings endpoints"""

    @pytest.mark.asyncio
    async def test_get_router_fields_success(self):
        """
        Test GET /router/fields endpoint successfully returns field definitions without values.
        """
        # Make request to router fields endpoint
        response = client.get(
            "/router/fields", headers={"Authorization": "Bearer sk-1234"}
        )

        # Verify response
        assert response.status_code == 200

        response_data = response.json()

        # Verify response structure
        assert "fields" in response_data
        assert "routing_strategy_descriptions" in response_data

        # Verify fields is a list
        assert isinstance(response_data["fields"], list)
        assert len(response_data["fields"]) > 0

        # Verify each field has required properties and field_value is None
        for field in response_data["fields"]:
            assert "field_name" in field
            assert "field_type" in field
            assert "field_description" in field
            assert "field_default" in field
            assert "ui_field_name" in field
            assert "field_value" in field
            assert field["field_value"] is None  # Ensure field_value is None

        # Verify routing_strategy_descriptions is a dict
        assert isinstance(response_data["routing_strategy_descriptions"], dict)
        assert len(response_data["routing_strategy_descriptions"]) > 0

        # Verify routing_strategy field has options populated
        routing_strategy_field = next(
            (
                f
                for f in response_data["fields"]
                if f["field_name"] == "routing_strategy"
            ),
            None,
        )
        assert routing_strategy_field is not None
        assert "options" in routing_strategy_field
        assert isinstance(routing_strategy_field["options"], list)
        assert len(routing_strategy_field["options"]) > 0

    @pytest.mark.asyncio
    async def test_get_router_settings_includes_routing_groups_from_live_router(
        self, monkeypatch
    ):
        """GET /router/settings returns routing_groups from the live router."""
        groups = [
            {
                "group_name": "test-group",
                "models": ["latency-model"],
                "routing_strategy": "latency-based-routing",
                "routing_strategy_args": {},
            }
        ]
        llm_router = Router(
            model_list=[
                {
                    "model_name": "latency-model",
                    "litellm_params": {
                        "model": "openai/gpt-4o",
                        "api_key": "sk-x",
                    },
                }
            ],
            routing_groups=groups,
        )

        monkeypatch.setattr(proxy_server, "llm_router", llm_router)

        async def fake_get_config(self, config_file_path=None):
            return {}

        monkeypatch.setattr(
            proxy_server.ProxyConfig, "get_config", fake_get_config, raising=True
        )

        admin_user = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-x"
        )
        response = await get_router_settings(user_api_key_dict=admin_user)

        assert response.current_values.get("routing_groups") == groups

        rg_field = next(f for f in response.fields if f.field_name == "routing_groups")
        assert rg_field.field_value == groups

    @pytest.mark.asyncio
    async def test_get_router_settings_empty_config_routing_groups_not_overwritten(
        self, monkeypatch
    ):
        """When config returns empty routing_groups, live router value is preserved."""
        live_groups = [
            {
                "group_name": "live-group",
                "models": ["gpt-4"],
                "routing_strategy": "cost-based-routing",
            }
        ]
        llm_router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"},
                }
            ],
            routing_groups=live_groups,
        )

        monkeypatch.setattr(proxy_server, "llm_router", llm_router)

        async def fake_get_config(self, config_file_path=None):
            return {"router_settings": {"routing_groups": []}}

        monkeypatch.setattr(
            proxy_server.ProxyConfig, "get_config", fake_get_config, raising=True
        )

        admin_user = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-x"
        )
        response = await get_router_settings(user_api_key_dict=admin_user)

        assert response.current_values.get("routing_groups") == live_groups

    @pytest.mark.asyncio
    async def test_get_router_settings_merge_routing_groups_config_wins(
        self, monkeypatch
    ):
        """When both live router and config have routing_groups, config values win for same group_name."""
        live_groups = [
            {
                "group_name": "shared-group",
                "models": ["gpt-4"],
                "routing_strategy": "cost-based-routing",
            },
            {
                "group_name": "live-only-group",
                "models": ["gpt-3.5"],
                "routing_strategy": "simple-shuffle",
            },
        ]
        config_groups = [
            {
                "group_name": "shared-group",
                "models": ["gpt-4o"],
                "routing_strategy": "latency-based-routing",
            },
            {
                "group_name": "config-only-group",
                "models": ["claude-3"],
                "routing_strategy": "least-busy",
            },
        ]
        llm_router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"},
                },
                {
                    "model_name": "gpt-3.5",
                    "litellm_params": {"model": "openai/gpt-3.5-turbo", "api_key": "sk-x"},
                },
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"},
                },
                {
                    "model_name": "claude-3",
                    "litellm_params": {"model": "anthropic/claude-3", "api_key": "sk-x"},
                },
            ],
            routing_groups=live_groups,
        )

        monkeypatch.setattr(proxy_server, "llm_router", llm_router)

        async def fake_get_config(self, config_file_path=None):
            return {"router_settings": {"routing_groups": config_groups}}

        monkeypatch.setattr(
            proxy_server.ProxyConfig, "get_config", fake_get_config, raising=True
        )

        admin_user = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-x"
        )
        response = await get_router_settings(user_api_key_dict=admin_user)

        result_groups = response.current_values.get("routing_groups", [])
        result_by_name = {g["group_name"]: g for g in result_groups}

        # shared-group should have config values
        assert result_by_name["shared-group"]["routing_strategy"] == "latency-based-routing"
        assert result_by_name["shared-group"]["models"] == ["gpt-4o"]

        # live-only-group should be preserved
        assert "live-only-group" in result_by_name
        assert result_by_name["live-only-group"]["routing_strategy"] == "simple-shuffle"

        # config-only-group should be added
        assert "config-only-group" in result_by_name
        assert result_by_name["config-only-group"]["routing_strategy"] == "least-busy"

    @pytest.mark.asyncio
    async def test_get_router_settings_fallback_to_config_routing_groups(
        self, monkeypatch
    ):
        """When live router has no routing_groups, config value is used."""
        llm_router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"},
                }
            ],
        )

        monkeypatch.setattr(proxy_server, "llm_router", llm_router)

        config_groups = [
            {
                "group_name": "config-group",
                "models": ["gpt-4"],
                "routing_strategy": "cost-based-routing",
            }
        ]

        async def fake_get_config(self, config_file_path=None):
            return {"router_settings": {"routing_groups": config_groups}}

        monkeypatch.setattr(
            proxy_server.ProxyConfig, "get_config", fake_get_config, raising=True
        )

        admin_user = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-x"
        )
        response = await get_router_settings(user_api_key_dict=admin_user)

        assert response.current_values.get("routing_groups") == config_groups

    @pytest.mark.asyncio
    async def test_get_router_settings_other_fields_overwritten_by_config(
        self, monkeypatch
    ):
        """Non-routing_groups fields should still be overwritten by config values."""
        llm_router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "sk-x"},
                }
            ],
            num_retries=5,
        )

        monkeypatch.setattr(proxy_server, "llm_router", llm_router)

        async def fake_get_config(self, config_file_path=None):
            return {"router_settings": {"num_retries": 10}}

        monkeypatch.setattr(
            proxy_server.ProxyConfig, "get_config", fake_get_config, raising=True
        )

        admin_user = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-x"
        )
        response = await get_router_settings(user_api_key_dict=admin_user)

        assert response.current_values.get("num_retries") == 10
