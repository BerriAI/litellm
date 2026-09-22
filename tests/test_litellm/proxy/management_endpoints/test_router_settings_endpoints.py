"""
Tests for router settings management endpoints.

Tests the GET endpoints for router settings and router fields.
"""

from collections.abc import Mapping
from typing import Any, Final

import pytest
from fastapi.testclient import TestClient

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.config_resolvers import SettingsStore
from litellm.proxy.management_endpoints.router_settings_endpoints import (
    RouterFieldsResponse,
    RouterSettingsResponse,
    get_router_fields,
    get_router_settings,
)
from litellm.proxy.proxy_server import app
from litellm.router import Router
from litellm.types.router import RoutingGroup

client = TestClient(app)


class _StubProxyConfig:
    def __init__(self, router_settings: SettingsStore, config_router_settings: Mapping[str, Any]) -> None:
        self.router_settings: Final = router_settings
        self._config_router_settings: Final = dict(config_router_settings)

    async def get_config(self, config_file_path: str | None = None) -> dict[str, Any]:
        del config_file_path
        return {"router_settings": dict(self._config_router_settings)}


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
    async def test_get_router_settings_reports_sources(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store = SettingsStore("router_settings")
        store.load_yaml({"routing_strategy": "simple-shuffle"})
        store.apply_db_row("router_settings", {"num_retries": 3})
        monkeypatch.setattr(
            proxy_server,
            "proxy_config",
            _StubProxyConfig(
                store,
                {"routing_strategy": "simple-shuffle", "num_retries": 3},
            ),
        )
        monkeypatch.setattr(proxy_server, "llm_router", None)

        admin_user = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-x"
        )
        response = await get_router_settings(user_api_key_dict=admin_user)

        assert response.source["routing_strategy"] == "config"
        assert response.source["num_retries"] == "db"

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
        monkeypatch.setattr(
            proxy_server,
            "proxy_config",
            _StubProxyConfig(SettingsStore("router_settings"), {}),
        )

        admin_user = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-x"
        )
        response = await get_router_settings(user_api_key_dict=admin_user)

        assert response.current_values.get("routing_groups") == groups
        assert response.current_values["timeout"] is not None
        assert response.source["timeout"] == "default"

        rg_field = next(f for f in response.fields if f.field_name == "routing_groups")
        assert rg_field.field_value == groups

    @pytest.mark.asyncio
    @pytest.mark.parametrize("metadata_only", (True, False))
    async def test_priority_is_advertised_for_groups_only(
        self, monkeypatch: pytest.MonkeyPatch, metadata_only: bool
    ) -> None:
        monkeypatch.setattr(proxy_server, "llm_router", None)
        monkeypatch.setattr(
            proxy_server,
            "proxy_config",
            _StubProxyConfig(SettingsStore("router_settings"), {}),
        )
        admin_user: Final = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test"
        )

        response: Final[RouterFieldsResponse | RouterSettingsResponse] = (
            await get_router_fields(user_api_key_dict=admin_user)
            if metadata_only
            else await get_router_settings(user_api_key_dict=admin_user)
        )

        global_options: Final = next(
            field.options
            for field in response.fields
            if field.field_name == "routing_strategy"
        )
        assert global_options is not None
        assert "priority" not in global_options
        assert response.model_dump(mode="json")["routing_group_strategies"] == [*global_options, "priority"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("from_config", (False, True))
    async def test_settings_retains_explicit_model_priorities(
        self, monkeypatch: pytest.MonkeyPatch, from_config: bool
    ) -> None:
        group: Final = RoutingGroup(
            group_name="ordered-chat",
            models=["primary", "backup"],
            routing_strategy="priority",
            model_priorities={"primary": 1, "backup": 2},
        )
        llm_router: Final = Router(
            model_list=[
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": "openai/gpt-5.4-nano",
                        "api_key": "sk-test",
                    },
                }
                for model in group.models
            ],
            routing_groups=[group],
        )
        expected: Final = [
            {
                "group_name": "ordered-chat",
                "models": ["primary", "backup"],
                "routing_strategy": "priority",
                "routing_strategy_args": None,
                "model_priorities": (
                    {"primary": 8, "backup": 3}
                    if from_config
                    else {"primary": 1, "backup": 2}
                ),
            }
        ]
        config: Final = {"routing_groups": expected} if from_config else {}
        store: Final = SettingsStore("router_settings")
        store.load_yaml(config)
        monkeypatch.setattr(proxy_server, "llm_router", llm_router)
        monkeypatch.setattr(
            proxy_server, "proxy_config", _StubProxyConfig(store, config)
        )
        admin_user: Final = UserAPIKeyAuth(
            user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-test"
        )

        response: Final[RouterSettingsResponse] = await get_router_settings(
            user_api_key_dict=admin_user
        )

        assert response.current_values["routing_groups"] == expected
        groups_field: Final = next(
            field for field in response.fields if field.field_name == "routing_groups"
        )
        assert groups_field.field_value == expected
