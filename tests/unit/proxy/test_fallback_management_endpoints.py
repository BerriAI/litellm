"""
Tests for fallback management endpoints

Tests:
1. Create fallback configuration
2. Get fallback configuration
3. Delete fallback configuration
4. Validation tests (invalid models, duplicate fallbacks, etc.)
"""

import copy
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm import Router
from litellm.proxy.utils import evict_config_param, get_config_param
from litellm.proxy.management_endpoints.fallback_management_endpoints import (
    FallbackCreateRequest,
    FallbackDeleteResponse,
    FallbackResponse,
    create_fallback,
    delete_fallback,
    get_fallback,
)
from litellm.types.router import DeploymentTypedDict, LiteLLMParamsTypedDict


class TestFallbackCreateRequest:
    """Test the FallbackCreateRequest validation"""

    def test_valid_request(self):
        """Test valid fallback request"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4", "claude-3-haiku"],
            fallback_type="general",
        )
        assert request.model == "gpt-3.5-turbo"
        assert request.fallback_models == ["gpt-4", "claude-3-haiku"]
        assert request.fallback_type == "general"

    def test_default_fallback_type(self):
        """Test default fallback type is 'general'"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )
        assert request.fallback_type == "general"

    def test_empty_fallback_models(self):
        """Test that empty fallback_models raises validation error"""
        with pytest.raises(ValueError, match="at least 1 item"):
            FallbackCreateRequest(
                model="gpt-3.5-turbo",
                fallback_models=[],
            )

    def test_duplicate_fallback_models(self):
        """Test that duplicate fallback models raise validation error"""
        with pytest.raises(
            ValueError, match="fallback_models must not contain duplicates"
        ):
            FallbackCreateRequest(
                model="gpt-3.5-turbo",
                fallback_models=["gpt-4", "gpt-4"],
            )

    def test_empty_model_name(self):
        """Test that empty model name raises validation error"""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            FallbackCreateRequest(
                model="",
                fallback_models=["gpt-4"],
            )

    def test_whitespace_model_name(self):
        """Test that whitespace-only model name raises validation error"""
        with pytest.raises(ValueError, match="model must be a non-empty string"):
            FallbackCreateRequest(
                model="   ",
                fallback_models=["gpt-4"],
            )

    def test_model_name_trimmed(self):
        """Test that model name is trimmed"""
        request = FallbackCreateRequest(
            model="  gpt-3.5-turbo  ",
            fallback_models=["gpt-4"],
        )
        assert request.model == "gpt-3.5-turbo"

    def test_context_window_fallback_type(self):
        """Test context_window fallback type"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4-32k"],
            fallback_type="context_window",
        )
        assert request.fallback_type == "context_window"

    def test_content_policy_fallback_type(self):
        """Test content_policy fallback type"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
            fallback_type="content_policy",
        )
        assert request.fallback_type == "content_policy"


TEAM_ID: Final = "team-a"
PRIMARY_INTERNAL_NAME: Final = f"model_name_{TEAM_ID}_1a6437cb-4cab-432c-8099-1d7411731a8b"
FALLBACK_INTERNAL_NAME: Final = f"model_name_{TEAM_ID}_83151607-5556-4bbf-ac65-c474dbc64eba"
PRIMARY_DEPLOYMENT_ID: Final = "team-primary-id"
FALLBACK_DEPLOYMENT_ID: Final = "team-fallback-id"

FallbackRules = list[dict[str, list[str]] | dict[str, object] | str]
RouterSettings = dict[str, FallbackRules | None]


def _litellm_params(mock_response: str | None) -> LiteLLMParamsTypedDict:
    if mock_response is None:
        return {"model": "openai/gpt-5.4-mini", "api_key": "fake"}
    return {"model": "openai/gpt-5.4-mini", "api_key": "fake", "mock_response": mock_response}


def _team_scoped_deployment(
    internal_name: str, public_name: str, deployment_id: str, mock_response: str | None = None
) -> DeploymentTypedDict:
    return {
        "model_name": internal_name,
        "litellm_params": _litellm_params(mock_response),
        "model_info": {"id": deployment_id, "team_id": TEAM_ID, "team_public_model_name": public_name},
    }


def _team_router(primary_mock_response: str | None = None, fallback_mock_response: str | None = None) -> Router:
    return Router(
        model_list=[
            {"model_name": "gpt-5.4-mini", "litellm_params": _litellm_params(None)},
            _team_scoped_deployment(
                PRIMARY_INTERNAL_NAME, "team-primary", PRIMARY_DEPLOYMENT_ID, primary_mock_response
            ),
            _team_scoped_deployment(
                FALLBACK_INTERNAL_NAME, "team-fallback", FALLBACK_DEPLOYMENT_ID, fallback_mock_response
            ),
        ],
        num_retries=0,
    )


class TestCreateFallbackForTeamScopedModels:
    """POST /fallback takes the public name a caller invokes a team-scoped model by, not only the stored internal one"""

    @pytest.fixture
    def router(self) -> Router:
        return _team_router()

    @pytest.fixture
    def prisma_client(self) -> MagicMock:
        client: Final = MagicMock()
        client.db.litellm_config.upsert = AsyncMock()
        return client

    @pytest.fixture
    def proxy_config(self) -> MagicMock:
        config: Final = MagicMock()
        config.get_config = AsyncMock(return_value={"router_settings": {}})
        return config

    async def _create(
        self, request: FallbackCreateRequest, router: Router, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> FallbackResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", router),
            patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
            patch("litellm.proxy.proxy_server.proxy_config", proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await create_fallback(request, MagicMock())

    async def test_public_names_create_a_rule_keyed_on_the_public_name(
        self, router: Router, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        request: Final = FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"])

        response: Final = await self._create(request, router, prisma_client, proxy_config)

        assert response.model == "team-primary"
        assert response.fallback_models == ["team-fallback"]
        assert router.fallbacks == [{"team-primary": ["team-fallback"]}]
        persisted: Final = json.loads(
            prisma_client.db.litellm_config.upsert.call_args.kwargs["data"]["create"]["param_value"]
        )
        assert persisted["fallbacks"] == [{"team-primary": ["team-fallback"]}]

    async def test_internal_names_keep_working(
        self, router: Router, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        request: Final = FallbackCreateRequest(model=PRIMARY_INTERNAL_NAME, fallback_models=[FALLBACK_INTERNAL_NAME])

        response: Final = await self._create(request, router, prisma_client, proxy_config)

        assert router.fallbacks == [{PRIMARY_INTERNAL_NAME: [FALLBACK_INTERNAL_NAME]}]
        assert response.model == PRIMARY_INTERNAL_NAME

    async def test_public_fallback_target_behind_a_gateway_primary(
        self, router: Router, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        request: Final = FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"])

        await self._create(request, router, prisma_client, proxy_config)

        assert router.fallbacks == [{"gpt-5.4-mini": ["team-fallback"]}]

    async def test_unknown_name_is_rejected_and_the_error_names_the_public_names(
        self, router: Router, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        request: Final = FallbackCreateRequest(model="team-missing", fallback_models=["team-fallback"])

        with pytest.raises(HTTPException) as exc_info:
            await self._create(request, router, prisma_client, proxy_config)

        assert exc_info.value.status_code == 404
        assert {"team-primary", "team-fallback", "gpt-5.4-mini"} <= set(exc_info.value.detail["available_models"])

    async def test_a_team_request_fails_over_to_the_public_name_fallback(
        self, prisma_client: MagicMock, proxy_config: MagicMock
    ) -> None:
        router: Final = _team_router(primary_mock_response="litellm.RateLimitError", fallback_mock_response="pong")
        request: Final = FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"])
        await self._create(request, router, prisma_client, proxy_config)

        response: Final = await router.acompletion(
            model="team-primary",
            messages=[{"role": "user", "content": "ping"}],
            metadata={"user_api_key_team_id": TEAM_ID},
        )

        assert response.choices[0].message.content == "pong"
        assert response._hidden_params["model_id"] == FALLBACK_DEPLOYMENT_ID


def _config_row(router_settings: RouterSettings) -> SimpleNamespace:
    return SimpleNamespace(param_name="router_settings", param_value=router_settings)


class _StoredRouterSettings:
    """The LiteLLM_Config router_settings row, with the proxy objects that read it through the config cache"""

    def __init__(self, router_settings: RouterSettings | None) -> None:
        self.row: SimpleNamespace | None = None if router_settings is None else _config_row(router_settings)
        self.prisma_client: Final = MagicMock()
        self.prisma_client.get_generic_data = AsyncMock(side_effect=lambda **_: self.row)
        self.prisma_client.db.litellm_config.upsert = AsyncMock(side_effect=self._upsert)
        self.proxy_config: Final = MagicMock()
        self.proxy_config.get_config = AsyncMock(side_effect=self._get_config)

    async def _upsert(self, where: dict[str, str], data: dict[str, dict[str, str]]) -> None:
        self.row = _config_row(json.loads(data["update"]["param_value"]))

    async def _get_config(self) -> dict[str, RouterSettings]:
        row: Final = await get_config_param(self.prisma_client, "router_settings")
        return {"router_settings": copy.deepcopy(row.param_value) if row is not None else {}}

    def written_by_another_instance(self, router_settings: RouterSettings) -> None:
        self.row = _config_row(router_settings)

    def stored_fallbacks(self) -> FallbackRules:
        assert self.row is not None
        return self.row.param_value["fallbacks"]

    async def cached_fallbacks(self) -> FallbackRules:
        row: Final = await get_config_param(self.prisma_client, "router_settings")
        return row.param_value["fallbacks"]


TEAM_RULE: Final = {"team-primary": ["team-fallback"]}
GATEWAY_RULE: Final = {"gpt-5.4-mini": ["team-fallback"]}
MALFORMED_GATEWAY_RULE: Final = {"gpt-5.4-mini": "team-fallback"}
NON_STANDARD_RULES: Final = [
    "claude-3-haiku",
    {"model": "gpt-5.4-mini", "messages": [{"role": "user", "content": "retry"}]},
]


class TestFallbackWritesSeeTheLatestStoredRules:
    """A write reads the rules the database holds now and leaves no stale copy in the config cache behind"""

    @pytest.fixture(autouse=True)
    async def clean_config_cache(self) -> AsyncIterator[None]:
        await evict_config_param("router_settings")
        yield
        await evict_config_param("router_settings")

    async def _create(self, request: FallbackCreateRequest, stored: _StoredRouterSettings) -> FallbackResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", _team_router()),
            patch("litellm.proxy.proxy_server.prisma_client", stored.prisma_client),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await create_fallback(request, MagicMock())

    async def _delete(self, model: str, stored: _StoredRouterSettings) -> FallbackDeleteResponse:
        with (
            patch("litellm.proxy.proxy_server.llm_router", _team_router()),
            patch("litellm.proxy.proxy_server.prisma_client", stored.prisma_client),
            patch("litellm.proxy.proxy_server.proxy_config", stored.proxy_config),
            patch("litellm.proxy.proxy_server.store_model_in_db", True),
        ):
            return await delete_fallback(model, "general", MagicMock())

    async def test_a_second_create_keeps_the_first_rule(self) -> None:
        stored: Final = _StoredRouterSettings(None)

        await self._create(FallbackCreateRequest(model="team-primary", fallback_models=["team-fallback"]), stored)
        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [TEAM_RULE, GATEWAY_RULE]
        assert await stored.cached_fallbacks() == [TEAM_RULE, GATEWAY_RULE]

    async def test_a_create_keeps_a_rule_another_instance_stored_since_this_one_last_read(self) -> None:
        stored: Final = _StoredRouterSettings({})
        await get_config_param(stored.prisma_client, "router_settings")
        stored.written_by_another_instance({"fallbacks": [TEAM_RULE]})

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [TEAM_RULE, GATEWAY_RULE]

    async def test_a_delete_keeps_a_rule_another_instance_stored_since_this_one_last_read(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [TEAM_RULE]})
        await get_config_param(stored.prisma_client, "router_settings")
        stored.written_by_another_instance({"fallbacks": [TEAM_RULE, GATEWAY_RULE]})

        await self._delete("team-primary", stored)

        assert stored.stored_fallbacks() == [GATEWAY_RULE]
        assert await stored.cached_fallbacks() == [GATEWAY_RULE]

    async def test_writes_keep_the_non_standard_rules_the_router_accepts(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [*NON_STANDARD_RULES, TEAM_RULE]})

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)
        assert stored.stored_fallbacks() == [*NON_STANDARD_RULES, TEAM_RULE, GATEWAY_RULE]

        await self._delete("team-primary", stored)
        assert stored.stored_fallbacks() == [*NON_STANDARD_RULES, GATEWAY_RULE]

    async def test_a_create_replaces_a_same_key_rule_whatever_its_targets_shape(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [MALFORMED_GATEWAY_RULE, TEAM_RULE]})

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [GATEWAY_RULE, TEAM_RULE]

    async def test_a_delete_removes_a_same_key_rule_whatever_its_targets_shape(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": [MALFORMED_GATEWAY_RULE, TEAM_RULE]})

        await self._delete("gpt-5.4-mini", stored)

        assert stored.stored_fallbacks() == [TEAM_RULE]

    async def test_a_create_treats_a_null_rule_list_as_empty(self) -> None:
        stored: Final = _StoredRouterSettings({"fallbacks": None})

        await self._create(FallbackCreateRequest(model="gpt-5.4-mini", fallback_models=["team-fallback"]), stored)

        assert stored.stored_fallbacks() == [GATEWAY_RULE]


@pytest.mark.asyncio
class TestCreateFallback:
    """Test the create_fallback endpoint"""

    @pytest.fixture
    def mock_router(self):
        """Create a mock router"""
        router = MagicMock()
        router.model_names = {"gpt-3.5-turbo", "gpt-4", "claude-3-haiku"}
        router.team_public_model_names = frozenset()
        router.fallbacks = []
        router.context_window_fallbacks = []
        router.content_policy_fallbacks = []
        return router

    @pytest.fixture
    def mock_prisma_client(self):
        """Create a mock prisma client"""
        client = MagicMock()
        client.db.litellm_config.upsert = AsyncMock()
        client.jsonify_object = lambda x: x
        return client

    @pytest.fixture
    def mock_proxy_config(self):
        """Create a mock proxy config"""
        config = MagicMock()
        config.get_config = AsyncMock(return_value={"router_settings": {}})
        return config

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_create_fallback_success(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test successful fallback creation"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4", "claude-3-haiku"],
            fallback_type="general",
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await create_fallback(request, mock_user_api_key_dict)

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_models == ["gpt-4", "claude-3-haiku"]
            assert response.fallback_type == "general"
            assert (
                "created" in response.message.lower()
                or "updated" in response.message.lower()
            )

            # Verify database was updated
            mock_prisma_client.db.litellm_config.upsert.assert_called_once()

    async def test_create_fallback_router_not_initialized(
        self, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when router is not initialized"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)

    async def test_create_fallback_model_not_found(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when model is not found in router"""
        request = FallbackCreateRequest(
            model="invalid-model",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 404
        assert "not found in router" in str(exc_info.value.detail)

    async def test_create_fallback_invalid_fallback_model(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when fallback model is not found in router"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["invalid-fallback-model"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "Invalid fallback models" in str(exc_info.value.detail)

    async def test_create_fallback_model_is_own_fallback(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test error when model is its own fallback"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-3.5-turbo", "gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "cannot be its own fallback" in str(exc_info.value.detail)

    async def test_create_fallback_db_not_enabled(
        self, mock_router, mock_user_api_key_dict
    ):
        """Test error when database storage is not enabled"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                False,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await create_fallback(request, mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "Database storage not enabled" in str(exc_info.value.detail)

    async def test_create_fallback_context_window_type(
        self, mock_router, mock_prisma_client, mock_proxy_config, mock_user_api_key_dict
    ):
        """Test creating context_window fallback"""
        request = FallbackCreateRequest(
            model="gpt-3.5-turbo",
            fallback_models=["gpt-4"],
            fallback_type="context_window",
        )

        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await create_fallback(request, mock_user_api_key_dict)

            assert response.fallback_type == "context_window"
            # Verify the correct attribute was updated
            assert hasattr(mock_router, "context_window_fallbacks")


@pytest.mark.asyncio
class TestGetFallback:
    """Test the get_fallback endpoint"""

    @pytest.fixture
    def mock_router_with_fallbacks(self):
        """Create a mock router with fallbacks configured"""
        router = MagicMock()
        router.fallbacks = [{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}]
        router.context_window_fallbacks = []
        router.content_policy_fallbacks = []
        return router

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_get_fallback_success(
        self, mock_router_with_fallbacks, mock_user_api_key_dict
    ):
        """Test successful fallback retrieval"""
        with patch(
            "litellm.proxy.proxy_server.llm_router",
            mock_router_with_fallbacks,
        ):
            response = await get_fallback(
                "gpt-3.5-turbo", "general", mock_user_api_key_dict
            )

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_models == ["gpt-4", "claude-3-haiku"]
            assert response.fallback_type == "general"

    async def test_get_fallback_not_found(
        self, mock_router_with_fallbacks, mock_user_api_key_dict
    ):
        """Test error when fallback is not found"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_fallback("gpt-4", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 404
        assert "No general fallbacks configured" in str(exc_info.value.detail)

    async def test_get_fallback_router_not_initialized(self, mock_user_api_key_dict):
        """Test error when router is not initialized"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await get_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)


@pytest.mark.asyncio
class TestDeleteFallback:
    """Test the delete_fallback endpoint"""

    @pytest.fixture
    def mock_router_with_fallbacks(self):
        """Create a mock router with fallbacks configured"""
        router = MagicMock()
        router.fallbacks = [{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}]
        router.context_window_fallbacks = []
        router.content_policy_fallbacks = []
        return router

    @pytest.fixture
    def mock_prisma_client(self):
        """Create a mock prisma client"""
        client = MagicMock()
        client.db.litellm_config.upsert = AsyncMock()
        client.jsonify_object = lambda x: x
        return client

    @pytest.fixture
    def mock_proxy_config(self):
        """Create a mock proxy config"""
        config = MagicMock()
        config.get_config = AsyncMock(
            return_value={
                "router_settings": {
                    "fallbacks": [{"gpt-3.5-turbo": ["gpt-4", "claude-3-haiku"]}]
                }
            }
        )
        return config

    @pytest.fixture
    def mock_user_api_key_dict(self):
        """Create a mock user API key dict"""
        return MagicMock()

    async def test_delete_fallback_success(
        self,
        mock_router_with_fallbacks,
        mock_prisma_client,
        mock_proxy_config,
        mock_user_api_key_dict,
    ):
        """Test successful fallback deletion"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
        ):
            response = await delete_fallback(
                "gpt-3.5-turbo", "general", mock_user_api_key_dict
            )

            assert response.model == "gpt-3.5-turbo"
            assert response.fallback_type == "general"
            assert "deleted" in response.message.lower()

            # Verify database was updated
            mock_prisma_client.db.litellm_config.upsert.assert_called_once()

    async def test_delete_fallback_not_found(
        self,
        mock_router_with_fallbacks,
        mock_prisma_client,
        mock_proxy_config,
        mock_user_api_key_dict,
    ):
        """Test error when fallback to delete is not found"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.prisma_client",
                mock_prisma_client,
            ),
            patch(
                "litellm.proxy.proxy_server.proxy_config",
                mock_proxy_config,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                True,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback("gpt-4", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 404
        assert "No general fallbacks configured" in str(exc_info.value.detail)

    async def test_delete_fallback_router_not_initialized(self, mock_user_api_key_dict):
        """Test error when router is not initialized"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                None,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 500
        assert "Router not initialized" in str(exc_info.value.detail)

    async def test_delete_fallback_db_not_enabled(
        self, mock_router_with_fallbacks, mock_user_api_key_dict
    ):
        """Test error when database storage is not enabled"""
        with (
            patch(
                "litellm.proxy.proxy_server.llm_router",
                mock_router_with_fallbacks,
            ),
            patch(
                "litellm.proxy.proxy_server.store_model_in_db",
                False,
            ),
            pytest.raises(HTTPException) as exc_info,
        ):
            await delete_fallback("gpt-3.5-turbo", "general", mock_user_api_key_dict)

        assert exc_info.value.status_code == 400
        assert "Database storage not enabled" in str(exc_info.value.detail)
