"""
Tests for budget bypass on model-discovery / info routes.

Model-discovery and info endpoints (e.g. GET /v1/models, /models, /model/info)
perform no inference and incur no spend, so an exhausted budget must never block
them. Otherwise OpenAI-compatible clients (Open WebUI, Cursor, Aider, Continue,
LibreChat, ...) that call GET /v1/models for model discovery break entirely once
any team/key/org/user budget is exceeded.

Regression test for https://github.com/BerriAI/litellm/issues/27923
"""

from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import common_checks
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router

# Routes that must never be blocked by a budget. Sourced from
# LiteLLMRoutes.info_routes (the same group RouteChecks.is_info_route checks).
MODEL_DISCOVERY_ROUTES = [
    "/v1/models",
    "/models",
    "/model/info",
    "/v1/model/info",
    "/v2/model/info",
    "/model_group/info",
]


@pytest.fixture
def mock_router():
    """A router with a single paid (non-zero-cost) model."""
    return Router(
        model_list=[
            {
                "model_name": "cloud-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-test",
                },
                "model_info": {"id": "cloud-model-id"},
            }
        ]
    )


@pytest.fixture
def mock_proxy_logging():
    proxy_logging = ProxyLogging(user_api_key_cache=None)

    async def mock_budget_alerts(*args, **kwargs):
        pass

    proxy_logging.budget_alerts = mock_budget_alerts
    return proxy_logging


class TestTeamBudgetInfoRouteBypass:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", MODEL_DISCOVERY_ROUTES)
    async def test_team_over_budget_can_list_models(
        self, route, mock_router, mock_proxy_logging
    ):
        """A team that is over budget can still reach model-discovery routes."""
        team_object = LiteLLM_TeamTable(
            team_id="test-team", spend=150.0, max_budget=100.0
        )

        result = await common_checks(
            request_body={},
            team_object=team_object,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=mock_router,
            proxy_logging_obj=mock_proxy_logging,
            valid_token=UserAPIKeyAuth(token="test-token", team_id="test-team"),
            request=MagicMock(),
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_team_over_budget_still_blocked_on_inference(
        self, mock_router, mock_proxy_logging
    ):
        """Control: inference routes are still budget-enforced (no regression)."""
        team_object = LiteLLM_TeamTable(
            team_id="test-team", spend=150.0, max_budget=100.0
        )

        with pytest.raises(litellm.BudgetExceededError):
            await common_checks(
                request_body={"model": "cloud-model"},
                team_object=team_object,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/v1/chat/completions",
                llm_router=mock_router,
                proxy_logging_obj=mock_proxy_logging,
                valid_token=UserAPIKeyAuth(token="test-token", team_id="test-team"),
                request=MagicMock(),
            )


class TestUserBudgetInfoRouteBypass:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", MODEL_DISCOVERY_ROUTES)
    async def test_user_over_budget_can_list_models(
        self, route, mock_router, mock_proxy_logging
    ):
        user_object = LiteLLM_UserTable(
            user_id="test-user", spend=100.0, max_budget=50.0
        )

        result = await common_checks(
            request_body={},
            team_object=None,
            user_object=user_object,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=mock_router,
            proxy_logging_obj=mock_proxy_logging,
            valid_token=UserAPIKeyAuth(token="test-token", user_id="test-user"),
            request=MagicMock(),
        )
        assert result is True


class TestEndUserBudgetInfoRouteBypass:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", MODEL_DISCOVERY_ROUTES)
    async def test_end_user_over_budget_can_list_models(
        self, route, mock_router, mock_proxy_logging
    ):
        end_user_object = LiteLLM_EndUserTable(
            user_id="end-user-123",
            spend=50.0,
            litellm_budget_table=LiteLLM_BudgetTable(max_budget=20.0),
            blocked=False,
        )

        result = await common_checks(
            request_body={"user": "end-user-123"},
            team_object=None,
            user_object=None,
            end_user_object=end_user_object,
            global_proxy_spend=None,
            general_settings={},
            route=route,
            llm_router=mock_router,
            proxy_logging_obj=mock_proxy_logging,
            valid_token=UserAPIKeyAuth(token="test-token"),
            request=MagicMock(),
        )
        assert result is True


class TestGlobalProxyBudgetInfoRouteBypass:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("route", MODEL_DISCOVERY_ROUTES)
    async def test_global_proxy_over_budget_can_list_models(
        self, route, mock_router, mock_proxy_logging
    ):
        with patch.object(litellm, "max_budget", 100.0):
            result = await common_checks(
                request_body={},
                team_object=None,
                user_object=None,
                end_user_object=None,
                global_proxy_spend=150.0,
                general_settings={},
                route=route,
                llm_router=mock_router,
                proxy_logging_obj=mock_proxy_logging,
                valid_token=UserAPIKeyAuth(token="test-token"),
                request=MagicMock(),
            )
        assert result is True
