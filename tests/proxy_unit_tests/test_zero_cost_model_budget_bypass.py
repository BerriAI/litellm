"""
Tests for zero-cost model budget bypass functionality.

When a user exceeds their budget, the system should still allow requests
to models with zero cost (e.g., on-premises models).
"""

import asyncio
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    LiteLLM_TeamMembership,
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import (
    _check_team_member_budget,
    _is_model_cost_zero,
    _team_max_budget_check,
    common_checks,
)
from litellm.proxy.utils import ProxyLogging
from litellm.router import Router
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo


@pytest.fixture
def mock_router_with_zero_cost_model():
    """Create a mock router with a zero-cost model."""
    router = Router(
        model_list=[
            {
                "model_name": "on-prem-model",
                "litellm_params": {
                    "model": "ollama/llama2",
                    "api_base": "http://localhost:11434",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
                "model_info": {
                    "id": "on-prem-model-id",
                    "input_cost_per_token": 0.0,
                    "output_cost_per_token": 0.0,
                },
            },
            {
                "model_name": "cloud-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-test",
                },
                "model_info": {
                    "id": "cloud-model-id",
                },
            },
        ]
    )
    return router


@pytest.fixture
def mock_router_with_paid_model():
    """Create a mock router with only paid models."""
    router = Router(
        model_list=[
            {
                "model_name": "cloud-model",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "sk-test",
                },
                "model_info": {
                    "id": "cloud-model-id",
                },
            }
        ]
    )
    return router


@pytest.fixture
def mock_proxy_logging():
    """Create a mock ProxyLogging instance."""
    proxy_logging = ProxyLogging(user_api_key_cache=None)
    
    async def mock_budget_alerts(*args, **kwargs):
        pass
    
    proxy_logging.budget_alerts = mock_budget_alerts
    return proxy_logging


class TestIsModelCostZero:
    """Tests for _is_model_cost_zero helper function."""

    def test_zero_cost_model_in_router(self, mock_router_with_zero_cost_model):
        """Test that a zero-cost model in router is correctly identified."""
        result = _is_model_cost_zero(
            model="on-prem-model", llm_router=mock_router_with_zero_cost_model
        )
        assert result is True

    def test_paid_model_in_router(self, mock_router_with_zero_cost_model):
        """Test that a paid model is correctly identified as non-zero cost."""
        with patch("litellm.get_model_info") as mock_get_model_info:
            # Mock the return value for gpt-3.5-turbo
            mock_get_model_info.return_value = {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
            }
            result = _is_model_cost_zero(
                model="cloud-model", llm_router=mock_router_with_zero_cost_model
            )
            assert result is False

    def test_none_model(self, mock_router_with_zero_cost_model):
        """Test that None model returns False."""
        result = _is_model_cost_zero(
            model=None, llm_router=mock_router_with_zero_cost_model
        )
        assert result is False

    def test_none_router(self):
        """Test that None router returns False."""
        result = _is_model_cost_zero(model="some-model", llm_router=None)
        assert result is False

    def test_list_of_zero_cost_models(self, mock_router_with_zero_cost_model):
        """Test that a list of zero-cost models returns True."""
        result = _is_model_cost_zero(
            model=["on-prem-model"], llm_router=mock_router_with_zero_cost_model
        )
        assert result is True

    def test_mixed_cost_models(self, mock_router_with_zero_cost_model):
        """Test that a list with mixed cost models returns False."""
        with patch("litellm.get_model_info") as mock_get_model_info:
            mock_get_model_info.return_value = {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
            }
            result = _is_model_cost_zero(
                model=["on-prem-model", "cloud-model"],
                llm_router=mock_router_with_zero_cost_model,
            )
            assert result is False


class TestUserBudgetBypass:
    """Tests for user budget bypass with zero-cost models."""

    @pytest.mark.asyncio
    async def test_user_over_budget_with_zero_cost_model_allowed(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that user over budget can still use zero-cost models."""
        user_object = LiteLLM_UserTable(
            user_id="test-user",
            spend=100.0,
            max_budget=50.0,
        )

        request_body = {"model": "on-prem-model"}

        # Should not raise BudgetExceededError
        result = await common_checks(
            request_body=request_body,
            team_object=None,
            user_object=user_object,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/v1/chat/completions",
            llm_router=mock_router_with_zero_cost_model,
            proxy_logging_obj=mock_proxy_logging,
            valid_token=UserAPIKeyAuth(
                token="test-token",
                user_id="test-user",
            ),
            request=MagicMock(),
            skip_budget_checks=True,  # This is set by user_api_key_auth for zero-cost models
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_user_over_budget_with_paid_model_blocked(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that user over budget cannot use paid models."""
        user_object = LiteLLM_UserTable(
            user_id="test-user",
            spend=100.0,
            max_budget=50.0,
        )

        request_body = {"model": "cloud-model"}

        with patch("litellm.get_model_info") as mock_get_model_info:
            mock_get_model_info.return_value = {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
            }
            with pytest.raises(litellm.BudgetExceededError) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=user_object,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route="/v1/chat/completions",
                    llm_router=mock_router_with_zero_cost_model,
                    proxy_logging_obj=mock_proxy_logging,
                    valid_token=UserAPIKeyAuth(
                        token="test-token",
                        user_id="test-user",
                    ),
                    request=MagicMock(),
                )

            assert exc_info.value.current_cost == 100.0
            assert exc_info.value.max_budget == 50.0
            assert "test-user" in str(exc_info.value)


class TestEndUserBudgetBypass:
    """Tests for end user budget bypass with zero-cost models."""

    @pytest.mark.asyncio
    async def test_end_user_over_budget_with_zero_cost_model_allowed(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that end user over budget can still use zero-cost models."""
        end_user_budget = LiteLLM_BudgetTable(max_budget=20.0)
        end_user_object = LiteLLM_EndUserTable(
            user_id="end-user-123",
            spend=50.0,
            litellm_budget_table=end_user_budget,
            blocked=False,
        )

        request_body = {"model": "on-prem-model", "user": "end-user-123"}

        # In the real flow, skip_budget_checks would be set to True for zero-cost models
        result = await common_checks(
            request_body=request_body,
            team_object=None,
            user_object=None,
            end_user_object=end_user_object,
            global_proxy_spend=None,
            general_settings={},
            route="/v1/chat/completions",
            llm_router=mock_router_with_zero_cost_model,
            proxy_logging_obj=mock_proxy_logging,
            valid_token=UserAPIKeyAuth(
                token="test-token",
            ),
            request=MagicMock(),
            skip_budget_checks=True,  # This is set by user_api_key_auth for zero-cost models
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_end_user_over_budget_with_paid_model_blocked(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that end user over budget cannot use paid models."""
        end_user_budget = LiteLLM_BudgetTable(max_budget=20.0)
        end_user_object = LiteLLM_EndUserTable(
            user_id="end-user-123",
            spend=50.0,
            litellm_budget_table=end_user_budget,
            blocked=False,
        )

        request_body = {"model": "cloud-model", "user": "end-user-123"}

        with patch("litellm.get_model_info") as mock_get_model_info:
            mock_get_model_info.return_value = {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
            }
            with pytest.raises(litellm.BudgetExceededError) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=None,
                    user_object=None,
                    end_user_object=end_user_object,
                    global_proxy_spend=None,
                    general_settings={},
                    route="/v1/chat/completions",
                    llm_router=mock_router_with_zero_cost_model,
                    proxy_logging_obj=mock_proxy_logging,
                    valid_token=UserAPIKeyAuth(
                        token="test-token",
                    ),
                    request=MagicMock(),
                )

            assert exc_info.value.current_cost == 50.0
            assert exc_info.value.max_budget == 20.0
            assert "end-user-123" in str(exc_info.value)


class TestTeamBudgetBypass:
    """Tests for team budget bypass with zero-cost models."""

    @pytest.mark.asyncio
    async def test_team_over_budget_with_zero_cost_model_allowed(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that team over budget can still use zero-cost models."""
        team_object = LiteLLM_TeamTable(
            team_id="test-team",
            spend=150.0,
            max_budget=100.0,
        )

        valid_token = UserAPIKeyAuth(
            token="test-token",
            team_id="test-team",
        )

        request_body = {"model": "on-prem-model"}

        # In the real flow, skip_budget_checks would be set to True for zero-cost models
        result = await common_checks(
            request_body=request_body,
            team_object=team_object,
            user_object=None,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/v1/chat/completions",
            llm_router=mock_router_with_zero_cost_model,
            proxy_logging_obj=mock_proxy_logging,
            valid_token=valid_token,
            request=MagicMock(),
            skip_budget_checks=True,  # This is set by user_api_key_auth for zero-cost models
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_team_over_budget_with_paid_model_blocked(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that team over budget cannot use paid models."""
        team_object = LiteLLM_TeamTable(
            team_id="test-team",
            spend=150.0,
            max_budget=100.0,
        )

        valid_token = UserAPIKeyAuth(
            token="test-token",
            team_id="test-team",
        )

        request_body = {"model": "cloud-model"}

        with patch("litellm.get_model_info") as mock_get_model_info:
            mock_get_model_info.return_value = {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
            }
            with pytest.raises(litellm.BudgetExceededError) as exc_info:
                await common_checks(
                    request_body=request_body,
                    team_object=team_object,
                    user_object=None,
                    end_user_object=None,
                    global_proxy_spend=None,
                    general_settings={},
                    route="/v1/chat/completions",
                    llm_router=mock_router_with_zero_cost_model,
                    proxy_logging_obj=mock_proxy_logging,
                    valid_token=valid_token,
                    request=MagicMock(),
                )

            assert exc_info.value.current_cost == 150.0
            assert exc_info.value.max_budget == 100.0
            assert "test-team" in str(exc_info.value)


class TestTeamMemberBudgetBypass:
    """Tests for team member budget bypass with zero-cost models."""

    @pytest.mark.asyncio
    async def test_team_member_over_budget_with_zero_cost_model_allowed(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that team member over budget can still use zero-cost models."""
        team_object = LiteLLM_TeamTable(
            team_id="test-team",
        )

        user_object = LiteLLM_UserTable(
            user_id="test-user",
        )

        valid_token = UserAPIKeyAuth(
            token="test-token",
            user_id="test-user",
            team_id="test-team",
        )

        member_budget = LiteLLM_BudgetTable(max_budget=30.0)
        team_membership = LiteLLM_TeamMembership(
            user_id="test-user",
            team_id="test-team",
            spend=60.0,
            litellm_budget_table=member_budget,
        )

        request_body = {"model": "on-prem-model"}

        # Mock get_team_membership
        with patch(
            "litellm.proxy.auth.auth_checks.get_team_membership"
        ) as mock_get_membership:
            mock_get_membership.return_value = team_membership

            # In the real flow, skip_budget_checks would be set to True for zero-cost models
            result = await common_checks(
                request_body=request_body,
                team_object=team_object,
                user_object=user_object,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/v1/chat/completions",
                llm_router=mock_router_with_zero_cost_model,
                proxy_logging_obj=mock_proxy_logging,
                valid_token=valid_token,
                request=MagicMock(),
                skip_budget_checks=True,  # This is set by user_api_key_auth for zero-cost models
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_team_member_over_budget_with_paid_model_blocked(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that team member over budget cannot use paid models."""
        team_object = LiteLLM_TeamTable(
            team_id="test-team",
        )

        user_object = LiteLLM_UserTable(
            user_id="test-user",
        )

        valid_token = UserAPIKeyAuth(
            token="test-token",
            user_id="test-user",
            team_id="test-team",
        )

        member_budget = LiteLLM_BudgetTable(max_budget=30.0)
        team_membership = LiteLLM_TeamMembership(
            user_id="test-user",
            team_id="test-team",
            spend=60.0,
            litellm_budget_table=member_budget,
        )

        request_body = {"model": "cloud-model"}

        with patch(
            "litellm.proxy.auth.auth_checks.get_team_membership"
        ) as mock_get_membership:
            mock_get_membership.return_value = team_membership

            with patch("litellm.get_model_info") as mock_get_model_info:
                mock_get_model_info.return_value = {
                    "input_cost_per_token": 0.0000015,
                    "output_cost_per_token": 0.000002,
                }
                with pytest.raises(litellm.BudgetExceededError) as exc_info:
                    await common_checks(
                        request_body=request_body,
                        team_object=team_object,
                        user_object=user_object,
                        end_user_object=None,
                        global_proxy_spend=None,
                        general_settings={},
                        route="/v1/chat/completions",
                        llm_router=mock_router_with_zero_cost_model,
                        proxy_logging_obj=mock_proxy_logging,
                        valid_token=valid_token,
                        request=MagicMock(),
                    )

                assert exc_info.value.current_cost == 60.0
                assert exc_info.value.max_budget == 30.0
                assert "test-user" in str(exc_info.value)
                assert "test-team" in str(exc_info.value)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_model_not_in_router(self, mock_router_with_zero_cost_model):
        """Test behavior when model is not found in router."""
        with patch("litellm.get_model_info") as mock_get_model_info:
            # Simulate model not found
            mock_get_model_info.side_effect = Exception("Model not found")
            result = _is_model_cost_zero(
                model="nonexistent-model", llm_router=mock_router_with_zero_cost_model
            )
            # Should return False (conservative approach)
            assert result is False

    @pytest.mark.asyncio
    async def test_user_under_budget_with_paid_model_allowed(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that user under budget can use paid models normally."""
        user_object = LiteLLM_UserTable(
            user_id="test-user",
            spend=30.0,
            max_budget=100.0,
        )

        request_body = {"model": "cloud-model"}

        with patch("litellm.get_model_info") as mock_get_model_info:
            mock_get_model_info.return_value = {
                "input_cost_per_token": 0.0000015,
                "output_cost_per_token": 0.000002,
            }
            # Should not raise BudgetExceededError
            result = await common_checks(
                request_body=request_body,
                team_object=None,
                user_object=user_object,
                end_user_object=None,
                global_proxy_spend=None,
                general_settings={},
                route="/v1/chat/completions",
                llm_router=mock_router_with_zero_cost_model,
                proxy_logging_obj=mock_proxy_logging,
                valid_token=UserAPIKeyAuth(
                    token="test-token",
                    user_id="test-user",
                ),
                request=MagicMock(),
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_user_under_budget_with_zero_cost_model_allowed(
        self, mock_router_with_zero_cost_model, mock_proxy_logging
    ):
        """Test that user under budget can use zero-cost models normally."""
        user_object = LiteLLM_UserTable(
            user_id="test-user",
            spend=30.0,
            max_budget=100.0,
        )

        request_body = {"model": "on-prem-model"}

        # Should not raise BudgetExceededError
        result = await common_checks(
            request_body=request_body,
            team_object=None,
            user_object=user_object,
            end_user_object=None,
            global_proxy_spend=None,
            general_settings={},
            route="/v1/chat/completions",
            llm_router=mock_router_with_zero_cost_model,
            proxy_logging_obj=mock_proxy_logging,
            valid_token=UserAPIKeyAuth(
                token="test-token",
                user_id="test-user",
            ),
            request=MagicMock(),
        )
        assert result is True
