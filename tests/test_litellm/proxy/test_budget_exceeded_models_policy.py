import pytest
import litellm
from unittest.mock import MagicMock, patch
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.proxy_server import _apply_budget_exceeded_models_policy


@pytest.mark.asyncio
async def test_apply_budget_exceeded_models_policy_all():
    """
    When policy is "all", the full model list is returned regardless of budget.
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-123",
        max_budget=10.0,
        spend=15.0,  # Over budget
    )

    all_models = ["gpt-4", "gpt-3.5-turbo"]

    mock_settings = {"budget_exceeded_models_policy": "all"}
    mock_router = MagicMock()

    with (
        patch("litellm.proxy.proxy_server.general_settings", mock_settings),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):

        result = _apply_budget_exceeded_models_policy(user_api_key_dict, all_models)
        assert result == all_models


@pytest.mark.asyncio
async def test_apply_budget_exceeded_models_policy_default_is_all():
    """
    When no policy is configured, an over-budget caller still gets the full model
    list. Model-discovery routes are exempt from budget enforcement by default
    (MODEL_DISCOVERY_ROUTES / skip_budget_checks), so the default must not raise.
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-123",
        max_budget=10.0,
        spend=15.0,  # Over budget
    )

    all_models = ["gpt-4", "gpt-3.5-turbo"]

    mock_settings = {}  # policy not set -> default behavior
    mock_router = MagicMock()

    with (
        patch("litellm.proxy.proxy_server.general_settings", mock_settings),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):

        result = _apply_budget_exceeded_models_policy(user_api_key_dict, all_models)
        assert result == all_models


@pytest.mark.asyncio
async def test_apply_budget_exceeded_models_policy_blocked():
    """
    When policy is "blocked" and budget is exceeded, BudgetExceededError is raised.
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-123",
        max_budget=10.0,
        spend=15.0,  # Over budget
    )

    all_models = ["gpt-4", "gpt-3.5-turbo"]

    mock_settings = {"budget_exceeded_models_policy": "blocked"}
    mock_router = MagicMock()

    with (
        patch("litellm.proxy.proxy_server.general_settings", mock_settings),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):

        with pytest.raises(litellm.BudgetExceededError):
            _apply_budget_exceeded_models_policy(user_api_key_dict, all_models)


@pytest.mark.asyncio
async def test_apply_budget_exceeded_models_policy_free_only():
    """
    When policy is "free_only" and budget is exceeded, only zero-cost models are returned.
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-123",
        max_budget=10.0,
        spend=15.0,  # Over budget
    )

    all_models = ["gpt-4", "free-model"]

    mock_settings = {"budget_exceeded_models_policy": "free_only"}
    mock_router = MagicMock()

    def mock_is_model_cost_zero(model, router):
        return model == "free-model"

    with (
        patch("litellm.proxy.proxy_server.general_settings", mock_settings),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch(
            "litellm.proxy.auth.auth_checks._is_model_cost_zero",
            side_effect=mock_is_model_cost_zero,
        ),
    ):

        result = _apply_budget_exceeded_models_policy(user_api_key_dict, all_models)
        assert result == ["free-model"]


@pytest.mark.asyncio
async def test_apply_budget_exceeded_models_policy_under_budget():
    """
    When the caller is under budget, all models are returned regardless of policy.
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-123",
        max_budget=10.0,
        spend=5.0,  # Under budget
    )

    all_models = ["gpt-4", "free-model"]

    mock_settings = {"budget_exceeded_models_policy": "free_only"}
    mock_router = MagicMock()

    with (
        patch("litellm.proxy.proxy_server.general_settings", mock_settings),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
    ):

        result = _apply_budget_exceeded_models_policy(user_api_key_dict, all_models)
        assert result == all_models


@pytest.mark.asyncio
async def test_apply_budget_exceeded_models_policy_team_budget_exceeded():
    """
    Tests that team budget exceedance is also honored.
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-123",
        team_max_budget=100.0,
        team_spend=120.0,  # Over budget
    )

    all_models = ["gpt-4", "free-model"]

    mock_settings = {"budget_exceeded_models_policy": "free_only"}
    mock_router = MagicMock()

    def mock_is_model_cost_zero(model, router):
        return model == "free-model"

    with (
        patch("litellm.proxy.proxy_server.general_settings", mock_settings),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch(
            "litellm.proxy.auth.auth_checks._is_model_cost_zero",
            side_effect=mock_is_model_cost_zero,
        ),
    ):

        result = _apply_budget_exceeded_models_policy(user_api_key_dict, all_models)
        assert result == ["free-model"]


@pytest.mark.asyncio
async def test_apply_budget_exceeded_models_policy_user_budget_exceeded():
    """
    Tests that user budget exceedance is also honored.
    """
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-123",
        user_max_budget=50.0,
        user_spend=60.0,  # Over budget
    )

    all_models = ["gpt-4", "free-model"]

    mock_settings = {"budget_exceeded_models_policy": "free_only"}
    mock_router = MagicMock()

    def mock_is_model_cost_zero(model, router):
        return model == "free-model"

    with (
        patch("litellm.proxy.proxy_server.general_settings", mock_settings),
        patch("litellm.proxy.proxy_server.llm_router", mock_router),
        patch(
            "litellm.proxy.auth.auth_checks._is_model_cost_zero",
            side_effect=mock_is_model_cost_zero,
        ),
    ):

        result = _apply_budget_exceeded_models_policy(user_api_key_dict, all_models)
        assert result == ["free-model"]
