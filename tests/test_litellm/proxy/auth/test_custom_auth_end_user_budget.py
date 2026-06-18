import pytest
from unittest.mock import AsyncMock, patch
import litellm
from litellm.proxy.auth.user_api_key_auth import (
    _run_post_custom_auth_checks,
    update_valid_token_with_end_user_params,
)
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_EndUserTable,
    UserAPIKeyAuth,
)


@pytest.mark.asyncio
async def test_custom_auth_run_post_custom_auth_checks_without_end_user_id():
    # common_checks now runs in the user_api_key_auth wrapper via
    # _run_centralized_common_checks, gated by
    # custom_auth_run_common_checks for custom-auth deployments. The
    # helper itself no longer calls common_checks.
    valid_token = UserAPIKeyAuth(token="test_token")

    # Default: common_checks should NOT be called inside the helper
    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ) as mock_common:
        mock_common.return_value = True
        result = await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data={},
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        assert result.token == "test_token"
        assert getattr(result, "end_user_id", None) is None
        mock_common.assert_not_awaited()

    # With opt-in flag: still not from the helper — the centralized gate
    # in the wrapper handles it.
    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
        ) as mock_common,
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ),
    ):
        mock_common.return_value = True
        result = await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data={},
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        assert result.token == "test_token"
        mock_common.assert_not_awaited()


@pytest.mark.asyncio
async def test_custom_auth_run_post_custom_auth_checks_with_end_user_budget_exceeded():
    valid_token = UserAPIKeyAuth(
        token="test_token",
        end_user_id="test_user",
        end_user_model_max_budget={
            "gpt-4": {"budget_limit": 10.0, "time_period": "1d"}
        },
    )
    request_data = {"model": "gpt-4"}

    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ):
        with patch(
            "litellm.proxy.proxy_server.model_max_budget_limiter.is_end_user_within_model_budget",
            new_callable=AsyncMock,
        ) as mock_budget_check:
            mock_budget_check.side_effect = litellm.BudgetExceededError(
                message="Exceeded budget", current_cost=20.0, max_budget=10.0
            )

            with pytest.raises(litellm.BudgetExceededError):
                await _run_post_custom_auth_checks(
                    valid_token=valid_token,
                    request=None,
                    request_data=request_data,
                    route="/v1/chat/completions",
                    parent_otel_span=None,
                )
            mock_budget_check.assert_awaited_once()


@pytest.mark.asyncio
async def test_custom_auth_enforces_end_user_budget_when_common_checks_skipped():
    # custom-auth deployments with custom_auth_run_common_checks unset skip
    # common_checks() (and its end-user budget enforcement) in the centralized
    # gate, so the helper must enforce the end-user budget itself. Regression:
    # an over-budget end user must be rejected on this path.
    valid_token = UserAPIKeyAuth(token="test_token", end_user_id="customer-1")
    over_budget_end_user = LiteLLM_EndUserTable(
        user_id="customer-1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    async def mock_get_current_spend(counter_key, fallback_spend, max_budget=None, **kwargs):
        if counter_key == "spend:end_user:customer-1":
            return 5.0
        return fallback_spend

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
            new_callable=AsyncMock,
            return_value=over_budget_end_user,
        ),
        patch("litellm.proxy.proxy_server.get_current_spend", mock_get_current_spend),
        patch("litellm.proxy.proxy_server.general_settings", {}),
    ):
        with pytest.raises(litellm.BudgetExceededError):
            await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=None,
                request_data={"model": "gpt-4"},
                route="/v1/chat/completions",
                parent_otel_span=None,
            )


@pytest.mark.asyncio
async def test_custom_auth_defers_end_user_budget_to_common_checks_when_enabled():
    # With custom_auth_run_common_checks set, the wrapper's common_checks()
    # enforces the end-user budget, so the helper must not double-enforce it.
    valid_token = UserAPIKeyAuth(token="test_token", end_user_id="customer-1")
    end_user_obj = LiteLLM_EndUserTable(
        user_id="customer-1",
        blocked=False,
        spend=0.0,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=1.0),
    )

    with (
        patch(
            "litellm.proxy.auth.user_api_key_auth.get_end_user_object",
            new_callable=AsyncMock,
            return_value=end_user_obj,
        ),
        patch(
            "litellm.proxy.auth.user_api_key_auth._check_end_user_budget",
            new_callable=AsyncMock,
        ) as mock_check,
        patch(
            "litellm.proxy.auth.user_api_key_auth._enforce_key_and_fallback_model_access",
            new_callable=AsyncMock,
        ),
        patch(
            "litellm.proxy.proxy_server.general_settings",
            {"custom_auth_run_common_checks": True},
        ),
    ):
        await _run_post_custom_auth_checks(
            valid_token=valid_token,
            request=None,
            request_data={"model": "gpt-4"},
            route="/v1/chat/completions",
            parent_otel_span=None,
        )
        mock_check.assert_not_awaited()


def test_update_valid_token_does_not_override_custom_auth_values_with_none():
    """
    Greptile feedback: if custom auth sets end_user_model_max_budget on the token,
    but the DB end_user has no model_max_budget in their budget table, the DB lookup
    should NOT clear the custom-auth-provided value.
    """
    custom_auth_budget = {"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}}
    valid_token = UserAPIKeyAuth(
        token="test_token",
        end_user_id="user_1",
        end_user_tpm_limit=100,
        end_user_rpm_limit=50,
        end_user_model_max_budget=custom_auth_budget,
    )

    # Simulate DB lookup that found the end_user but budget table has no limits set
    end_user_params = {
        "end_user_id": "user_1",
        "allowed_model_region": None,
        # No tpm_limit, rpm_limit, or model_max_budget from DB
    }

    result = update_valid_token_with_end_user_params(valid_token, end_user_params)

    # Custom-auth-provided values should be preserved, not cleared to None
    assert result.end_user_tpm_limit == 100
    assert result.end_user_rpm_limit == 50
    assert result.end_user_model_max_budget == custom_auth_budget
    assert result.end_user_id == "user_1"


def test_update_valid_token_db_values_override_custom_auth_when_set():
    """
    When the DB budget table has explicit values, they should override
    whatever the custom auth function set (DB is source of truth).
    """
    valid_token = UserAPIKeyAuth(
        token="test_token",
        end_user_id="user_1",
        end_user_tpm_limit=100,
        end_user_model_max_budget={"gpt-4": {"budget_limit": 5.0, "time_period": "1d"}},
    )

    db_budget = {"gpt-4": {"budget_limit": 20.0, "time_period": "1d"}}
    end_user_params = {
        "end_user_id": "user_1",
        "end_user_tpm_limit": 500,
        "end_user_model_max_budget": db_budget,
    }

    result = update_valid_token_with_end_user_params(valid_token, end_user_params)

    # DB values should win
    assert result.end_user_tpm_limit == 500
    assert result.end_user_model_max_budget == db_budget
