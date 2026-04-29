import pytest
from unittest.mock import AsyncMock, patch
import litellm
from litellm.proxy.auth.user_api_key_auth import (
    _run_post_custom_auth_checks,
    update_valid_token_with_end_user_params,
)
from litellm.proxy._types import UserAPIKeyAuth


@pytest.mark.asyncio
async def test_custom_auth_run_post_custom_auth_checks_without_end_user_id():
    # Test backwards compatibility — common_checks only runs when opt-in flag is set
    valid_token = UserAPIKeyAuth(token="test_token")

    # Default: common_checks should NOT be called
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

    # With opt-in flag: common_checks SHOULD be called
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
        mock_common.assert_awaited_once()


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


def test_update_valid_token_does_not_mutate_original_token():
    """
    Request-scoped end-user limits must not mutate the cached UserAPIKeyAuth object.
    """
    valid_token = UserAPIKeyAuth(
        token="test_token",
        end_user_id=None,
        end_user_tpm_limit=None,
        end_user_rpm_limit=None,
        allowed_model_region=None,
    )
    end_user_params = {
        "end_user_id": "attacker-user",
        "end_user_tpm_limit": 1,
        "end_user_rpm_limit": 1,
        "allowed_model_region": "eu",
    }

    result = update_valid_token_with_end_user_params(valid_token, end_user_params)

    assert result is not valid_token
    assert result.end_user_id == "attacker-user"
    assert result.end_user_tpm_limit == 1
    assert result.end_user_rpm_limit == 1
    assert result.allowed_model_region == "eu"
    assert valid_token.end_user_id is None
    assert valid_token.end_user_tpm_limit is None
    assert valid_token.end_user_rpm_limit is None
    assert valid_token.allowed_model_region is None
