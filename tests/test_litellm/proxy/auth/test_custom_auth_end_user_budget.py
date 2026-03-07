import pytest
from unittest.mock import AsyncMock, patch
import litellm
from litellm.proxy.auth.user_api_key_auth import (
    _apply_custom_auth_team_overrides,
    _run_post_custom_auth_checks,
    update_valid_token_with_end_user_params,
)
from litellm.proxy._types import (
    LiteLLM_TeamTableCachedObj,
    UserAPIKeyAuth,
)


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
    with patch(
        "litellm.proxy.auth.user_api_key_auth.common_checks", new_callable=AsyncMock
    ) as mock_common, patch(
        "litellm.proxy.proxy_server.general_settings",
        {"custom_auth_run_common_checks": True},
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


def test_apply_custom_auth_team_overrides_unit():
    """
    Direct unit test of _apply_custom_auth_team_overrides: custom auth values
    fill in DB defaults, but DB explicit values are preserved.
    """
    team_obj = LiteLLM_TeamTableCachedObj(
        team_id="team-1",
        models=[],
        max_budget=None,
        tpm_limit=None,
        rpm_limit=None,
        soft_budget=None,
        blocked=False,
    )
    valid_token = UserAPIKeyAuth(
        token="test_token",
        team_id="team-1",
        team_models=["gpt-4", "gpt-3.5-turbo"],
        team_max_budget=200.0,
        team_tpm_limit=5000,
        team_rpm_limit=100,
        team_soft_budget=150.0,
        team_blocked=True,
    )

    result = _apply_custom_auth_team_overrides(team_obj, valid_token)

    assert result.models == ["gpt-4", "gpt-3.5-turbo"]
    assert result.max_budget == 200.0
    assert result.tpm_limit == 5000
    assert result.rpm_limit == 100
    assert result.soft_budget == 150.0
    assert result.blocked is True


def test_apply_custom_auth_team_overrides_db_wins():
    """
    When DB team has explicit values, they should not be overridden
    by custom auth values.
    """
    team_obj = LiteLLM_TeamTableCachedObj(
        team_id="team-1",
        models=["gpt-3.5-turbo"],
        max_budget=500.0,
        tpm_limit=10000,
        rpm_limit=200,
    )
    valid_token = UserAPIKeyAuth(
        token="test_token",
        team_id="team-1",
        team_models=["gpt-4"],
        team_max_budget=100.0,
        team_tpm_limit=5000,
        team_rpm_limit=50,
    )

    result = _apply_custom_auth_team_overrides(team_obj, valid_token)

    # DB values should be preserved
    assert result.models == ["gpt-3.5-turbo"]
    assert result.max_budget == 500.0
    assert result.tpm_limit == 10000
    assert result.rpm_limit == 200


def test_apply_custom_auth_team_overrides_backward_compat():
    """
    When neither custom auth nor DB sets values, defaults are preserved
    (empty models = all access).
    """
    team_obj = LiteLLM_TeamTableCachedObj(
        team_id="team-1",
        models=[],
        max_budget=None,
    )
    valid_token = UserAPIKeyAuth(
        token="test_token",
        team_id="team-1",
        # team_models defaults to []
    )

    result = _apply_custom_auth_team_overrides(team_obj, valid_token)

    assert result.models == []
    assert result.max_budget is None


@pytest.mark.asyncio
async def test_custom_auth_team_models_enforced_when_db_has_empty_models():
    """
    When custom auth sets team_models and the DB team has models=[],
    the custom auth team_models should be applied to the team object
    passed to common_checks.
    """
    valid_token = UserAPIKeyAuth(
        token="test_token",
        team_id="test-team",
        team_models=["gpt-4"],
    )

    mock_team_obj = LiteLLM_TeamTableCachedObj(
        team_id="test-team",
        models=[],
    )

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team_obj,
    ):
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common:
            mock_common.return_value = True
            await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=None,
                request_data={"model": "gpt-3.5-turbo"},
                route="/v1/chat/completions",
                parent_otel_span=None,
            )
            call_kwargs = mock_common.call_args[1]
            assert call_kwargs["team_object"].models == ["gpt-4"]


@pytest.mark.asyncio
async def test_custom_auth_team_models_not_overridden_when_db_has_models():
    """
    When the DB team has explicit models, those should be used even if
    custom auth set different team_models.
    """
    valid_token = UserAPIKeyAuth(
        token="test_token",
        team_id="test-team",
        team_models=["gpt-4"],
    )

    mock_team_obj = LiteLLM_TeamTableCachedObj(
        team_id="test-team",
        models=["gpt-3.5-turbo"],
    )

    with patch(
        "litellm.proxy.auth.user_api_key_auth.get_team_object",
        new_callable=AsyncMock,
        return_value=mock_team_obj,
    ):
        with patch(
            "litellm.proxy.auth.user_api_key_auth.common_checks",
            new_callable=AsyncMock,
        ) as mock_common:
            mock_common.return_value = True
            await _run_post_custom_auth_checks(
                valid_token=valid_token,
                request=None,
                request_data={"model": "gpt-4"},
                route="/v1/chat/completions",
                parent_otel_span=None,
            )
            call_kwargs = mock_common.call_args[1]
            assert call_kwargs["team_object"].models == ["gpt-3.5-turbo"]
