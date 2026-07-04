import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import update_valid_token_with_end_user_params


def test_update_valid_token_applies_end_user_model_max_budget_from_params():
    valid_token = UserAPIKeyAuth(token="test-key")
    end_user_params = {
        "end_user_id": "customer-1",
        "end_user_model_max_budget": {
            "google/gemini-2.5-flash-lite": {"max_budget": 1e-05, "budget_duration": "1d"}
        },
    }

    result = update_valid_token_with_end_user_params(valid_token, end_user_params)

    assert result.end_user_id == "customer-1"
    assert result.end_user_model_max_budget == end_user_params["end_user_model_max_budget"]


@pytest.mark.asyncio
async def test_enforce_end_user_model_max_budget_raises_when_over_budget():
    from litellm.proxy.auth.user_api_key_auth import _enforce_end_user_model_max_budget_checks

    valid_token = UserAPIKeyAuth(
        token="master-key",
        end_user_id="customer-1",
        end_user_model_max_budget={
            "google/gemini-2.5-flash-lite": {"max_budget": 1e-05, "budget_duration": "1d"}
        },
    )
    request = MagicMock()
    request_data = {"model": "google/gemini-2.5-flash-lite"}

    with patch(
        "litellm.proxy.auth.user_api_key_auth._get_model_from_request_context",
        return_value="google/gemini-2.5-flash-lite",
    ):
        with patch(
            "litellm.proxy.proxy_server.model_max_budget_limiter.is_end_user_within_model_budget",
            new_callable=AsyncMock,
        ) as mock_check:
            mock_check.side_effect = litellm.BudgetExceededError(
                message="Exceeded budget", current_cost=0.0002, max_budget=1e-05
            )

            with pytest.raises(litellm.BudgetExceededError):
                await _enforce_end_user_model_max_budget_checks(
                    valid_token=valid_token,
                    request_data=request_data,
                    route="/v1/chat/completions",
                    request=request,
                )

            mock_check.assert_awaited_once()
