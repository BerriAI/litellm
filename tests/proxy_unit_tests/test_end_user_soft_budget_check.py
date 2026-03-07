import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from litellm.proxy.auth.auth_checks import _end_user_soft_budget_check
from litellm.proxy._types import (
    LiteLLM_EndUserTable,
    LiteLLM_BudgetTable,
    UserAPIKeyAuth,
)


@pytest.mark.asyncio
async def test_end_user_soft_budget_alert_triggers():
    # Setup
    mock_proxy_logging = MagicMock()
    mock_proxy_logging.budget_alerts = AsyncMock()

    end_user_obj = LiteLLM_EndUserTable(
        user_id="end-user-soft-budget",
        spend=15.0,
        blocked=False,
        litellm_budget_table=LiteLLM_BudgetTable(
            budget_id="budget-1", max_budget=20.0, soft_budget=10.0
        ),
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user",
        team_id="test-team",
        team_alias="team-alias",
        org_id="org-id",
        key_alias="key-alias",
    )

    # Execute
    await _end_user_soft_budget_check(
        end_user_object=end_user_obj,
        valid_token=valid_token,
        proxy_logging_obj=mock_proxy_logging,
    )

    # Let the background task run
    await asyncio.sleep(0.1)

    # Verify
    mock_proxy_logging.budget_alerts.assert_called_once()
    args, kwargs = mock_proxy_logging.budget_alerts.call_args
    assert kwargs["type"] == "soft_budget"
    assert kwargs["user_info"].customer_id == "end-user-soft-budget"
    assert kwargs["user_info"].spend == 15.0
    assert kwargs["user_info"].soft_budget == 10.0
