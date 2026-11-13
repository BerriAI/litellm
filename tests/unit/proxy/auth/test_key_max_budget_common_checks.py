"""
Regression tests for #45605: key max_budget enforcement in common_checks.

With custom_auth_run_common_checks enabled, custom-auth requests only pass
through common_checks() - the stock virtual-key branch that calls
virtual_key_max_budget_check is never reached. common_checks must therefore
enforce the key's own max_budget itself, or a budgeted key overspends.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import common_checks
from litellm.proxy.utils import ProxyLogging


def _make_common_checks_kwargs(valid_token: UserAPIKeyAuth, route: str) -> dict:
    mock_request = MagicMock()
    mock_request.url.path = route

    mock_proxy_logging = MagicMock(spec=ProxyLogging)
    mock_proxy_logging.budget_alerts = AsyncMock()

    return {
        "request_body": {"model": "gpt-4"},
        "team_object": None,
        "user_object": None,
        "end_user_object": None,
        "global_proxy_spend": None,
        "general_settings": {},
        "route": route,
        "llm_router": None,
        "proxy_logging_obj": mock_proxy_logging,
        "valid_token": valid_token,
        "request": mock_request,
    }


@pytest.mark.asyncio
async def test_key_max_budget_enforced_in_common_checks():
    """
    A key over its own max_budget must be rejected by common_checks on LLM
    routes. On unfixed main this raises nothing - the check only ran in the
    stock virtual-key branch of user_api_key_auth_builder.
    """
    valid_token = UserAPIKeyAuth(
        token="sk-test-key-over-budget",
        max_budget=100.0,
        spend=150.0,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client"),
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.get_current_spend", new_callable=AsyncMock) as mock_spend,
    ):
        mock_spend.return_value = 150.0

        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await common_checks(**_make_common_checks_kwargs(valid_token=valid_token, route="/v1/chat/completions"))

        assert exc_info.value.max_budget == 100.0
        assert exc_info.value.current_cost == 150.0


@pytest.mark.asyncio
async def test_key_under_max_budget_passes_common_checks():
    """A key under its max_budget sails through."""
    valid_token = UserAPIKeyAuth(
        token="sk-test-key-under-budget",
        max_budget=100.0,
        spend=10.0,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client"),
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.get_current_spend", new_callable=AsyncMock) as mock_spend,
    ):
        mock_spend.return_value = 10.0

        result = await common_checks(
            **_make_common_checks_kwargs(valid_token=valid_token, route="/v1/chat/completions")
        )
        assert result is True


@pytest.mark.asyncio
async def test_key_max_budget_not_enforced_on_non_llm_routes():
    """
    Parity with the stock branch: the key max_budget check is gated on LLM
    API routes, so key-management routes stay reachable even over budget.
    """
    valid_token = UserAPIKeyAuth(
        token="sk-test-key-admin-route",
        max_budget=100.0,
        spend=150.0,
    )

    with (
        patch("litellm.proxy.proxy_server.prisma_client"),
        patch("litellm.proxy.proxy_server.user_api_key_cache"),
        patch("litellm.proxy.proxy_server.get_current_spend", new_callable=AsyncMock) as mock_spend,
    ):
        mock_spend.return_value = 150.0

        result = await common_checks(**_make_common_checks_kwargs(valid_token=valid_token, route="/key/info"))
        assert result is True
        mock_spend.assert_not_called()
