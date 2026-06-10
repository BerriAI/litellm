"""
Regression coverage for per-organization-member budget enforcement.

Mirrors test_team_member_budget.py. These tests call _check_organization_member_budget
directly; they fail until that helper exists and is wired into common_checks.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import litellm
from litellm.proxy._types import (
    LiteLLM_BudgetTable,
    LiteLLM_OrganizationMembershipTable,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
def _org_member_budget_check():
    import litellm.proxy.auth.auth_checks as auth_checks

    fn = getattr(auth_checks, "_check_organization_member_budget", None)
    if fn is None:
        pytest.fail("_check_organization_member_budget is not implemented")
    return fn


@pytest.mark.asyncio
async def test_organization_member_budget_check_exceeds_budget():
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        organization_id="test-org-1",
        spend=0.0,
        max_budget=None,
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        org_id="test-org-1",
        team_id="test-team-1",
        models=["gpt-3.5-turbo"],
    )

    now = datetime.now(timezone.utc)
    org_membership = LiteLLM_OrganizationMembershipTable(
        user_id="test-user-1",
        organization_id="test-org-1",
        spend=0.0000002,
        created_at=now,
        updated_at=now,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=0.0000001),
    )

    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    with (
        patch(
            "litellm.proxy.auth.auth_checks.get_organization_membership",
            new_callable=AsyncMock,
            return_value=org_membership,
        ),
        patch(
            "litellm.proxy.proxy_server.get_current_spend",
            new_callable=AsyncMock,
            return_value=0.0000002,
        ),
    ):
        with pytest.raises(litellm.BudgetExceededError) as exc_info:
            await _org_member_budget_check()(
                team_object=team_object,
                valid_token=valid_token,
                prisma_client=mock_prisma_client,
                user_api_key_cache=mock_user_api_key_cache,
                proxy_logging_obj=mock_proxy_logging_obj,
            )

        assert "Budget has been exceeded" in str(exc_info.value)
        assert "test-user-1" in str(exc_info.value)
        assert "test-org-1" in str(exc_info.value)


@pytest.mark.asyncio
async def test_organization_member_budget_check_within_budget():
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        organization_id="test-org-1",
        spend=0.0,
        max_budget=None,
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        org_id="test-org-1",
        team_id="test-team-1",
        models=["gpt-3.5-turbo"],
    )

    now = datetime.now(timezone.utc)
    org_membership = LiteLLM_OrganizationMembershipTable(
        user_id="test-user-1",
        organization_id="test-org-1",
        spend=0.00000005,
        created_at=now,
        updated_at=now,
        litellm_budget_table=LiteLLM_BudgetTable(max_budget=0.0000001),
    )

    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    with (
        patch(
            "litellm.proxy.auth.auth_checks.get_organization_membership",
            new_callable=AsyncMock,
            return_value=org_membership,
        ),
        patch(
            "litellm.proxy.proxy_server.get_current_spend",
            new_callable=AsyncMock,
            return_value=0.00000005,
        ),
    ):
        await _org_member_budget_check()(
            team_object=team_object,
            valid_token=valid_token,
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_user_api_key_cache,
            proxy_logging_obj=mock_proxy_logging_obj,
        )


@pytest.mark.asyncio
async def test_organization_member_budget_check_no_budget_set():
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        organization_id="test-org-1",
        spend=0.0,
        max_budget=None,
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        org_id="test-org-1",
        team_id="test-team-1",
        models=["gpt-3.5-turbo"],
    )

    now = datetime.now(timezone.utc)
    org_membership = LiteLLM_OrganizationMembershipTable(
        user_id="test-user-1",
        organization_id="test-org-1",
        spend=0.0,
        created_at=now,
        updated_at=now,
        litellm_budget_table=None,
    )

    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    with patch(
        "litellm.proxy.auth.auth_checks.get_organization_membership",
        new_callable=AsyncMock,
        return_value=org_membership,
    ):
        await _org_member_budget_check()(
            team_object=team_object,
            valid_token=valid_token,
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_user_api_key_cache,
            proxy_logging_obj=mock_proxy_logging_obj,
        )


@pytest.mark.asyncio
async def test_organization_member_budget_check_no_membership():
    team_object = LiteLLM_TeamTable(
        team_id="test-team-1",
        organization_id="test-org-1",
        spend=0.0,
        max_budget=None,
    )

    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        org_id="test-org-1",
        team_id="test-team-1",
        models=["gpt-3.5-turbo"],
    )

    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    with patch(
        "litellm.proxy.auth.auth_checks.get_organization_membership",
        new_callable=AsyncMock,
        return_value=None,
    ):
        await _org_member_budget_check()(
            team_object=team_object,
            valid_token=valid_token,
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_user_api_key_cache,
            proxy_logging_obj=mock_proxy_logging_obj,
        )


@pytest.mark.asyncio
async def test_organization_member_budget_check_skipped_without_org_context():
    valid_token = UserAPIKeyAuth(
        token="test-token",
        user_id="test-user-1",
        org_id=None,
        team_id=None,
        models=["gpt-3.5-turbo"],
    )

    mock_prisma_client = MagicMock()
    mock_user_api_key_cache = MagicMock()
    mock_proxy_logging_obj = MagicMock()

    with patch(
        "litellm.proxy.auth.auth_checks.get_organization_membership",
        new_callable=AsyncMock,
    ) as mock_get_org_membership:
        await _org_member_budget_check()(
            team_object=None,
            valid_token=valid_token,
            prisma_client=mock_prisma_client,
            user_api_key_cache=mock_user_api_key_cache,
            proxy_logging_obj=mock_proxy_logging_obj,
        )

        mock_get_org_membership.assert_not_called()
