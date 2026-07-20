"""Tests for the authorization-scoped usage data provider.

The security property under test: a non-admin (UserScope) provider can only
ever read its own user_id, no matter what filter a tool passes, and team/tag
breakdowns are refused for non-admins.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.proxy.management_endpoints.dashboard_ai.scoped_data import (
    AdminScope,
    ScopedUsageDataProvider,
    UserScope,
    summarise_entity_data,
    summarise_usage_data,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    SpendAnalyticsPaginatedResponse,
)


def _as_response(payload) -> SpendAnalyticsPaginatedResponse:
    return SpendAnalyticsPaginatedResponse.model_validate(payload)


SAMPLE_AGGREGATED_RESPONSE = {
    "results": [
        {
            "date": "2025-01-15",
            "metrics": {"spend": 50.25, "total_tokens": 30000, "api_requests": 500},
            "breakdown": {
                "models": {
                    "gpt-4": {"metrics": {"spend": 40.0, "api_requests": 300, "total_tokens": 25000}, "metadata": {}},
                },
                "providers": {
                    "openai": {"metrics": {"spend": 50.25, "api_requests": 500}, "metadata": {}},
                },
                "entities": {},
            },
        },
    ],
    "metadata": {
        "total_spend": 50.25,
        "total_api_requests": 500,
        "total_successful_requests": 480,
        "total_failed_requests": 20,
        "total_tokens": 30000,
    },
}

SAMPLE_TEAM_RESPONSE = {
    "results": [
        {
            "date": "2025-01-15",
            "metrics": {"spend": 100.0},
            "breakdown": {
                "entities": {
                    "team-1": {
                        "metrics": {"spend": 60.0, "api_requests": 600, "total_tokens": 30000},
                        "metadata": {"alias": "Engineering"},
                    },
                    "team-2": {
                        "metrics": {"spend": 40.0, "api_requests": 400, "total_tokens": 20000},
                        "metadata": {"alias": "Marketing"},
                    },
                },
            },
        },
    ],
    "metadata": {"total_spend": 100.0},
}


def _response_mock(payload):
    resp = MagicMock()
    resp.model_dump.return_value = payload
    return resp


class TestUserScopeForcesOwnUserId:
    @pytest.mark.asyncio
    async def test_user_scope_ignores_model_supplied_user_id_filter(self):
        """Cross-tenant regression: a non-admin query must be scoped to the
        caller's own user_id even when the tool arguments name a different one."""
        provider = ScopedUsageDataProvider(scope=UserScope(user_id="my-user"), prisma_client=MagicMock())

        with patch(
            "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity_aggregated",
            new_callable=AsyncMock,
        ) as mock_agg:
            mock_agg.return_value = _response_mock(SAMPLE_AGGREGATED_RESPONSE)

            await provider.usage(start_date="2025-01-01", end_date="2025-01-31", user_id_filter="other-user")

        assert mock_agg.call_args.kwargs["entity_id"] == "my-user"

    @pytest.mark.asyncio
    async def test_admin_scope_honors_supplied_user_id_filter(self):
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="admin-1"), prisma_client=MagicMock())

        with patch(
            "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity_aggregated",
            new_callable=AsyncMock,
        ) as mock_agg:
            mock_agg.return_value = _response_mock(SAMPLE_AGGREGATED_RESPONSE)

            await provider.usage(start_date="2025-01-01", end_date="2025-01-31", user_id_filter="target-user")

        assert mock_agg.call_args.kwargs["entity_id"] == "target-user"

    @pytest.mark.asyncio
    async def test_admin_scope_global_view_when_no_filter(self):
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id="admin-1"), prisma_client=MagicMock())

        with patch(
            "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity_aggregated",
            new_callable=AsyncMock,
        ) as mock_agg:
            mock_agg.return_value = _response_mock(SAMPLE_AGGREGATED_RESPONSE)

            await provider.usage(start_date="2025-01-01", end_date="2025-01-31", user_id_filter=None)

        assert mock_agg.call_args.kwargs["entity_id"] is None


class TestAdminOnlyBreakdowns:
    @pytest.mark.asyncio
    async def test_user_scope_cannot_query_team_data(self):
        provider = ScopedUsageDataProvider(scope=UserScope(user_id="u1"), prisma_client=MagicMock())
        with pytest.raises(PermissionError):
            await provider.team(start_date="2025-01-01", end_date="2025-01-31", team_ids=None)

    @pytest.mark.asyncio
    async def test_user_scope_cannot_query_tag_data(self):
        provider = ScopedUsageDataProvider(scope=UserScope(user_id="u1"), prisma_client=MagicMock())
        with pytest.raises(PermissionError):
            await provider.tag(start_date="2025-01-01", end_date="2025-01-31", tags=None)

    @pytest.mark.asyncio
    async def test_admin_scope_can_query_team_data_with_parsed_ids(self):
        provider = ScopedUsageDataProvider(scope=AdminScope(caller_user_id=None), prisma_client=MagicMock())
        with patch(
            "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity",
            new_callable=AsyncMock,
        ) as mock_paginated:
            mock_paginated.return_value = _response_mock(SAMPLE_TEAM_RESPONSE)
            await provider.team(start_date="2025-01-01", end_date="2025-01-31", team_ids="team-1, team-2")

        assert mock_paginated.call_args.kwargs["entity_id"] == ["team-1", "team-2"]


class TestIsAdminFlag:
    def test_admin_scope_is_admin(self):
        assert ScopedUsageDataProvider(scope=AdminScope(caller_user_id=None), prisma_client=None).is_admin is True

    def test_user_scope_is_not_admin(self):
        assert ScopedUsageDataProvider(scope=UserScope(user_id="u1"), prisma_client=None).is_admin is False


class TestSummarisers:
    def test_usage_summary_includes_totals_models_providers(self):
        summary = summarise_usage_data(_as_response(SAMPLE_AGGREGATED_RESPONSE))
        assert "$50.25" in summary
        assert "gpt-4" in summary
        assert "openai" in summary

    def test_usage_summary_handles_empty(self):
        assert "no data" in summarise_usage_data(_as_response({"results": [], "metadata": {}})).lower()

    def test_entity_summary_ranks_by_spend(self):
        summary = summarise_entity_data(_as_response(SAMPLE_TEAM_RESPONSE), "Team")
        assert "Engineering" in summary
        assert "Marketing" in summary
        # Engineering (higher spend) must appear before Marketing
        assert summary.index("Engineering") < summary.index("Marketing")

    def test_entity_summary_empty(self):
        assert "No Team usage data" in summarise_entity_data(_as_response({"results": [], "metadata": {}}), "Team")
