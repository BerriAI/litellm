import datetime
from unittest.mock import AsyncMock, patch

import pytest

from litellm.proxy.spend_tracking.cost_savings_report import (
    CostSavingsReport,
    get_cost_savings_report_for_time_range,
)
from litellm.types.proxy.management_endpoints.common_daily_activity import (
    DailySpendMetadata,
    SpendAnalyticsPaginatedResponse,
)


def _aggregated_response(
    total_spend: float,
    autorouter_savings_spend: float,
    compression_savings_spend: float,
    prompt_caching_savings_spend: float,
) -> SpendAnalyticsPaginatedResponse:
    return SpendAnalyticsPaginatedResponse(
        results=[],
        metadata=DailySpendMetadata(
            total_spend=total_spend,
            total_compression_savings_spend=compression_savings_spend,
            total_prompt_caching_savings_spend=prompt_caching_savings_spend,
            total_autorouter_savings_spend=autorouter_savings_spend,
        ),
    )


@pytest.mark.asyncio
async def test_get_cost_savings_report_maps_aggregated_totals_onto_report():
    mock_response = _aggregated_response(
        total_spend=100.0,
        autorouter_savings_spend=10.5,
        compression_savings_spend=2.25,
        prompt_caching_savings_spend=1.25,
    )
    start_date = datetime.date(2026, 1, 1)
    end_date = datetime.date(2026, 1, 8)

    with patch(
        "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity_aggregated",
        new=AsyncMock(return_value=mock_response),
    ):
        report = await get_cost_savings_report_for_time_range(
            prisma_client=AsyncMock(),
            start_date=start_date,
            end_date=end_date,
        )

    assert report == CostSavingsReport(
        start_date=start_date,
        end_date=end_date,
        total_spend=100.0,
        autorouter_savings_spend=10.5,
        compression_savings_spend=2.25,
        prompt_caching_savings_spend=1.25,
    )
    assert report.total_savings == pytest.approx(14.0)


@pytest.mark.asyncio
async def test_get_cost_savings_report_returns_none_when_nothing_to_report():
    mock_response = _aggregated_response(
        total_spend=0.0,
        autorouter_savings_spend=0.0,
        compression_savings_spend=0.0,
        prompt_caching_savings_spend=0.0,
    )

    with patch(
        "litellm.proxy.management_endpoints.common_daily_activity.get_daily_activity_aggregated",
        new=AsyncMock(return_value=mock_response),
    ):
        report = await get_cost_savings_report_for_time_range(
            prisma_client=AsyncMock(),
            start_date=datetime.date(2026, 1, 1),
            end_date=datetime.date(2026, 1, 8),
        )

    assert report is None
