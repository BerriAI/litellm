"""Vendor §9.20: GET /team/daily/activity structure and required query params (LIT-4778).

The spend-route breadth probe only checks that the path responds. These cases pin
the customer-facing contract: a valid date range returns results+metadata, and
missing start/end dates are rejected.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import BaseModel

from e2e_http import ProbeResult
from models import DateRangeParams
from spend_e2e_client import SpendClient

pytestmark = pytest.mark.e2e

ROUTE = "/team/daily/activity"


class TeamDailyActivityParams(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    page: int = 1


class TeamDailyActivityRow(BaseModel):
    date: str | None = None
    metrics: dict[str, object] | None = None


class TeamDailyActivityResponse(BaseModel):
    results: list[TeamDailyActivityRow] = []
    metadata: dict[str, object] | None = None


def _range_days(days: int) -> DateRangeParams:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return DateRangeParams(start_date=start.isoformat(), end_date=end.isoformat())


def _probe(client: SpendClient, params: BaseModel) -> ProbeResult:
    return client.proxy.transport.probe(ROUTE, params=params)


class TestTeamDailyActivity:
    @pytest.mark.covers("mgmt.team.daily_activity.happy_path")
    @pytest.mark.parametrize("days", [1, 7, 30])
    def test_valid_date_range_returns_results_and_metadata(
        self, client: SpendClient, days: int
    ) -> None:
        result = _probe(client, _range_days(days))
        assert result.status_code == 200, (
            f"{ROUTE} range={days}d must be 200, got {result.status_code}: {result.body[:600]}"
        )
        parsed = TeamDailyActivityResponse.model_validate_json(result.body)
        assert parsed.results is not None, f"results field required: {result.body[:600]}"
        assert parsed.metadata is not None, f"metadata field required: {result.body[:600]}"
        if parsed.results:
            first = parsed.results[0]
            assert first.date is not None, f"result row needs date: {result.body[:600]}"
            assert first.metrics is not None, f"result row needs metrics: {result.body[:600]}"

    @pytest.mark.covers("mgmt.team.daily_activity.missing_start_date_rejected")
    def test_missing_start_date_is_rejected(self, client: SpendClient) -> None:
        end = datetime.now(timezone.utc).date().isoformat()
        result = _probe(client, TeamDailyActivityParams(end_date=end, page=1))
        assert result.status_code == 400, (
            f"missing start_date must be 400, got {result.status_code}: {result.body[:600]}"
        )

    @pytest.mark.covers("mgmt.team.daily_activity.missing_end_date_rejected")
    def test_missing_end_date_is_rejected(self, client: SpendClient) -> None:
        start = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
        result = _probe(client, TeamDailyActivityParams(start_date=start, page=1))
        assert result.status_code == 400, (
            f"missing end_date must be 400, got {result.status_code}: {result.body[:600]}"
        )
