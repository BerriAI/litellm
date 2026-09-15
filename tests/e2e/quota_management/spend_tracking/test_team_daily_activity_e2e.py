"""Vendor §9.20: GET /team/daily/activity structure and required query params (LIT-4778).

The spend-route breadth probe only checks that the path responds. These cases pin
the customer-facing contract: a valid date range returns results+metadata, and
missing start/end dates are rejected.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from math import isclose
from typing import Final

import pytest
from e2e_http import ProbeResult
from lifecycle import ResourceManager
from pydantic import BaseModel
from spend_e2e_client import SpendClient
from spend_reconciliation import assert_logs_match, create_traffic

pytestmark = pytest.mark.e2e

ROUTE = "/team/daily/activity"


class TeamDailyActivityParams(BaseModel):
    start_date: str | None = None
    end_date: str | None = None
    page: int = 1
    page_size: int = 1
    team_ids: str | None = None


class TeamDailyActivityRow(BaseModel):
    date: str
    metrics: TeamDailyActivityMetrics
    breakdown: TeamDailyActivityBreakdown


class TeamDailyActivityMetrics(BaseModel):
    spend: float
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    api_requests: int
    successful_requests: int
    failed_requests: int


class TeamDailyActivityEntity(BaseModel):
    metrics: TeamDailyActivityMetrics


class TeamDailyActivityBreakdown(BaseModel):
    entities: dict[str, TeamDailyActivityEntity]


class TeamDailyActivityMetadata(BaseModel):
    page: int
    total_pages: int
    has_more: bool
    total_spend: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_api_requests: int
    total_successful_requests: int
    total_failed_requests: int


class TeamDailyActivityResponse(BaseModel):
    results: list[TeamDailyActivityRow]
    metadata: TeamDailyActivityMetadata


def _probe(client: SpendClient, params: BaseModel) -> ProbeResult:
    return client.proxy.transport.probe(ROUTE, params=params)


class TestTeamDailyActivity:
    @pytest.mark.replayable
    @pytest.mark.covers("mgmt.team.daily_activity.happy_path")
    def test_valid_date_range_returns_results_and_metadata(
        self, client: SpendClient, resources: ResourceManager
    ) -> None:
        started: Final = datetime.now(timezone.utc).date()
        traffic: Final = create_traffic(client, resources)
        for team in traffic:
            assert_logs_match(client, team)
        ended: Final = datetime.now(timezone.utc).date()
        team_ids: Final = ",".join(team.team_id for team in traffic)

        def fetch(
            page: int, start: str = started.isoformat(), end: str = ended.isoformat()
        ) -> TeamDailyActivityResponse:
            result: Final = _probe(
                client,
                TeamDailyActivityParams(
                    start_date=start,
                    end_date=end,
                    page=page,
                    page_size=1,
                    team_ids=team_ids,
                ),
            )
            assert result.status_code == 200, f"daily activity failed: {result.status_code} {result.body[:300]}"
            return TeamDailyActivityResponse.model_validate_json(result.body)

        def pages() -> tuple[TeamDailyActivityResponse, ...]:
            first: Final = fetch(1)
            assert first.metadata.total_pages <= len(traffic) * 2, "unexpected extra scoped daily groups"
            return (first, *(fetch(page) for page in range(2, first.metadata.total_pages + 1)))

        deadline: Final = time.monotonic() + client.proxy.poll_timeout
        while True:
            observed = pages()
            if sum(page.metadata.total_api_requests for page in observed) >= sum(len(t.responses) for t in traffic):
                break
            if time.monotonic() >= deadline:
                break
            time.sleep(client.proxy.poll_interval)

        assert len(observed) >= 2, "two teams must exercise a page boundary"
        for index, page in enumerate(observed, 1):
            assert page.metadata.page == index
            assert page.metadata.total_pages == len(observed)
            assert page.metadata.has_more == (index < len(observed))
            assert len(page.results) == 1, "each fetched daily group must appear in results"
            row = page.results[0]
            assert started <= datetime.fromisoformat(row.date).date() <= ended
            assert len(row.breakdown.entities) == 1
            assert row.metrics.total_tokens == page.metadata.total_tokens
            assert row.metrics.prompt_tokens == page.metadata.total_prompt_tokens
            assert row.metrics.completion_tokens == page.metadata.total_completion_tokens
            assert row.metrics.api_requests == page.metadata.total_api_requests
            assert row.metrics.successful_requests == page.metadata.total_successful_requests
            assert row.metrics.failed_requests == page.metadata.total_failed_requests
            assert isclose(row.metrics.spend, page.metadata.total_spend, rel_tol=1e-6, abs_tol=1e-9)

        entities: Final = tuple(
            (team_id, entity.metrics)
            for page in observed
            for row in page.results
            for team_id, entity in row.breakdown.entities.items()
        )
        assert frozenset(team_id for team_id, _ in entities) == frozenset(team.team_id for team in traffic)
        for team in traffic:
            metrics = tuple(metrics for team_id, metrics in entities if team_id == team.team_id)
            assert sum(m.api_requests for m in metrics) == len(team.responses)
            assert sum(m.successful_requests for m in metrics) == len(team.responses)
            assert sum(m.failed_requests for m in metrics) == 0
            assert sum(m.prompt_tokens for m in metrics) == team.prompt_tokens
            assert sum(m.completion_tokens for m in metrics) == team.completion_tokens
            assert sum(m.total_tokens for m in metrics) == team.prompt_tokens + team.completion_tokens
            assert isclose(sum(m.spend for m in metrics), team.spend, rel_tol=1e-6, abs_tol=1e-9)
        assert isclose(
            sum(page.metadata.total_spend for page in observed),
            sum(team.spend for team in traffic),
            rel_tol=1e-6,
            abs_tol=1e-9,
        )
        assert sum(page.metadata.total_tokens for page in observed) == sum(
            team.prompt_tokens + team.completion_tokens for team in traffic
        )

        empty_date: Final = (started - timedelta(days=7)).isoformat()
        empty: Final = fetch(1, empty_date, empty_date)
        assert empty.results == []
        assert empty.metadata.total_pages == 0
        assert empty.metadata.page == 1
        assert not empty.metadata.has_more
        assert empty.metadata.total_spend == 0
        assert empty.metadata.total_tokens == 0
        assert empty.metadata.total_api_requests == 0
        assert empty.metadata.total_prompt_tokens == 0
        assert empty.metadata.total_completion_tokens == 0
        assert empty.metadata.total_successful_requests == 0
        assert empty.metadata.total_failed_requests == 0

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
        assert result.status_code == 400, f"missing end_date must be 400, got {result.status_code}: {result.body[:600]}"
