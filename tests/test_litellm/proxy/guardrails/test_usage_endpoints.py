from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.guardrails.usage_endpoints import (
    UsageChartPoint,
    UsageDetailResponse,
    UsageLogEntry,
    UsageOverviewRow,
    UsageTimeSeriesPoint,
    _action_from_guardrail_entry,
    _chart_from_metrics,
    _status_from_fail_rate,
    _trend_from_comparison,
    router,
)


def _metric(date: str, passed: int, blocked: int) -> SimpleNamespace:
    return SimpleNamespace(date=date, passed_count=passed, blocked_count=blocked, requests_evaluated=passed + blocked)


class TestStatusFromFailRate:
    @pytest.mark.parametrize(
        "fail_rate, expected",
        [
            (0.0, "healthy"),
            (5.0, "healthy"),
            (5.1, "warning"),
            (15.0, "warning"),
            (15.1, "critical"),
            (100.0, "critical"),
        ],
    )
    def test_boundaries(self, fail_rate: float, expected: str) -> None:
        assert _status_from_fail_rate(fail_rate) == expected


class TestTrendFromComparison:
    @pytest.mark.parametrize(
        "current, previous, expected",
        [
            (10.0, 0.0, "stable"),
            (10.0, -1.0, "stable"),
            (10.0, 9.5, "stable"),
            (10.0, 9.4, "up"),
            (10.0, 10.5, "stable"),
            (10.0, 10.6, "down"),
        ],
    )
    def test_boundaries(self, current: float, previous: float, expected: str) -> None:
        assert _trend_from_comparison(current, previous) == expected


class TestActionFromGuardrailEntry:
    @pytest.mark.parametrize(
        "status, expected",
        [
            ("guardrail_intervened", "blocked"),
            ("BLOCKED", "blocked"),
            ("guardrail_failed_to_run", "flagged"),
            ("ERROR", "flagged"),
            ("success", "passed"),
            ("", "passed"),
        ],
    )
    def test_maps_guardrail_status(self, status: str, expected: str) -> None:
        assert _action_from_guardrail_entry({"guardrail_status": status}) == expected

    def test_missing_entry_is_passed(self) -> None:
        assert _action_from_guardrail_entry(None) == "passed"

    def test_null_status_is_passed(self) -> None:
        assert _action_from_guardrail_entry({"guardrail_status": None}) == "passed"

    def test_blocked_takes_precedence_over_failed(self) -> None:
        assert _action_from_guardrail_entry({"guardrail_status": "blocked_after_error"}) == "blocked"


class TestChartFromMetrics:
    def test_aggregates_duplicate_dates_and_sorts(self) -> None:
        chart = _chart_from_metrics(
            [
                _metric("2026-07-03", passed=1, blocked=2),
                _metric("2026-07-01", passed=3, blocked=4),
                _metric("2026-07-01", passed=5, blocked=6),
            ]
        )

        assert chart == [
            UsageChartPoint(date="2026-07-01", passed=8, blocked=10),
            UsageChartPoint(date="2026-07-03", passed=1, blocked=2),
        ]

    def test_null_counts_are_zero(self) -> None:
        metric = SimpleNamespace(date="2026-07-01", passed_count=None, blocked_count=None, requests_evaluated=None)

        assert _chart_from_metrics([metric]) == [UsageChartPoint(date="2026-07-01", passed=0, blocked=0)]

    def test_empty_metrics(self) -> None:
        assert _chart_from_metrics([]) == []


class TestResponseContractIsNarrow:
    """The dashboard indexes record lookups by these literals, so the schema must not widen back to `str`."""

    def test_overview_row_rejects_unknown_status(self) -> None:
        with pytest.raises(ValidationError):
            UsageOverviewRow(
                id="g1",
                name="g1",
                type="Guardrail",
                provider="Custom",
                requestsEvaluated=1,
                failRate=0.0,
                avgScore=None,
                avgLatency=None,
                status="degraded",
                trend="stable",
            )

    def test_overview_row_rejects_unknown_trend(self) -> None:
        with pytest.raises(ValidationError):
            UsageOverviewRow(
                id="g1",
                name="g1",
                type="Guardrail",
                provider="Custom",
                requestsEvaluated=1,
                failRate=0.0,
                avgScore=None,
                avgLatency=None,
                status="healthy",
                trend="sideways",
            )

    def test_log_entry_rejects_unknown_action(self) -> None:
        with pytest.raises(ValidationError):
            UsageLogEntry(
                id="req-1",
                timestamp="2026-07-01T00:00:00+00:00",
                action="allowed",
                score=None,
                latency_ms=None,
                model=None,
                input_snippet=None,
                output_snippet=None,
                reason=None,
            )

    def test_detail_time_series_rejects_unshaped_point(self) -> None:
        with pytest.raises(ValidationError):
            UsageDetailResponse(
                guardrail_id="g1",
                guardrail_name="g1",
                type="Guardrail",
                provider="Custom",
                requestsEvaluated=0,
                failRate=0.0,
                avgScore=None,
                avgLatency=None,
                status="healthy",
                trend="stable",
                description=None,
                time_series=[{"day": "2026-07-01"}],
            )

    def test_time_series_point_carries_chart_fields_plus_score(self) -> None:
        point = UsageTimeSeriesPoint(date="2026-07-01", passed=2, blocked=1, score=None)

        assert point.model_dump() == {"date": "2026-07-01", "passed": 2, "blocked": 1, "score": None}


class _FakeTable:
    def __init__(self, rows: tuple[SimpleNamespace, ...]) -> None:
        self._rows = rows

    async def find_many(self, **_kwargs: object) -> list[SimpleNamespace]:
        return list(self._rows)


class _FakeDb:
    def __init__(self, guardrails: tuple[SimpleNamespace, ...], metrics: tuple[SimpleNamespace, ...]) -> None:
        self.litellm_guardrailstable = _FakeTable(guardrails)
        self.litellm_dailyguardrailmetrics = _FakeTable(metrics)


class _FakePrisma:
    def __init__(self, guardrails: tuple[SimpleNamespace, ...], metrics: tuple[SimpleNamespace, ...]) -> None:
        self.db = _FakeDb(guardrails, metrics)


@pytest.fixture
def overview_client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    guardrail = SimpleNamespace(
        guardrail_id="g-1",
        guardrail_name="pii-detector",
        litellm_params={"guardrail": "presidio"},
        guardrail_info={"type": "PII"},
    )
    metrics = (
        SimpleNamespace(
            guardrail_id="pii-detector",
            date="2026-07-01",
            requests_evaluated=100,
            passed_count=80,
            blocked_count=20,
            flagged_count=0,
        ),
    )

    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "prisma_client", _FakePrisma((guardrail,), metrics), raising=False)

    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(api_key="sk-test")
    return TestClient(app)


class TestOverviewEndpointContract:
    """Exercises FastAPI's response_model serialization, not just the helpers."""

    def test_serializes_a_row_through_the_narrowed_response_model(self, overview_client: TestClient) -> None:
        response = overview_client.get("/guardrails/usage/overview?start_date=2026-07-01&end_date=2026-07-08")

        assert response.status_code == 200
        body = response.json()
        assert body["rows"] == [
            {
                "id": "g-1",
                "name": "pii-detector",
                "type": "PII",
                "provider": "presidio",
                "requestsEvaluated": 100,
                "failRate": 20.0,
                "avgScore": None,
                "avgLatency": None,
                "status": "critical",
                "trend": "stable",
            }
        ]

    def test_chart_serializes_as_objects_not_opaque_dicts(self, overview_client: TestClient) -> None:
        response = overview_client.get("/guardrails/usage/overview?start_date=2026-07-01&end_date=2026-07-08")

        assert response.json()["chart"] == [{"date": "2026-07-01", "passed": 80, "blocked": 20}]

    def test_totals_are_computed_from_metrics(self, overview_client: TestClient) -> None:
        body = overview_client.get("/guardrails/usage/overview").json()

        assert (body["totalRequests"], body["totalBlocked"], body["passRate"]) == (100, 20, 80.0)
