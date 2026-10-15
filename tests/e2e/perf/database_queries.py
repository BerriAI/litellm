from __future__ import annotations

import math
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import Final, Literal

from e2e_http import without_retries
from pydantic import BaseModel, ConfigDict, JsonValue, RootModel, ValidationError
from transport import Transport

from .database_seed import SeedManifest


class QueryParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    start_date: str | None = None
    end_date: str | None = None
    page: int | None = None
    page_size: int | None = None
    api_key: str | None = None
    team_id: str | None = None
    model: str | None = None
    status_filter: str | None = None
    min_spend: float | None = None
    max_spend: float | None = None
    search: str | None = None
    session_id: str | None = None
    group_by_session: bool | None = None
    sort_by: str | None = None
    sort_order: str | None = None
    api_key_limit: int | None = None
    offset: int | None = None
    limit: int | None = None
    timezone: int | None = None


class LogRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: str
    api_key: str
    team_id: str
    model: str
    status: str
    spend: float
    startTime: datetime
    session_id: str | None


class LogPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[LogRow, ...]
    total: int
    page: int
    page_size: int


class Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_requests: int
    spend: float


class MetricRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics: Metrics


class DailyRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    date: str
    metrics: Metrics
    breakdown: Mapping[str, Mapping[str, MetricRow]]


class DailyMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_api_requests: int
    total_spend: float


class DailyPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: tuple[DailyRow, ...]
    metadata: DailyMetadata


class KeyPage(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_keys: tuple[MetricRow, ...]
    total_api_keys: int
    offset: int
    limit: int


class QueryCase(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    path: str
    params: QueryParams
    shape: Literal["logs", "detail", "daily", "keys"]
    expected_requests: int | None = None
    breakdowns: tuple[str, ...] = ()


class QueryBody(RootModel[JsonValue]):
    model_config = ConfigDict(frozen=True)


class QuerySample(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    seconds: float
    status: int | None
    cache_state: Literal["first_request", "repeated_request"]
    failure: str | None = None


class QuerySummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    samples: int
    errors: int
    first_request_seconds: float
    repeated_p50_seconds: float
    repeated_p95_seconds: float
    repeated_p99_seconds: float


def latency_regressed(baseline_seconds: float, target_seconds: float, ratio: float) -> bool:
    return target_seconds > max(0.25, baseline_seconds) * ratio


def query_cases(manifest: SeedManifest, sample: LogRow) -> tuple[QueryCase, ...]:
    start: Final = manifest.spec.anchor - timedelta(days=90)
    log_params: Final = QueryParams(
        start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
        end_date=manifest.spec.anchor.strftime("%Y-%m-%d %H:%M:%S"),
        page=1,
        page_size=50,
        sort_by="startTime",
        sort_order="desc",
    )
    daily_params: Final = QueryParams(
        start_date=start.date().isoformat(),
        end_date=(manifest.spec.anchor - timedelta(microseconds=1)).date().isoformat(),
        timezone=0,
        api_key_limit=20,
    )
    filters: Final = (
        ("logs_first_page", QueryParams()),
        ("logs_second_page", QueryParams(page=2)),
        (
            "logs_day",
            QueryParams(
                start_date=f"{sample.startTime.date()} 00:00:00", end_date=f"{sample.startTime.date()} 23:59:59"
            ),
        ),
        ("logs_key", QueryParams(api_key=sample.api_key)),
        ("logs_team", QueryParams(team_id=sample.team_id)),
        ("logs_model", QueryParams(model=sample.model)),
        ("logs_failure", QueryParams(status_filter="failure")),
        ("logs_spend", QueryParams(min_spend=0.001, max_spend=0.1, sort_by="spend", sort_order="asc")),
        ("logs_search", QueryParams(search=sample.request_id)),
        ("logs_grouped_sessions", QueryParams(group_by_session=True)),
    )
    return (
        *(
            QueryCase(
                name=name,
                path="/spend/logs/ui",
                shape="logs",
                params=log_params.model_copy(update=values.model_dump(exclude_none=True)),
            )
            for name, values in filters
        ),
        QueryCase(
            name="session",
            path="/spend/logs/session/ui",
            shape="logs",
            params=QueryParams(session_id=sample.session_id, page=1, page_size=50),
        ),
        QueryCase(
            name="request_detail", path=f"/spend/logs/ui/{sample.request_id}", shape="detail", params=QueryParams()
        ),
        QueryCase(
            name="daily_activity",
            path="/user/daily/activity",
            shape="daily",
            params=daily_params.model_copy(update={"page_size": 1000}),
            expected_requests=manifest.progress.rows,
        ),
        *(
            QueryCase(
                name=f"{entity}_aggregated",
                path=f"/{entity}/daily/activity/aggregated",
                shape="daily",
                params=daily_params,
                expected_requests=manifest.progress.rows * multiplier,
                breakdowns=("models", "providers", "api_keys") if entity == "user" else (),
            )
            for entity, multiplier in (("user", 1), ("team", 1), ("tag", 2), ("customer", 1))
        ),
        QueryCase(
            name="key_page",
            path="/user/daily/activity/aggregated/keys",
            shape="keys",
            params=daily_params.model_copy(update={"offset": 20, "limit": 20}),
        ),
    )


def check_logs(case: QueryCase, body: JsonValue) -> None:
    page: Final = LogPage.model_validate(body)
    assert page.data and page.total >= len(page.data), "Empty or inconsistent log page"
    assert page.page == case.params.page and page.page_size == case.params.page_size, "Pagination changed"
    assert len(page.data) <= page.page_size, "Page exceeds the requested limit"
    assert len({row.request_id for row in page.data}) == len(page.data), "Repeated request on one page"
    for row in page.data:
        for attribute, expected in (
            ("api_key", case.params.api_key),
            ("team_id", case.params.team_id),
            ("model", case.params.model),
            ("status", case.params.status_filter),
            ("session_id", case.params.session_id),
        ):
            assert expected is None or getattr(row, attribute) == expected, f"Ignored {attribute} filter"
        assert case.params.min_spend is None or row.spend >= case.params.min_spend, "Ignored minimum spend"
        assert case.params.max_spend is None or row.spend <= case.params.max_spend, "Ignored maximum spend"
        assert case.params.search is None or row.request_id == case.params.search, "Search returned another request"
        if case.params.start_date is not None and case.params.end_date is not None:
            assert datetime.fromisoformat(case.params.start_date).replace(tzinfo=timezone.utc) <= row.startTime.replace(
                tzinfo=timezone.utc
            ), "Log precedes date range"
            assert row.startTime.replace(tzinfo=timezone.utc) <= datetime.fromisoformat(case.params.end_date).replace(
                tzinfo=timezone.utc
            ), "Log follows date range"
    values: Final = tuple(row.spend if case.params.sort_by == "spend" else row.startTime for row in page.data)
    assert values == tuple(sorted(values, reverse=case.params.sort_order != "asc")), "Log order changed"
    if case.params.group_by_session:
        sessions: Final = tuple((row.api_key, row.session_id or row.request_id) for row in page.data)
        assert len(sessions) == len(set(sessions)), "Grouped page contains repeated sessions"


def check_body(case: QueryCase, body: JsonValue) -> None:
    match case.shape:
        case "logs":
            check_logs(case, body)
        case "detail":
            assert body == {"messages": {}, "response": {}, "proxy_server_request": {}}, "Unexpected prompt payload"
        case "daily":
            page: Final = DailyPage.model_validate(body)
            assert page.results, "Empty daily activity"
            assert page.metadata.total_api_requests == case.expected_requests, "Daily request total differs from seed"
            assert sum(row.metrics.api_requests for row in page.results) == case.expected_requests, "Missing daily rows"
            assert page.metadata.total_spend > 0, "Empty spend total"
            assert math.isclose(sum(row.metrics.spend for row in page.results), page.metadata.total_spend), (
                "Spend mismatch"
            )
            for dimension in case.breakdowns:
                assert any(row.breakdown.get(dimension) for row in page.results), f"Empty {dimension} breakdown"
        case "keys":
            keys: Final = KeyPage.model_validate(body)
            assert keys.api_keys and keys.total_api_keys > 20, "Empty key page"
            assert keys.offset == case.params.offset and keys.limit == case.params.limit, "Key pagination changed"
            assert all(row.metrics.api_requests > 0 for row in keys.api_keys), "Empty key metrics"


def query_once(transport: Transport, case: QueryCase, *, repeated: bool = False) -> QuerySample:
    started: Final = time.monotonic()
    with without_retries():
        result: Final = transport.probe(case.path, headers=transport.master, params=case.params)
    elapsed: Final = time.monotonic() - started
    cache: Final = "repeated_request" if repeated else "first_request"
    if result.status_code != 200:
        return QuerySample(
            name=case.name,
            seconds=elapsed,
            cache_state=cache,
            status=result.status_code,
            failure="HTTP request failed" if result.status_code != -1 else "Network error",
        )
    try:
        check_body(case, QueryBody.model_validate_json(result.body).root)
    except (AssertionError, ValidationError) as exc:
        return QuerySample(
            name=case.name, seconds=elapsed, status=result.status_code, cache_state=cache, failure=str(exc)
        )
    return QuerySample(name=case.name, seconds=elapsed, status=result.status_code, cache_state=cache)


def percentile(values: tuple[float, ...], fraction: float) -> float:
    assert values, "Cannot report a percentile without measurements"
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def summarize(samples: tuple[QuerySample, ...]) -> QuerySummary:
    assert len(samples) >= 2, "Need a first request and repeated measurements"
    repeated: Final = tuple(sample.seconds for sample in samples[1:])
    return QuerySummary(
        name=samples[0].name,
        samples=len(samples),
        errors=sum(sample.failure is not None for sample in samples),
        first_request_seconds=samples[0].seconds,
        repeated_p50_seconds=percentile(repeated, 0.5),
        repeated_p95_seconds=percentile(repeated, 0.95),
        repeated_p99_seconds=percentile(repeated, 0.99),
    )
