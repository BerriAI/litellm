from typing import Final

import pytest
from pydantic import JsonValue, ValidationError

from perf.database_queries import QueryCase, QueryParams, QuerySample, check_body, summarize


@pytest.mark.parametrize("shape", ("logs", "detail", "daily", "keys"))
def test_empty_responses_never_pass_query_correctness(shape: str) -> None:
    case: Final = QueryCase.model_validate({"name": shape, "path": "/test", "shape": shape, "params": {}})
    with pytest.raises((AssertionError, ValidationError)):
        check_body(case, {})


def test_log_filter_validation_rejects_unfiltered_results() -> None:
    body: Final[JsonValue] = {
        "data": [
            {
                "request_id": "perf-seed-1",
                "api_key": "seed-key",
                "team_id": "seed-team",
                "model": "perf-chat",
                "status": "success",
                "spend": 0.1,
                "startTime": "2026-01-01T00:00:00Z",
                "session_id": "seed-session",
            }
        ],
        "total": 1,
        "page": 1,
        "page_size": 1,
    }
    matching: Final = QueryCase(
        name="model", path="/spend/logs/ui", shape="logs", params=QueryParams(page=1, page_size=1, model="perf-chat")
    )
    assert check_body(matching, body) is None
    filtered: Final = matching.model_copy(update={"params": QueryParams(page=1, page_size=1, model="perf-code")})
    with pytest.raises(AssertionError, match="Ignored model filter"):
        check_body(filtered, body)


def test_daily_totals_must_equal_the_seed_not_just_each_other() -> None:
    body: Final[JsonValue] = {
        "results": [{"date": "2026-01-01", "metrics": {"api_requests": 10, "spend": 1}, "breakdown": {}}],
        "metadata": {"total_api_requests": 10, "total_spend": 1},
    }
    case: Final = QueryCase(
        name="daily", path="/user/daily/activity", shape="daily", params=QueryParams(), expected_requests=11
    )
    with pytest.raises(AssertionError, match="Daily request total differs from seed"):
        check_body(case, body)
    assert check_body(case.model_copy(update={"expected_requests": 10}), body) is None


def test_query_percentiles_exclude_first_request_but_include_errors() -> None:
    samples: Final = (
        QuerySample(name="logs", seconds=99, status=200, cache_state="first_request"),
        QuerySample(name="logs", seconds=1, status=200, cache_state="repeated_request"),
        QuerySample(name="logs", seconds=2, status=500, cache_state="repeated_request", failure="HTTP request failed"),
        QuerySample(name="logs", seconds=3, status=200, cache_state="repeated_request"),
        QuerySample(name="logs", seconds=4, status=200, cache_state="repeated_request"),
    )
    result: Final = summarize(samples)
    assert (result.samples, result.errors, result.first_request_seconds) == (5, 1, 99)
    assert (result.repeated_p50_seconds, result.repeated_p95_seconds, result.repeated_p99_seconds) == (2, 4, 4)
