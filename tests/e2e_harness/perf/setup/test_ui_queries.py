from __future__ import annotations

from datetime import UTC, datetime
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
from e2e_http import NetworkError, Result, Success, UnknownApiError
from setup.ui_queries import (
    Contract,
    Document,
    Sample,
    database_failures,
    measure,
    query_path,
    summarize,
    ui_junit_properties,
)
from setup.ui_query_cases import Params, cases, next_page


def test_query_string_preserves_ui_arrays_and_boolean_strings() -> None:
    requests: Final = cases(datetime(2026, 1, 1, tzinfo=UTC), "key")
    cache: Final = next(case for case in requests if case.name == "cache_activity_filtered")
    assert parse_qs(urlsplit(query_path(cache)).query) == {
        "start_date": ["2025-12-25"],
        "end_date": ["2026-01-01"],
        "key_aliases": ["perf-key-0", "perf-key-1"],
        "models": ["perf-chat"],
    }
    health: Final = next(case for case in requests if case.name == "logs_exclude_internal_health_checks_30d")
    query: Final = parse_qs(urlsplit(query_path(health)).query)
    assert query["exclude_internal_health_checks"] == ["true"]
    assert query["group_by_session"] == ["true"]


@pytest.mark.parametrize("entity", ("user", "team", "tag", "customer"))
@pytest.mark.parametrize(
    ("route", "extra"),
    (
        ("aggregated", {}),
        ("aggregated/keys", {"offset": 0, "limit": 50}),
        ("aggregated/search", {"search": "perf-key"}),
        ("aggregated/model_top_keys", {"model_group": "perf-chat", "by_model_group": "true"}),
    ),
)
def test_usage_requests_list_keys_without_a_single_key_filter(
    entity: str, route: str, extra: dict[str, str | int]
) -> None:
    case: Final = next(
        case
        for case in cases(datetime(2026, 1, 1, tzinfo=UTC), "key")
        if case.path == f"/{entity}/daily/activity/{route}"
    )
    assert case.params.model_dump(exclude_none=True) == {
        "start_date": "2025-12-25",
        "end_date": "2026-01-01",
        "timezone": "0",
        **({"exclude_team_ids": "litellm-dashboard"} if entity == "team" else {}),
        **extra,
    }


@pytest.mark.parametrize(("name", "value"), (("api_key", "key"), ("key_alias", "perf-key-0")))
def test_key_filters_request_30_days_of_logs(name: str, value: str) -> None:
    case: Final = next(
        case for case in cases(datetime(2026, 1, 1, tzinfo=UTC), "key") if case.name == f"logs_{name}_30d"
    )
    assert parse_qs(urlsplit(query_path(case)).query) == {
        "start_date": ["2025-12-02 00:00:00"],
        "end_date": ["2026-01-01 00:00:00"],
        "page_size": ["25"],
        "page": ["1"],
        "sort_by": ["startTime"],
        "sort_order": ["desc"],
        "group_by_session": ["true"],
        name: [value],
    }


@pytest.mark.parametrize("invalid", ("not-a-number", True, float("nan"), float("inf")))
def test_positive_row_does_not_hide_a_malformed_later_row(invalid: str | bool | float) -> None:
    with pytest.raises(AssertionError, match=r"Expected (numeric|finite) spend"):
        Contract(positive=("spend",)).count(Document([{"spend": 1}, {"spend": invalid}]))


def test_session_cursor_is_reused_and_repeated_rows_rejected() -> None:
    case: Final = next(case for case in cases(datetime(2026, 1, 1, tzinfo=UTC), "key") if case.cursor == "session")
    first: Final = Document.model_validate_json('{"data":[{"request_id":"one"}],"next_session_cursor":"opaque"}')
    second: Final = next_page(case, first)
    assert Params.model_validate(second.params).session_cursor == "opaque"
    assert Params.model_validate(second.params).page == 2
    with pytest.raises(AssertionError, match="Repeated page-one row"):
        _ = second.contract.count(first)


def test_first_query_is_separate_from_repeated_latency() -> None:
    samples: Final = (Sample(seconds=80, status=200), *(Sample(seconds=i, status=200) for i in range(1, 101)))
    result: Final = summarize(samples)
    assert result.first_seconds == 80
    assert result.repeated_samples == 100
    assert result.repeated_p50_seconds == 50.5
    assert result.repeated_p95_seconds == 95
    assert result.repeated_p99_seconds == 99
    assert result.repeated_max_seconds == 100


def test_failed_repeats_do_not_look_like_fast_successes() -> None:
    result: Final = summarize((Sample(seconds=10, status=200), Sample(seconds=0.01, status=503, error="HTTP 503")))
    assert result.repeated_samples == 0
    assert result.repeated_p50_seconds is None
    assert result.repeated_p95_seconds is None
    assert result.repeated_p99_seconds is None
    assert result.repeated_max_seconds is None


def test_small_sample_does_not_report_tail_percentiles() -> None:
    result: Final = summarize(tuple(Sample(seconds=1, status=200) for _ in range(6)))
    assert result.repeated_samples == 5
    assert result.repeated_p50_seconds == 1
    assert result.repeated_p95_seconds is None
    assert result.repeated_p99_seconds is None


def test_failed_first_request_emits_error_code_without_latency() -> None:
    sample: Final = Sample(seconds=30, status=500, error="HTTP 500")
    assert summarize((sample,)).first_seconds is None
    assert dict(ui_junit_properties((sample,))) == {
        "perf_ui_requests": 1,
        "perf_ui_failures": 1,
        "perf_ui_repeated_samples": 0,
        "perf_ui_error_http_500": 1,
    }


def test_junit_counts_failures_but_only_measures_successful_latency() -> None:
    samples: Final = (
        Sample(seconds=2, status=200),
        Sample(seconds=3, status=200),
        Sample(seconds=30, status=-1, error="timeout"),
        Sample(seconds=30, status=503, error="HTTP 503"),
        Sample(seconds=1, status=200, error="Empty collection"),
    )
    assert dict(ui_junit_properties(samples, Sample(seconds=4, status=200))) == {
        "perf_ui_requests": 6,
        "perf_ui_failures": 3,
        "perf_ui_repeated_samples": 1,
        "perf_ui_error_timeout": 1,
        "perf_ui_error_http_503": 1,
        "perf_ui_error_contract": 1,
        "perf_ui_first_seconds": 2,
        "perf_ui_repeated_seconds_p50": 3,
        "perf_ui_repeated_seconds_max": 3,
    }


def test_failed_cursor_prerequisite_is_counted_once() -> None:
    sample: Final = Sample(seconds=30, status=503, error="HTTP 503")
    assert dict(ui_junit_properties((sample,), sample)) == dict(ui_junit_properties((sample,)))


@pytest.mark.parametrize("body", ("{}", '{"data": []}', '{"data": [{"spend": 1}]}', '{"data": [null]}'))
def test_empty_or_malformed_rows_fail_even_with_http_200(body: str) -> None:
    ticks: Final = iter((10.0, 12.5))
    result, _ = measure(
        lambda: Success(status_code=200, data=Document.model_validate_json(body)),
        Contract(("data",), ("request_id",)),
        clock=lambda: next(ticks),
    )
    assert result.status == 200
    assert result.seconds == 2.5
    assert result.rows == 0
    assert result.error


def test_valid_response_records_elapsed_seconds_and_rows() -> None:
    ticks: Final = iter((5.0, 5.125))
    result, document = measure(
        lambda: Success(status_code=200, data=Document.model_validate_json('{"data":[{"request_id":"one"}]}')),
        Contract(("data",), ("request_id",)),
        clock=lambda: next(ticks),
    )
    assert result == Sample(seconds=0.125, status=200, rows=1)
    assert document is not None


def test_zero_spend_rows_are_allowed_alongside_nonzero_fixture_data() -> None:
    contract: Final = Contract(positive=("spend",))
    assert contract.count(Document.model_validate_json('[{"spend":0},{"spend":2}]')) == 2
    with pytest.raises(AssertionError, match="positive spend"):
        _ = contract.count(Document.model_validate_json('[{"spend":0}]'))


@pytest.mark.parametrize(
    "failure,error,status",
    (
        (UnknownApiError(status_code=500, body="sensitive error detail"), "HTTP 500", 500),
        (NetworkError(message="Read timed out"), "timeout", -1),
        (NetworkError(message="connection refused"), "network", -1),
    ),
)
def test_failed_requests_are_recorded_without_response_bodies(
    failure: Result[Document],
    error: str,
    status: int,
) -> None:
    ticks: Final = iter((1.0, 2.0))
    result, document = measure(lambda: failure, Contract(), clock=lambda: next(ticks))
    assert result == Sample(seconds=1, status=status, error=error)
    assert document is None


def _metrics(postgres: str, batch: str) -> str:
    return (
        "# TYPE litellm_postgres_failed_requests_total counter\n"
        f'litellm_postgres_failed_requests_total{{call_type="query_raw"}} {postgres}\n'
        "# TYPE litellm_batch_write_to_db_failed_requests_total counter\n"
        f'litellm_batch_write_to_db_failed_requests_total{{call_type="batch"}} {batch}\n'
    )


def test_postgres_failures_include_background_batch_writes() -> None:
    assert database_failures(_metrics("0", "0")) == 0
    assert database_failures(_metrics("2", "3")) == 5


def test_lazy_failure_counters_require_evidence_of_database_activity() -> None:
    assert (
        database_failures(
            "# TYPE litellm_postgres_total_requests_total counter\nlitellm_postgres_total_requests_total 3\n"
        )
        == 0
    )


@pytest.mark.parametrize(
    "text", ("", "# TYPE unrelated counter\nunrelated_total 0\n", _metrics("NaN", "0"), _metrics("-1", "0"))
)
def test_missing_or_invalid_metrics_cannot_pass_as_zero(text: str) -> None:
    with pytest.raises(AssertionError):
        _ = database_failures(text)
