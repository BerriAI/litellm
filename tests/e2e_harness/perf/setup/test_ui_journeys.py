from __future__ import annotations

from typing import Final

import pytest
from setup.ui_journeys import Sample, database_failures, journeys, summarize, ui_junit_properties


def test_first_journey_is_separate_from_repeated_latency() -> None:
    samples: Final = (Sample(seconds=80), *(Sample(seconds=i) for i in range(1, 101)))
    result: Final = summarize(samples)
    assert result.first_seconds == 80
    assert result.repeated_samples == 100
    assert result.repeated_p50_seconds == 50.5
    assert result.repeated_p95_seconds == 95
    assert result.repeated_p99_seconds == 99
    assert result.repeated_max_seconds == 100


def test_failed_repeats_do_not_look_like_fast_successes() -> None:
    result: Final = summarize((Sample(seconds=10), Sample(seconds=0.01, error="AssertionError: no rows")))
    assert result.repeated_samples == 0
    assert result.repeated_p50_seconds is None
    assert result.repeated_p95_seconds is None
    assert result.repeated_p99_seconds is None
    assert result.repeated_max_seconds is None


def test_small_sample_does_not_report_tail_percentiles() -> None:
    result: Final = summarize(tuple(Sample(seconds=1) for _ in range(6)))
    assert result.repeated_samples == 5
    assert result.repeated_p50_seconds == 1
    assert result.repeated_p95_seconds is None
    assert result.repeated_p99_seconds is None


def test_failed_first_journey_emits_error_code_without_latency() -> None:
    sample: Final = Sample(seconds=120, error="timeout")
    assert summarize((sample,)).first_seconds is None
    assert dict(ui_junit_properties((sample,))) == {
        "perf_ui_journeys": 1,
        "perf_ui_failures": 1,
        "perf_ui_samples": 0,
        "perf_ui_repeated_samples": 0,
        "perf_ui_error_timeout": 1,
    }


def test_junit_counts_failures_but_only_measures_successful_latency() -> None:
    samples: Final = (
        Sample(seconds=2, rows=25),
        Sample(seconds=3, rows=25),
        Sample(seconds=120, error="timeout"),
        Sample(seconds=1, error="AssertionError: Logs table rendered empty"),
        Sample(seconds=4, rows=25),
    )
    assert dict(ui_junit_properties(samples)) == {
        "perf_ui_journeys": 5,
        "perf_ui_failures": 2,
        "perf_ui_samples": 3,
        "perf_ui_repeated_samples": 2,
        "perf_ui_error_timeout": 1,
        "perf_ui_error_ui": 1,
        "perf_ui_seconds_p50": 3,
        "perf_ui_seconds_p90": 4,
        "perf_ui_seconds_max": 4,
        "perf_ui_first_seconds": 2,
        "perf_ui_repeated_seconds_p50": 3.5,
        "perf_ui_repeated_seconds_max": 4,
    }


def test_all_journey_percentiles_include_first_and_use_nearest_rank_p90() -> None:
    samples: Final = tuple(Sample(seconds=seconds) for seconds in (10, 1, 9, 2, 8, 3, 7, 4, 6, 5))
    metrics: Final = dict(ui_junit_properties(samples))
    assert {key: value for key, value in metrics.items() if key.startswith("perf_ui_seconds_")} == {
        "perf_ui_seconds_p50": 5.5,
        "perf_ui_seconds_p90": 9,
        "perf_ui_seconds_max": 10,
    }
    assert metrics["perf_ui_samples"] == 10


def test_journeys_have_unique_names_and_cover_every_dashboard_surface() -> None:
    names: Final = tuple(journey.name for journey in journeys("key"))
    assert len(names) == len(frozenset(names))
    assert frozenset(journey.surface for journey in journeys("key")) == {"logs", "usage", "caching"}


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
