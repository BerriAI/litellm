from __future__ import annotations

from pathlib import Path
from typing import Final

from locust_load import (
    LoadError,
    LoadResult,
    LocustStatEntry,
    aggregate_stats,
    percentile_seconds,
    read_errors,
    read_generator_warnings,
)

_FAILURES_HEADER = "Method,Name,Error,Occurrences,First Seen,Last Seen\n"


def _entry(
    *,
    num_requests: int,
    name: str = "/chat/completions",
    num_failures: int = 0,
    start_time: float = 1000.0,
    last_request_timestamp: float = 1010.0,
    response_times: dict[int, int] | None = None,
) -> LocustStatEntry:
    return LocustStatEntry(
        name=name,
        num_requests=num_requests,
        num_failures=num_failures,
        start_time=start_time,
        last_request_timestamp=last_request_timestamp,
        response_times=response_times if response_times is not None else {50: num_requests},
    )


def _result(
    *,
    errors: tuple[LoadError, ...] = (),
    generator_warnings: tuple[str, ...] = (),
) -> LoadResult:
    return LoadResult(
        requests=10,
        failures=10,
        requests_per_second=1.0,
        p50_seconds=0.05,
        p90_seconds=0.08,
        p99_seconds=0.1,
        endpoints=(),
        errors=errors,
        generator_warnings=generator_warnings,
    )


class TestPercentiles:
    def test_median_is_the_middle_sample_not_the_mean_a_slow_tail_would_drag(self) -> None:
        # Nine fast requests and one very slow one: the mean is 1.99s, the median is 20ms.
        entry = _entry(num_requests=10, response_times={20: 9, 20000: 1})

        assert percentile_seconds([entry], 0.5) == 0.02

    def test_the_tail_percentiles_reach_the_slow_samples_the_median_hides(self) -> None:
        # 100 samples: 89 fast, 10 slow, 1 very slow. p50 sits in the fast bucket, p90 in the
        # slow one, and p99 lands on the single very slow sample.
        entry = _entry(num_requests=100, response_times={20: 89, 500: 10, 20000: 1})

        assert percentile_seconds([entry], 0.5) == 0.02
        assert percentile_seconds([entry], 0.9) == 0.5
        assert percentile_seconds([entry], 0.99) == 0.5
        assert percentile_seconds([entry], 1.0) == 20.0

    def test_percentiles_merge_the_histograms_of_every_stats_entry(self) -> None:
        # Per entry the median would be 10ms and 90ms; merged, the middle of the five samples is 90ms.
        entries = [
            _entry(num_requests=2, response_times={10: 2}),
            _entry(num_requests=3, response_times={90: 3}),
        ]

        assert percentile_seconds(entries, 0.5) == 0.09

    def test_an_even_split_takes_the_lower_middle_sample_as_locust_itself_does(self) -> None:
        entry = _entry(num_requests=4, response_times={10: 2, 90: 2})

        assert percentile_seconds([entry], 0.5) == 0.01

    def test_no_samples_reports_zero_rather_than_dividing_by_an_empty_histogram(self) -> None:
        assert percentile_seconds([], 0.5) == 0.0


class TestAggregate:
    def test_throughput_spans_the_whole_window_and_latency_comes_from_the_histogram(self) -> None:
        entry = _entry(
            num_requests=180,
            start_time=1000.0,
            last_request_timestamp=1060.0,
            response_times={57: 180},
        )

        result = aggregate_stats([entry], (), ())

        assert result.requests_per_second == 3.0
        assert result.p50_seconds == 0.057
        assert result.p99_seconds == 0.057
        assert result.failure_ratio == 0.0

    def test_tail_percentiles_come_from_the_slow_end_of_the_histogram(self) -> None:
        entry = _entry(num_requests=100, response_times={20: 89, 500: 10, 3000: 1})

        result = aggregate_stats([entry], (), ())

        assert result.p50_seconds == 0.02
        assert result.p90_seconds == 0.5
        assert result.p99_seconds == 0.5
        assert result.latency_summary() == "p50 0.020s, p90 0.500s, p99 0.500s"

    def test_throughput_spans_from_the_earliest_start_when_locust_reports_several_entries(self) -> None:
        entries = [
            _entry(num_requests=60, start_time=1000.0, last_request_timestamp=1030.0),
            _entry(num_requests=60, start_time=1020.0, last_request_timestamp=1060.0),
        ]

        result = aggregate_stats(entries, (), ())

        assert result.requests_per_second == 2.0

    def test_a_run_that_drove_no_traffic_reports_a_total_failure_ratio(self) -> None:
        result = aggregate_stats([], (), ())

        assert result.requests == 0
        assert result.requests_per_second == 0.0
        assert result.failure_ratio == 1.0
        assert result.endpoints == ()


class TestPerEndpoint:
    def test_each_route_keeps_its_own_requests_failures_and_median(self) -> None:
        entries: Final = (
            _entry(name="/chat/completions", num_requests=100, response_times={20: 100}),
            _entry(name="/v1/messages", num_requests=40, num_failures=3, response_times={900: 40}),
        )

        result: Final = aggregate_stats(entries, (), ())

        assert tuple((one.name, one.requests, one.failures, one.p50_seconds) for one in result.endpoints) == (
            ("/chat/completions", 100, 0, 0.02),
            ("/v1/messages", 40, 3, 0.9),
        )

    def test_several_stats_entries_for_one_route_fold_into_a_single_row(self) -> None:
        entries: Final = (
            _entry(name="/v1/messages", num_requests=10, response_times={30: 10}),
            _entry(name="/v1/messages", num_requests=30, num_failures=1, response_times={30: 30}),
        )

        result: Final = aggregate_stats(entries, (), ())

        assert tuple((one.name, one.requests, one.failures) for one in result.endpoints) == (("/v1/messages", 40, 1),)

    def test_a_route_that_never_ran_is_absent_so_a_one_sided_run_cannot_pass_unnoticed(self) -> None:
        result: Final = aggregate_stats((_entry(name="/chat/completions", num_requests=10),), (), ())

        assert tuple(one.name for one in result.endpoints) == ("/chat/completions",)

    def test_the_summary_names_every_route_with_its_counts(self) -> None:
        entries: Final = (
            _entry(name="/chat/completions", num_requests=2, response_times={20: 2}),
            _entry(name="/v1/messages", num_requests=1, num_failures=1, response_times={500: 1}),
        )

        result: Final = aggregate_stats(entries, (), ())

        assert result.endpoint_summary() == (
            "/chat/completions 2 requests, 0 failures, p50 0.020s, /v1/messages 1 requests, 1 failures, p50 0.500s"
        )


class TestErrorBreakdown:
    def test_locust_failure_rows_become_the_error_breakdown(self, tmp_path: Path) -> None:
        failures_csv = tmp_path / "locust_failures.csv"
        failures_csv.write_text(
            _FAILURES_HEADER
            + 'POST,/chat/completions,"LocustBadStatusCode(code=401)",381,2026-07-30 12:42:01,2026-07-30 12:45:00\n'
        )

        assert read_errors(failures_csv) == (
            LoadError(name="/chat/completions", error="LocustBadStatusCode(code=401)", occurrences=381),
        )

    def test_a_run_with_no_failures_writes_no_csv_and_reports_no_errors(self, tmp_path: Path) -> None:
        assert read_errors(tmp_path / "locust_failures.csv") == ()

    def test_diagnosis_leads_with_the_most_common_error(self) -> None:
        result = _result(
            errors=(
                LoadError(name="/chat/completions", error="ConnectionRefused", occurrences=12),
                LoadError(name="/chat/completions", error="LocustBadStatusCode(code=503)", occurrences=43675),
            )
        )

        assert result.diagnosis().startswith("43675x /chat/completions: LocustBadStatusCode(code=503)")

    def test_diagnosis_caps_the_list_and_says_how_many_it_left_out(self) -> None:
        result = _result(
            errors=tuple(
                LoadError(name="/chat/completions", error=f"error-{index}", occurrences=index) for index in range(1, 9)
            )
        )

        assert result.diagnosis().count("x /chat/completions") == 5
        assert "and 3 more distinct errors" in result.diagnosis()

    def test_diagnosis_says_so_when_locust_recorded_nothing(self) -> None:
        assert _result().diagnosis() == "locust recorded no error breakdown"


class TestGeneratorSaturation:
    def test_repeated_cpu_warnings_collapse_to_one_and_reach_the_diagnosis(self) -> None:
        stderr = (
            "[2026-07-31 12:47:01] WARNING/locust.runners: CPU usage above 90%!\n"
            "[2026-07-31 12:47:02] INFO/locust.main: Run time limit reached\n"
            "[2026-07-31 12:47:03] WARNING/locust.runners: CPU usage above 90%!\n"
        )

        warnings = read_generator_warnings(stderr)

        assert len(warnings) == 1
        assert "CPU usage above 90%!" in warnings[0]
        assert "CPU usage above 90%!" in _result(generator_warnings=warnings).diagnosis()

    def test_ordinary_locust_chatter_is_not_reported_as_a_warning(self) -> None:
        assert read_generator_warnings("[2026-07-31] INFO/locust.main: Shutting down (exit code 0)\n") == ()
