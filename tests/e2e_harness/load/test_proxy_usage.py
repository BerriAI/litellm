from __future__ import annotations

from typing import Final

from proxy_usage import UsageSample, UsageWindow

_MB: Final = 2**20


def _window(*points: tuple[float, int, float]) -> UsageWindow:
    return UsageWindow(
        samples=tuple(
            UsageSample(elapsed_seconds=elapsed, rss_bytes=rss, cpu_seconds=cpu) for elapsed, rss, cpu in points
        )
    )


class TestRssPercentiles:
    def test_the_tail_percentiles_reach_the_peak_the_median_hides(self) -> None:
        # 100 one-second samples: 89 flat, 10 elevated, 1 spike. The median stays flat, p90 sees the
        # elevated plateau, and only the max reaches the spike.
        window: Final = _window(
            *((float(i), 100 * _MB, float(i)) for i in range(89)),
            *((float(89 + i), 300 * _MB, float(89 + i)) for i in range(10)),
            (99.0, 900 * _MB, 99.0),
        )

        assert window.rss_percentile(0.5) == 100 * _MB
        assert window.rss_percentile(0.9) == 300 * _MB
        assert window.rss_percentile(0.99) == 300 * _MB
        assert window.rss_percentile(1.0) == 900 * _MB

    def test_an_empty_window_reports_zero_rather_than_indexing_nothing(self) -> None:
        assert _window().rss_percentile(0.5) == 0


class TestCpuUtilization:
    def test_utilization_is_the_counter_delta_over_the_interval_not_the_counter_itself(self) -> None:
        # The counter climbs 0.5 CPU seconds per second, then 4.0 per second: half a core, then four.
        window: Final = _window((0.0, _MB, 0.0), (1.0, _MB, 0.5), (2.0, _MB, 1.0), (3.0, _MB, 5.0))

        p50, p90, p99 = window.cpu_utilization_percentiles()

        assert (p50, p90, p99) == (0.5, 4.0, 4.0)
        assert window.cpu_seconds_consumed() == 5.0

    def test_a_single_sample_has_no_interval_and_reports_zero(self) -> None:
        window: Final = _window((0.0, _MB, 3.0))

        assert window.cpu_utilization_percentiles() == (0.0, 0.0, 0.0)
        assert window.cpu_seconds_consumed() == 0.0

    def test_cost_per_request_separates_runs_that_cores_busy_reports_identically(self) -> None:
        # Both windows pin 4 cores for 10 seconds, so utilization cannot tell them apart. The
        # second one served a tenth of the traffic for the same CPU, which is the regression shape.
        window: Final = _window(*((float(i), _MB, 4.0 * i) for i in range(11)))

        assert window.cpu_utilization_percentiles()[0] == 4.0
        assert window.cpu_seconds_per_request(4000) == 0.01
        assert window.cpu_seconds_per_request(400) == 0.1

    def test_no_requests_reports_zero_cost_rather_than_dividing_by_zero(self) -> None:
        assert _window((0.0, _MB, 0.0), (1.0, _MB, 1.0)).cpu_seconds_per_request(0) == 0.0

    def test_summary_reports_every_percentile_in_human_units(self) -> None:
        window: Final = _window((0.0, 200 * _MB, 0.0), (1.0, 200 * _MB, 1.5), (2.0, 200 * _MB, 3.0))

        assert window.summary() == (
            "RSS p50 200 MB, p90 200 MB, p99 200 MB; "
            "CPU cores busy p50 1.50, p90 1.50, p99 1.50; 3.0 CPU seconds consumed"
        )
