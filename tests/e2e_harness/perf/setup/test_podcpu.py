from __future__ import annotations

import math
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Final

import pytest
from setup.podcpu import GatewayCpuSampler, PodMetrics, PodMetricsList


def _pod(name: str, end: float, window: str, *quantities: str) -> PodMetrics:
    return PodMetrics.model_validate(
        {
            "metadata": {"name": name},
            "timestamp": datetime.fromtimestamp(1000.0 + end, tz=UTC),
            "window": window,
            "containers": [{"usage": {"cpu": quantity}} for quantity in quantities],
        }
    )


def _metrics(*pods: PodMetrics) -> PodMetricsList:
    return PodMetricsList(items=list(pods))


def _sampler(responses: Iterator[PodMetricsList | None], *, settle_timeout_s: float = 0.0) -> GatewayCpuSampler:
    times: Final = iter((1000.0, 1020.0))
    return GatewayCpuSampler(
        "app=gateway",
        "perf",
        "test-token",
        "test-ca",
        read_metrics=lambda: next(responses, None),
        clock=lambda: next(times),
        pause=lambda _: None,
        settle_timeout_s=settle_timeout_s,
    )


def test_stale_start_window_is_not_extrapolated_and_stop_collects_the_final_partial_window() -> None:
    stale: Final = _metrics(_pod("gateway", 4, "18.091s", "800m"))
    sampler: Final = _sampler(
        iter(
            (
                _metrics(_pod("gateway", -5, "18.091s", "0")),
                stale,
                stale,
                _metrics(_pod("gateway", 24, "20s", "3500m")),
            )
        ),
        settle_timeout_s=1.0,
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is not None and math.isclose(sampler.seconds, 0.8 * 4 + 3.5 * 16)
    assert [(sample.at, sample.value) for sample in sampler.series.samples] == [
        (1002.0, 3.2),
        (1012.0, 56.0),
    ]


def test_overlapping_source_windows_charge_each_second_once() -> None:
    sampler: Final = _sampler(
        iter(
            (
                _metrics(_pod("gateway", 0, "10s", "2")),
                _metrics(_pod("gateway", 10, "10s", "2")),
                _metrics(_pod("gateway", 24, "18s", "3")),
            )
        ),
        settle_timeout_s=1.0,
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is not None and math.isclose(sampler.seconds, 2 * 10 + 3 * 10)
    assert math.isclose(sum(sample.value for sample in sampler.series.samples), 50.0)


def test_each_pod_uses_its_own_window_and_all_containers_are_counted() -> None:
    sampler: Final = _sampler(
        iter(
            (
                _metrics(_pod("a", 0, "10s", "0"), _pod("b", 0, "10s", "0")),
                _metrics(_pod("a", 22, "30s", "500m", "1500m"), _pod("b", 26, "40s", "250m", "750m")),
            )
        )
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is not None and math.isclose(sampler.seconds, (0.5 + 1.5 + 0.25 + 0.75) * 20)
    assert len(sampler.series.samples) == 2
    assert tuple(sample.at for sample in sampler.series.samples) == (1010.0, 1010.0)


@pytest.mark.parametrize("quantity", ("2", "2000m", "2000000u", "2000000000n"))
@pytest.mark.parametrize("window", ("20s", "20000ms", "20000000us", "20000000000ns", "1m10s"))
def test_cpu_quantities_and_source_duration_units_are_converted_to_cpu_seconds(quantity: str, window: str) -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", 0, "10s", "0")), _metrics(_pod("gateway", 20, window, quantity))))
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is not None and math.isclose(sampler.seconds, 40.0)


@pytest.mark.parametrize(
    "final",
    (
        None,
        _metrics(),
        _metrics(_pod("gateway", 4, "18.091s", "800m")),
        _metrics(_pod("gateway", 25, "20s", "3")),
        _metrics(_pod("replacement", 20, "20s", "3")),
    ),
    ids=("network-error", "empty-list", "stale-start-only", "coverage-gap", "pod-replaced"),
)
def test_missing_coverage_is_missing_cpu_instead_of_a_low_passing_number(final: PodMetricsList | None) -> None:
    sampler: Final = _sampler(iter((_metrics(_pod("gateway", 0, "10s", "0")), final)))
    sampler.start()
    sampler.stop()

    assert sampler.seconds is None
    assert sampler.series.samples == ()


@pytest.mark.parametrize("window", ("", "0s", "-20s", "NaNs", "20seconds", "20s-junk"))
def test_invalid_source_windows_do_not_become_cpu_measurements(window: str) -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", 0, "10s", "0")), _metrics(_pod("gateway", 20, window, "2"))))
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is None
    assert sampler.series.samples == ()


@pytest.mark.parametrize("quantity", ("-1", "nan", "inf", "not-cpu"))
def test_invalid_cpu_rates_do_not_become_cpu_measurements(quantity: str) -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", 0, "10s", "0")), _metrics(_pod("gateway", 20, "20s", quantity))))
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is None
    assert sampler.series.samples == ()


def test_zero_cpu_with_complete_coverage_is_a_measurement() -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", 0, "10s", "0")), _metrics(_pod("gateway", 20, "20s", "0"))))
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds == 0.0
    assert len(sampler.series.samples) == 1


def test_a_pod_without_metrics_cannot_be_hidden_by_a_complete_other_pod() -> None:
    sampler: Final = _sampler(
        iter(
            (
                _metrics(_pod("a", 0, "10s", "0"), _pod("b", 0, "10s", "0")),
                _metrics(_pod("a", 20, "20s", "2"), _pod("b", 4, "20s", "1")),
            )
        )
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is None
    assert sampler.series.samples == ()


def test_whole_second_timestamps_do_not_turn_subsecond_rounding_into_a_missing_pass() -> None:
    sampler: Final = _sampler(
        iter(
            (
                _metrics(_pod("gateway", 0, "10s", "0")),
                _metrics(_pod("gateway", 10, "10s", "2")),
                _metrics(_pod("gateway", 20, "9.5s", "3")),
            )
        ),
        settle_timeout_s=1.0,
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds == 50.0
    assert tuple(sample.at for sample in sampler.series.samples) == (1005.0, 1015.0)
