from __future__ import annotations

import json
import math
import threading
from collections.abc import Iterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import Final

import pytest
from cases.small_bodies_5rps import CASE
from memory_readings import RssCapture
from setup.observe import Observation, evaluate, junit_properties, render
from setup.performance_test_case import Limit
from setup.podcpu import GatewayCpuSampler, PodMetrics, PodMetricsList
from setup.wire import WireRecord


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


@pytest.mark.parametrize(("gap", "expected", "coverage"), ((1.0, 63.0, 0.95), (2.0, 62.0, 0.9)))
def test_short_internal_gaps_use_both_neighbouring_rates_even_at_ninety_percent(
    gap: float, expected: float, coverage: float
) -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", 8, "8s", "2")), _metrics(_pod("gateway", 20, f"{12 - gap}s", "4"))))
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is not None and math.isclose(sampler.seconds, expected)
    assert sampler.coverage is not None and math.isclose(sampler.coverage, coverage)


def test_a_gap_below_ninety_percent_is_missing_but_a_shorter_boundary_gap_is_bridged() -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", 8, "8s", "2")), _metrics(_pod("gateway", 20, "9.8s", "4"))))
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is None
    assert sampler.coverage is not None and math.isclose(sampler.coverage, 0.89)
    coverage: Final = sampler.series.coverage_between(1010.0, 1020.0)
    assert coverage is not None and math.isclose(coverage, 0.98)
    assert math.isclose(sum(sampler.series.between(1010.0, 1020.0) or ()), 40.0)


@pytest.mark.parametrize(("first", "last"), ((0, 20), (19, 19)), ids=("start-gap", "end-gap"))
def test_a_boundary_gap_uses_the_one_available_source_rate(first: int, last: int) -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", first, "19s", "3")), _metrics(_pod("gateway", last, "19s", "3"))))
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds == 60.0
    assert sampler.coverage == 0.95


@pytest.mark.parametrize(("window", "expected", "coverage"), (("19s", 60.0, 0.95), ("17s", None, 0.85)))
def test_cpu_coverage_is_reported_next_to_the_reading_in_junit_and_logs(
    capsys: pytest.CaptureFixture[str], window: str, expected: float | None, coverage: float
) -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", 0, "20s", "0")), _metrics(_pod("gateway", 20, window, "3"))))
    )
    sampler.start()
    sampler.stop()
    record: Final = WireRecord(
        seq=1,
        marker="request",
        caller="key",
        endpoint="chat_completions",
        stream=False,
        blocked=False,
        started_at=1000.0,
        status=200,
        header_s=1.0,
        first_byte_s=1.0,
        first_token_s=None,
        total_s=1.0,
        text="",
        ok=True,
        mock_delay_s=0.0,
        call_id=None,
        error=None,
    )
    observation: Final = Observation(
        records=(record,),
        mock_requests=(),
        guardrail_calls=(),
        rss_before=RssCapture(readings=(), failures=()),
        rss_after=RssCapture(readings=(), failures=()),
        wall_seconds=20.0,
        gateway_cpu_seconds=sampler.seconds,
        gateway_cpu_coverage=sampler.coverage,
    )
    case: Final = replace(CASE, limits=(Limit("cpu_seconds_per_request", max=100.0),))
    rows, _ = evaluate(case, observation)
    properties: Final = dict(junit_properties(rows))
    line: Final = next(
        line for line in render(case, observation, rows).splitlines() if "cpu_seconds_per_request" in line
    )

    assert properties["perf_cpu_seconds_per_request_coverage"] == coverage
    assert properties.get("perf_cpu_seconds_per_request") == expected
    assert f"coverage {coverage:.4f}" in line
    assert ("n/a" if expected is None else f"{expected:.4f}") in line
    assert f'"coverage": {coverage}' in capsys.readouterr().out
    assert math.isclose(sum(sampler.series.between(1002.9, 1020.0) or ()), 51.3)


def test_disjoint_pod_gaps_count_against_the_same_gateway_coverage_budget() -> None:
    sampler: Final = _sampler(
        iter(
            (
                _metrics(_pod("a", 0, "1s", "0"), _pod("b", 18, "18s", "3")),
                _metrics(_pod("a", 20, "18s", "2"), _pod("b", 18, "18s", "3")),
            )
        )
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is None
    assert sampler.coverage == 0.8
    assert sum(sampler.series.between(1001.5, 1018.5) or ()) == 85.0


@pytest.mark.parametrize(
    ("last_end", "reason", "uncovered_start", "uncovered_end", "next_start", "next_end"),
    ((20, "source_gap", 1004.0, 1010.0, 1010.0, 1020.0), (10, "tail_not_covered", 1010.0, 1020.0, None, None)),
)
def test_cpu_diagnostics_report_source_windows_and_exact_uncovered_interval(
    capsys: pytest.CaptureFixture[str],
    last_end: int,
    reason: str,
    uncovered_start: float,
    uncovered_end: float,
    next_start: float | None,
    next_end: float | None,
) -> None:
    first: Final = _pod("gateway", 4, "4s", "2")
    sampler: Final = _sampler(iter((_metrics(first), _metrics(_pod("gateway", last_end, "10s", "2")))))
    sampler.start()
    sampler.stop()
    assert sampler.seconds is None

    lines: Final = capsys.readouterr().out.splitlines()
    assert (
        "gateway_cpu "
        + json.dumps(
            {
                "event": "response",
                "pod": "gateway",
                "timestamp": first.timestamp.isoformat(),
                "window": "4s",
                "cpu": ["2"],
                "valid": True,
            },
            sort_keys=True,
        )
        in lines
    )
    assert (
        "gateway_cpu "
        + json.dumps(
            {
                "event": "missing",
                "reason": reason,
                "pod": "gateway",
                "started_at": 1000.0,
                "stopped_at": 1020.0,
                "uncovered_start": uncovered_start,
                "uncovered_end": uncovered_end,
                "gap_seconds": uncovered_end - uncovered_start,
                "next_window_start": next_start,
                "next_window_end": next_end,
            },
            sort_keys=True,
        )
        in lines
    )


def test_cpu_diagnostics_distinguish_population_change_from_unavailable_reads(
    capsys: pytest.CaptureFixture[str],
) -> None:
    changed: Final = _sampler(iter((_metrics(_pod("gateway", 0, "20s", "2")), _metrics(_pod("new", 20, "20s", "2")))))
    changed.start()
    changed.stop()
    assert changed.seconds is None
    assert changed.coverage is None
    changed_log: Final = capsys.readouterr().out
    assert '"reason": "pod_population_changed"' in changed_log
    assert '"expected": ["gateway"]' in changed_log
    assert '"observed": ["new"]' in changed_log

    unavailable: Final = _sampler(iter((None, None)))
    unavailable.start()
    unavailable.stop()
    assert unavailable.seconds is None
    assert unavailable.coverage is None
    unavailable_log: Final = capsys.readouterr().out
    assert '"state": "read_failed"' in unavailable_log
    assert '"reason": "no_pod_metrics"' in unavailable_log


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

    assert sampler.seconds == 49.75
    assert sampler.coverage == 0.975
    assert tuple(sample.at for sample in sampler.series.samples) == (1005.0, 1010.25, 1015.25)


def test_collecting_final_cpu_does_not_hold_up_the_other_post_run_snapshots() -> None:
    snapshots_read: Final = threading.Event()
    final: Final = _metrics(_pod("gateway", 20, "20s", "2"))
    responses: Final = iter((_metrics(_pod("gateway", 0, "20s", "0")), final))
    times: Final = iter((1000.0, 1020.0))

    def read() -> PodMetricsList | None:
        response: Final = next(responses, None)
        if response is final and not snapshots_read.wait(1.0):
            return None
        return response

    sampler: Final = GatewayCpuSampler(
        "gateway", "perf", "token", "ca.pem", read_metrics=read, clock=lambda: next(times), settle_timeout_s=0.0
    )
    sampler.start()
    sampler.stop()
    snapshots_read.set()

    assert sampler.seconds == 40.0


@pytest.mark.parametrize(("last_pod", "last_window"), (("gateway", "10s"), ("replacement", "20s")))
def test_incomplete_cpu_coverage_does_not_become_zero_in_ramp_windows(last_pod: str, last_window: str) -> None:
    sampler: Final = _sampler(
        iter((_metrics(_pod("gateway", 4, "4s", "2")), _metrics(_pod(last_pod, 20, last_window, "2"))))
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is None
    assert sampler.series.between(1000.0, 1010.0) is None
    assert sampler.series.between(1010.0, 1020.0) == ((20.0,) if last_pod == "gateway" else None)


def test_pod_replacement_ends_final_collection_without_waiting_or_reading_again() -> None:
    paused: Final = threading.Event()
    extra_read: Final = threading.Event()
    responses: Final = iter((_metrics(_pod("gateway", 0, "20s", "2")), _metrics(_pod("replacement", 20, "20s", "2"))))
    times: Final = iter((1000.0, 1020.0))

    def read() -> PodMetricsList | None:
        response: Final = next(responses, None)
        if response is None:
            extra_read.set()
        return response

    sampler: Final = GatewayCpuSampler(
        "gateway",
        "perf",
        "token",
        "ca.pem",
        read_metrics=read,
        clock=lambda: next(times),
        pause=lambda _: paused.set(),
        settle_timeout_s=1.0,
    )
    sampler.start()
    sampler.stop()

    assert sampler.seconds is None
    assert not paused.is_set()
    assert not extra_read.is_set()
