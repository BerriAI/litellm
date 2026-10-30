import threading
from collections.abc import Callable
from typing import Final, TypeAlias

import pytest

from litellm.integrations.prometheus_helpers.deployment_state_series import (
    MAX_SERIES_PER_DEPLOYMENT,
    MAX_TRACKED_DEPLOYMENTS,
    DeploymentState,
    DeploymentStateSeries,
    DeploymentStateSeriesTracker,
)

HEALTHY: Final = DeploymentState.HEALTHY
PARTIAL_OUTAGE: Final = DeploymentState.PARTIAL_OUTAGE
COMPLETE_OUTAGE: Final = DeploymentState.COMPLETE_OUTAGE

Gauge: TypeAlias = dict[DeploymentStateSeries, DeploymentState]
StateChange: TypeAlias = Callable[[DeploymentStateSeriesTracker, DeploymentStateSeries], None]


def gauge_writer(
    gauge: Gauge, before_write: Callable[[DeploymentState], None] = lambda state: None
) -> Callable[[DeploymentState, str, str | None, str | None, str], None]:
    def write(
        state: DeploymentState, litellm_model_name: str, model_id: str | None, api_base: str | None, api_provider: str
    ) -> None:
        before_write(state)
        gauge[DeploymentStateSeries(litellm_model_name, model_id, api_base, api_provider)] = state

    return write


def tracker_and_gauge() -> tuple[DeploymentStateSeriesTracker, Gauge]:
    gauge: Final[Gauge] = {}
    return DeploymentStateSeriesTracker(write=gauge_writer(gauge)), gauge


def request_series(model_id: str, api_base: str = "https://api.example/v1") -> DeploymentStateSeries:
    return DeploymentStateSeries("gpt", model_id, api_base, "openai")


def config_series(model_id: str) -> DeploymentStateSeries:
    return DeploymentStateSeries("group", model_id, "", "openai")


def distinct_series(count: int) -> list[DeploymentStateSeries]:
    return [request_series("dep-1", f"https://api.example/{index}") for index in range(count)]


class ContentionRecordingLock:
    def __init__(self, on_contention: threading.Event) -> None:
        self.contended: Final = threading.Event()
        self._on_contention = on_contention
        self._lock = threading.Lock()

    def __enter__(self) -> None:
        if not self._lock.acquire(blocking=False):
            self.contended.set()
            self._on_contention.set()
            self._lock.acquire()

    def __exit__(self, *exc_info: object) -> None:
        self._lock.release()


def test_a_deployment_is_remembered_until_more_than_the_cap_of_deployments_were_seen():
    tracker, gauge = tracker_and_gauge()
    for index in range(MAX_TRACKED_DEPLOYMENTS):
        tracker.healthy(request_series(f"dep-{index}"))

    tracker.complete_outage(config_series("dep-0"))

    assert gauge == {
        **{request_series(f"dep-{index}"): HEALTHY for index in range(1, MAX_TRACKED_DEPLOYMENTS)},
        request_series("dep-0"): COMPLETE_OUTAGE,
    }


def test_the_deployment_unused_for_longest_is_forgotten_first():
    tracker, gauge = tracker_and_gauge()
    tracker.healthy(request_series("busy"))
    for index in range(MAX_TRACKED_DEPLOYMENTS - 1):
        tracker.healthy(request_series(f"dep-{index}"))
    tracker.healthy(request_series("busy"))
    tracker.healthy(request_series("newest"))

    tracker.complete_outage(config_series("busy"))
    tracker.complete_outage(config_series("dep-0"))

    assert gauge == {
        **{request_series(f"dep-{index}"): HEALTHY for index in range(MAX_TRACKED_DEPLOYMENTS - 1)},
        request_series("busy"): COMPLETE_OUTAGE,
        request_series("newest"): HEALTHY,
        config_series("dep-0"): COMPLETE_OUTAGE,
    }


def test_a_cooldown_marks_only_the_most_recent_series_of_a_deployment():
    tracker, gauge = tracker_and_gauge()
    failing: Final = distinct_series(MAX_SERIES_PER_DEPLOYMENT + 3)
    for series in failing:
        tracker.partial_outage(series)

    tracker.complete_outage(config_series("dep-1"))

    assert gauge == {
        **{series: PARTIAL_OUTAGE for series in failing[:-MAX_SERIES_PER_DEPLOYMENT]},
        **{series: COMPLETE_OUTAGE for series in failing[-MAX_SERIES_PER_DEPLOYMENT:]},
    }


def test_repeated_failures_of_one_series_do_not_push_another_series_out_of_the_cooldown():
    tracker, gauge = tracker_and_gauge()
    first, repeating = request_series("dep-1", "first"), request_series("dep-1", "repeating")
    tracker.partial_outage(first)
    for _ in range(MAX_SERIES_PER_DEPLOYMENT + 2):
        tracker.partial_outage(repeating)

    tracker.complete_outage(config_series("dep-1"))

    assert gauge == {first: COMPLETE_OUTAGE, repeating: COMPLETE_OUTAGE}


def test_a_success_under_a_new_label_clears_every_series_a_full_cooldown_marked():
    tracker, gauge = tracker_and_gauge()
    failing: Final = distinct_series(MAX_SERIES_PER_DEPLOYMENT)
    for series in failing:
        tracker.partial_outage(series)
    tracker.complete_outage(config_series("dep-1"))
    recovered: Final = request_series("dep-1", "https://api.example/recovered")

    tracker.healthy(recovered)

    assert gauge == {**{series: HEALTHY for series in failing}, recovered: HEALTHY}


def test_no_series_stays_at_two_after_recovery_when_a_cooldown_saw_more_labels_than_it_tracks():
    tracker, gauge = tracker_and_gauge()
    labels: Final = distinct_series(MAX_SERIES_PER_DEPLOYMENT + 1)
    for series in labels[:MAX_SERIES_PER_DEPLOYMENT]:
        tracker.partial_outage(series)
    tracker.complete_outage(config_series("dep-1"))
    tracker.partial_outage(labels[-1])

    tracker.healthy(labels[-1])

    assert gauge == {series: HEALTHY for series in labels}


def test_a_series_dropped_during_a_cooldown_stays_cleared_through_a_second_cooldown():
    tracker, gauge = tracker_and_gauge()
    labels: Final = distinct_series(MAX_SERIES_PER_DEPLOYMENT + 1)
    for series in labels[:MAX_SERIES_PER_DEPLOYMENT]:
        tracker.partial_outage(series)
    tracker.complete_outage(config_series("dep-1"))
    tracker.partial_outage(labels[-1])

    tracker.complete_outage(config_series("dep-1"))

    assert gauge == {labels[0]: HEALTHY, **{series: COMPLETE_OUTAGE for series in labels[1:]}}


def test_failures_beyond_the_series_cap_outside_a_cooldown_leave_the_older_series_at_one():
    tracker, gauge = tracker_and_gauge()
    labels: Final = distinct_series(MAX_SERIES_PER_DEPLOYMENT + 1)
    for series in labels:
        tracker.partial_outage(series)

    assert gauge == {series: PARTIAL_OUTAGE for series in labels}


def test_a_deployment_pushed_out_of_the_tracker_keeps_the_state_its_requests_last_reported():
    tracker, gauge = tracker_and_gauge()
    failing: Final = request_series("failing")
    tracker.partial_outage(failing)
    for index in range(MAX_TRACKED_DEPLOYMENTS):
        tracker.healthy(request_series(f"dep-{index}"))

    assert gauge == {
        failing: PARTIAL_OUTAGE,
        **{request_series(f"dep-{index}"): HEALTHY for index in range(MAX_TRACKED_DEPLOYMENTS)},
    }


def test_a_cooling_deployment_pushed_out_of_the_tracker_is_cleared_instead_of_left_at_two():
    tracker, gauge = tracker_and_gauge()
    cooling: Final = request_series("cooling")
    tracker.healthy(cooling)
    tracker.complete_outage(config_series("cooling"))
    for index in range(MAX_TRACKED_DEPLOYMENTS):
        tracker.healthy(request_series(f"dep-{index}"))

    assert gauge == {
        cooling: HEALTHY,
        **{request_series(f"dep-{index}"): HEALTHY for index in range(MAX_TRACKED_DEPLOYMENTS)},
    }


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    ("blocked_state", "in_flight_change", "in_flight_series", "waiting_change", "waiting_series", "expected_gauge"),
    [
        pytest.param(
            HEALTHY,
            DeploymentStateSeriesTracker.healthy,
            request_series("dep-1"),
            DeploymentStateSeriesTracker.complete_outage,
            config_series("dep-1"),
            {request_series("dep-1"): COMPLETE_OUTAGE},
            id="cooldown_waits_for_a_success_write",
        ),
        pytest.param(
            COMPLETE_OUTAGE,
            DeploymentStateSeriesTracker.complete_outage,
            config_series("dep-1"),
            DeploymentStateSeriesTracker.healthy,
            request_series("dep-1"),
            {config_series("dep-1"): HEALTHY, request_series("dep-1"): HEALTHY},
            id="success_waits_for_a_cooldown_write",
        ),
        pytest.param(
            PARTIAL_OUTAGE,
            DeploymentStateSeriesTracker.partial_outage,
            request_series("dep-1"),
            DeploymentStateSeriesTracker.complete_outage,
            config_series("dep-1"),
            {request_series("dep-1"): COMPLETE_OUTAGE},
            id="cooldown_waits_for_a_failure_write",
        ),
        pytest.param(
            COMPLETE_OUTAGE,
            DeploymentStateSeriesTracker.complete_outage,
            config_series("dep-1"),
            DeploymentStateSeriesTracker.partial_outage,
            request_series("dep-1"),
            {config_series("dep-1"): COMPLETE_OUTAGE, request_series("dep-1"): COMPLETE_OUTAGE},
            id="failure_waits_for_a_cooldown_write",
        ),
    ],
)
def test_a_write_in_flight_holds_back_the_next_state_change_of_the_deployment(
    blocked_state: DeploymentState,
    in_flight_change: StateChange,
    in_flight_series: DeploymentStateSeries,
    waiting_change: StateChange,
    waiting_series: DeploymentStateSeries,
    expected_gauge: Gauge,
):
    in_flight: Final = threading.Event()
    release: Final = threading.Event()
    second_settled: Final = threading.Event()
    gauge: Final[Gauge] = {}

    def block_on_the_blocked_state(state: DeploymentState) -> None:
        if state == blocked_state:
            in_flight.set()
            release.wait()

    def change_state_of_waiting_call() -> None:
        try:
            waiting_change(tracker, waiting_series)
        finally:
            second_settled.set()

    lock: Final = ContentionRecordingLock(second_settled)
    tracker: Final = DeploymentStateSeriesTracker(write=gauge_writer(gauge, block_on_the_blocked_state), lock=lock)
    first: Final = threading.Thread(target=in_flight_change, args=(tracker, in_flight_series), daemon=True)
    second: Final = threading.Thread(target=change_state_of_waiting_call, daemon=True)

    try:
        first.start()
        in_flight.wait()
        second.start()
        second_settled.wait()
    finally:
        release.set()
    first.join()
    second.join()

    assert lock.contended.is_set()
    assert gauge == expected_gauge
