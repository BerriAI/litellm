from __future__ import annotations

from types import MappingProxyType
from typing import Final

import pytest
from memory_readings import RssReading, WorkerKey, assert_same_rss_workers


def _workers(*identities: tuple[str | None, int]) -> MappingProxyType[WorkerKey, RssReading]:
    return MappingProxyType(
        {("http://router", pod, pid): RssReading("http://router", pod, pid, 1000.0) for pod, pid in identities}
    )


def test_same_workers_can_be_compared_in_any_order() -> None:
    before: Final = _workers(("pod-a", 11), ("pod-a", 12), ("pod-b", 11))
    after: Final = _workers(("pod-b", 11), ("pod-a", 12), ("pod-a", 11))

    assert assert_same_rss_workers(before, after) is None


@pytest.mark.parametrize(
    "after",
    [
        _workers(("pod-a", 11)),
        _workers(),
        _workers(("pod-a", 11), ("pod-a", 12), ("pod-a", 13)),
        _workers(("pod-a", 11), ("pod-a", 13), ("pod-a", 14)),
    ],
)
def test_unmatched_capture_is_probe_incomplete(after: MappingProxyType[WorkerKey, RssReading]) -> None:
    before: Final = _workers(("pod-a", 11), ("pod-a", 12))

    with pytest.raises(AssertionError, match="probe incomplete: RSS growth limit set but nothing observed"):
        assert_same_rss_workers(before, after)


@pytest.mark.parametrize("before", [_workers(), _workers((None, 11))])
def test_missing_baseline_or_pod_identity_is_probe_incomplete(before: MappingProxyType[WorkerKey, RssReading]) -> None:
    with pytest.raises(AssertionError, match="probe incomplete"):
        assert_same_rss_workers(before, _workers((None, 12)))


@pytest.mark.parametrize("after", [_workers(("pod-a", 13)), _workers(("pod-a", 11), ("pod-a", 13))])
def test_new_pid_on_the_same_pod_is_worker_death(after: MappingProxyType[WorkerKey, RssReading]) -> None:
    before: Final = _workers(("pod-a", 11), ("pod-a", 12))

    with pytest.raises(AssertionError, match="worker death: new pid on the same pod"):
        assert_same_rss_workers(before, after)


@pytest.mark.parametrize("after", [_workers(("pod-b", 11)), _workers(("pod-a", 11), ("pod-a", 12), ("pod-b", 13))])
def test_new_hostname_after_warm_window_is_worker_death(after: MappingProxyType[WorkerKey, RssReading]) -> None:
    before: Final = _workers(("pod-a", 11), ("pod-a", 12))

    with pytest.raises(AssertionError, match="worker death: new pod hostname after the warm window"):
        assert_same_rss_workers(before, after)
