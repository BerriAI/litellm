"""
`least-busy` sent everything to the first-registered deployment and left the rest idle.

Two defects in `_get_available_deployments`:
- ties were compared with a strict `<`, so an equal count never displaced the first deployment. The
  counts drain to zero between requests, so in light traffic every request is a tie
- the counts came from a cache that can still hold ids for deployments that are no longer healthy.
  When a stale id held the minimum, selection fell through to an arbitrary pick
"""

import collections

import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.least_busy import LeastBusyLoggingHandler

HEALTHY = [{"model_info": {"id": "A"}}, {"model_info": {"id": "B"}}]


def _pick_counts(request_count: dict, healthy: list = HEALTHY, n: int = 1000) -> collections.Counter:
    handler = LeastBusyLoggingHandler(router_cache=DualCache())
    return collections.Counter(
        handler._get_available_deployments(healthy_deployments=healthy, all_deployments=dict(request_count))[
            "model_info"
        ]["id"]
        for _ in range(n)
    )


@pytest.mark.parametrize("request_count", [{}, {"A": 0, "B": 0}, {"A": 7, "B": 7}])
def test_ties_are_spread_across_the_tied_deployments(request_count):
    """Every deployment tied at the minimum must be reachable, not just the first one."""
    picks = _pick_counts(request_count)

    assert set(picks) == {"A", "B"}
    assert min(picks.values()) > 300


def test_a_three_way_tie_reaches_every_deployment():
    healthy = [{"model_info": {"id": name}} for name in ("A", "B", "C")]

    picks = _pick_counts({}, healthy=healthy, n=1200)

    assert set(picks) == {"A", "B", "C"}
    assert min(picks.values()) > 250


@pytest.mark.parametrize(
    "request_count, expected",
    [
        ({"A": 5, "B": 1}, "B"),
        ({"A": 1, "B": 5}, "A"),
        ({"A": 0, "B": 3}, "A"),
    ],
)
def test_the_least_busy_deployment_wins(request_count, expected):
    assert set(_pick_counts(request_count, n=200)) == {expected}


def test_an_unhealthy_deployment_cannot_win_the_minimum():
    """
    `C` is the least busy of the three but is in cooldown, so it is not in `healthy_deployments`. It
    must neither be returned nor make the pick arbitrary: the answer is the least busy healthy one.
    """
    picks = _pick_counts({"A": 5, "B": 3, "C": 0}, n=500)

    assert set(picks) == {"B"}


def test_a_deployment_with_no_recorded_traffic_counts_as_idle():
    """`B` has never been used, so it has no cache entry and must beat `A`."""
    picks = _pick_counts({"A": 4}, n=200)

    assert set(picks) == {"B"}
