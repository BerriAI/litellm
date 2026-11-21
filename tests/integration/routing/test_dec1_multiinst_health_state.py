"""Deployment health from background health checks, shared by proxy instances through one Redis entry.

With `general_settings.background_health_checks: true` an instance probes every deployment and merges the
results into the single Redis key `litellm:health_check:deployment_health_state`. Its value is a dict keyed by
deployment id whose records hold `is_healthy`, `timestamp` (epoch seconds) and `reason`, written with
`str(value)` (a Python literal) and a TTL of `int(staleness * 1.5)` seconds, where the staleness is
`router_settings.health_check_staleness_threshold` or `DEFAULT_HEALTH_CHECK_INTERVAL * 2` (600 seconds).
A write merges into what Redis already holds instead of replacing it, so pods that probe different deployments
do not erase each other, and records older than `staleness * 1.5` are pruned on write.

An instance with `router_settings.enable_health_check_routing: true` leaves out every deployment whose record is
unhealthy and younger than the staleness threshold, whether it probed it itself or read it from Redis. A record
with extra fields is still read, a missing `is_healthy` counts as healthy, and a timestamp that is no number
counts as 0 (stale).

Here only the first instance runs background health checks; the second only routes. The store moved to Rust in
#50357 (`litellm/rust/crates/redis-store/src/deployment_health/`), and `DeploymentHealthCache` in
`litellm/router_utils/health_state_cache.py` keeps the merge and staleness rules.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, held, object_value
from tests.integration.routing.dec1_multiinst_support import HEALTH_KEY, Fleet, decoded, fleet, served
from tests.integration.routing.dec1_routing_support import chat, deployment

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

STALENESS_SECONDS: Final = 40
HEALTH_TTL_SECONDS: Final = int(STALENESS_SECONDS * 1.5)
CONVERGENCE_SECONDS: Final = 60
ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "disable_cooldowns": True,
    "enable_health_check_routing": True,
    "health_check_staleness_threshold": STALENESS_SECONDS,
}
PROBING: Final[dict[str, JsonValue]] = {"background_health_checks": True, "health_check_interval": 2}
ROUTING_ONLY: Final[dict[str, JsonValue]] = {"background_health_checks": False}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("health-shared", "status-500-health-shared", api_base, weight=1),
        deployment("health-shared", "ok-health-shared", api_base),
        deployment("health-foreign", "ok-health-foreign-target", api_base, weight=1),
        deployment("health-foreign", "ok-health-foreign-sibling", api_base),
        deployment("health-stale", "ok-health-stale-target", api_base, weight=1),
        deployment("health-stale", "ok-health-stale-sibling", api_base),
        deployment("health-lenient", "ok-health-lenient-target", api_base, weight=1),
        deployment("health-lenient", "ok-health-lenient-sibling", api_base),
    )


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-multiinst-health"),
        models=_models,
        router_settings=ROUTER_SETTINGS,
        general_settings=PROBING,
        second_general_settings=ROUTING_ONLY,
    ) as value:
        yield value


def _states(pair: Fleet) -> dict[str, JsonValue]:
    raw: Final = pair.raw(HEALTH_KEY)
    return {} if raw is None else object_value(decoded(raw))


def _record(pair: Fleet, deployment_id: str) -> dict[str, JsonValue] | None:
    found: Final = _states(pair).get(deployment_id)
    return None if found is None else object_value(found)


def _write_states(pair: Fleet, extra: Mapping[str, Mapping[str, object]]) -> None:
    """Merges records into the shared entry the way another Python pod would: `str(dict)` with the entry's TTL."""
    with pair.store() as client:
        raw = client.get(HEALTH_KEY)
        current = {} if raw is None else dict(object_value(decoded(bytes(raw))))
        client.set(HEALTH_KEY, str({**current, **extra}), ex=HEALTH_TTL_SECONDS)


def test_a_failed_probe_is_stored_in_the_shared_entry_as_unhealthy_with_a_reason(pair: Fleet) -> None:
    """The first instance probes `dec1-status-500-health-shared`, whose upstream always answers 500.

    Within 60 seconds `litellm:health_check:deployment_health_state` must hold a record for that id with
    `is_healthy` false, a positive numeric `timestamp` (epoch seconds) and a non-empty string `reason`.
    """
    record: Final = eventually(
        lambda: _record(pair, "dec1-status-500-health-shared"),
        lambda found: found is not None and found.get("is_healthy") is False,
        seconds=CONVERGENCE_SECONDS,
    )
    assert record is not None
    assert isinstance(record["timestamp"], int | float)
    assert float(record["timestamp"]) > 0  # test-quality-ok: epoch seconds, freshness is what the routing tests observe
    assert isinstance(record["reason"], str) and record["reason"]


def test_a_passed_probe_is_stored_as_healthy(pair: Fleet) -> None:
    """The first instance probes `dec1-ok-health-shared`, whose upstream answers 200.

    Within 60 seconds the shared entry must hold a record for that id with `is_healthy` true.
    """
    record: Final = eventually(
        lambda: _record(pair, "dec1-ok-health-shared"), lambda found: found is not None, seconds=CONVERGENCE_SECONDS
    )
    assert record is not None and record["is_healthy"] is True


def test_the_shared_health_entry_expires_after_one_and_a_half_staleness_thresholds(pair: Fleet) -> None:
    """`health_check_staleness_threshold: 40`. Once the shared entry exists, its TTL must be between 1 and 60
    seconds (`int(40 * 1.5)`, `DeploymentHealthCache.set_deployment_health_states`).
    """
    eventually(lambda: pair.raw(HEALTH_KEY), lambda raw: raw is not None, seconds=CONVERGENCE_SECONDS)
    assert 0 < pair.ttl(HEALTH_KEY) <= HEALTH_TTL_SECONDS


def test_an_instance_without_health_checks_avoids_a_deployment_its_peer_found_unhealthy(pair: Fleet) -> None:
    """Only the first instance runs background health checks. Once the shared entry marks
    `dec1-status-500-health-shared` unhealthy, the second instance must serve eight requests to `health-shared` all
    from `ok-health-shared`, within 60 seconds of polling. It never called the failing deployment's probe; the shared
    entry is its only source.
    """
    eventually(
        lambda: _record(pair, "dec1-status-500-health-shared"),
        lambda found: found is not None and found.get("is_healthy") is False,
        seconds=CONVERGENCE_SECONDS,
    )
    names: Final = eventually(
        lambda: tuple(chat(pair.second, "health-shared").text for _ in range(8)),
        lambda found: all("served ok-health-shared" in text for text in found),
        seconds=CONVERGENCE_SECONDS,
    )
    assert all("served ok-health-shared" in text for text in names)


def test_an_unhealthy_record_written_by_another_pod_is_honoured(pair: Fleet) -> None:
    """The test plays another pod: it merges `{"dec1-ok-health-foreign-target": {"is_healthy": False, "timestamp":
    <now>, "reason": "scripted"}}` into the shared entry as a Python literal.

    Within 30 seconds eight requests to `health-foreign` through the second instance must all be served by
    `ok-health-foreign-sibling`, although the target's own upstream is healthy. This pins the value format a Rust
    reader must accept from a Python writer.
    """
    _write_states(
        pair, {"dec1-ok-health-foreign-target": {"is_healthy": False, "timestamp": time.time(), "reason": "scripted"}}
    )
    names: Final = eventually(
        lambda: tuple(served(chat(pair.second, "health-foreign")) for _ in range(8)),
        lambda found: found == ("ok-health-foreign-sibling",) * 8,
        seconds=30,
    )
    assert names == ("ok-health-foreign-sibling",) * 8


def test_a_record_older_than_the_staleness_threshold_is_ignored(pair: Fleet) -> None:
    """The test merges an unhealthy record for `dec1-ok-health-stale-target` whose `timestamp` is 50 seconds ago,
    older than the 40 second staleness threshold.

    For 10 seconds, requests to `health-stale` through the second instance must keep reaching
    `ok-health-stale-target` (it has `weight: 1`, its sibling none). A stale record means "unknown", which is
    healthy (`_extract_unhealthy_ids`).
    """
    _write_states(
        pair, {"dec1-ok-health-stale-target": {"is_healthy": False, "timestamp": time.time() - 50, "reason": "old"}}
    )
    target: Final = held(
        lambda: served(chat(pair.second, "health-stale")),
        lambda name: name == "ok-health-stale-target",
        holding=10,
    )
    assert target == "ok-health-stale-target"


def test_a_record_with_extra_fields_and_an_integer_timestamp_is_still_read(pair: Fleet) -> None:
    """The test merges `{"is_healthy": False, "timestamp": <int now>, "reason": "x", "probe": "other-stack"}` for
    `dec1-ok-health-lenient-target`.

    Within 30 seconds requests to `health-lenient` through the second instance must all be served by
    `ok-health-lenient-sibling`. The store ignores extra fields and accepts any number as a timestamp
    (`deployment_health/types.rs`, `HealthState::read`).
    """
    _write_states(
        pair,
        {
            "dec1-ok-health-lenient-target": {
                "is_healthy": False,
                "timestamp": int(time.time()),
                "reason": "x",
                "probe": "other-stack",
            }
        },
    )
    names: Final = eventually(
        lambda: tuple(served(chat(pair.second, "health-lenient")) for _ in range(6)),
        lambda found: found == ("ok-health-lenient-sibling",) * 6,
        seconds=30,
    )
    assert names == ("ok-health-lenient-sibling",) * 6


def test_a_health_write_merges_into_records_other_pods_wrote(pair: Fleet) -> None:
    """The test merges a fresh healthy record for an id no instance knows, `dec1-written-by-another-pod`, and then
    waits for the first instance to write its next round of probe results (health check interval 2 seconds).

    After the first instance's record for `dec1-ok-health-shared` has a newer timestamp than the test's write, the
    shared entry must still contain `dec1-written-by-another-pod`. A writer that replaced the entry with only its own
    results would erase the results of pods that probe other model groups.
    """
    written_at: Final = time.time()
    _write_states(pair, {"dec1-written-by-another-pod": {"is_healthy": True, "timestamp": written_at, "reason": ""}})
    eventually(
        lambda: _record(pair, "dec1-ok-health-shared"),
        lambda found: found is not None and float(found["timestamp"]) > written_at,  # type: ignore[arg-type]
        seconds=CONVERGENCE_SECONDS,
    )
    assert "dec1-written-by-another-pod" in _states(pair)


def test_a_health_write_prunes_records_older_than_one_and_a_half_staleness_thresholds(pair: Fleet) -> None:
    """The test merges a record for `dec1-long-gone` whose timestamp is 100 seconds ago, beyond `40 * 1.5 = 60`
    seconds.

    Once the first instance has written its next round of results, the shared entry must no longer contain
    `dec1-long-gone` (`_unexpired` drops it on every merge).
    """
    _write_states(pair, {"dec1-long-gone": {"is_healthy": False, "timestamp": time.time() - 100, "reason": "old"}})
    states: Final = eventually(
        lambda: _states(pair), lambda found: "dec1-long-gone" not in found, seconds=CONVERGENCE_SECONDS
    )
    assert "dec1-long-gone" not in states
