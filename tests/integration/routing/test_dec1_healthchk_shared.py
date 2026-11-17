"""`use_shared_health_check`: two pods run one set of background probes through Redis.

With `general_settings.use_shared_health_check: true` and a Redis cache, each background cycle goes
through `SharedHealthCheckManager.perform_shared_health_check`
(litellm/proxy/health_check_utils/shared_health_check_manager.py:206): a pod first reads the Redis key
`health_check_results` and uses it if it is younger than `DEFAULT_SHARED_HEALTH_CHECK_TTL` (300 s);
otherwise it takes the lock `health_check_lock` (TTL `DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL`, 60 s), probes,
writes the results with `checked_by: pod_<epoch ms>` and a 300 s TTL, and releases the lock. A pod that
loses the lock polls the cache instead of probing.

The consequence pinned here and flagged in the index: while the shared results are fresh, no pod probes at
all, so `health_check_interval` is effectively raised to 300 seconds and an outage stays invisible to
`/health` for up to five minutes. This file is distinct from `test_dec1_multiinst_health_state.py`, which
covers the per deployment routing state key `litellm:health_check:deployment_health_state`.

One fleet of two proxies on one scratch database and one owned Redis, both with background checks every 2
seconds and sharing on. Redis is moving to the Rust `redis-store` families, which must keep these key names,
the JSON shape and the TTLs, or mixed fleets will each probe on their own.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, eventually, held, object_value, string_value
from tests.integration.routing.dec1_multiinst_support import Fleet, decoded, fleet
from tests.integration.routing.dec1_routing_support import deployment

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

RUN: Final = uuid.uuid4().hex[:6]
RESULTS_KEY: Final = "health_check_results"
LOCK_KEY: Final = "health_check_lock"
CACHE_TTL_SECONDS: Final = 300
LOCK_TTL_SECONDS: Final = 60
CONVERGENCE_SECONDS: Final = 60
OK_GROUP: Final = f"shared-ok-{RUN}"
FLIP_GROUP: Final = f"shared-flip-{RUN}"
OK_MODEL: Final = f"ok-shared-{RUN}"
FLIP_MODEL: Final = f"script-shared-{RUN}"
SHARED: Final[dict[str, JsonValue]] = {
    "background_health_checks": True,
    "health_check_interval": 2,
    "use_shared_health_check": True,
}


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-healthchk-shared"),
        models=lambda api_base: (
            deployment(OK_GROUP, OK_MODEL, api_base),
            deployment(FLIP_GROUP, FLIP_MODEL, api_base),
        ),
        router_settings={"num_retries": 0, "disable_cooldowns": True},
        general_settings=SHARED,
    ) as value:
        yield value


def _shared(raw: bytes | None) -> dict[str, JsonValue]:
    """The stored results; the writer passes an already `safe_dumps`-ed string to the cache, so the value may be
    JSON inside a JSON string. UNVERIFIED which of the two the Redis layer stores."""
    value: Final = decoded(raw)
    return JSON_OBJECT.validate_json(value) if isinstance(value, str) else object_value(value)


def _results(pair: Fleet) -> dict[str, JsonValue]:
    raw: Final = eventually(lambda: pair.raw(RESULTS_KEY), lambda value: value is not None, seconds=CONVERGENCE_SECONDS)
    return _shared(raw)


def _health_ids(pair: Fleet, which: int, kind: str) -> frozenset[str]:
    response: Final = pair.instances[which].request("GET", "/health")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    entries: Final = body.get(f"{kind}_endpoints")
    assert isinstance(entries, list)
    return frozenset(string_value(object_value(entry)["model_id"]) for entry in entries)


def test_a_cycle_stores_its_results_under_health_check_results_with_the_probing_pod(pair: Fleet) -> None:
    """Both pods run background checks with sharing on.

    Within 60 seconds Redis must hold `health_check_results`, a JSON object with exactly the keys
    `healthy_endpoints, unhealthy_endpoints, healthy_count, unhealthy_count, timestamp, checked_by`, where
    `checked_by` starts with `pod_` and `healthy_count` equals the length of `healthy_endpoints`.
    """
    results: Final = _results(pair)
    assert set(results) == {
        "healthy_endpoints",
        "unhealthy_endpoints",
        "healthy_count",
        "unhealthy_count",
        "timestamp",
        "checked_by",
    }
    assert string_value(results["checked_by"]).startswith("pod_")
    healthy: Final = results["healthy_endpoints"]
    assert isinstance(healthy, list) and results["healthy_count"] == len(healthy)


def test_the_shared_results_live_for_the_shared_ttl(pair: Fleet) -> None:
    """Once `health_check_results` exists its Redis TTL must be positive and at most 300 seconds
    (`DEFAULT_SHARED_HEALTH_CHECK_TTL`), and above 200 right after a write.
    """
    _results(pair)
    ttl: Final = pair.ttl(RESULTS_KEY)
    assert 200 < ttl <= CACHE_TTL_SECONDS


def test_the_lock_is_released_after_a_cycle(pair: Fleet) -> None:
    """After the probing pod has written `health_check_results`, `health_check_lock` must be gone from Redis
    within 30 seconds: the holder deletes it in a `finally` instead of letting it run out its 60 second TTL.
    """
    _results(pair)
    assert eventually(lambda: pair.raw(LOCK_KEY), lambda value: value is None, seconds=30) is None


def test_two_pods_probe_a_deployment_at_most_once_each_while_results_are_fresh(pair: Fleet) -> None:
    """Both pods run a cycle every 2 seconds.

    Once the first probe of `ok-shared-<run>` has arrived, the upstream's count for it must stay at most 2
    for another 12 seconds (six cycles on each pod): every later cycle reads the fresh shared
    results instead of probing. CI run 44176682961 saw exactly 2 probes that then held, with the fleet's pods
    started about 40 seconds apart; where the second one comes from (a startup probe of the later pod) is
    UNVERIFIED, so the test allows one per pod and pins that nothing probes per cycle.
    """
    eventually(lambda: pair.upstream.count(OK_MODEL), lambda count: count >= 1, seconds=CONVERGENCE_SECONDS)
    assert held(lambda: pair.upstream.count(OK_MODEL), lambda count: count <= 2, holding=12, seconds=30) <= 2
    assert pair.upstream.count(OK_MODEL) <= 2


def test_the_pod_that_did_not_probe_reports_the_shared_result(pair: Fleet) -> None:
    """After the single shared cycle, the master key calls `GET /health` on each pod.

    Both answers must list deployment `dec1-ok-shared-<run>` among the healthy endpoints, although only one
    pod ever called the provider.
    """
    _results(pair)
    for which in (0, 1):
        eventually(
            lambda which=which: _health_ids(pair, which, "healthy"), lambda ids: f"dec1-{OK_MODEL}" in ids, seconds=30
        )


def test_shared_status_reports_redis_the_ttls_and_the_last_prober(pair: Fleet) -> None:
    """The master key calls `GET /health/shared-status` on the first pod after results exist.

    It must answer 200 with `shared_health_check_enabled: true` and a `status` object whose `redis_available` is
    true, `lock_ttl` 60, `cache_ttl` 300, `cache_available` true and `last_checked_by` starting `pod_`.
    """
    _results(pair)
    body: Final = pair.first.get("/health/shared-status")
    assert body["shared_health_check_enabled"] is True
    status: Final = object_value(body["status"])
    assert status["redis_available"] is True
    assert (status["lock_ttl"], status["cache_ttl"]) == (LOCK_TTL_SECONDS, CACHE_TTL_SECONDS)
    assert status["cache_available"] is True
    assert string_value(status["last_checked_by"]).startswith("pod_")


def test_shared_status_names_a_new_pod_id_on_every_call(pair: Fleet) -> None:
    """Two consecutive `GET /health/shared-status` calls on the same pod.

    Their `status.pod_id` values must differ: the route builds a fresh `SharedHealthCheckManager` per call
    (_health_endpoints.py:1859), so `pod_id` names the request, not the pod, and cannot be compared with
    `last_checked_by`. Pinned so a port that fixes this does it on purpose. UNVERIFIED that two calls never land
    in the same millisecond.
    """
    _results(pair)
    first: Final = object_value(pair.first.get("/health/shared-status")["status"])["pod_id"]
    second: Final = eventually(
        lambda: object_value(pair.first.get("/health/shared-status")["status"])["pod_id"],
        lambda value: value != first,
        seconds=5,
    )
    assert string_value(second).startswith("pod_")


def test_an_outage_after_the_shared_cycle_stays_invisible_while_the_results_are_fresh(pair: Fleet) -> None:
    """`script-shared-<run>` passed its first shared probe; its upstream is then scripted to answer 500 to the
    next twenty requests.

    For the next 10 seconds both pods' `GET /health` must still list `dec1-script-shared-<run>` as healthy and
    the upstream must receive no new probe for it: fresh shared results suppress every probe for up to 300
    seconds regardless of `health_check_interval: 2`. Pinned as current behaviour and listed as a suspected bug.
    """
    eventually(lambda: pair.upstream.count(FLIP_MODEL), lambda count: count >= 1, seconds=CONVERGENCE_SECONDS)
    before: Final = pair.upstream.count(FLIP_MODEL)
    pair.upstream.script(FLIP_MODEL, *([500] * 20))
    try:
        held(
            lambda: (
                f"dec1-{FLIP_MODEL}" in _health_ids(pair, 0, "healthy"),
                f"dec1-{FLIP_MODEL}" in _health_ids(pair, 1, "healthy"),
                pair.upstream.count(FLIP_MODEL),
            ),
            lambda seen: seen == (True, True, before),
            holding=10,
            seconds=30,
        )
    finally:
        pair.upstream.script(FLIP_MODEL)


def test_deleting_the_shared_results_makes_the_next_cycle_probe_again(pair: Fleet) -> None:
    """With fresh shared results in place, the test deletes `health_check_results` from Redis.

    Within 30 seconds the upstream must receive another probe for `ok-shared-<run>` and Redis must hold a new
    `health_check_results` with a later `timestamp`: a cache miss is what triggers a real check.
    """
    old: Final = _results(pair)
    before: Final = pair.upstream.count(OK_MODEL)
    with pair.store() as client:
        client.delete(RESULTS_KEY)
    eventually(lambda: pair.upstream.count(OK_MODEL), lambda count: count > before, seconds=30)
    fresh: Final = eventually(
        lambda: pair.raw(RESULTS_KEY),
        lambda raw: raw is not None and _shared(raw)["timestamp"] != old["timestamp"],
        seconds=30,
    )
    assert float(str(_shared(fresh)["timestamp"])) > float(str(old["timestamp"]))
