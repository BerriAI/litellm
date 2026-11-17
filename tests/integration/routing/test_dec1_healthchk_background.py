"""The background health check loop: cadence, scope, concurrency and the cached `/health` answer.

With `general_settings.background_health_checks: true` the proxy starts `_run_background_health_check`
(litellm/proxy/proxy_server.py:5204) once models are loaded. Each cycle deep copies the model list, drops
deployments with `model_info.disable_background_health_check: true` and, when
`background_health_check_model_groups` is set, every group not listed, probes the rest (at most
`health_check_concurrency` at a time), stores the result in the process wide `health_check_results`, and then
sleeps `health_check_interval` seconds. The first cycle runs right away, not after one interval.

In this mode `GET /health` never probes: it answers from the cached result, narrowed to the targeted
deployment when `model` or `model_id` is given, and still answers 503 for a targeted group with no healthy
endpoint. As the process model moves to Rust, the loop's owner changes; these are the outside effects that
must not.

Two owned proxies: `fast` probes every 2 seconds with a group allowlist and no concurrency cap; `slow` uses
a one hour interval with `health_check_concurrency: 1`, so it runs exactly one cycle during the test.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from itertools import pairwise
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import eventually, held
from tests.integration.routing.dec1_healthchk_support import (
    LAG_SECONDS,
    Rig,
    deployment,
    endpoint_for,
    endpoints,
    health,
    health_rig,
    report,
)

pytestmark: Final = [pytest.mark.timeout(400), pytest.mark.shares_worker_state]

RUN: Final = uuid.uuid4().hex[:6]
FAST_INTERVAL_SECONDS: Final = 2
CONVERGENCE_SECONDS: Final = 60
OK_GROUP: Final = f"bg-ok-{RUN}"
BAD_GROUP: Final = f"bg-bad-{RUN}"
OPTOUT_GROUP: Final = f"bg-optout-{RUN}"
UNLISTED_GROUP: Final = f"bg-unlisted-{RUN}"
LAG_GROUP: Final = f"bg-lag-{RUN}"
OK_MODEL: Final = f"ok-bg-{RUN}"
BAD_MODEL: Final = f"status-500-bg-{RUN}"
OPTOUT_MODEL: Final = f"ok-bg-optout-{RUN}"
UNLISTED_MODEL: Final = f"ok-bg-unlisted-{RUN}"
LAG_MODELS: Final = tuple(f"lag-bg-{index}-{RUN}" for index in range(3))
SLOW_OK_MODEL: Final = f"ok-bg-hourly-{RUN}"
SLOW_LAG_MODELS: Final = tuple(f"lag-bg-hourly-{index}-{RUN}" for index in range(3))


def _fast_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment(OK_GROUP, OK_MODEL, api_base),
        deployment(BAD_GROUP, BAD_MODEL, api_base),
        deployment(OPTOUT_GROUP, OPTOUT_MODEL, api_base, model_info={"disable_background_health_check": True}),
        deployment(UNLISTED_GROUP, UNLISTED_MODEL, api_base),
        *(deployment(LAG_GROUP, model, api_base) for model in LAG_MODELS),
    )


def _slow_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment(OK_GROUP, SLOW_OK_MODEL, api_base),
        *(deployment(LAG_GROUP, model, api_base) for model in SLOW_LAG_MODELS),
    )


@pytest.fixture(scope="module")
def fast(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-bg-fast"),
        _fast_models,
        general_settings={
            "background_health_checks": True,
            "health_check_interval": FAST_INTERVAL_SECONDS,
            "background_health_check_model_groups": [OK_GROUP, BAD_GROUP, OPTOUT_GROUP, LAG_GROUP],
        },
    ) as value:
        yield value


@pytest.fixture(scope="module")
def slow(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with health_rig(
        tmp_path_factory.mktemp("dec1-healthchk-bg-slow"),
        _slow_models,
        general_settings={
            "background_health_checks": True,
            "health_check_interval": 3600,
            "health_check_concurrency": 1,
        },
    ) as value:
        yield value


def test_the_background_loop_keeps_probing_on_its_interval(fast: Rig) -> None:
    """`health_check_interval: 2`; nobody calls `/health` or the model.

    Within 60 seconds the upstream must have received at least three probes for `ok-bg-<run>`, and every gap
    between consecutive probes must be at least 1.5 seconds: the loop repeats, but sleeps a full interval
    between cycles instead of hammering the provider.
    """
    probes: Final = eventually(
        lambda: fast.upstream.probes(OK_MODEL), lambda seen: len(seen) >= 3, seconds=CONVERGENCE_SECONDS
    )
    gaps: Final = [later.at - earlier.at for earlier, later in pairwise(probes)]
    assert min(gaps) >= FAST_INTERVAL_SECONDS * 0.75, gaps


def test_the_background_loop_never_probes_an_opted_out_deployment(fast: Rig) -> None:
    """Deployment `bg-optout` sets `model_info.disable_background_health_check: true`.

    Once the loop has run several cycles (three probes of `ok-bg-<run>`), the opted-out upstream must still have
    received nothing, and must keep receiving nothing for another 6 seconds.
    """
    eventually(lambda: len(fast.upstream.probes(OK_MODEL)), lambda count: count >= 3, seconds=CONVERGENCE_SECONDS)
    assert held(lambda: fast.upstream.count(OPTOUT_MODEL), lambda count: count == 0, holding=6, seconds=20) == 0


def test_the_background_loop_never_probes_a_group_outside_the_allowlist(fast: Rig) -> None:
    """`background_health_check_model_groups` lists every group except `bg-unlisted`.

    After several cycles the upstream of `bg-unlisted` must have received nothing, and must keep receiving
    nothing for another 6 seconds (`filter_deployments_to_model_groups`).
    """
    eventually(lambda: len(fast.upstream.probes(OK_MODEL)), lambda count: count >= 3, seconds=CONVERGENCE_SECONDS)
    assert held(lambda: fast.upstream.count(UNLISTED_MODEL), lambda count: count == 0, holding=6, seconds=20) == 0


def test_the_cached_report_leaves_out_deployments_the_loop_skipped(fast: Rig) -> None:
    """After a few cycles the master key calls `GET /health` with no model.

    The answer must list `bg-ok` healthy and `bg-bad` unhealthy, and must contain no entry for the opted-out
    deployment or the group outside the allowlist: the cached result only has what the loop probed.
    """
    eventually(lambda: len(fast.upstream.probes(OK_MODEL)), lambda count: count >= 2, seconds=CONVERGENCE_SECONDS)
    response: Final = health(fast.gateway)
    assert response.status_code == 200, response.text
    ids: Final = {entry.get("model_id") for kind in ("healthy", "unhealthy") for entry in endpoints(response, kind)}
    assert f"hc-{OK_MODEL}" in ids and f"hc-{BAD_MODEL}" in ids
    assert f"hc-{OPTOUT_MODEL}" not in ids and f"hc-{UNLISTED_MODEL}" not in ids
    assert endpoint_for(response, f"hc-{BAD_MODEL}")[0] == "unhealthy"


def test_a_targeted_check_of_a_failing_group_is_503_even_while_other_groups_are_healthy(fast: Rig) -> None:
    """In background mode the master key asks `GET /health?model=bg-bad-<run>`, whose only deployment fails.

    It must answer 503 with `healthy_count: 0` and `unhealthy_count: 1`: the cache is narrowed to the target
    before the 503 rule is applied (_health_endpoints.py:1705-1740), so healthy `bg-ok` cannot mask it.
    """
    eventually(lambda: len(fast.upstream.probes(BAD_MODEL)), lambda count: count >= 1, seconds=CONVERGENCE_SECONDS)

    def ask() -> tuple[int, JsonValue, JsonValue]:
        response = health(fast.gateway, model=BAD_GROUP)
        body = report(response)
        return response.status_code, body.get("healthy_count"), body.get("unhealthy_count")

    assert eventually(ask, lambda seen: seen[2] == 1, seconds=30) == (503, 0, 1)


def test_a_targeted_check_of_a_skipped_group_is_503_with_nothing_in_it(fast: Rig) -> None:
    """In background mode the master key asks `GET /health?model=bg-unlisted-<run>`, a group the loop never
    probes.

    It must answer 503 with `healthy_count: 0` and `unhealthy_count: 0`, and the call itself must not probe
    the upstream: background mode never runs a live check from `/health`. Read from code; operators should
    know a skipped group reads as down on a targeted query.
    """
    before: Final = fast.upstream.count(UNLISTED_MODEL)
    response: Final = health(fast.gateway, model=UNLISTED_GROUP)
    assert response.status_code == 503, response.text
    assert (report(response)["healthy_count"], report(response)["unhealthy_count"]) == (0, 0)
    assert fast.upstream.count(UNLISTED_MODEL) == before


def test_without_a_concurrency_cap_one_cycle_probes_deployments_in_parallel(fast: Rig) -> None:
    """Group `bg-lag` has three deployments whose upstream takes 0.6 seconds to answer; `fast` sets no
    `health_check_concurrency`.

    The upstream's peak number of requests in flight must reach at least 3: without a cap every probe of a
    cycle is started at once (`_perform_health_check`, unbounded branch).
    """
    assert eventually(lambda: fast.upstream.peak_in_flight, lambda peak: peak >= 3, seconds=CONVERGENCE_SECONDS) >= 3


def test_the_first_cycle_runs_at_startup_without_waiting_an_interval(slow: Rig) -> None:
    """`slow` sets `health_check_interval: 3600`.

    The upstream must receive a probe for `ok-bg-hourly-<run>` within 60 seconds of boot: the loop probes
    first and sleeps after, so a fresh pod knows its deployments' health immediately.
    """
    first: Final = eventually(
        lambda: slow.upstream.probes(SLOW_OK_MODEL), lambda seen: len(seen) >= 1, seconds=CONVERGENCE_SECONDS
    )
    assert first[0].path.endswith("/chat/completions")


def test_a_long_interval_means_exactly_one_probe_per_deployment(slow: Rig) -> None:
    """With `health_check_interval: 3600`, once the first probe of `ok-bg-hourly-<run>` has arrived, the
    count must stay exactly 1 for another 8 seconds.
    """
    eventually(lambda: slow.upstream.count(SLOW_OK_MODEL), lambda count: count >= 1, seconds=CONVERGENCE_SECONDS)
    assert held(lambda: slow.upstream.count(SLOW_OK_MODEL), lambda count: count == 1, holding=8, seconds=20) == 1


def test_background_mode_answers_health_from_the_cache_without_probing(slow: Rig) -> None:
    """After the single startup cycle, the master key calls `GET /health` and `GET /health?model=bg-ok-<run>`.

    Both must answer 200 listing `hc-ok-bg-hourly-<run>` healthy, and the upstream count for that model must
    stay 1: in background mode `/health` reads `health_check_results` and never calls the provider.
    """
    eventually(lambda: slow.upstream.count(SLOW_OK_MODEL), lambda count: count >= 1, seconds=CONVERGENCE_SECONDS)
    eventually(lambda: health(slow.gateway).status_code, lambda status: status == 200, seconds=30)
    for params in ({}, {"model": OK_GROUP}):
        response = health(slow.gateway, **params)
        assert response.status_code == 200, response.text
        assert endpoint_for(response, f"hc-{SLOW_OK_MODEL}")[0] == "healthy"
    assert slow.upstream.count(SLOW_OK_MODEL) == 1


def test_health_check_concurrency_one_probes_one_deployment_at_a_time(slow: Rig) -> None:
    """`slow` sets `health_check_concurrency: 1` and has three deployments whose upstream takes 0.6 seconds.

    Once all three have been probed, the upstream's peak number of requests in flight must be exactly 1, and
    the three probes must be at least 0.5 seconds apart: the bounded dispatcher starts the next probe only
    when one finishes.
    """
    eventually(
        lambda: [slow.upstream.count(model) for model in SLOW_LAG_MODELS],
        lambda counts: all(count >= 1 for count in counts),
        seconds=CONVERGENCE_SECONDS,
    )
    assert slow.upstream.peak_in_flight == 1
    starts: Final = sorted(slow.upstream.calls(model)[0].at for model in SLOW_LAG_MODELS)
    assert all(later - earlier >= LAG_SECONDS * 0.8 for earlier, later in pairwise(starts))
