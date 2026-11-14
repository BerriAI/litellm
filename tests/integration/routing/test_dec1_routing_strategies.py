"""`router_settings.routing_strategy`: what each non-default strategy does with a two-deployment group.

Each test boots its own proxy, because the strategy is a router-wide setting. Retries are off and cooldowns
are disabled, so the strategy alone decides which deployment a request reaches, and the scripted upstream
reports that as the text `served <upstream model>`.

`least-busy` sends a request to the deployment with fewer requests in flight. `latency-based-routing` settles
on the deployment that answers faster. `usage-based-routing-v2` stops picking a deployment that has used up
its `rpm` for the current minute. `cost-based-routing` settles on the deployment with the lower configured
token price. The default strategy, `simple-shuffle`, and its `weight`, `rpm`, `tpm` and `order` inputs are
pinned in `test_dec1_routing_selection.py`.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
from integration.routing.dec1_routing_support import Rig, chat, content_of, deployment, routing_rig
from pydantic import JsonValue

from tests.integration._support.client import eventually, object_value

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

SETTLED_REQUESTS: Final = 8
LEAST_BUSY_ROUNDS: Final = 3


def _settings(strategy: str) -> dict[str, JsonValue]:
    return {"num_retries": 0, "disable_cooldowns": True, "routing_strategy": strategy}


def _served(rig: Rig, group: str) -> str:
    response: Final = chat(rig.gateway, group)
    assert response.status_code == 200, response.text
    return content_of(response).removeprefix("served ")


def _reported_strategy(rig: Rig) -> JsonValue:
    return object_value(rig.gateway.get("/router/settings")["current_values"])["routing_strategy"]


def _least_busy_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("strategy-busy", "held-busy-first", api_base),
        deployment("strategy-busy", "held-busy-second", api_base),
    )


def _two_overlapping_requests(rig: Rig, pool: ThreadPoolExecutor, models: tuple[str, ...]) -> frozenset[str]:
    """The upstream models that received two requests sent one after the other while both stay in flight."""
    for model in models:
        rig.upstream.hold(model)
    before: Final = len(rig.upstream.order(*models))
    try:
        first: Final = pool.submit(chat, rig.gateway, "strategy-busy")
        eventually(lambda: len(rig.upstream.order(*models)), lambda seen: seen == before + 1)
        second: Final = pool.submit(chat, rig.gateway, "strategy-busy")
        eventually(lambda: len(rig.upstream.order(*models)), lambda seen: seen == before + 2)
        reached: Final = frozenset(rig.upstream.order(*models)[before:])
    finally:
        for model in models:
            rig.upstream.release(model)
    assert first.result().status_code == 200
    assert second.result().status_code == 200
    return reached


def test_least_busy_sends_a_second_concurrent_request_to_the_idle_deployment(tmp_path: Path) -> None:
    """`routing_strategy: least-busy`; both deployments hold their answer until the test releases them.

    In each of three rounds the test sends one request, waits until an upstream model has received it, then
    sends a second one. The second request must reach the other deployment, because the first deployment has a
    request in flight. Both requests return 200 once released. Random selection would pass all three rounds
    one time in eight. `/router/settings` must report the strategy as `least-busy`.
    """
    models: Final = ("held-busy-first", "held-busy-second")
    with routing_rig(tmp_path, _least_busy_models, _settings("least-busy")) as rig, ThreadPoolExecutor(2) as pool:
        assert _reported_strategy(rig) == "least-busy"
        rounds: Final = tuple(_two_overlapping_requests(rig, pool, models) for _ in range(LEAST_BUSY_ROUNDS))
        assert rounds == (frozenset(models),) * LEAST_BUSY_ROUNDS


def _latency_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("strategy-latency", "lag-latency", api_base),
        deployment("strategy-latency", "ok-latency", api_base),
    )


def test_latency_based_routing_settles_on_the_faster_deployment(tmp_path: Path) -> None:
    """`routing_strategy: latency-based-routing`; one deployment answers after 0.4 seconds, the other at once.

    After a warm-up of twelve requests that lets both deployments record a latency, the test polls with
    batches of eight sequential requests until a whole batch is served by the fast deployment. That must
    happen within 60 seconds, and the slow deployment must have been sampled at least once along the way.
    """
    with routing_rig(tmp_path, _latency_models, _settings("latency-based-routing")) as rig:
        assert _reported_strategy(rig) == "latency-based-routing"
        for _ in range(12):
            _served(rig, "strategy-latency")
        eventually(
            lambda: tuple(_served(rig, "strategy-latency") for _ in range(SETTLED_REQUESTS)),
            lambda served: set(served) == {"ok-latency"},
            seconds=60,
        )
        assert rig.upstream.count("lag-latency") >= 1


def _usage_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("strategy-usage", "ok-usage-capped", api_base, rpm=1),
        deployment("strategy-usage", "ok-usage-open", api_base, rpm=1000),
    )


def test_usage_based_routing_v2_stops_picking_a_deployment_that_used_its_rpm_this_minute(tmp_path: Path) -> None:
    """`routing_strategy: usage-based-routing-v2`; one deployment has `rpm: 1`, the other `rpm: 1000`.

    Six sequential requests must all return 200. The capped deployment may serve at most two of them (one per
    clock minute the test can touch) and the open deployment serves the rest. UNVERIFIED: that the capped
    deployment is filtered rather than answered with a 429; see `lowest_tpm_rpm_v2.py`.
    """
    with routing_rig(tmp_path, _usage_models, _settings("usage-based-routing-v2")) as rig:
        assert _reported_strategy(rig) == "usage-based-routing-v2"
        served: Final = tuple(_served(rig, "strategy-usage") for _ in range(6))
        assert served.count("ok-usage-capped") <= 2, served
        assert served.count("ok-usage-open") >= 4, served


def _cost_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("strategy-cost", "ok-cost-pricey", api_base, input_cost_per_token=0.01, output_cost_per_token=0.01),
        deployment(
            "strategy-cost", "ok-cost-cheap", api_base, input_cost_per_token=0.000001, output_cost_per_token=0.000001
        ),
    )


def test_cost_based_routing_settles_on_the_cheaper_deployment(tmp_path: Path) -> None:
    """`routing_strategy: cost-based-routing`; the deployments set token prices of 0.01 and 0.000001.

    The test polls with batches of eight sequential requests until a whole batch is served by the cheaper
    deployment, which must happen within 60 seconds. The first request may land anywhere because the strategy
    has no usage record yet.
    """
    with routing_rig(tmp_path, _cost_models, _settings("cost-based-routing")) as rig:
        assert _reported_strategy(rig) == "cost-based-routing"
        eventually(
            lambda: tuple(_served(rig, "strategy-cost") for _ in range(SETTLED_REQUESTS)),
            lambda served: set(served) == {"ok-cost-cheap"},
            seconds=60,
        )
