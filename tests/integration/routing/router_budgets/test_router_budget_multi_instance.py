"""Router budgets across processes: two two-worker proxies sharing one owned Redis, restarts, window resets,
a Redis outage mid burst and a burst that is all in flight before any spend lands.

The burst cells hold every request at the upstream behind a barrier until the whole burst has arrived, so
every request passes the budget filter before the first success is logged; the result does not depend on
scheduling. Probes carry ``PROVIDER_FAILURE`` and are never charged.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.routing.router_budgets import _rig as rig
from pydantic import JsonValue

BURST: Final = 12
NEAR_CAP: Final = 3 * rig.CALL_COST
BARRIERS: Final = MappingProxyType(
    {f"router-budget-burst-{label}": threading.Barrier(BURST) for label in ("accounting", "hard-cap")}
)
MIXED: Final = ("chat", "chat_stream", "messages", "responses")


def _respond(request: Request) -> Reply:
    barrier: Final = next((barrier for marker, barrier in BARRIERS.items() if marker.encode() in request.body), None)
    if barrier is not None:
        barrier.wait(timeout=20)
    return rig.upstream(request)


def _deployments(url: str) -> tuple[dict[str, JsonValue], ...]:
    capped: Final = (
        ("shared-tiny", 0.01, "1d"),
        ("restart-tiny", 0.01, "1d"),
        ("window-tiny", 0.01, "3s"),
        ("near-cap-accounting", NEAR_CAP, "1d"),
        ("near-cap-hard-cap", NEAR_CAP, "1d"),
        ("outage-tiny", 0.01, "1d"),
        ("outage-roomy", 100, "1d"),
    )
    return (
        *(
            rig.deployment(name, f"hosted_vllm/{name}", url, model_id=name, max_budget=cap, budget_duration=window)
            for name, cap, window in capped
        ),
        rig.deployment("roomy", "openai/roomy", url, model_id="roomy"),
    )


@dataclass(frozen=True, slots=True)
class FleetRig:
    first: Gateway
    second: Gateway
    budget: rig.BudgetRig
    upstream: Wire
    config: Path
    tmp_path: Path
    environment: Gateway


def _config(tmp_path: Path, url: str) -> Path:
    return rig.write_config(
        tmp_path / "fleet.yaml",
        _deployments(url),
        provider_budget_config={"openai": {"budget_limit": 100, "time_period": "1d"}},
    )


@pytest.fixture(scope="module")
def fleet(tmp_path_factory: pytest.TempPathFactory) -> Iterator[FleetRig]:
    tmp_path: Final = tmp_path_factory.mktemp("budget-fleet")
    with gateway_from_environment() as gateway, wire_server(_respond) as upstream, rig.redis_for(tmp_path) as cache:
        config: Final = _config(tmp_path, upstream.url)
        with (
            rig.proxy_on(gateway, tmp_path, config, cache) as first,
            rig.proxy_on(gateway, tmp_path, config, cache) as second,
        ):
            yield FleetRig(first, second, rig.BudgetRig(first, cache), upstream, config, tmp_path, gateway)


def _fresh_send(gateway: Gateway, endpoint: str, model: str, text: str) -> httpx.Response:
    with httpx.Client(base_url=str(gateway.client.base_url), trust_env=False, timeout=60) as client:
        return Gateway(client, gateway.key, gateway.upstream_url).request(
            "POST", rig.path_for(endpoint), rig.body_for(endpoint, model, text)
        )


def _burst(targets: tuple[Gateway, ...], model: str, text: Callable[[int], str]) -> tuple[httpx.Response, ...]:
    with ThreadPoolExecutor(max_workers=BURST) as pool:
        futures: Final = tuple(
            pool.submit(_fresh_send, targets[index % len(targets)], MIXED[index % len(MIXED)], model, text(index))
            for index in range(BURST)
        )
        return tuple(future.result() for future in futures)


def test_spend_on_one_proxy_is_enforced_by_its_peer(fleet: FleetRig) -> None:
    first: Final = rig.chat(fleet.first, "shared-tiny", "shared first")
    assert first.status_code == 200, first.text

    blocked: Final = rig.until_rejected(fleet.second, "shared-tiny", rig.probe_text("shared on peer"))

    assert "model_id: shared-tiny" in blocked.text
    assert fleet.budget.settled("deployment_spend:shared-tiny:1d", rig.CALL_COST) == rig.CALL_COST


def test_a_concurrent_burst_across_both_proxies_is_counted_exactly_once(fleet: FleetRig) -> None:
    responses: Final = _burst((fleet.first, fleet.second), "roomy", lambda index: f"roomy burst {index}")

    assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
    assert fleet.budget.settled("provider_spend:openai:1d", BURST * rig.CALL_COST) == BURST * rig.CALL_COST
    for proxy in (fleet.first, fleet.second):
        reported = eventually(
            lambda proxy=proxy: object_value(object_value(proxy.get("/provider/budgets")["providers"])["openai"]),
            lambda entry: entry["spend"] == BURST * rig.CALL_COST,
        )
        assert reported["budget_limit"] == 100.0


def test_a_burst_in_flight_past_the_cap_is_charged_in_full_and_then_blocked(fleet: FleetRig) -> None:
    marker: Final = "router-budget-burst-accounting"
    responses: Final = _burst((fleet.first, fleet.second), "near-cap-accounting", lambda index: f"{marker} {index}")
    admitted: Final = sum(response.status_code == 200 for response in responses)

    assert admitted >= 3, [response.text for response in responses]
    assert (
        fleet.budget.settled("deployment_spend:near-cap-accounting:1d", admitted * rig.CALL_COST)
        == admitted * rig.CALL_COST
    )
    for proxy in (fleet.first, fleet.second):
        rig.until_rejected(proxy, "near-cap-accounting", rig.probe_text("near cap after burst"))


@pytest.mark.xfail(
    strict=True,
    reason="budgets are checked before the call and charged after it, so a concurrent burst overshoots the cap",
)
def test_a_burst_in_flight_never_admits_more_than_the_cap_allows(fleet: FleetRig) -> None:
    marker: Final = "router-budget-burst-hard-cap"
    responses: Final = _burst((fleet.first, fleet.second), "near-cap-hard-cap", lambda index: f"{marker} {index}")

    assert sum(response.status_code == 200 for response in responses) <= 3


def test_an_exhausted_budget_survives_a_proxy_restart(fleet: FleetRig) -> None:
    first: Final = rig.chat(fleet.first, "restart-tiny", "restart first")
    assert first.status_code == 200, first.text
    assert fleet.budget.settled("deployment_spend:restart-tiny:1d", rig.CALL_COST) == rig.CALL_COST

    with rig.proxy_on(fleet.environment, fleet.tmp_path, fleet.config, fleet.budget.redis) as restarted:
        blocked: Final = rig.until_rejected(restarted, "restart-tiny", rig.probe_text("restart"))

    assert "model_id: restart-tiny" in blocked.text


def test_a_spent_budget_admits_traffic_again_after_its_window_resets(fleet: FleetRig) -> None:
    first: Final = rig.chat(fleet.first, "window-tiny", "window first")
    assert first.status_code == 200, first.text
    rig.until_rejected(fleet.first, "window-tiny", rig.probe_text("window blocked"))

    def five_probes() -> tuple[int, ...]:
        return tuple(rig.chat(fleet.first, "window-tiny", rig.probe_text("window reset")).status_code for _ in range(5))

    assert eventually(five_probes, lambda statuses: statuses == (500,) * 5, seconds=30) == (500,) * 5


def test_a_redis_outage_mid_burst_keeps_serving_and_reconciles_spend_exactly_once(
    tmp_path: Path, fleet: FleetRig
) -> None:
    with rig.redis_for(tmp_path) as cache:
        config: Final = _config(tmp_path, fleet.upstream.url)
        with rig.proxy_on(
            fleet.environment, tmp_path, config, cache, extra={"REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "1"}
        ) as proxy:
            budget: Final = rig.BudgetRig(proxy, cache)
            warm: Final = rig.chat(proxy, "outage-roomy", "outage warm")
            assert warm.status_code == 200, warm.text
            assert budget.settled("deployment_spend:outage-roomy:1d", rig.CALL_COST) == rig.CALL_COST
            cache.stop()

            responses: Final = _burst((proxy,), "outage-roomy", lambda index: f"outage burst {index}")
            exhausted_during: Final = rig.chat(proxy, "outage-tiny", "outage exhaust")

            cache.start()

            assert all(response.status_code == 200 for response in responses), [r.text for r in responses]
            assert exhausted_during.status_code == 200, exhausted_during.text
            assert budget.settled("deployment_spend:outage-tiny:1d", rig.CALL_COST, seconds=30) == rig.CALL_COST
            assert budget.settled("deployment_spend:outage-roomy:1d", BURST * rig.CALL_COST, seconds=30) == (
                BURST * rig.CALL_COST
            )
            rig.until_rejected(proxy, "outage-tiny", rig.probe_text("outage after recovery"))


@pytest.mark.xfail(
    strict=True,
    reason="router budget spend lives only in Redis and process memory, so an emptied Redis resets every budget",
)
def test_an_exhausted_budget_survives_a_redis_restart_for_a_new_process(tmp_path: Path, fleet: FleetRig) -> None:
    with rig.redis_for(tmp_path) as cache:
        config: Final = _config(tmp_path, fleet.upstream.url)
        with rig.proxy_on(fleet.environment, tmp_path, config, cache) as before:
            exhausted: Final = rig.chat(before, "restart-tiny", "redis restart exhaust")
            assert exhausted.status_code == 200, exhausted.text
            spent: Final = rig.BudgetRig(before, cache).settled("deployment_spend:restart-tiny:1d", rig.CALL_COST)
            assert spent == rig.CALL_COST
        cache.stop()
        cache.start()
        with rig.proxy_on(fleet.environment, tmp_path, config, cache) as after:
            probe: Final = rig.chat(after, "restart-tiny", rig.probe_text("redis restart"))

    assert rig.is_budget_rejection(probe), probe.text
