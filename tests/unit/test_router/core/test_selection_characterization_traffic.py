"""
Characterization of the stateful routing strategies once traffic has gone through the core:
least-busy follows in-flight requests, latency-based follows measured latency, and usage-based
routing follows the tokens and requests each deployment served. The host here is a
``TrafficHost``, which feeds the strategy loggers the way litellm's logging does around a call.
"""

import asyncio
import re
from typing import Final

import pytest

import litellm
from tests.unit.test_router.core.scripted_host import bad_request, deployment
from tests.unit.test_router.core.selection_helpers import MESSAGES, MakeCore, TrafficHost, picked_ids


@pytest.fixture
def host() -> TrafficHost:
    return TrafficHost()


async def _until_called(host: TrafficHost, count: int) -> None:
    for _ in range(200):
        if len(host.started) >= count:
            return
        await asyncio.sleep(0.005)
    raise AssertionError(f"expected {count} started calls, saw {len(host.started)}")


# --- least-busy -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_least_busy_avoids_a_deployment_with_a_request_in_flight(make_core: MakeCore, host: TrafficHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], routing_strategy="least-busy")
    host.gates["openai/a"] = asyncio.Event()
    in_flight: Final = asyncio.create_task(core.acompletion(model="g", messages=MESSAGES))  # pyright: ignore[reportAttributeAccessIssue]
    await _until_called(host, 1)
    assert host.started == ["openai/a"]
    assert await picked_ids(core, "g", times=5, messages=MESSAGES) == {"b"}
    host.gates["openai/a"].set()
    await in_flight
    assert await picked_ids(core, "g", times=5, messages=MESSAGES) == {"a"}


@pytest.mark.asyncio
async def test_least_busy_releases_a_deployment_whose_request_failed(make_core: MakeCore, host: TrafficHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], routing_strategy="least-busy")
    host.gates["openai/a"] = asyncio.Event()
    host.answer("openai/a", bad_request())
    in_flight: Final = asyncio.create_task(core.acompletion(model="g", messages=MESSAGES, num_retries=0))  # pyright: ignore[reportAttributeAccessIssue]
    await _until_called(host, 1)
    assert await picked_ids(core, "g", times=5, messages=MESSAGES) == {"b"}
    host.gates["openai/a"].set()
    with pytest.raises(litellm.BadRequestError):
        await in_flight
    assert await picked_ids(core, "g", times=5, messages=MESSAGES) == {"a"}


@pytest.mark.asyncio
async def test_least_busy_spreads_concurrent_requests_over_idle_deployments(
    make_core: MakeCore, host: TrafficHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b"), deployment("g", "c")], routing_strategy="least-busy"
    )
    gate: Final = asyncio.Event()
    for model in ("openai/a", "openai/b", "openai/c"):
        host.gates[model] = gate
    calls: Final = []
    for count in range(1, 4):
        calls.append(asyncio.create_task(core.acompletion(model="g", messages=MESSAGES)))  # pyright: ignore[reportAttributeAccessIssue]
        await _until_called(host, count)
    assert host.started == ["openai/a", "openai/b", "openai/c"]
    gate.set()
    await asyncio.gather(*calls)


# --- latency-based ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_latency_based_routing_moves_to_the_faster_deployment(make_core: MakeCore, host: TrafficHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "slow"), deployment("g", "fast")], routing_strategy="latency-based-routing"
    )
    host.delays["openai/slow"] = 0.15
    for _ in range(6):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert "fast" in host.deployments_called()
    assert await picked_ids(core, "g", times=10, messages=MESSAGES) == {"fast"}


# --- usage-based ---------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_strategy", ["usage-based-routing", "usage-based-routing-v2"])
async def test_usage_based_routing_alternates_between_deployments_by_tokens_served(
    make_core: MakeCore, host: TrafficHost, routing_strategy: str
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], routing_strategy=routing_strategy)
    for _ in range(4):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    called: Final = host.deployments_called()
    assert sorted(called[:2]) == ["a", "b"]
    assert sorted(called[2:]) == ["a", "b"]


@pytest.mark.asyncio
async def test_usage_based_v2_stops_picking_a_deployment_once_its_tpm_is_used(
    make_core: MakeCore, host: TrafficHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a", tpm=15), deployment("g", "b", tpm=1000)],
        routing_strategy="usage-based-routing-v2",
    )
    for _ in range(4):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    called: Final = host.deployments_called()
    assert called.count("a") <= 1
    assert called[-2:] == ["b", "b"]


@pytest.mark.asyncio
async def test_usage_based_v2_rate_limits_once_every_deployment_used_its_rpm(
    make_core: MakeCore, host: TrafficHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a", rpm=1), deployment("g", "b", rpm=1)],
        routing_strategy="usage-based-routing-v2",
    )
    await core.acompletion(model="g", messages=MESSAGES, num_retries=0)  # pyright: ignore[reportAttributeAccessIssue]
    await core.acompletion(model="g", messages=MESSAGES, num_retries=0)  # pyright: ignore[reportAttributeAccessIssue]
    assert sorted(host.deployments_called()) == ["a", "b"]
    with pytest.raises(litellm.RateLimitError) as raised:
        await core.acompletion(model="g", messages=MESSAGES, num_retries=0)  # pyright: ignore[reportAttributeAccessIssue]
    assert re.search(
        r"No deployments available for selected model\. Passed model=g\. Deployments=\{'[ab]': \{'current_tpm': 10, "
        r"'tpm_limit': inf, 'current_rpm': 1, 'rpm_limit': 1\}",
        str(raised.value),
    )
    assert len(host.calls) == 2
