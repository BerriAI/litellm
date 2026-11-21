"""Health check driven routing: `background_health_checks` with `enable_health_check_routing`.

One owned proxy runs with `general_settings.background_health_checks: true`, `health_check_interval: 2`, and
`router_settings.enable_health_check_routing: true`. Retries are off and cooldowns are disabled, so a failing
deployment can only lose traffic because a background probe marked it unhealthy.

The proxy probes every deployment in the background by calling its upstream. A deployment whose probe fails
is left out of selection while a healthy sibling exists. When every deployment of a group is unhealthy the
group keeps being tried, so the caller sees the provider's error rather than a routing refusal. A request
that names a deployment by its id skips the health filter.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    NO_DEPLOYMENTS,
    PROVIDER_KEY,
    Rig,
    chat,
    content_of,
    deployment,
    routing_rig,
)
from pydantic import JsonValue

from tests.integration._support.client import eventually

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

HEALTH_CHECK_INTERVAL_SECONDS: Final = 2
CONVERGENCE_SECONDS: Final = 90
ROUTER_SETTINGS: Final[dict[str, JsonValue]] = {
    "num_retries": 0,
    "disable_cooldowns": True,
    "enable_health_check_routing": True,
}
GENERAL_SETTINGS: Final[dict[str, JsonValue]] = {
    "background_health_checks": True,
    "health_check_interval": HEALTH_CHECK_INTERVAL_SECONDS,
}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("health-probed", "ok-health-probed", api_base),
        deployment("health-mixed", "status-500-health-mixed", api_base, weight=1),
        deployment("health-mixed", "ok-health-mixed", api_base),
        deployment("health-all-down", "status-500-health-lone", api_base),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with routing_rig(
        tmp_path_factory.mktemp("dec1-routing-health"),
        _models,
        ROUTER_SETTINGS,
        general_settings=GENERAL_SETTINGS,
    ) as value:
        yield value


def test_background_health_checks_call_each_deployments_upstream_without_any_client_request(rig: Rig) -> None:
    """Background health checks run every 2 seconds and no client ever calls group `health-probed`.

    Within 90 seconds the upstream must have received at least two requests for that deployment's model
    `ok-health-probed`, each carrying the deployment's own provider key as a bearer token: the probes
    themselves.
    """
    probes: Final = eventually(
        lambda: rig.upstream.calls("ok-health-probed"), lambda calls: len(calls) >= 2, seconds=CONVERGENCE_SECONDS
    )
    assert {probe.model for probe in probes} == {"ok-health-probed"}
    assert all(probe.headers.get("authorization") == f"Bearer {PROVIDER_KEY}" for probe in probes)


def test_a_deployment_whose_health_probe_fails_stops_receiving_client_traffic(rig: Rig) -> None:
    """Group `health-mixed` has a deployment with `weight: 1` whose upstream always answers 500 and a healthy
    sibling without a weight.

    By weight alone every request would reach the failing deployment and return 500. Once a probe has marked
    it unhealthy, a batch of six sequential requests must all return 200 with the sibling's content. That
    must happen within 90 seconds.
    """

    def batch() -> tuple[tuple[int, str], ...]:
        responses: Final = tuple(chat(rig.gateway, "health-mixed") for _ in range(6))
        return tuple(
            (response.status_code, content_of(response) if response.status_code == 200 else "")
            for response in responses
        )

    settled: Final = eventually(
        batch, lambda seen: set(seen) == {(200, "served ok-health-mixed")}, seconds=CONVERGENCE_SECONDS
    )
    assert len(settled) == 6


def test_a_group_whose_every_deployment_is_unhealthy_is_still_tried(rig: Rig) -> None:
    """Group `health-all-down` has one deployment and its upstream always answers 500, to probes too.

    After at least two probes have failed, a client request must still reach the upstream and return the
    provider's HTTP 500, not a 429 `No deployments available`. Filtering every deployment out would turn a
    flaky health probe into a full outage.
    """
    eventually(
        lambda: rig.upstream.count("status-500-health-lone"), lambda count: count >= 2, seconds=CONVERGENCE_SECONDS
    )
    before: Final = rig.upstream.count("status-500-health-lone")
    response: Final = chat(rig.gateway, "health-all-down", messages=[{"role": "user", "content": "client call"}])
    assert response.status_code == 500, response.text
    assert NO_DEPLOYMENTS not in response.text
    client_calls: Final = tuple(
        seen
        for seen in rig.upstream.calls("status-500-health-lone")[before:]
        if seen.body.get("messages") == [{"role": "user", "content": "client call"}]
    )
    assert len(client_calls) == 1


def test_a_call_by_deployment_id_reaches_a_deployment_the_health_check_marked_unhealthy(rig: Rig) -> None:
    """The weighted deployment of `health-mixed`, id `dec1-status-500-health-mixed`, fails its probes.

    Once group traffic has moved to the healthy sibling, a request whose `model` is the unhealthy deployment's
    id must still be sent to it and return its HTTP 500. Operators use this to test a deployment directly.
    """
    eventually(
        lambda: chat(rig.gateway, "health-mixed").status_code, lambda status: status == 200, seconds=CONVERGENCE_SECONDS
    )
    response: Final = chat(
        rig.gateway, "dec1-status-500-health-mixed", messages=[{"role": "user", "content": "by id call"}]
    )
    assert response.status_code == 500, response.text
    by_id_calls: Final = tuple(
        seen
        for seen in rig.upstream.calls("status-500-health-mixed")
        if seen.body.get("messages") == [{"role": "user", "content": "by id call"}]
    )
    assert len(by_id_calls) == 1
