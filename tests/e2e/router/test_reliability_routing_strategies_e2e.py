"""Live e2e: each routing strategy sends traffic where its own rule says, not
where the shuffle weights point.

Every test registers a two-deployment group on the real gpt-5.5 whose members
differ only in the signal the strategy under test reads: the configured cost, the
tpm headroom, the measured latency, or the in-flight request count. For the
strategies that read a static or accumulated signal, deployment A holds all of
the group's shuffle weight and B none, so the plain weighted shuffle always opens
on A; a strategy that then sends every call to B has demonstrably read its own
signal, and the closing simple-shuffle control call landing on A proves A was
healthy the whole time, so the B picks cannot be explained by a cooldown.

Least-busy reads live traffic, so its pair carries equal weights: one long
streaming request is opened and held (its head names the deployment it landed
on), and every short call sent while it is in flight must land on the other one.

The per-request strategy comes in through `router_settings_override`, the same
knob a key or team's `router_settings` feeds, so one long-lived proxy configured
for simple-shuffle serves every strategy. The proxy builds a strategy's selector
the first time a request asks for it, so the latency and least-busy tests open
with a warm-up call under their strategy before seeding the signal they read.
"""

from __future__ import annotations

import pytest

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import StreamHead
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, ModelInfoBody, ModelNewBody, RouterSettingsOverride, RoutingStrategy
from reliability_support import REAL_KEY, REAL_MODEL, chat_override, model_id_of, open_chat_stream

pytestmark = pytest.mark.e2e

STRATEGY_CALLS = 3
LATENCY_SEED_CALLS = 2


def _register(client: ComplexityRouterClient, resources: ResourceManager, group: str, params: LiteLLMParamsBody) -> str:
    model_id = client.proxy.register_model(
        ModelNewBody(model_name=group, litellm_params=params, model_info=ModelInfoBody())
    )
    resources.defer(lambda: client.proxy.delete_model(model_id))
    return model_id


def _real(
    weight: int,
    *,
    tpm: int | None = None,
    input_cost_per_token: float | None = None,
    output_cost_per_token: float | None = None,
) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=REAL_MODEL,
        api_key=REAL_KEY,
        weight=weight,
        tpm=tpm,
        input_cost_per_token=input_cost_per_token,
        output_cost_per_token=output_cost_per_token,
    )


def _pick(client: ComplexityRouterClient, key: str, group: str, strategy: RoutingStrategy) -> str:
    resp = chat_override(
        client.proxy,
        key,
        group,
        f"say hi {unique_marker()}",
        override=RouterSettingsOverride(routing_strategy=strategy),
    )
    assert resp.status_code == 200, f"{strategy} call failed with {resp.status_code}: {resp.body[:300]}"
    model_id = model_id_of(resp)
    assert model_id is not None, f"{strategy} response is missing the x-litellm-model-id header"
    return model_id


def _assert_every_pick(
    client: ComplexityRouterClient, key: str, group: str, strategy: RoutingStrategy, expected: str, why: str
) -> None:
    picks = [_pick(client, key, group, strategy) for _ in range(STRATEGY_CALLS)]
    assert picks == [expected] * STRATEGY_CALLS, f"{strategy} picked {picks}, expected every call on {expected} ({why})"


def _assert_shuffle_control_lands_on(client: ComplexityRouterClient, key: str, group: str, weighted: str) -> None:
    control = _pick(client, key, group, "simple-shuffle")
    assert control == weighted, (
        f"the simple-shuffle control landed on {control}, not the weighted deployment {weighted}: "
        "the weighted deployment was unhealthy, so the strategy picks above prove nothing"
    )


class TestReliabilityRoutingStrategies:
    @pytest.mark.covers("reliability.routing.simple_shuffle.picks_healthy_deployment")
    def test_simple_shuffle_honors_weights(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-shuffle-{unique_marker()}"
        weighted = _register(client, resources, group, _real(weight=1))
        _ = _register(client, resources, group, _real(weight=0))

        _assert_every_pick(
            client, scoped_key, group, "simple-shuffle", weighted, "it holds all of the group's shuffle weight"
        )

    @pytest.mark.covers("reliability.routing.cost_based.picks_lowest_cost")
    def test_cost_based_picks_cheapest_deployment(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-cost-{unique_marker()}"
        pricey = _register(
            client, resources, group, _real(weight=1, input_cost_per_token=1e-3, output_cost_per_token=1e-3)
        )
        cheap = _register(
            client, resources, group, _real(weight=0, input_cost_per_token=1e-9, output_cost_per_token=1e-9)
        )

        _assert_every_pick(client, scoped_key, group, "cost-based-routing", cheap, "it is priced a million times lower")
        _assert_shuffle_control_lands_on(client, scoped_key, group, pricey)

    @pytest.mark.covers("reliability.routing.usage_based.picks_under_tpm")
    def test_usage_based_picks_deployment_with_tpm_headroom(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-usage-{unique_marker()}"
        capped = _register(client, resources, group, _real(weight=1, tpm=1))
        open_ended = _register(client, resources, group, _real(weight=0))

        _assert_every_pick(
            client, scoped_key, group, "usage-based-routing-v2", open_ended, "the other has a 1 tpm cap no prompt fits"
        )
        _assert_shuffle_control_lands_on(client, scoped_key, group, capped)

    @pytest.mark.covers("reliability.routing.latency_based.picks_lowest_latency")
    def test_latency_based_avoids_deployment_that_timed_out(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-latency-{unique_marker()}"
        slow = _register(client, resources, group, _real(weight=1))
        fast = _register(client, resources, group, _real(weight=0))

        _ = _pick(client, scoped_key, group, "latency-based-routing")
        for _ in range(LATENCY_SEED_CALLS):
            seed = chat_override(
                client.proxy,
                scoped_key,
                group,
                f"say hi {unique_marker()}",
                override=RouterSettingsOverride(routing_strategy="simple-shuffle", timeout=0.001, num_retries=0),
            )
            assert seed.status_code == 408, (
                f"the 1ms deadline should have timed out on the weighted deployment, got {seed.status_code}: "
                f"{seed.body[:300]}"
            )

        _assert_every_pick(
            client, scoped_key, group, "latency-based-routing", fast, "the other was measured timing out"
        )
        _assert_shuffle_control_lands_on(client, scoped_key, group, slow)

    @pytest.mark.covers("reliability.routing.least_busy.picks_lowest_traffic")
    def test_least_busy_avoids_deployment_with_request_in_flight(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-leastbusy-{unique_marker()}"
        deployments = {
            _register(client, resources, group, _real(weight=1)),
            _register(client, resources, group, _real(weight=1)),
        }

        _ = _pick(client, scoped_key, group, "least-busy")
        head = open_chat_stream(
            client.proxy,
            scoped_key,
            group,
            f"Write a 1500 word essay on the history of the telegraph. {unique_marker()}",
            override=RouterSettingsOverride(routing_strategy="least-busy"),
            max_tokens=3000,
        )
        assert isinstance(head, StreamHead), f"opening the long stream failed: {head}"
        try:
            assert head.status_code == 200, f"the long stream should have opened with a 200, got {head.status_code}"
            busy = head.headers.get("x-litellm-model-id")
            assert busy in deployments, f"the long stream landed on {busy!r}, not one of {deployments}"
            idle = (deployments - {busy}).pop()
            _assert_every_pick(
                client, scoped_key, group, "least-busy", idle, f"{busy} still has the long stream in flight"
            )
        finally:
            for _ in head.steps:
                pass
