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

Latency-based reads a signal each proxy process accumulates itself (a timeout
counts as a 1000s latency) and, like least-busy, reads the shared copy from Redis
only on a process's first look at a group. So its slow deployment carries a 1ms
deadline that times out every call it gets, and the test keeps calling under
latency-based routing until it has seen that timeout and three picks in a row
then land on the fast one: any process meets the slow deployment at most once
before routing around it. The control call's timeout proves the slow deployment
was still routable, so the fast picks were latency's doing, not a cooldown's.

Least-busy reads live traffic, so its group of four equal deployments gets one
long streaming request, opened under least-busy and held unread (its head names
the deployment it landed on), and every short least-busy call sent while it is
in flight must land on one of the other three. The stream itself goes through
least-busy because a proxy process only starts counting in-flight requests once
it has routed a least-busy request, which is what registers the counting
callback, so a stream opened under another strategy would go uncounted in a
process that has never routed one. Three idle deployments rather than one
because a process counts in its own memory, reads the shared count from Redis
only on its first look at a group, and releases a call's count in a success
callback that runs some time after the response leaves it, so a process can
still count the previous call or two against whichever deployment took them;
with three calls and three idle deployments, every process's view keeps some
idle deployment at zero, strictly below the one holding the stream, so no call
can tie with it and lose the tie on insertion order. The group gets no warm-up
call for the same reason: a process that served it before the stream opened
would route on its own stale copy, in which nothing is busy. Draining the stream
to its terminator afterwards proves the deployment holding it was healthy the
whole time.

The per-request strategy comes in through `router_settings_override`, the same
knob a key or team's `router_settings` feeds, so one long-lived proxy configured
for simple-shuffle serves every strategy.
"""

from __future__ import annotations

import pytest
from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import StreamChunk, StreamHead, StreamStep, StreamTruncation
from lifecycle import ResourceManager
from models import LiteLLMParamsBody, ModelInfoBody, ModelNewBody, RouterSettingsOverride, RoutingStrategy
from reliability_support import REAL_KEY, REAL_MODEL, chat_override, model_id_of, open_chat_stream

pytestmark = pytest.mark.e2e

STRATEGY_CALLS = 3
LATENCY_CONVERGENCE_CALLS = 12


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
    timeout: float | None = None,
    input_cost_per_token: float | None = None,
    output_cost_per_token: float | None = None,
) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=REAL_MODEL,
        api_key=REAL_KEY,
        weight=weight,
        tpm=tpm,
        timeout=timeout,
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


def _latency_pick(client: ComplexityRouterClient, key: str, group: str, slow: str, fast: str) -> str:
    resp = chat_override(
        client.proxy,
        key,
        group,
        f"say hi {unique_marker()}",
        override=RouterSettingsOverride(routing_strategy="latency-based-routing", num_retries=0),
    )
    if resp.status_code == 408:
        return slow
    assert resp.status_code == 200, f"latency-based call failed with {resp.status_code}: {resp.body[:300]}"
    assert model_id_of(resp) == fast, (
        f"a 200 came from {model_id_of(resp)!r}, but only {fast} can answer inside its deadline"
    )
    return fast


def _latency_picks(
    client: ComplexityRouterClient, key: str, group: str, slow: str, fast: str, history: tuple[str, ...] = ()
) -> tuple[str, ...]:
    settled = slow in history and history[-STRATEGY_CALLS:] == (fast,) * STRATEGY_CALLS
    if settled or len(history) == LATENCY_CONVERGENCE_CALLS:
        return history
    return _latency_picks(client, key, group, slow, fast, (*history, _latency_pick(client, key, group, slow, fast)))


def _assert_streamed_to_the_end(drained: tuple[StreamStep, ...], busy: str | None) -> None:
    truncations = [step for step in drained if isinstance(step, StreamTruncation)]
    body = b"".join(step.data for step in drained if isinstance(step, StreamChunk))
    assert not truncations and b"[DONE]" in body, (
        f"the long stream on {busy} did not run to its terminator, so that deployment may not have been healthy: "
        f"{truncations or body[-200:]!r}"
    )


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
    def test_latency_based_routes_around_deployment_that_times_out(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-latency-{unique_marker()}"
        slow = _register(client, resources, group, _real(weight=1, timeout=0.001))
        fast = _register(client, resources, group, _real(weight=0))

        picks = _latency_picks(client, scoped_key, group, slow, fast)
        assert slow in picks and picks[-STRATEGY_CALLS:] == (fast,) * STRATEGY_CALLS, (
            f"latency-based routing never both saw {slow} time out and settled on {fast} for {STRATEGY_CALLS} "
            f"calls in a row within {LATENCY_CONVERGENCE_CALLS} calls, it picked {picks}"
        )

        control = chat_override(
            client.proxy,
            scoped_key,
            group,
            f"say hi {unique_marker()}",
            override=RouterSettingsOverride(routing_strategy="simple-shuffle", num_retries=0),
        )
        assert control.status_code == 408, (
            f"the simple-shuffle control should have timed out on the weighted deployment {slow}, got "
            f"{control.status_code}: it was benched, so the fast picks above prove nothing"
        )

    @pytest.mark.covers("reliability.routing.least_busy.picks_lowest_traffic")
    def test_least_busy_avoids_deployment_with_request_in_flight(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        group = f"reliability-leastbusy-{unique_marker()}"
        deployments = frozenset(_register(client, resources, group, _real(weight=1)) for _ in range(STRATEGY_CALLS + 1))

        head = open_chat_stream(
            client.proxy,
            scoped_key,
            group,
            f"Write a 1500 word essay on the history of the telegraph. {unique_marker()}",
            override=RouterSettingsOverride(routing_strategy="least-busy"),
            max_tokens=3000,
        )
        assert isinstance(head, StreamHead), f"opening the long stream failed: {head}"
        busy = head.headers.get("x-litellm-model-id")
        try:
            assert head.status_code == 200, f"the long stream should have opened with a 200, got {head.status_code}"
            assert busy in deployments, f"the long stream landed on {busy!r}, not one of {sorted(deployments)}"
            idle = deployments - {busy}
            picks = [_pick(client, scoped_key, group, "least-busy") for _ in range(STRATEGY_CALLS)]
            assert all(pick in idle for pick in picks), (
                f"least-busy picked {picks}, expected every call on one of {sorted(idle)} while {busy} still has the "
                "long stream in flight"
            )
        finally:
            drained = tuple(head.steps)
        _assert_streamed_to_the_end(drained, busy)
