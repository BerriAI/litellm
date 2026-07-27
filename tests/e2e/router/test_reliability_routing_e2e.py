"""Live e2e: each routing strategy picks the deployment that strategy implies.

Every test builds its own model group of real `openai/gpt-5.5` deployments that
differ in exactly one dimension the strategy under test reads - configured price,
tpm ceiling, shuffle weight, or recorded latency - so only one
deployment can win and the pick is a statement about the strategy rather than
about luck. The strategy is selected per request through a
`router_settings_override` on the /chat/completions body, so one long-lived proxy
serves every strategy, and the deployment that actually served a call is read
back from the x-litellm-model-id response header.

Every prompt carries a unique marker: the proxy under test has its response cache
on, and a repeated prompt would be answered from cache before the router ever
routed it.
"""

from __future__ import annotations

import time
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict

from complexity_router_client import ComplexityRouterClient
from e2e_config import unique_marker
from e2e_http import StreamingResponse, unwrap
from lifecycle import ResourceManager
from models import ChatBody, ChatMessage, LiteLLMParamsBody, ModelInfoBody, ModelNewResponse
from proxy_client import ProxyClient
from reliability_support import REAL_KEY, REAL_MODEL

pytestmark = pytest.mark.e2e

StrategyName = Literal[
    "simple-shuffle",
    "usage-based-routing-v2",
    "latency-based-routing",
    "cost-based-routing",
]

PRICEY_PER_TOKEN = 1e-3
CHEAP_PER_TOKEN = 1e-9
EXHAUSTED_TPM = 1
UNREACHABLE_BASE = "http://127.0.0.1:9/v1"
UNMEETABLE_DEADLINE_SECONDS = 0.001

ANSWER_TOKENS = 256
SHUFFLE_CALLS = 4

GROUP_READY_TIMEOUT_SECONDS = 30.0
GROUP_READY_POLL_SECONDS = 0.5

SHORT_PROMPT = "answer in one word: what colour is a clear midday sky?"


class RoutingParams(LiteLLMParamsBody):
    """/model/new litellm_params plus the two per-deployment routing knobs the
    shared body does not model: the tpm ceiling usage-based routing reads, and the
    pick weight simple-shuffle reads."""

    tpm: int | None = None
    weight: int | None = None


class RoutingModelNewBody(BaseModel):
    """POST /model/new carrying RoutingParams.

    The shared ModelNewBody types `litellm_params` as LiteLLMParamsBody, and pydantic
    serializes a field by its declared type, so handing it a RoutingParams would drop
    tpm and weight from the request body without a word - every routing strategy
    would then see two identical deployments."""

    model_config = ConfigDict(protected_namespaces=())
    model_name: str
    litellm_params: RoutingParams
    model_info: ModelInfoBody = ModelInfoBody()


class StrategyOverride(BaseModel):
    """The `router_settings_override` that picks a routing strategy for one call,
    and optionally the deadline that call is held to."""

    routing_strategy: StrategyName
    timeout: float | None = None


class StrategyChatBody(ChatBody):
    """A /chat/completions body routed by one strategy. Composes ChatBody."""

    router_settings_override: StrategyOverride


def _group_size(client: ComplexityRouterClient, group: str) -> int:
    return sum(1 for entry in client.proxy.model_info() if entry.model_name == group)


def _await_group_size(client: ComplexityRouterClient, group: str, size: int) -> None:
    """Block until the proxy serves `size` deployments under `group`.

    /model/new answers before the new deployment is necessarily on the router that
    serves the next call, and a group name shows up on /v1/models as soon as its
    first deployment lands, so routing without this gate can credit a strategy for a
    pick it never had a choice in."""
    deadline = time.monotonic() + GROUP_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _group_size(client, group) >= size:
            return
        time.sleep(GROUP_READY_POLL_SECONDS)
    raise AssertionError(
        f"the proxy never reported {size} deployments under {group!r} within "
        f"{GROUP_READY_TIMEOUT_SECONDS}s of registering them"
    )


def _deploy(
    client: ComplexityRouterClient,
    resources: ResourceManager,
    group: str,
    *,
    tpm: int | None = None,
    weight: int | None = None,
    input_cost_per_token: float | None = None,
    output_cost_per_token: float | None = None,
    api_base: str | None = None,
) -> str:
    """Register one real gpt-5.5 deployment under `group`, wait for the proxy to
    serve it alongside the group's existing deployments, delete it on teardown, and
    return the model_id the router reports as the server of a call."""
    expected = _group_size(client, group) + 1
    model_id = unwrap(
        client.proxy.transport.post(
            "/model/new",
            headers=client.proxy.transport.master,
            json=RoutingModelNewBody(
                model_name=group,
                litellm_params=RoutingParams(
                    model=REAL_MODEL,
                    api_key=REAL_KEY,
                    tpm=tpm,
                    weight=weight,
                    input_cost_per_token=input_cost_per_token,
                    output_cost_per_token=output_cost_per_token,
                    api_base=api_base,
                ),
            ),
            response_type=ModelNewResponse,
        )
    ).model_id
    resources.defer(lambda: client.proxy.delete_model(model_id))
    _await_group_size(client, group, expected)
    return model_id


def _route(
    proxy: ProxyClient,
    key: str,
    group: str,
    strategy: StrategyName,
    *,
    prompt: str = SHORT_PROMPT,
    max_tokens: int = ANSWER_TOKENS,
    timeout: float | None = None,
) -> StreamingResponse:
    """Drive one real completion through `group` under `strategy`, returning the raw
    outcome so the caller can read the deployment header off it."""
    return proxy.transport.send(
        "/chat/completions",
        headers=proxy.transport.bearer(key),
        json=StrategyChatBody(
            model=group,
            messages=[ChatMessage(role="user", content=f"{prompt} [{unique_marker()}]")],
            max_tokens=max_tokens,
            router_settings_override=StrategyOverride(routing_strategy=strategy, timeout=timeout),
        ),
    )


def _served_by(resp: StreamingResponse) -> str:
    """The model_id of the deployment that served a successful call."""
    assert resp.status_code == 200, f"routed call failed with {resp.status_code}: {resp.body[:300]}"
    model_id = resp.headers.get("x-litellm-model-id")
    assert model_id, (
        f"response carries no x-litellm-model-id, so the served deployment is unknowable; "
        f"headers={sorted(resp.headers)}"
    )
    return model_id


class TestReliabilityRoutingStrategies:
    @pytest.mark.covers("reliability.routing.cost_based.picks_lowest_cost")
    def test_cost_based_picks_lowest_cost(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """Two deployments of the same model priced a million-fold apart: cost-based
        routing must spend the request on the cheap one."""
        group = f"routing-cost-{unique_marker()}"
        pricey = _deploy(
            client,
            resources,
            group,
            input_cost_per_token=PRICEY_PER_TOKEN,
            output_cost_per_token=PRICEY_PER_TOKEN,
        )
        cheap = _deploy(
            client,
            resources,
            group,
            input_cost_per_token=CHEAP_PER_TOKEN,
            output_cost_per_token=CHEAP_PER_TOKEN,
        )

        served = _served_by(_route(client.proxy, scoped_key, group, "cost-based-routing"))
        assert served == cheap, (
            f"cost-based routing served {served}, but the cheap deployment is {cheap} "
            f"(the other, {pricey}, costs {PRICEY_PER_TOKEN / CHEAP_PER_TOKEN:.0e}x more per token)"
        )

    @pytest.mark.covers("reliability.routing.usage_based.picks_under_tpm")
    def test_usage_based_picks_deployment_under_tpm(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """One deployment's tpm ceiling is a single token, so no real prompt fits under
        it; usage-based routing must place the request on the deployment that has
        token budget left rather than over-allocating the capped one."""
        group = f"routing-tpm-{unique_marker()}"
        capped = _deploy(client, resources, group, tpm=EXHAUSTED_TPM)
        uncapped = _deploy(client, resources, group)

        served = _served_by(_route(client.proxy, scoped_key, group, "usage-based-routing-v2"))
        assert served == uncapped, (
            f"usage-based routing served {served}, but only {uncapped} had tpm budget for the "
            f"request; {capped} is capped at {EXHAUSTED_TPM} tpm and cannot fit any prompt"
        )

    @pytest.mark.covers("reliability.routing.simple_shuffle.picks_healthy_deployment")
    def test_simple_shuffle_picks_healthy_deployment(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """A zero-weight deployment pointed at an unreachable base sits next to the
        healthy one: simple-shuffle's weighted pick must land every call on the
        healthy deployment, so every call is answered instead of dying on a dead
        connection."""
        group = f"routing-shuffle-{unique_marker()}"
        unreachable = _deploy(client, resources, group, weight=0, api_base=UNREACHABLE_BASE)
        healthy = _deploy(client, resources, group, weight=1)

        served = tuple(
            _served_by(_route(client.proxy, scoped_key, group, "simple-shuffle"))
            for _ in range(SHUFFLE_CALLS)
        )
        assert set(served) == {healthy}, (
            f"simple-shuffle served {served}, but every call had to land on the healthy "
            f"deployment {healthy}; {unreachable} carries weight 0 and an unreachable base"
        )

    @pytest.mark.covers("reliability.routing.latency_based.picks_lowest_latency")
    def test_latency_based_picks_lowest_latency(
        self, client: ComplexityRouterClient, resources: ResourceManager, scoped_key: str
    ) -> None:
        """The first deployment answers a request held to a deadline it cannot meet,
        which is the router's worst latency observation; the second then joins the
        group and answers normally. Both deployments are configured identically and
        stay healthy - the deadline lived on the request, not on the deployment - so
        the only thing separating them is the latency the router measured, and the
        assertion call has to go to the deployment that was quick.

        The simple-shuffle control in the middle is there to rule out the other reason
        a router skips a deployment: it forces the pick onto the timed-out deployment
        by weight and gets an answer, which no cooled-down deployment would give."""
        group = f"routing-latency-{unique_marker()}"
        timed_out = _deploy(client, resources, group, weight=1)

        missed_deadline = _route(
            client.proxy,
            scoped_key,
            group,
            "latency-based-routing",
            timeout=UNMEETABLE_DEADLINE_SECONDS,
        )
        assert missed_deadline.status_code == 408, (
            f"the seeding call was meant to exceed its {UNMEETABLE_DEADLINE_SECONDS}s deadline on "
            f"{timed_out}, got {missed_deadline.status_code}: {missed_deadline.body[:300]}"
        )

        quick = _deploy(client, resources, group, weight=0)
        assert _served_by(_route(client.proxy, scoped_key, group, "latency-based-routing")) == quick, (
            f"a deployment with no latency history is the router's lowest known latency, so this "
            f"call should have gone to {quick}"
        )

        control = _route(client.proxy, scoped_key, group, "simple-shuffle")
        assert _served_by(control) == timed_out, (
            f"the weighted control call should have been served by {timed_out}, proving it is "
            f"still healthy and selectable after its timeout"
        )

        served = _served_by(_route(client.proxy, scoped_key, group, "latency-based-routing"))
        assert served == quick, (
            f"latency-based routing served {served}, but {quick} answered in the time {timed_out} "
            f"blew a deadline in, so it is the lower-latency deployment"
        )
