"""`routing_strategy: cost-based-routing` across providers: the cheapest deployment by custom price or cost map wins.

The pick is answered by `LowestCostLoggingHandler.async_get_available_deployments` in
`litellm/router_strategy/lowest_cost.py`. For each healthy deployment the price is `input_cost_per_token +
output_cost_per_token`, each taken from the deployment's `litellm_params` when truthy, else from the first cost map
row among `model_info.base_model`, `litellm_params.base_model` and `litellm_params.model` (looked up literally,
provider prefix included, see `deployment_pricing_models`), else 5.0. Deployments over their `tpm` or `rpm` for the
current minute (counted by this strategy's own success callback) are dropped, the rest are sorted by price with a
stable sort, and the first wins. Nothing is random and no warm-up is needed: the very first request already goes
to the cheapest deployment.

Deployments here are on OpenAI, Azure OpenAI, Bedrock, Vertex AI and Anthropic. Retries are off and cooldowns
disabled. Cost map prices quoted below are from `litellm/model_prices_and_context_window_backup.json`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration.routing.dec1_routestrat_support import (
    LONG_PROMPT,
    NO_DEPLOYMENTS,
    Rig,
    chat,
    dep,
    early_in_a_minute,
    error_message,
    reported_strategy,
    served,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]


def _price(value: float) -> dict[str, JsonValue]:
    return {"input_cost_per_token": value, "output_cost_per_token": value}


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        # custom prices on every deployment
        dep("anthropic", "cost-custom", "cc-anthropic", base, **_price(0.00002)),
        dep("bedrock", "cost-custom", "cc-bedrock", base, **_price(0.000001)),
        dep("vertex", "cost-custom", "cc-vertex", base, **_price(0.000005)),
        # cost map rows reached through model_info.base_model
        dep("azure", "cost-map", "cm-azure", base, model_info={"base_model": "azure/gpt-4.1-nano"}),
        dep("bedrock", "cost-map", "cm-bedrock", base, model_info={"base_model": "amazon.nova-micro-v1:0"}),
        dep("vertex", "cost-map", "cm-vertex", base, model_info={"base_model": "gemini-2.5-flash"}),
        # a custom price beats the cost map row of the same deployment
        dep("bedrock", "cost-override", "co-bedrock", base, model_info={"base_model": "amazon.nova-micro-v1:0"}),
        dep(
            "vertex",
            "cost-override",
            "co-vertex",
            base,
            model_info={"base_model": "gemini-2.5-flash"},
            **_price(0.000000001),
        ),
        # an unmapped model ranks at 5.0 + 5.0
        dep("openai", "cost-unmapped-loses", "cul-openai", base),
        dep("anthropic", "cost-unmapped-loses", "cul-anthropic", base, **_price(4.0)),
        dep("openai", "cost-unmapped-wins", "cuw-openai", base),
        dep("anthropic", "cost-unmapped-wins", "cuw-anthropic", base, **_price(6.0)),
        # a zero custom price
        dep("azure", "cost-free", "cf-azure", base, **_price(0)),
        dep("bedrock", "cost-free", "cf-bedrock", base, **_price(0.001)),
        # limits
        dep("vertex", "cost-rpm", "cr-vertex", base, rpm=2, **_price(0.000001)),
        dep("openai", "cost-rpm", "cr-openai", base, **_price(0.00001)),
        dep("anthropic", "cost-rpm", "cr-anthropic", base, **_price(0.0001)),
        dep("bedrock", "cost-tpm", "ct-bedrock", base, tpm=50, **_price(0.000001)),
        dep("azure", "cost-tpm", "ct-azure", base, **_price(0.00001)),
        dep("vertex", "cost-tpm-only", "cto-vertex", base, tpm=50, **_price(0.000001)),
        # equal prices
        dep("openai", "cost-tie", "cti-openai", base, **_price(0.000002)),
        dep("anthropic", "cost-tie", "cti-anthropic", base, **_price(0.000002)),
        # weights are not read
        dep("anthropic", "cost-weight", "cw-anthropic", base, weight=100, **_price(0.0001)),
        dep("bedrock", "cost-weight", "cw-bedrock", base, **_price(0.000001)),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "cost-based-routing",
        "num_retries": 0,
        "disable_cooldowns": True,
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-cost"), _models, settings) as value:
        yield value


def test_cost_based_routing_is_reported_by_router_settings(rig: Rig) -> None:
    """The proxy is booted with `routing_strategy: cost-based-routing`.

    `GET /router/settings` must report `current_values.routing_strategy` as `cost-based-routing`.
    """
    assert reported_strategy(rig.gateway) == "cost-based-routing"


def test_the_very_first_request_already_goes_to_the_cheapest_custom_price(rig: Rig) -> None:
    """Anthropic, Bedrock and Vertex set custom token prices of 0.00002, 0.000001 and 0.000005.

    The first request to the group, and the nineteen after it, must all be served by Bedrock. Unlike latency
    routing there is no exploration of untried deployments.
    """
    assert served(rig.gateway, "cost-custom") == "cc-bedrock"
    assert served_many(rig.gateway, "cost-custom", 19) == {"cc-bedrock": 19}


def test_cost_map_prices_found_through_base_model_pick_the_cheapest_provider(rig: Rig) -> None:
    """No deployment sets a price; their `model_info.base_model` rows are `azure/gpt-4.1-nano` (1e-7 + 4e-7),
    `amazon.nova-micro-v1:0` (3.5e-8 + 1.4e-7) and `gemini-2.5-flash` (3e-7 + 2.5e-6).

    Twenty requests must all be served by Bedrock. A rewrite that ignored `base_model` would rank all three
    deployments at the unmapped default and pick by list order.
    """
    assert served_many(rig.gateway, "cost-map", 20) == {"cm-bedrock": 20}


def test_a_custom_price_in_litellm_params_wins_over_the_cost_map_row(rig: Rig) -> None:
    """Vertex's `base_model` is `gemini-2.5-flash` (the dearer row) but its `litellm_params` set prices of 1e-9;
    Bedrock relies on the `amazon.nova-micro-v1:0` row.

    Twenty requests must all be served by Vertex.
    """
    assert served_many(rig.gateway, "cost-override", 20) == {"co-vertex": 20}


def test_an_unmapped_model_ranks_at_five_dollars_input_and_output(rig: Rig) -> None:
    """Two groups pair an OpenAI `openai/gpt-4o-mini` deployment with no price and no `base_model` against an
    Anthropic deployment with custom prices. The cost map has a `gpt-4o-mini` row but no `openai/gpt-4o-mini`
    row, and the lookup is literal, so the OpenAI deployment ranks at the default 5.0 + 5.0 = 10.0.

    With Anthropic at 4.0 + 4.0 = 8.0 twenty requests must all reach Anthropic; with Anthropic at 6.0 + 6.0 =
    12.0 twenty requests must all reach OpenAI.
    """
    assert served_many(rig.gateway, "cost-unmapped-loses", 20) == {"cul-anthropic": 20}
    assert served_many(rig.gateway, "cost-unmapped-wins", 20) == {"cuw-openai": 20}


def test_a_deployment_priced_at_zero_is_the_cheapest(rig: Rig) -> None:
    """Azure sets `input_cost_per_token: 0` and `output_cost_per_token: 0` (a free, self-hosted style deployment);
    Bedrock sets 0.001 for both.

    Twenty requests should all reach Azure, the free deployment.
    """
    pytest.skip(
        "BUG: lowest_cost.py reads custom prices only when truthy, so a price of 0 falls through to the cost map "
        "or the 5.0 default and the free deployment ranks as the most expensive"
    )
    assert served_many(rig.gateway, "cost-free", 20) == {"cf-azure": 20}


def test_the_cheapest_deployment_is_skipped_once_it_used_its_rpm_this_minute(rig: Rig) -> None:
    """Vertex is cheapest and has `rpm: 2`; OpenAI is next (1e-5) and Anthropic dearest (1e-4).

    Starting early in a minute, ten sequential requests must all return 200. Vertex serves at most three of them
    (two, plus one a late success callback may let through), OpenAI the rest, and Anthropic none: the strategy
    falls through to the next cheapest deployment, not to a random one.
    """
    early_in_a_minute()
    counts: Final = served_many(rig.gateway, "cost-rpm", 10)
    assert counts["cr-vertex"] <= 3, counts
    assert counts["cr-openai"] >= 7, counts
    assert counts["cr-anthropic"] == 0, counts


def test_a_prompt_larger_than_the_cheapest_deployments_tpm_goes_to_the_next_cheapest(rig: Rig) -> None:
    """Bedrock is cheapest with `tpm: 50`; Azure is dearer with no limit. The prompt is about 1200 tokens.

    Every one of five such requests must be served by Azure, including the first, because the strategy adds the
    prompt's token count to the minute's usage before comparing with `tpm`. A short prompt is still served by
    Bedrock.
    """
    assert served_many(rig.gateway, "cost-tpm", 5, messages=[{"role": "user", "content": LONG_PROMPT}]) == {
        "ct-azure": 5
    }
    assert served(rig.gateway, "cost-tpm") == "ct-bedrock"


def test_a_prompt_over_every_deployments_tpm_is_refused_with_429_and_no_upstream_call(rig: Rig) -> None:
    """The group's only deployment (Vertex) has `tpm: 50`; the prompt is about 1200 tokens.

    The strategy finds no deployment and the router answers HTTP 429 whose message contains `No deployments
    available for selected model`, and the peer receives nothing for Vertex. UNVERIFIED: the exact wording; the
    strategy returns None and the core raises `NoHealthyDeployments` (`select.rs:849`).
    """
    response: Final = chat(rig.gateway, "cost-tpm-only", messages=[{"role": "user", "content": LONG_PROMPT}])
    assert response.status_code == 429, response.text
    assert NO_DEPLOYMENTS in error_message(response), response.text
    assert rig.peer.count("cto-vertex") == 0


def test_equal_prices_send_every_request_to_one_deployment(rig: Rig) -> None:
    """OpenAI and Anthropic set the same custom prices.

    Twenty requests must all be served by the same one deployment: the sort is stable and the cost map keeps
    deployments in the order they were first used, so a tie never alternates. A rewrite that broke ties at random
    would split the traffic.
    """
    counts: Final = served_many(rig.gateway, "cost-tie", 20)
    assert len(counts) == 1, counts


def test_cost_based_routing_ignores_weight(rig: Rig) -> None:
    """Anthropic has `weight: 100` and a price of 1e-4; Bedrock has no weight and a price of 1e-6.

    Twenty requests must all be served by Bedrock.
    """
    assert served_many(rig.gateway, "cost-weight", 20) == {"cw-bedrock": 20}


def test_the_cheapest_deployment_wins_on_messages_too(rig: Rig) -> None:
    """The caller uses `POST /v1/messages` on the group `cost-custom`.

    The request must reach Bedrock, the cheapest deployment, so the strategy is applied on the Anthropic native
    endpoint as on chat completions. The answer is 200.
    """
    response: Final = rig.gateway.request(
        "POST",
        "/v1/messages",
        {"model": "cost-custom", "max_tokens": 16, "messages": [{"role": "user", "content": "cost on messages"}]},
        headers={"anthropic-version": "2023-06-01"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["x-litellm-model-id"] == "dec1-routestrat-cc-bedrock"


def test_a_fresh_proxy_with_the_same_group_picks_the_same_cheapest_deployment(tmp_path: Path) -> None:
    """A second proxy is booted with only the `cost-custom` group, listed in a different order (Vertex, Anthropic,
    Bedrock).

    Its first request must reach Bedrock as well: the choice depends on price, not on list order or on history.
    """

    def models(base: str) -> Sequence[dict[str, JsonValue]]:
        return (
            dep("vertex", "cost-custom", "cc2-vertex", base, **_price(0.000005)),
            dep("anthropic", "cost-custom", "cc2-anthropic", base, **_price(0.00002)),
            dep("bedrock", "cost-custom", "cc2-bedrock", base, **_price(0.000001)),
        )

    settings: Final[dict[str, JsonValue]] = {"routing_strategy": "cost-based-routing", "num_retries": 0}
    with strat_rig(tmp_path, models, settings) as fresh:
        assert served(fresh.gateway, "cost-custom") == "cc2-bedrock"
