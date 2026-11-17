"""A key's or team's saved `routing_strategy` overrides the router's strategy for that caller's requests only.

Router settings are static after the router is built (the 2026-10-30 ruling): the proxy never swaps the router's
own strategy per request. Instead `hierarchical_router_settings_updates`
(`litellm/proxy/common_request_processing.py:1070`) puts the key's, else the team's, saved `router_settings` on the
request, `_merge_router_settings_override` (`litellm/proxy/route_llm_request.py:796`) copies the keys in
`_PER_REQUEST_ROUTER_SETTINGS` (among them `routing_strategy`) onto the request body unless the body already has
them, and the core honours a request strategy only when it is one of `OVERRIDABLE` (`strategy.rs:13`):
simple-shuffle, least-busy, usage-based-routing, usage-based-routing-v2, latency-based-routing and
cost-based-routing. Anything else, for example `lar1` or a typo, is ignored and the router's strategy decides.

The rig makes the two strategies disagree: the router runs simple-shuffle, and the group `override-group` gives
`weight: 1` to an expensive Anthropic deployment and no weight to a cheap Bedrock one. Simple-shuffle therefore
sends everything to Anthropic, while cost-based routing sends everything to Bedrock.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration.routing.dec1_routestrat_support import (
    Rig,
    dep,
    reported_strategy,
    served_many,
    strat_rig,
)

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

GROUP: Final = "override-group"
SHUFFLED: Final = {"ov-anthropic": 20}
CHEAPEST: Final = {"ov-bedrock": 20}


def _models(base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        dep(
            "anthropic", GROUP, "ov-anthropic", base, weight=1, input_cost_per_token=0.001, output_cost_per_token=0.001
        ),
        dep("bedrock", GROUP, "ov-bedrock", base, input_cost_per_token=0.000001, output_cost_per_token=0.000001),
    )


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    settings: Final[dict[str, JsonValue]] = {
        "routing_strategy": "simple-shuffle",
        "num_retries": 0,
        "disable_cooldowns": True,
    }
    with strat_rig(tmp_path_factory.mktemp("dec1-routestrat-override"), _models, settings) as value:
        yield value


def test_without_an_override_simple_shuffle_follows_the_weight(rig: Rig) -> None:
    """The master key sends twenty requests to `override-group`.

    All twenty must reach the weighted Anthropic deployment. This is the baseline the override tests depart from.
    """
    assert served_many(rig.gateway, GROUP, 20) == SHUFFLED


def test_a_keys_saved_routing_strategy_applies_to_its_requests(rig: Rig) -> None:
    """A key saves `router_settings: {"routing_strategy": "cost-based-routing"}`.

    Twenty requests with that key must all reach the cheap Bedrock deployment, and twenty with the master key
    right after must still all reach Anthropic: the override is per request.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"routing_strategy": "cost-based-routing"})
        assert served_many(rig.gateway, GROUP, 20, key=key) == CHEAPEST
    assert served_many(rig.gateway, GROUP, 20) == SHUFFLED


def test_a_teams_saved_routing_strategy_applies_to_its_keys(rig: Rig) -> None:
    """A team saves `router_settings: {"routing_strategy": "cost-based-routing"}`; its key saves nothing.

    Twenty requests with the team's key must all reach Bedrock.
    """
    with rig.gateway.scenario() as scenario:
        team: Final = scenario.team(router_settings={"routing_strategy": "cost-based-routing"})
        key: Final = scenario.key(team_id=team)
        assert served_many(rig.gateway, GROUP, 20, key=key) == CHEAPEST


@pytest.mark.parametrize("strategy", ["lar1", "lowest-cost", "priority"])
def test_a_saved_strategy_outside_the_overridable_list_is_ignored(rig: Rig, strategy: str) -> None:
    """A key saves a `routing_strategy` that is not in `OVERRIDABLE` (`lar1`, a misspelling, or `priority`).

    Requests with that key must still succeed and follow the router's simple-shuffle: twenty requests all reach
    Anthropic. A rewrite that rejected or applied unknown names would change the answer or the routing.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"routing_strategy": strategy})
        assert served_many(rig.gateway, GROUP, 20, key=key) == SHUFFLED


def test_a_saved_simple_shuffle_override_keeps_the_weighted_pick(rig: Rig) -> None:
    """A key saves `routing_strategy: simple-shuffle`, the router's own strategy.

    Twenty requests must all reach Anthropic, exactly as without an override.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"routing_strategy": "simple-shuffle"})
        assert served_many(rig.gateway, GROUP, 20, key=key) == SHUFFLED


def test_a_routing_strategy_in_the_request_body_overrides_like_a_saved_one(rig: Rig) -> None:
    """The master key sends `routing_strategy: "cost-based-routing"` in the chat request body.

    Twenty such requests must all reach Bedrock: the body field is read as the per-request strategy, and a saved
    key setting is only copied onto the body when the body does not already carry the field
    (`route_llm_request.py:820`). UNVERIFIED: that no earlier proxy step strips `routing_strategy` from a body.
    """
    assert served_many(rig.gateway, GROUP, 20, routing_strategy="cost-based-routing") == CHEAPEST


def test_a_body_routing_strategy_wins_over_the_keys_saved_one(rig: Rig) -> None:
    """A key saves `routing_strategy: cost-based-routing` and the request body sends `simple-shuffle`.

    Twenty requests must all reach Anthropic, because the body field is kept and the saved one is not copied over
    it. UNVERIFIED for the same reason as the test above.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"routing_strategy": "cost-based-routing"})
        assert served_many(rig.gateway, GROUP, 20, key=key, routing_strategy="simple-shuffle") == SHUFFLED


def test_per_request_overrides_never_change_the_routers_reported_strategy(rig: Rig) -> None:
    """After keys with saved overrides have sent requests, `GET /router/settings` is read.

    `current_values.routing_strategy` must still be `simple-shuffle`: router settings are static after
    construction and an override applies to one request.
    """
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(router_settings={"routing_strategy": "least-busy"})
        served_many(rig.gateway, GROUP, 3, key=key)
    assert reported_strategy(rig.gateway) == "simple-shuffle"
