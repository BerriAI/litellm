"""
Characterization of how a router core's routing strategies pick a deployment: strategy validation,
simple-shuffle weighting, the stateful strategies without traffic, per-request strategy overrides,
runtime strategy changes and custom strategies registered with ``register_plugin``.
"""

import re
from typing import Final

import pydantic
import pytest

import litellm
from litellm.router_api import ChangeSet
from litellm.types.router import CustomRoutingStrategyBase, RouterRateLimitError, RoutingStrategy
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment
from tests.unit.test_router.core.selection_helpers import MESSAGES, MakeCore, picked_ids

SPREAD_PICKS: Final = 60
"""Enough uniform picks between two deployments that missing one is a 2**-59 event."""

INVALID_STRATEGY_MESSAGE: Final = (
    "Invalid routing_strategy: '{name}'. Valid options: ['simple-shuffle', 'lar1', 'least-busy', "
    "'latency-based-routing', 'cost-based-routing', 'usage-based-routing-v2', 'usage-based-routing', "
    "'provider-budget-routing']. Check 'router_settings.routing_strategy' in your config.yaml or the "
    "'routing_strategy' parameter if using the Router SDK directly."
)

NO_DEPLOYMENTS_MESSAGE: Final = (
    "No deployments available for selected model, Try again in 5 seconds. Passed model=g. "
    "pre-call-checks=False, cooldown_list=[]"
)


def _two(**params: object) -> list[dict[str, object]]:  # mutable-ok: a model_list  # kwargs-ok: shared litellm_params
    return [deployment("g", "a", **params), deployment("g", "b", **params)]


# --- strategy validation ---------------------------------------------------------------------


@pytest.mark.parametrize("routing_strategy", ["bogus", "priority", "Simple-Shuffle"])
def test_an_unknown_routing_strategy_is_rejected_at_construction(make_core: MakeCore, routing_strategy: str) -> None:
    with pytest.raises(ValueError, match="Invalid routing_strategy") as raised:
        make_core(model_list=_two(), routing_strategy=routing_strategy)
    assert str(raised.value) == INVALID_STRATEGY_MESSAGE.format(name=routing_strategy)


@pytest.mark.asyncio
async def test_lar1_is_an_accepted_strategy_that_picks_within_the_group(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[*_two(), deployment("other", "c")], routing_strategy="lar1")
    assert await picked_ids(core, "g", messages=MESSAGES) <= {"a", "b"}


@pytest.mark.asyncio
async def test_provider_budget_routing_without_budgets_has_no_selector_and_rate_limits_every_pick(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_two(), routing_strategy="provider-budget-routing")
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g", messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert str(raised.value) == NO_DEPLOYMENTS_MESSAGE
    with pytest.raises(RouterRateLimitError):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [("a", "b"), ("b", "a")])
async def test_a_routing_strategy_enum_member_is_accepted_like_its_string(
    make_core: MakeCore, ids: tuple[str, str]
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", deployment_id) for deployment_id in ids],
        routing_strategy=RoutingStrategy.LEAST_BUSY,
    )
    assert await picked_ids(core, "g", messages=MESSAGES) == {ids[0]}


@pytest.mark.asyncio
async def test_select_with_an_unknown_kind_raises_value_error(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two())
    with pytest.raises(ValueError, match=re.escape("unknown select kind 'nope'")):
        await core.select("g", kind="nope")  # pyright: ignore[reportCallIssue, reportArgumentType]


# --- simple-shuffle --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_simple_shuffle_spreads_picks_over_unweighted_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two())
    assert await picked_ids(core, "g", times=SPREAD_PICKS) == {"a", "b"}


@pytest.mark.asyncio
async def test_simple_shuffle_spreads_picks_when_every_weight_is_zero(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two(weight=0, rpm=0, tpm=0))
    assert await picked_ids(core, "g", times=SPREAD_PICKS) == {"a", "b"}


@pytest.mark.asyncio
async def test_simple_shuffle_falls_through_all_zero_weights_to_rpm(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", weight=0, rpm=5), deployment("g", "b", weight=0, rpm=0)])
    assert await picked_ids(core, "g") == {"a"}


@pytest.mark.asyncio
async def test_simple_shuffle_falls_through_unset_weight_and_rpm_to_tpm(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", tpm=0), deployment("g", "b", tpm=100)])
    assert await picked_ids(core, "g") == {"b"}


@pytest.mark.asyncio
async def test_rpm_takes_precedence_over_tpm(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", rpm=1, tpm=0), deployment("g", "b", rpm=0, tpm=1000)])
    assert await picked_ids(core, "g") == {"a"}


@pytest.mark.asyncio
async def test_simple_shuffle_with_a_non_numeric_weight_raises_type_error_at_pick(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", weight="heavy"), deployment("g", "b")])
    with pytest.raises(TypeError, match="Deployment weight must be numeric"):
        await core.select("g")


@pytest.mark.asyncio
async def test_request_router_weights_are_keyed_by_the_alias_target_group(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two(), model_group_alias={"alias": "g"})
    assert await picked_ids(core, "alias", request_kwargs={"_router_weights": {"g": {"b": 1.0}}}) == {"b"}


@pytest.mark.asyncio
async def test_request_router_weights_for_another_group_are_ignored(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", weight=1), deployment("g", "b", weight=0)])
    assert await picked_ids(core, "g", request_kwargs={"_router_weights": {"other": {"b": 1.0}}}) == {"a"}


@pytest.mark.asyncio
@pytest.mark.parametrize("weights", [{"g": {"a": 0.0, "b": 0.0}}, {"g": {"a": -1.0}}, {"g": {"a": "1"}}])
async def test_invalid_request_router_weights_fail_the_pick_with_a_validation_error(
    make_core: MakeCore, weights: dict[str, object]
) -> None:
    core: Final = make_core(model_list=_two())
    with pytest.raises(pydantic.ValidationError):
        await core.select("g", {"_router_weights": weights})


@pytest.mark.asyncio
async def test_an_empty_request_router_weights_group_falls_back_to_deployment_weights(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", weight=0), deployment("g", "b", weight=2)])
    assert await picked_ids(core, "g", request_kwargs={"_router_weights": {"g": {}}}) == {"b"}


@pytest.mark.asyncio
async def test_request_router_weights_give_unlisted_deployments_zero_weight(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[*_two(), deployment("g", "c")])
    assert await picked_ids(core, "g", request_kwargs={"_router_weights": {"g": {"c": 0.5}}}) == {"c"}


# --- stateful strategies without traffic ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_strategy", ["latency-based-routing", "usage-based-routing-v2"])
async def test_strategy_without_traffic_spreads_picks(make_core: MakeCore, routing_strategy: str) -> None:
    core: Final = make_core(model_list=_two(), routing_strategy=routing_strategy)
    assert await picked_ids(core, "g", times=SPREAD_PICKS, messages=MESSAGES) == {"a", "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [("a", "b"), ("b", "a")])
async def test_cost_based_routing_without_known_costs_picks_the_first_configured(
    make_core: MakeCore, ids: tuple[str, str]
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", deployment_id) for deployment_id in ids], routing_strategy="cost-based-routing"
    )
    assert await picked_ids(core, "g", messages=MESSAGES) == {ids[0]}


def test_sync_pick_under_usage_based_routing_v1_picks_the_first_configured(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two(), routing_strategy="usage-based-routing")
    for _ in range(5):
        assert core.get_available_deployment("g", messages=MESSAGES)["model_info"]["id"] == "a"  # pyright: ignore[reportAttributeAccessIssue]


def test_sync_completion_under_least_busy_calls_the_first_configured(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=_two(), routing_strategy="least-busy")
    for _ in range(3):
        core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("completion") == ["a", "a", "a"]


# --- per-request routing strategy override --------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("override", ["least-busy", RoutingStrategy.LEAST_BUSY], ids=["string", "enum"])
async def test_a_request_routing_strategy_overrides_simple_shuffle(make_core: MakeCore, override: object) -> None:
    core: Final = make_core(model_list=_two())
    assert await picked_ids(core, "g", request_kwargs={"routing_strategy": override}, messages=MESSAGES) == {"a"}


@pytest.mark.asyncio
async def test_a_simple_shuffle_request_override_spreads_picks_on_a_least_busy_router(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two(), routing_strategy="least-busy")
    picked: Final = await picked_ids(
        core, "g", times=SPREAD_PICKS, request_kwargs={"routing_strategy": "simple-shuffle"}, messages=MESSAGES
    )
    assert picked == {"a", "b"}


@pytest.mark.asyncio
async def test_a_cost_based_request_override_picks_the_cheapest(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "pricey", input_cost_per_token=0.00001, output_cost_per_token=0.00002),
            deployment("g", "cheap", input_cost_per_token=0.000001, output_cost_per_token=0.000002),
        ]
    )
    picked: Final = await picked_ids(
        core, "g", request_kwargs={"routing_strategy": "cost-based-routing"}, messages=MESSAGES
    )
    assert picked == {"cheap"}


@pytest.mark.asyncio
@pytest.mark.parametrize("override", ["bogus", "lar1", "provider-budget-routing", 5])
async def test_an_unsupported_request_routing_strategy_is_ignored(make_core: MakeCore, override: object) -> None:
    core: Final = make_core(model_list=_two())
    picked: Final = await picked_ids(
        core, "g", times=SPREAD_PICKS, request_kwargs={"routing_strategy": override}, messages=MESSAGES
    )
    assert picked == {"a", "b"}


@pytest.mark.asyncio
async def test_a_request_routing_strategy_overrides_the_routing_group_strategy(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=_two(), routing_groups=[{"group_name": "grp", "models": ["g"], "routing_strategy": "least-busy"}]
    )
    assert await picked_ids(core, "g", messages=MESSAGES) == {"a"}
    picked: Final = await picked_ids(
        core, "g", times=SPREAD_PICKS, request_kwargs={"routing_strategy": "simple-shuffle"}, messages=MESSAGES
    )
    assert picked == {"a", "b"}


@pytest.mark.asyncio
async def test_completion_with_a_request_routing_strategy_calls_the_overridden_pick(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_two())
    for _ in range(5):
        await core.acompletion(model="g", messages=MESSAGES, routing_strategy="least-busy")  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == ["a"] * 5


# --- runtime strategy changes ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_applying_a_routing_strategy_switches_how_picks_are_made(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two())
    core.apply(ChangeSet(settings={"routing_strategy": "least-busy"}))
    assert core.settings().routing_strategy == "least-busy"
    assert await picked_ids(core, "g", messages=MESSAGES) == {"a"}


@pytest.mark.asyncio
async def test_applying_an_invalid_routing_strategy_raises_and_keeps_the_previous_one(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two(), routing_strategy="least-busy")
    with pytest.raises(ValueError, match="Invalid routing_strategy") as raised:
        core.apply(ChangeSet(settings={"routing_strategy": "bogus"}))
    assert str(raised.value) == INVALID_STRATEGY_MESSAGE.format(name="bogus")
    assert core.settings().routing_strategy == "least-busy"
    assert await picked_ids(core, "g", messages=MESSAGES) == {"a"}


# --- custom routing strategies -----------------------------------------------------------------


def _fixed(deployment_id: str) -> dict[str, object]:  # mutable-ok: the strategy interface's own type
    return {
        "model_name": "x",
        "litellm_params": {"model": f"openai/{deployment_id}"},
        "model_info": {"id": deployment_id},
    }


class FixedStrategy(CustomRoutingStrategyBase):
    """Always answers ``custom`` from the async pick and ``sync-custom`` from the sync pick."""

    def __init__(self) -> None:
        self.asked: list[str] = []  # mutable-ok: the models each pick was asked for

    async def async_get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,  # mutable-ok: the strategy interface's own type
        input: str | list | None = None,  # mutable-ok: the strategy interface's own type
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,  # mutable-ok: the strategy interface's own type
    ) -> dict:  # mutable-ok: the strategy interface's own type
        self.asked.append(model)
        return _fixed("custom")

    def get_available_deployment(
        self,
        model: str,
        messages: list[dict[str, str]] | None = None,  # mutable-ok: the strategy interface's own type
        input: str | list | None = None,  # mutable-ok: the strategy interface's own type
        specific_deployment: bool | None = False,
        request_kwargs: dict | None = None,  # mutable-ok: the strategy interface's own type
    ) -> dict:  # mutable-ok: the strategy interface's own type
        self.asked.append(model)
        return _fixed("sync-custom")


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["g", "unknown-model"])
async def test_a_custom_strategy_answer_is_returned_as_is_for_any_model(make_core: MakeCore, model: str) -> None:
    core: Final = make_core(model_list=_two())
    strategy: Final = FixedStrategy()
    core.register_plugin(strategy)
    chosen: Final = await core.select(model)
    assert chosen == _fixed("custom")
    assert strategy.asked == [model]


@pytest.mark.asyncio
async def test_a_custom_strategy_does_not_change_the_healthy_listing(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two())
    core.register_plugin(FixedStrategy())
    healthy: Final = await core.select("g", kind="healthy")
    assert sorted(d["model_info"]["id"] for d in healthy) == ["a", "b"]  # pyright: ignore[reportIndexIssue, reportArgumentType]


@pytest.mark.asyncio
async def test_completion_calls_the_deployment_a_custom_strategy_returns(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_two())
    core.register_plugin(FixedStrategy())
    await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == ["custom"]
    assert host.calls[-1].model == "openai/custom"


def test_sync_completion_uses_the_custom_strategys_sync_pick(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=_two())
    core.register_plugin(FixedStrategy())
    core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("completion") == ["sync-custom"]
    assert core.get_available_deployment("g") == _fixed("sync-custom")  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_set_custom_routing_strategy_registers_like_register_plugin(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two())
    core.set_custom_routing_strategy(FixedStrategy())  # pyright: ignore[reportAttributeAccessIssue]
    assert await core.select("g") == _fixed("custom")


@pytest.mark.asyncio
async def test_applying_a_different_routing_strategy_drops_the_custom_strategy(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two())
    core.register_plugin(FixedStrategy())
    core.apply(ChangeSet(settings={"routing_strategy": "least-busy"}))
    assert await picked_ids(core, "g", messages=MESSAGES) == {"a"}


@pytest.mark.asyncio
async def test_applying_the_current_routing_strategy_keeps_the_custom_strategy(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_two())
    core.register_plugin(FixedStrategy())
    core.apply(ChangeSet(settings={"routing_strategy": "simple-shuffle"}))
    assert await core.select("g") == _fixed("custom")


@pytest.mark.asyncio
async def test_a_custom_strategy_error_reaches_the_caller(make_core: MakeCore, host: ScriptedHost) -> None:
    class Refusing(FixedStrategy):
        async def async_get_available_deployment(
            self, model: str, *args: object, **kwargs: object
        ) -> dict:  # mutable-ok: the strategy interface's own type  # kwargs-ok: the strategy interface
            raise litellm.BadRequestError(message="refused by strategy", model=model, llm_provider="")

    core: Final = make_core(model_list=_two())
    core.register_plugin(Refusing())
    with pytest.raises(litellm.BadRequestError, match="refused by strategy"):
        await core.acompletion(model="g", messages=MESSAGES, num_retries=0)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls == []
