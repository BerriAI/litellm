"""
Gap tests for the first part of ``litellm/router_python_core.py`` (module helpers, the entry points,
``__init__``, callback registration, strategy setup, routing groups and the strategy dispatch):
behaviors no other characterization test pinned down.

Each test asserts what the Python core does today; oddities are named in the test's docstring.
"""

import datetime
import re
from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.router_api import ChangeSet
from litellm.types.router import Deployment, RouterRateLimitError
from litellm.utils import reapply_runtime_model_cost_registrations
from tests.unit.test_router.core.construction_precall_helpers import collect_other_routers, host_callback_names
from tests.unit.test_router.core.registry_helpers import unique_id
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment
from tests.unit.test_router.core.selection_helpers import MESSAGES, MakeCore, picked_ids

NO_DEPLOYMENTS_MESSAGE: Final = (
    "No deployments available for selected model, Try again in 5 seconds. Passed model={model}. "
    "pre-call-checks=False, cooldown_list=[]"
)


def _group(
    name: str, models: list[str], strategy: str, **extra: object
) -> dict[str, object]:  # mutable-ok: a routing_groups entry  # kwargs-ok: extra RoutingGroup fields
    return {"group_name": name, "models": models, "routing_strategy": strategy, **extra}


def _logging_obj() -> LiteLLMLogging:
    return LiteLLMLogging(
        model="g",
        messages=MESSAGES,
        stream=False,
        call_type="acompletion",
        start_time=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        litellm_call_id="c1",
        function_id="f1",
    )


def _dynamic_callback_names(logging_obj: LiteLLMLogging) -> list[str]:  # mutable-ok: a fresh list
    return [type(callback).__name__ for callback in logging_obj.dynamic_async_success_callbacks or ()]


# --- construction: routing_strategy=None and legacy attributes -------------------------------------


@pytest.mark.asyncio
async def test_routing_strategy_none_is_accepted_but_leaves_every_pick_without_a_deployment(
    make_core: MakeCore,
) -> None:
    """Oddity: ``routing_strategy=None`` passes validation and is stored as None (not the
    ``simple-shuffle`` default), so no strategy arm matches and every pick rate-limits, while the
    healthy listing still lists the deployment."""
    core: Final = make_core(model_list=[deployment("g", "a")], routing_strategy=None)
    assert core.settings().routing_strategy is None
    assert core.settings().as_dict()["routing_strategy"] is None
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g")
    assert str(raised.value) == NO_DEPLOYMENTS_MESSAGE.format(model="g")
    healthy: Final = await core.select("g", kind="healthy")
    assert isinstance(healthy, list)
    assert [entry["model_info"]["id"] for entry in healthy] == ["a"]


LEGACY_ATTRIBUTE_DEFAULTS: Final = {
    "total_calls": {},
    "fail_calls": {},
    "success_calls": {},
    "deployment_stats": {},
    "default_cache_time_seconds": 3600,
    "tenacity": None,
    "lowesttpm_logger_v2": None,
    "lowestlatency_logger": None,
    "lowestcost_logger": None,
}


@pytest.mark.parametrize(("name", "expected"), sorted(LEGACY_ATTRIBUTE_DEFAULTS.items()))
def test_legacy_attribute_default(make_core: MakeCore, name: str, expected: object) -> None:
    assert getattr(make_core(model_list=[deployment("g", "a")]), name) == expected


def test_healthy_deployments_is_the_live_deployment_list(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    assert core.healthy_deployments is core.registry().all()  # pyright: ignore[reportAttributeAccessIssue]


def test_chat_completions_create_calls_the_router_with_the_default_litellm_params(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], default_litellm_params={"temperature": 0.3})
    core.chat.completions.create(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert [(call.op, call.deployment_id, call.kwargs.get("temperature")) for call in host.calls] == [
        ("completion", "a", 0.3)
    ]


@pytest.mark.asyncio
async def test_a_provider_budget_config_adds_the_budget_limiter_after_the_listed_checks(make_core: MakeCore) -> None:
    """A provider budget appends ``router_budget_limiting`` to the optional pre-call checks given."""
    core: Final = make_core(
        model_list=[deployment("g", "a")],
        provider_budget_config={"openai": {"budget_limit": 1, "time_period": "1d"}},
        optional_pre_call_checks=["prompt_caching"],
    )
    assert [type(callback).__name__ for callback in core.optional_callbacks] == [  # pyright: ignore[reportAttributeAccessIssue]
        "PromptCachingDeploymentCheck",
        "RouterBudgetLimiting",
    ]
    core.retire()


# --- price data reloads ---------------------------------------------------------------------------


def test_a_price_data_reload_restores_a_live_routers_deployment_pricing_until_it_retires(
    make_core: MakeCore,
) -> None:
    model_id: Final = unique_id("l0001")
    core: Final = make_core(model_list=[deployment("gpt", model_id, model="openai/gpt-4o", input_cost_per_token=1e-3)])
    try:
        litellm.model_cost.pop(model_id)
        reapply_runtime_model_cost_registrations()
        assert litellm.model_cost[model_id]["input_cost_per_token"] == 1e-3
        core.retire()
        litellm.model_cost.pop(model_id)
        reapply_runtime_model_cost_registrations()
        assert model_id not in litellm.model_cost
    finally:
        litellm.model_cost.pop(model_id, None)


# --- runtime routing_strategy_args ----------------------------------------------------------------


def test_new_routing_strategy_args_rebuild_the_selector_and_replace_it_on_the_host(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], routing_strategy="latency-based-routing", routing_strategy_args={"ttl": 5}
    )
    (before,) = host.lists["callbacks"]
    core.apply(ChangeSet(settings={"routing_strategy_args": {"ttl": 9}}))
    (after,) = host.lists["callbacks"]
    assert after is not before
    assert after is core.lowestlatency_logger  # pyright: ignore[reportAttributeAccessIssue]
    assert after.routing_args.ttl == 9  # pyright: ignore[reportAttributeAccessIssue]
    assert core.settings().as_dict()["routing_strategy_args"]["ttl"] == 9


def test_invalid_routing_strategy_args_at_runtime_keep_the_previous_selector(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: the invalid args are still stored as ``routing_strategy_args``, while the selector (and
    so ``as_dict``, which reads the selector) keeps the previous ones."""
    core: Final = make_core(
        model_list=[deployment("g", "a")], routing_strategy="latency-based-routing", routing_strategy_args={"ttl": 5}
    )
    (before,) = host.lists["callbacks"]
    core.apply(ChangeSet(settings={"routing_strategy_args": {"ttl": "abc"}}))
    assert host.lists["callbacks"] == [before]
    assert core.lowestlatency_logger is before  # pyright: ignore[reportAttributeAccessIssue]
    assert core.settings().as_dict()["routing_strategy_args"]["ttl"] == 5
    assert core.routing_strategy_args == {"ttl": "abc"}  # pyright: ignore[reportAttributeAccessIssue]


def test_new_least_busy_args_replace_the_selector_in_callbacks_and_input_callbacks(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], routing_strategy="least-busy")
    before: Final = core.leastbusy_logger  # pyright: ignore[reportAttributeAccessIssue]
    core.apply(ChangeSet(settings={"routing_strategy_args": {"x": 1}}))
    after: Final = core.leastbusy_logger  # pyright: ignore[reportAttributeAccessIssue]
    assert after is not before
    assert (host.lists["callbacks"], host.lists["input_callback"]) == ([after], [after])


def test_new_routing_strategy_args_under_simple_shuffle_register_nothing(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    core.apply(ChangeSet(settings={"routing_strategy_args": {"ttl": 9}}))
    assert core.settings().as_dict()["routing_strategy_args"] == {"ttl": 9}
    assert (host.lists["callbacks"], host.lists["input_callback"]) == ([], [])


# --- routing groups: selectors on the host, shadowing ---------------------------------------------


def test_routing_group_selectors_are_registered_on_the_host_and_replaced_when_groups_change(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("m1", "a"), deployment("m2", "b")],
        routing_strategy="least-busy",
        routing_groups=[_group("g1", ["m1"], "latency-based-routing", routing_strategy_args={"ttl": 7})],
    )
    assert host_callback_names(host)["callbacks"] == ["LeastBusyLoggingHandler", "LowestLatencyLoggingHandler"]
    (latency,) = (
        callback for callback in host.lists["callbacks"] if type(callback).__name__ == "LowestLatencyLoggingHandler"
    )
    assert latency.routing_args.ttl == 7  # pyright: ignore[reportAttributeAccessIssue]
    core.apply(ChangeSet(settings={"routing_groups": [_group("g2", ["m2"], "usage-based-routing-v2")]}))
    assert host_callback_names(host)["callbacks"] == ["LeastBusyLoggingHandler", "LowestTPMLoggingHandler_v2"]
    core.apply(ChangeSet(settings={"routing_groups": []}))
    assert host_callback_names(host)["callbacks"] == ["LeastBusyLoggingHandler"]
    assert host_callback_names(host)["input_callback"] == ["LeastBusyLoggingHandler"]


@pytest.mark.asyncio
async def test_a_group_shadowed_by_a_model_name_still_routes_its_members_with_its_strategy(
    make_core: MakeCore,
) -> None:
    """The shadowed group's name is not callable (calling it reaches the model), but its least-busy
    strategy still serves its member: with no traffic least-busy always picks the first deployment."""
    core: Final = make_core(
        model_list=[deployment("m1", "a"), deployment("m2", "b"), deployment("m2", "c")],
        routing_groups=[_group("m1", ["m2"], "least-busy")],
    )
    assert await picked_ids(core, "m2", times=30, messages=MESSAGES) == {"b"}
    assert core.registry().routing_group("m1") is None


@pytest.mark.asyncio
async def test_a_group_named_like_an_alias_is_shadowed_by_the_alias(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("m1", "a"), deployment("m2", "b")],
        model_group_alias={"al": "m1"},
        routing_groups=[_group("al", ["m2"], "simple-shuffle")],
    )
    assert await picked_ids(core, "al") == {"a"}
    assert core.registry().routing_group("al") is None


def test_a_priority_group_named_like_an_alias_is_rejected(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match=re.escape("Priority routing group names must not shadow a model or alias")):
        make_core(
            model_list=[deployment("m1", "a")],
            model_group_alias={"al": "m1"},
            routing_groups=[_group("al", ["m1"], "priority", model_priorities={"m1": 1})],
        )


@pytest.mark.asyncio
async def test_a_deployment_added_under_a_group_name_shadows_the_group(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("m1", "a"), deployment("m2", "b")],
        routing_groups=[_group("grp", ["m1", "m2"], "simple-shuffle")],
    )
    core.apply(ChangeSet(upsert=[Deployment(**deployment("grp", "z"))]))  # pyright: ignore[reportArgumentType]
    assert await picked_ids(core, "grp") == {"z"}
    assert core.registry().routing_group("grp") is None


# --- registry().resolve kinds backed by this range ------------------------------------------------


def test_routing_group_alternatives_count_the_deployments_of_every_member(make_core: MakeCore) -> None:
    """Only a callable routing group (or an alias to one) can have alternatives; a plain model group
    with two deployments has none by this rule."""
    core: Final = make_core(
        model_list=[deployment("m1", "a"), deployment("m1", "a2"), deployment("m2", "b"), deployment("m3", "c")],
        model_group_alias={"al": "grp"},
        routing_groups=[
            _group("grp", ["m2", "m3"], "simple-shuffle"),
            _group("one", ["m2"], "priority", model_priorities={"m2": 1}),
        ],
    )
    registry: Final = core.registry()
    answers: Final = {
        name: registry.resolve(name, kind="routing_group_alternatives")  # pyright: ignore[reportCallIssue, reportArgumentType]
        for name in (None, "grp", "one", "al", "m1", "zzz")
    }
    assert answers == {None: False, "grp": True, "one": False, "al": True, "m1": False, "zzz": False}


TEAM_MODELS: Final = [
    deployment("pub", "t1", model_info={"team_id": "T", "team_public_model_name": "pubm"}),
    deployment("pub2", "t2", model_info={"team_id": "T", "team_public_model_name": "pubm"}),
    deployment("solo", "t3", model_info={"team_id": "T", "team_public_model_name": "solo"}),
    deployment("plain", "p1"),
    deployment("half", "h1", model_info={"team_id": "T"}),
]


@pytest.mark.parametrize(
    ("deployment_id", "expected"),
    [("t1", True), ("t3", False), ("p1", False), ("h1", False), ("zzz", False)],
    ids=["two-siblings", "only-one", "no-team", "no-public-name", "unknown-id"],
)
def test_team_model_alternatives_need_another_routable_deployment_under_the_same_team_name(
    make_core: MakeCore, deployment_id: str, expected: bool
) -> None:
    core: Final = make_core(model_list=list(TEAM_MODELS))
    assert core.registry().resolve(deployment_id, kind="team_model_alternatives") is expected  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_a_blocked_team_sibling_is_no_alternative(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("pub", "t1", model_info={"team_id": "T", "team_public_model_name": "pubm"}),
            deployment("pub2", "t2", model_info={"team_id": "T", "team_public_model_name": "pubm", "blocked": True}),
        ]
    )
    assert core.registry().resolve("t1", kind="team_model_alternatives") is False  # pyright: ignore[reportCallIssue, reportArgumentType]


def test_search_tool_support_comes_from_model_info_else_an_anthropic_request_shape(make_core: MakeCore) -> None:
    """Per routable deployment: a configured ``supports_search_tool`` wins, a deployment whose chat
    config builds an Anthropic request answers True, anything else None; an unknown model has none."""
    core: Final = make_core(
        model_list=[
            deployment("g", "o1"),
            deployment("g", "an1", model="anthropic/claude-3-5-sonnet-20240620"),
            deployment(
                "g", "cfg", model="anthropic/claude-3-5-sonnet-20240620", model_info={"supports_search_tool": False}
            ),
        ]
    )
    registry: Final = core.registry()
    assert registry.resolve("g", kind="search_tool_support") == (None, True, False)  # pyright: ignore[reportCallIssue, reportArgumentType]
    assert registry.resolve("zzz", kind="search_tool_support") == ()  # pyright: ignore[reportCallIssue, reportArgumentType]


# --- per-request strategy overrides ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_override_selector_is_bound_to_the_request_logging_object_and_reused(make_core: MakeCore) -> None:
    """An override strategy the router does not register globally is bound to the request's logging
    object; one selector per strategy serves every request. An override matching the router's own
    strategy reuses the global selector and binds nothing."""
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], routing_strategy="least-busy")
    own: Final = _logging_obj()
    await core.select("g", {"routing_strategy": "least-busy", "litellm_logging_obj": own})
    first: Final = _logging_obj()
    await core.select("g", {"routing_strategy": "usage-based-routing-v2", "litellm_logging_obj": first})
    second: Final = _logging_obj()
    await core.select("g", {"routing_strategy": "usage-based-routing-v2", "litellm_logging_obj": second})
    assert _dynamic_callback_names(own) == []
    assert _dynamic_callback_names(first) == ["LowestTPMLoggingHandler_v2"]
    assert (first.dynamic_async_success_callbacks or [None])[0] is (second.dynamic_async_success_callbacks or [])[0]


@pytest.mark.asyncio
async def test_an_override_matching_a_routing_group_strategy_is_not_bound_to_the_request(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("m", "b")],
        routing_groups=[_group("gg", ["m"], "usage-based-routing-v2")],
    )
    logging_obj: Final = _logging_obj()
    await core.select("g", {"routing_strategy": "usage-based-routing-v2", "litellm_logging_obj": logging_obj})
    assert _dynamic_callback_names(logging_obj) == []


def test_a_sync_usage_based_v2_override_counts_rpm_before_the_call(make_core: MakeCore, host: ScriptedHost) -> None:
    """The override selector is on no callback list, so the sync pick runs its pre-call check itself:
    the first call is counted against ``rpm=1`` and the next pick finds no deployment, raising the same
    ``RateLimitError`` as the async path."""
    core: Final = make_core(model_list=[deployment("g", "a", rpm=1)], num_retries=0)
    core.completion(model="g", messages=MESSAGES, routing_strategy="usage-based-routing-v2")  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(  # pyright: ignore[reportUnknownMemberType]  # pytest and the cores' untyped members
        litellm.RateLimitError, match=re.escape("No deployments available for selected model. Passed model=g.")
    ):
        core.completion(model="g", messages=MESSAGES, routing_strategy="usage-based-routing-v2")  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["a"]


@pytest.mark.asyncio
async def test_an_async_usage_based_v2_override_counts_rpm_before_the_call(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", rpm=1)], num_retries=0)
    await core.acompletion(model="g", messages=MESSAGES, routing_strategy="usage-based-routing-v2")  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(litellm.RateLimitError, match=re.escape("No deployments available for selected model.")):
        await core.acompletion(model="g", messages=MESSAGES, routing_strategy="usage-based-routing-v2")  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["a"]


@pytest.mark.asyncio
async def test_request_scoped_routing_still_runs_the_strategy_rpm_check(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """After ``observe(request_scoped=True)`` the selector is off the host's lists, but the router's
    own pre-call checks still reach it, so ``rpm=1`` still holds."""
    collect_other_routers()
    core: Final = make_core(
        model_list=[deployment("g", "a", rpm=1)], routing_strategy="usage-based-routing-v2", num_retries=0
    )
    core.observe(request_scoped=True)
    assert host.lists["callbacks"] == []
    await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(litellm.RateLimitError):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["a"]


# --- strategy dispatch arms -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_pick_under_usage_based_routing_v1_picks_the_first_configured(make_core: MakeCore) -> None:
    """The async pick calls v1's sync selector inline; with no traffic it picks the first deployment."""
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], routing_strategy="usage-based-routing"
    )
    assert await picked_ids(core, "g", times=10, messages=MESSAGES) == {"a"}


def test_sync_completion_under_latency_based_routing_reaches_the_group(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], routing_strategy="latency-based-routing"
    )
    for _ in range(10):
        core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert set(host.deployments_called()) <= {"a", "b"}
    assert len(host.calls) == 10


def test_sync_completion_under_cost_based_routing_picks_the_cheapest_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """The cost-based selector picks on the sync path as it does on the async one."""
    core: Final = make_core(
        model_list=[
            deployment("g", "a", input_cost_per_token=0.5, output_cost_per_token=0.5),
            deployment("g", "b", input_cost_per_token=0.1, output_cost_per_token=0.1),
        ],
        routing_strategy="cost-based-routing",
        num_retries=0,
    )
    for _ in range(3):
        core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["b", "b", "b"]


@pytest.mark.asyncio
async def test_pass_through_pick_under_cost_based_routing_picks_the_cheapest_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "a", use_in_pass_through=True, input_cost_per_token=0.5, output_cost_per_token=0.5),
            deployment("g", "b", use_in_pass_through=True, input_cost_per_token=0.1, output_cost_per_token=0.1),
        ],
        routing_strategy="cost-based-routing",
    )
    picked: Final = await core.select("g", kind="pass_through")  # pyright: ignore[reportCallIssue, reportUnknownVariableType]  # pytest and the cores' untyped members
    assert picked["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue, reportCallIssue, reportArgumentType]
