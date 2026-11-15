"""
Characterization of which deployments a router core considers for a request before a strategy
picks: routing groups and priority routing groups, tag filtering, region filtering, model access
groups, team-only models, default deployments and default fallbacks, paused (blocked)
deployments, pass-through selection, the healthy listing and the request-scoped narrowing keys.
"""

import datetime
import re
from collections.abc import Mapping
from typing import Final

import pydantic
import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router_api import ChangeSet
from litellm.types.router import RouterRateLimitError
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit
from tests.unit.test_router.core.selection_helpers import MESSAGES, MakeCore, ids_of, picked_ids

SPREAD_PICKS: Final = 60

NO_DEPLOYMENTS_MESSAGE: Final = (
    "No deployments available for selected model, Try again in 5 seconds. Passed model={model}. "
    "pre-call-checks={checks}, cooldown_list=[]"
)

NO_HEALTHY_MESSAGE: Final = (
    "litellm.BadRequestError: You passed in model={model}. There are no healthy deployments for this model"
)

GROUP_MODELS: Final = [deployment("m1", "a"), deployment("m2", "b"), deployment("m3", "c")]


def _group(
    name: str, models: list[str], strategy: str, **extra: object
) -> dict[str, object]:  # mutable-ok: a routing_groups entry  # kwargs-ok: extra RoutingGroup fields
    return {"group_name": name, "models": models, "routing_strategy": strategy, **extra}


def _priority(name: str, priorities: dict[str, int]) -> dict[str, object]:  # mutable-ok: a routing_groups entry
    return _group(name, list(priorities), "priority", model_priorities=priorities)


# --- routing groups -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_routing_group_name_is_callable_and_unions_its_members(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=list(GROUP_MODELS), routing_groups=[_group("grp", ["m1", "m2"], "simple-shuffle")]
    )
    assert await picked_ids(core, "grp", times=SPREAD_PICKS) == {"a", "b"}
    assert ids_of(await core.select("grp", kind="healthy")) == ["a", "b"]


@pytest.mark.asyncio
async def test_a_routing_group_call_uses_the_groups_strategy(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_MODELS), routing_groups=[_group("grp", ["m1", "m2"], "least-busy")])
    assert await picked_ids(core, "grp", messages=MESSAGES) == {"a"}


@pytest.mark.asyncio
async def test_a_member_called_by_name_uses_its_groups_strategy(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("m1", "a"), deployment("m1", "b")],
        routing_groups=[_group("grp", ["m1"], "least-busy")],
    )
    assert await picked_ids(core, "m1", messages=MESSAGES) == {"a"}


@pytest.mark.asyncio
async def test_models_outside_every_group_keep_the_top_level_strategy(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("m1", "a"), deployment("free", "x"), deployment("free", "y")],
        routing_groups=[_group("grp", ["m1"], "least-busy")],
    )
    assert await picked_ids(core, "free", times=SPREAD_PICKS, messages=MESSAGES) == {"x", "y"}


@pytest.mark.asyncio
async def test_an_alias_to_a_routing_group_resolves_to_the_group(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=list(GROUP_MODELS),
        model_group_alias={"al": "grp"},
        routing_groups=[_group("grp", ["m1", "m2"], "simple-shuffle")],
    )
    assert await picked_ids(core, "al", times=SPREAD_PICKS) == {"a", "b"}


@pytest.mark.asyncio
async def test_a_group_named_like_a_model_is_shadowed_by_the_model(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_MODELS), routing_groups=[_group("m1", ["m2"], "least-busy")])
    assert await picked_ids(core, "m1") == {"a"}


@pytest.mark.asyncio
async def test_a_group_whose_members_have_no_deployments_has_no_healthy_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_MODELS), routing_groups=[_group("empty", ["zz"], "least-busy")])
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("empty")
    assert str(raised.value) == NO_HEALTHY_MESSAGE.format(model="empty")


@pytest.mark.asyncio
async def test_a_routing_group_completion_reports_the_group_as_model_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(GROUP_MODELS), routing_groups=[_group("grp", ["m1"], "simple-shuffle")])
    response: Final = await core.acompletion(model="grp", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["a"]
    assert host.calls[-1].model_group == "grp"
    assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "grp"


def test_routing_group_names_are_listed_among_the_routable_names(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_MODELS), routing_groups=[_group("grp", ["m1", "m2"], "least-busy")])
    # The group name is listed twice today.
    assert sorted(core.registry().names(kind="groups")) == ["grp", "grp", "m1", "m2", "m3"]


@pytest.mark.asyncio
async def test_routing_groups_applied_at_runtime_become_callable(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_MODELS))
    with pytest.raises(litellm.BadRequestError):
        await core.select("grp")
    core.apply(ChangeSet(settings={"routing_groups": [_group("grp", ["m1", "m2"], "least-busy")]}))
    assert await picked_ids(core, "grp", messages=MESSAGES) == {"a"}


@pytest.mark.parametrize(
    ("groups", "message"),
    [
        (
            [_group("x", ["m1"], "least-busy"), _group("x", ["m2"], "least-busy")],
            "routing_groups: group names must be unique, duplicate group_name 'x'.",
        ),
        (
            [_group("default", ["m1"], "least-busy")],
            "routing_groups: 'default' is reserved for the implicit fallback group.",
        ),
        ([_group("", ["m1"], "least-busy")], "routing_groups: group_name must be non-empty."),
        (
            [_group("x", ["m1"], "least-busy"), _group("y", ["m1"], "simple-shuffle")],
            "routing_groups: model_name 'm1' appears in 'x' and 'y'. Each model may belong to at most one group.",
        ),
        ([_group("x", ["m1"], "nope")], "Invalid routing_strategy: 'nope'."),
        ([_priority("m1", {"m2": 1})], "Priority routing group names must not shadow a model or alias"),
    ],
    ids=["duplicate", "reserved-default", "empty-name", "overlap", "bad-strategy", "priority-shadow"],
)
def test_invalid_routing_groups_are_rejected_at_construction(
    make_core: MakeCore, groups: list[dict[str, object]], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        make_core(model_list=list(GROUP_MODELS), routing_groups=groups)


@pytest.mark.parametrize(
    ("group", "message"),
    [
        (_group("p", ["m1", "m2"], "priority", model_priorities={"m1": 1}), "model_priorities must contain exactly"),
        (
            _group("p", ["m1"], "priority", model_priorities={"m1": 1}, routing_strategy_args={"x": 1}),
            "Priority routing groups use model_priorities, not routing_strategy_args",
        ),
        (_group("p", ["m1"], "least-busy", model_priorities={"m1": 1}), "model_priorities requires routing_strategy"),
        (_group("p", ["m1"], "priority", model_priorities={"m1": 0}), "greater than or equal to 1"),
        (
            _group("p", ["m1", "m1"], "priority", model_priorities={"m1": 1}),
            "Priority routing groups require nonempty, distinct models",
        ),
    ],
    ids=["missing-priority", "strategy-args", "priorities-without-priority", "zero-priority", "duplicate-member"],
)
def test_invalid_priority_groups_fail_validation_at_construction(
    make_core: MakeCore, group: dict[str, object], message: str
) -> None:
    with pytest.raises(pydantic.ValidationError, match=re.escape(message)):
        make_core(model_list=list(GROUP_MODELS), routing_groups=[group])


# --- priority routing groups ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_priority_group_picks_only_its_lowest_priority_members(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[*GROUP_MODELS, deployment("m2", "b2")],
        routing_groups=[_priority("p", {"m1": 2, "m2": 1, "m3": 2})],
    )
    assert await picked_ids(core, "p", times=SPREAD_PICKS) == {"b", "b2"}
    assert sorted(ids_of(await core.select("p", kind="healthy"))) == ["b", "b2"]


@pytest.mark.asyncio
async def test_a_priority_group_pick_carries_the_member_priority_as_order(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_MODELS), routing_groups=[_priority("p", {"m1": 3, "m2": 1})])
    chosen: Final = await core.select("p")
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]
    assert chosen["model_name"] == "m2"  # pyright: ignore[reportIndexIssue]
    assert chosen["litellm_params"]["order"] == 1  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_a_priority_group_member_called_directly_ignores_the_priority(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(GROUP_MODELS), routing_groups=[_priority("p", {"m1": 3, "m2": 1})])
    chosen: Final = await core.select("m1")
    assert chosen["model_info"]["id"] == "a"  # pyright: ignore[reportIndexIssue]
    assert "order" not in chosen["litellm_params"]  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_a_priority_group_fails_over_to_the_next_priority_and_stays_there_while_the_first_cools_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=list(GROUP_MODELS),
        routing_groups=[_priority("p", {"m1": 1, "m2": 2})],
        allowed_fails=0,
        num_retries=0,
        cooldown_time=60,
    )
    host.answer("openai/a", rate_limit())
    await core.acompletion(model="p", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["a", "b"]
    assert await picked_ids(core, "p") == {"b"}


# --- tag filtering -----------------------------------------------------------------------------------

TAGGED: Final = [
    deployment("g", "free", tags=["free"]),
    deployment("g", "paid", tags=["paid"]),
    deployment("g", "dflt", tags=["default"]),
    deployment("g", "untagged"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tags", "expected"),
    [(["free"], {"free"}), (["paid"], {"paid"}), (["free", "paid"], {"free", "paid"})],
)
async def test_tag_filtering_picks_deployments_carrying_a_request_tag(
    make_core: MakeCore, tags: list[str], expected: set[str]
) -> None:
    core: Final = make_core(model_list=list(TAGGED), enable_tag_filtering=True)
    assert await picked_ids(core, "g", times=SPREAD_PICKS, request_kwargs={"metadata": {"tags": tags}}) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", [{}, {"metadata": {"tags": ["unknown"]}}], ids=["untagged", "unmatched"])
async def test_tag_filtering_sends_untagged_or_unmatched_requests_to_default_tagged_deployments(
    make_core: MakeCore, request_kwargs: dict[str, object]
) -> None:
    core: Final = make_core(model_list=list(TAGGED), enable_tag_filtering=True)
    assert await picked_ids(core, "g", request_kwargs=request_kwargs) == {"dflt"}


@pytest.mark.asyncio
async def test_tag_filtering_without_a_default_rejects_an_unmatched_tag(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=TAGGED[:2], enable_tag_filtering=True)
    with pytest.raises(ValueError, match="Not allowed to access model") as raised:
        await core.select("g", {"metadata": {"tags": ["unknown"]}})
    assert str(raised.value) == (
        "Not allowed to access model due to tags configuration. Passed model=g and tags=['unknown']"
    )


@pytest.mark.asyncio
async def test_tag_filtering_without_a_default_serves_an_untagged_request_from_every_deployment(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(model_list=TAGGED[:2], enable_tag_filtering=True)
    assert await picked_ids(core, "g", times=SPREAD_PICKS) == {"free", "paid"}


@pytest.mark.asyncio
async def test_request_tags_are_ignored_without_tag_filtering(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TAGGED))
    picked: Final = await picked_ids(core, "g", times=SPREAD_PICKS, request_kwargs={"metadata": {"tags": ["free"]}})
    assert picked == {"free", "paid", "dflt", "untagged"}


@pytest.mark.asyncio
async def test_a_request_can_enable_tag_filtering_for_itself(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TAGGED))
    request: Final = {"metadata": {"tags": ["free"]}, "enable_tag_filtering": True}
    assert await picked_ids(core, "g", request_kwargs=request) == {"free"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("match_any", "expected"), [(True, {"ab", "a"}), (False, {"ab"})])
async def test_tag_filtering_match_any_decides_whether_one_shared_tag_is_enough(
    make_core: MakeCore, match_any: bool, expected: set[str]
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "ab", tags=["a", "b"]), deployment("g", "a", tags=["a"])],
        enable_tag_filtering=True,
        tag_filtering_match_any=match_any,
    )
    picked: Final = await picked_ids(core, "g", times=SPREAD_PICKS, request_kwargs={"metadata": {"tags": ["a", "b"]}})
    assert picked == expected


@pytest.mark.asyncio
async def test_tags_in_litellm_metadata_are_read_too(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TAGGED), enable_tag_filtering=True)
    assert await picked_ids(core, "g", request_kwargs={"litellm_metadata": {"tags": ["paid"]}}) == {"paid"}


@pytest.mark.asyncio
async def test_completion_with_tags_calls_the_tagged_deployment(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=list(TAGGED), enable_tag_filtering=True)
    for _ in range(5):
        await core.acompletion(model="g", messages=MESSAGES, metadata={"tags": ["paid"]})  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["paid"] * 5


# --- region filtering --------------------------------------------------------------------------------

REGIONAL: Final = [
    deployment("g", "eu", region_name="eu"),
    deployment("g", "us", region_name="us"),
    deployment("g", "unknown"),
]


@pytest.mark.asyncio
async def test_allowed_model_region_keeps_only_deployments_in_that_region(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(REGIONAL), enable_pre_call_checks=True)
    assert await picked_ids(core, "g", request_kwargs={"allowed_model_region": "eu"}, messages=MESSAGES) == {"eu"}


@pytest.mark.asyncio
async def test_allowed_model_region_is_ignored_without_request_input(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(REGIONAL), enable_pre_call_checks=True)
    picked: Final = await picked_ids(core, "g", times=SPREAD_PICKS, request_kwargs={"allowed_model_region": "eu"})
    assert picked == {"eu", "us", "unknown"}


@pytest.mark.asyncio
async def test_allowed_model_region_is_ignored_without_pre_call_checks(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(REGIONAL))
    picked: Final = await picked_ids(
        core, "g", times=SPREAD_PICKS, request_kwargs={"allowed_model_region": "eu"}, messages=MESSAGES
    )
    assert picked == {"eu", "us", "unknown"}


@pytest.mark.asyncio
async def test_an_allowed_model_region_no_deployment_is_in_rate_limits(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(REGIONAL), enable_pre_call_checks=True)
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g", {"allowed_model_region": "ap"}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert str(raised.value) == NO_DEPLOYMENTS_MESSAGE.format(model="g", checks=True)


# --- model access groups -------------------------------------------------------------------------------

ACCESS_GROUPED: Final = [
    deployment("g", "in_a", model_info={"access_groups": ["grpA"]}),
    deployment("g", "in_b", model_info={"access_groups": ["grpB"]}),
]


def _auth(
    bucket: str = "metadata", **fields: object
) -> dict[str, object]:  # mutable-ok: request kwargs  # kwargs-ok: UserAPIKeyAuth fields
    return {bucket: {"user_api_key_auth": UserAPIKeyAuth(**fields)}}  # pyright: ignore[reportArgumentType]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "expected"),
    [
        (_auth(models=["grpA"]), {"in_a"}),
        (_auth(models=["grpB"]), {"in_b"}),
        (_auth(team_models=["grpB"]), {"in_b"}),
        (_auth("litellm_metadata", models=["grpA"]), {"in_a"}),
    ],
    ids=["key-group-a", "key-group-b", "team-group-b", "litellm-metadata"],
)
async def test_an_access_group_only_caller_is_narrowed_to_its_groups_deployments(
    make_core: MakeCore, request_kwargs: dict[str, object], expected: set[str]
) -> None:
    core: Final = make_core(model_list=list(ACCESS_GROUPED))
    assert await picked_ids(core, "g", request_kwargs=request_kwargs) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "models", [["g"], ["*"], ["all-proxy-models"], [], ["unrelated"]], ids=["direct", "star", "all", "none", "other"]
)
async def test_a_caller_not_limited_to_access_groups_can_reach_every_deployment(
    make_core: MakeCore, models: list[str]
) -> None:
    core: Final = make_core(model_list=list(ACCESS_GROUPED))
    picked: Final = await picked_ids(core, "g", times=SPREAD_PICKS, request_kwargs=_auth(models=models))
    assert picked == {"in_a", "in_b"}


# --- team-only models ------------------------------------------------------------------------------------

TEAM_ONLY: Final = [
    deployment("internal-x", "tx", model_info={"team_id": "team-x", "team_public_model_name": "pub"}),
    deployment("internal-y", "ty", model_info={"team_id": "team-y", "team_public_model_name": "pub"}),
]


@pytest.mark.asyncio
async def test_a_team_id_in_litellm_metadata_selects_the_team_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TEAM_ONLY))
    assert await picked_ids(core, "pub", request_kwargs={"litellm_metadata": {"user_api_key_team_id": "team-y"}}) == {
        "ty"
    }


@pytest.mark.asyncio
async def test_a_proxy_admin_without_a_team_is_refused_a_public_name_several_teams_share(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TEAM_ONLY))
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("pub", _auth(user_role="proxy_admin"))
    assert str(raised.value) == (
        "litellm.BadRequestError: Model name 'pub' matches deployments from multiple teams. "
        "Specify the deployment ID directly to disambiguate."
    )


@pytest.mark.asyncio
async def test_a_proxy_admin_without_a_team_finds_one_teams_public_name_but_cannot_use_it(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=TEAM_ONLY[:1])
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("pub", _auth(user_role="proxy_admin"))
    assert str(raised.value) == NO_DEPLOYMENTS_MESSAGE.format(model="pub", checks=False)


# --- default deployment, default fallbacks ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_model_goes_to_a_catch_all_deployment_under_its_own_name(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("*", "catch_all", model="openai/*")])
    chosen: Final = await core.select("brand-new")
    assert chosen["model_info"]["id"] == "catch_all"  # pyright: ignore[reportIndexIssue]
    assert chosen["litellm_params"]["model"] == "openai/brand-new"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_an_unknown_model_goes_to_the_first_default_fallback(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("h", "b")], default_fallbacks=["h", "g"])
    chosen: Final = await core.select("unknown")
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]
    await core.acompletion(model="unknown", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["b"]


@pytest.mark.asyncio
async def test_an_unknown_model_goes_to_the_process_default_fallback(make_core: MakeCore, host: ScriptedHost) -> None:
    host.set_dynamic(default_fallbacks=["g"])
    core: Final = make_core(model_list=[deployment("g", "a")])
    chosen: Final = await core.select("unknown")
    assert chosen["model_info"]["id"] == "a"  # pyright: ignore[reportIndexIssue]


# --- paused (blocked) deployments ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_blocked_deployment_is_never_picked_for_its_group(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model_info={"blocked": True}), deployment("g", "b")])
    assert await picked_ids(core, "g") == {"b"}
    assert ids_of(await core.select("g", kind="healthy")) == ["b"]


@pytest.mark.asyncio
async def test_selecting_a_blocked_deployment_by_id_raises_service_unavailable(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model_info={"blocked": True}), deployment("g", "b")])
    with pytest.raises(litellm.ServiceUnavailableError) as raised:
        await core.select("a")
    # The message names the deployment's litellm model, not the id that was asked for.
    assert str(raised.value) == (
        "litellm.ServiceUnavailableError: Model 'openai/a' is currently paused and cannot accept requests."
    )
    assert raised.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["select", "acompletion", "completion"])
async def test_a_group_whose_every_deployment_is_blocked_rate_limits(
    make_core: MakeCore, host: ScriptedHost, entry: str
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model_info={"blocked": True})])
    call: Final = core.select("g") if entry == "select" else core.call(entry, {"model": "g", "messages": MESSAGES})
    with pytest.raises(RouterRateLimitError) as raised:
        await call
    assert str(raised.value) == NO_DEPLOYMENTS_MESSAGE.format(model="g", checks=False)
    assert host.calls == []


# --- pass-through selection ------------------------------------------------------------------------------------

PASS_THROUGH: Final = [deployment("g", "pt", use_in_pass_through=True), deployment("g", "plain")]


@pytest.mark.asyncio
async def test_pass_through_selection_picks_only_pass_through_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PASS_THROUGH))
    assert await picked_ids(core, "g", kind="pass_through") == {"pt"}


@pytest.mark.asyncio
async def test_pass_through_selection_by_id_returns_a_pass_through_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PASS_THROUGH))
    chosen: Final = await core.select("pt", kind="pass_through")
    assert chosen["model_info"]["id"] == "pt"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_pass_through_selection_by_id_refuses_a_plain_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PASS_THROUGH))
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("plain", kind="pass_through")
    assert str(raised.value) == (
        "litellm.BadRequestError: Deployment plain does not support pass-through endpoint (use_in_pass_through=False)"
    )


@pytest.mark.asyncio
async def test_pass_through_selection_without_pass_through_deployments_raises_bad_request(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("g", kind="pass_through")
    assert str(raised.value) == (
        "litellm.BadRequestError: Model g has no deployment configured with use_in_pass_through=True. "
        "Please add use_in_pass_through: true in the deployment configuration"
    )


@pytest.mark.asyncio
async def test_async_pass_through_selection_words_the_missing_deployment_error_differently(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.async_get_available_deployment_for_pass_through("g", {})  # pyright: ignore[reportAttributeAccessIssue]
    assert str(raised.value) == (
        "litellm.BadRequestError: Model g has no deployments configured with use_in_pass_through=True. "
        "Please add use_in_pass_through: true to the deployment configuration"
    )


@pytest.mark.asyncio
async def test_pass_through_selection_of_an_unknown_model_has_no_healthy_deployments(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PASS_THROUGH))
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("nope", kind="pass_through")
    assert str(raised.value) == NO_HEALTHY_MESSAGE.format(model="nope")


@pytest.mark.asyncio
async def test_pass_through_selection_skips_a_cooling_pass_through_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "pt1", use_in_pass_through=True),
            deployment("g", "pt2", use_in_pass_through=True),
        ],
        allowed_fails=0,
        num_retries=0,
        cooldown_time=60,
    )
    host.answer("openai/pt1", rate_limit())
    with pytest.raises(litellm.RateLimitError):
        await core.acompletion(model="pt1", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert await picked_ids(core, "g", kind="pass_through") == {"pt2"}


def test_deprecated_pass_through_pick_matches_select(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(PASS_THROUGH))
    assert core.get_available_deployment_for_pass_through("g")["model_info"]["id"] == "pt"  # pyright: ignore[reportAttributeAccessIssue]


# --- the healthy listing --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_healthy_listing_keeps_only_the_lowest_order(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", order=1), deployment("g", "b", order=2)])
    assert ids_of(await core.select("g", kind="healthy")) == ["a"]


@pytest.mark.asyncio
async def test_the_healthy_listing_for_a_deployment_id_is_that_single_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")])
    healthy: Final = await core.select("b", kind="healthy")
    assert isinstance(healthy, Mapping)
    assert healthy["model_info"]["id"] == "b"


@pytest.mark.asyncio
async def test_the_healthy_listing_of_an_unknown_model_raises_bad_request(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("nope", kind="healthy")
    assert str(raised.value) == NO_HEALTHY_MESSAGE.format(model="nope")


@pytest.mark.asyncio
async def test_the_healthy_listing_leaves_out_cooling_deployments(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0, num_retries=0, cooldown_time=60
    )
    host.answer("openai/a", rate_limit())
    with pytest.raises(litellm.RateLimitError):
        await core.acompletion(model="a", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert ids_of(await core.select("g", kind="healthy")) == ["b"]


# --- request-scoped narrowing keys -----------------------------------------------------------------------------------

THREE: Final = [deployment("g", "a"), deployment("g", "b"), deployment("g", "c")]


@pytest.mark.asyncio
async def test_target_order_picks_that_order_instead_of_the_lowest(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", order=1), deployment("g", "b", order=2)])
    assert await picked_ids(core, "g", request_kwargs={"_target_order": 2}) == {"b"}
    assert ids_of(await core.select("g", {"_target_order": 2}, kind="healthy")) == ["b"]


@pytest.mark.asyncio
async def test_a_target_order_no_deployment_has_rate_limits(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", order=1)])
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g", {"_target_order": 3})
    assert str(raised.value) == NO_DEPLOYMENTS_MESSAGE.format(model="g", checks=False)


@pytest.mark.asyncio
async def test_excluded_deployment_ids_are_never_picked(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(THREE))
    assert await picked_ids(core, "g", request_kwargs={"_excluded_deployment_ids": ["a", "b"]}) == {"c"}


@pytest.mark.asyncio
async def test_excluding_every_deployment_rate_limits(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(THREE))
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select("g", {"_excluded_deployment_ids": ["a", "b", "c"]})
    assert str(raised.value) == NO_DEPLOYMENTS_MESSAGE.format(model="g", checks=False)


@pytest.mark.asyncio
async def test_retry_skipped_deployment_ids_are_not_picked_while_others_remain(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(THREE))
    assert await picked_ids(core, "g", request_kwargs={"_retry_skipped_deployment_ids": ("a", "b")}) == {"c"}


@pytest.mark.asyncio
async def test_skipping_every_deployment_on_retry_keeps_them_all(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(THREE))
    healthy: Final = await core.select("g", {"_retry_skipped_deployment_ids": ("a", "b", "c")}, kind="healthy")
    assert ids_of(healthy) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_retry_skipped_deployment_ids_given_as_a_list_are_ignored(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(THREE))
    healthy: Final = await core.select("g", {"_retry_skipped_deployment_ids": ["a", "b"]}, kind="healthy")
    assert ids_of(healthy) == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_select_pops_the_narrowing_keys_from_the_request(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(THREE))
    request: Final[dict[str, object]] = {
        "_target_order": None,
        "_excluded_deployment_ids": ["a"],
        "_retry_skipped_deployment_ids": ("b",),
    }
    assert ids_of(await core.select("g", request, kind="healthy")) == ["c"]
    assert request == {}


# --- access windows (reserved deployments) ----------------------------------------------------------------------------


def _window_for(team_ids: list[str]) -> dict[str, object]:  # mutable-ok: an access_windows entry
    now: Final = datetime.datetime.now(datetime.timezone.utc)
    return {
        "start": (now - datetime.timedelta(hours=1)).strftime("%H:%M"),
        "end": (now + datetime.timedelta(hours=1)).strftime("%H:%M"),
        "timezone": "UTC",
        "team_ids": team_ids,
    }


def _reserved(window: dict[str, object]) -> dict[str, object]:  # mutable-ok: a model_list entry
    return deployment("g", "reserved", model_info={"access_windows": [window]})


@pytest.mark.asyncio
@pytest.mark.parametrize("request_kwargs", [{"metadata": {"user_api_key_team_id": "team-b"}}, {}])
async def test_a_deployment_reserved_for_another_team_is_hidden_during_its_window(
    make_core: MakeCore, request_kwargs: dict[str, object]
) -> None:
    core: Final = make_core(model_list=[_reserved(_window_for(["team-a"])), deployment("g", "open")])
    assert await picked_ids(core, "g", request_kwargs=request_kwargs) == {"open"}


@pytest.mark.asyncio
async def test_the_reserving_team_reaches_its_reserved_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[_reserved(_window_for(["team-a"]))])
    assert await picked_ids(core, "g", request_kwargs={"metadata": {"user_api_key_team_id": "team-a"}}) == {"reserved"}


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "named"), [("g", "g"), ("reserved", "openai/reserved")], ids=["group", "id"])
async def test_only_reserved_deployments_left_raises_bad_request_naming_the_window_end(
    make_core: MakeCore, model: str, named: str
) -> None:
    window: Final = _window_for(["team-a"])
    core: Final = make_core(model_list=[_reserved(window)])
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select(model, {"metadata": {"user_api_key_team_id": "team-b"}})
    assert str(raised.value) == (
        f"litellm.BadRequestError: Deployment {named} is reserved for another team until {window['end']} UTC"
    )


# --- specific_deployment ------------------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["openai/unknown", "g"], ids=["unknown-litellm-model", "group-name"])
async def test_specific_deployment_for_a_name_no_litellm_model_matches_rate_limits(
    make_core: MakeCore, model: str
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    with pytest.raises(RouterRateLimitError) as raised:
        await core.select(model, specific_deployment=True)  # pyright: ignore[reportCallIssue]
    assert str(raised.value) == NO_DEPLOYMENTS_MESSAGE.format(model=model, checks=False)
