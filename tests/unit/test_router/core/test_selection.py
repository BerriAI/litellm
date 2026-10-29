"""
Deployment selection on a router core: which deployment ``select`` picks and which one an endpoint
call reaches, across groups, aliases, wildcard routes, deployment ids, teams, weights, order,
routing strategies and cooldowns.
"""

from collections.abc import Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.types.router import RouterRateLimitError
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    auth_error,
    bad_request,
    context_window,
    deployment,
    rate_limit,
    server_error,
    timeout,
)

MESSAGES: Final = [{"role": "user", "content": "hello there, router"}]
PICKS: Final = 20

MakeCore = Callable[..., RouterCore]


async def _picked_ids(
    core: RouterCore, model: str, times: int = PICKS, **select_arguments: object
) -> set[str]:  # kwargs-ok: select's keyword arguments
    picked: Final[set[str]] = set()
    for _ in range(times):
        chosen = await core.select(model, **select_arguments)  # pyright: ignore[reportCallIssue]  # PythonCore's select takes keyword options
        assert isinstance(chosen, Mapping)
        picked.add(str(chosen["model_info"]["id"]))
    return picked


async def _fail_through(core: RouterCore, deployment_id: str, times: int = 1) -> None:
    for _ in range(times):
        with pytest.raises(Exception):
            await core.acompletion(model=deployment_id, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


# --- picking within a model group ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("model", "expected"), [("g1", "a"), ("g2", "b"), ("g3", "c")])
async def test_select_returns_the_only_deployment_of_the_requested_group(
    make_core: MakeCore, model: str, expected: str
) -> None:
    core: Final = make_core(model_list=[deployment("g1", "a"), deployment("g2", "b"), deployment("g3", "c")])
    chosen: Final = await core.select(model)
    assert chosen["model_info"]["id"] == expected  # pyright: ignore[reportIndexIssue]
    assert chosen["model_name"] == model  # pyright: ignore[reportIndexIssue]
    assert chosen["litellm_params"]["model"] == f"openai/{expected}"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_select_only_picks_deployments_of_the_requested_group(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b"), deployment("other", "c"), deployment("other", "d")]
    )
    assert await _picked_ids(core, "g") <= {"a", "b"}
    assert await _picked_ids(core, "other") <= {"c", "d"}


@pytest.mark.asyncio
async def test_select_does_not_call_the_provider(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    await core.select("g")
    assert host.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "request_payload"),
    [
        ("acompletion", {"messages": MESSAGES}),
        ("atext_completion", {"prompt": "hi"}),
        ("aembedding", {"input": "hi"}),
    ],
)
async def test_async_endpoint_calls_the_deployment_of_the_requested_group(
    make_core: MakeCore, host: ScriptedHost, op: str, request_payload: dict[str, object]
) -> None:
    core: Final = make_core(model_list=[deployment("g1", "a"), deployment("g2", "b")])
    await core.call(op, {"model": "g2", **request_payload})
    assert host.deployments_called(op) == ["b"]
    assert host.calls[-1].model == "openai/b"


@pytest.mark.parametrize(
    ("op", "request_payload"), [("completion", {"messages": MESSAGES}), ("embedding", {"input": "hi"})]
)
def test_sync_endpoint_calls_the_deployment_of_the_requested_group(
    make_core: MakeCore, host: ScriptedHost, op: str, request_payload: dict[str, object]
) -> None:
    core: Final = make_core(model_list=[deployment("g1", "a"), deployment("g2", "b")])
    getattr(core, op)(model="g1", **request_payload)
    assert host.deployments_called(op) == ["a"]


@pytest.mark.asyncio
async def test_completion_reports_the_requested_group_in_its_headers(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g1", "a"), deployment("g2", "b")])
    response: Final = await core.acompletion(model="g2", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "g2"
    assert host.calls[-1].model_group == "g2"


# --- model_group_alias ---------------------------------------------------------------------

ALIASES: Final = {
    "plain": "g",
    "visible": {"model": "g", "hidden": False},
    "hidden": {"model": "g", "hidden": True},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", sorted(ALIASES))
async def test_select_resolves_an_alias_to_its_target_group(make_core: MakeCore, alias: str) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("other", "b")], model_group_alias=dict(ALIASES)
    )
    chosen: Final = await core.select(alias)
    assert chosen["model_info"]["id"] == "a"  # pyright: ignore[reportIndexIssue]
    assert chosen["model_name"] == "g"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
@pytest.mark.parametrize("alias", sorted(ALIASES))
async def test_completion_through_an_alias_calls_the_target_group(
    make_core: MakeCore, host: ScriptedHost, alias: str
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("other", "b")], model_group_alias=dict(ALIASES)
    )
    await core.acompletion(model=alias, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.parametrize(("alias", "target"), [("plain", "g"), ("visible", "g"), ("hidden", "g"), ("g", None)])
def test_registry_resolves_alias_targets(make_core: MakeCore, alias: str, target: str | None) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], model_group_alias=dict(ALIASES))
    assert core.registry().resolve(alias, kind="alias") == target


def test_hidden_alias_is_left_out_of_the_routable_names(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], model_group_alias=dict(ALIASES))
    assert sorted(core.registry().names(kind="groups")) == ["g", "plain", "visible"]


@pytest.mark.asyncio
async def test_alias_to_an_unknown_group_fails_naming_the_target(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], model_group_alias={"dangling": "missing"})
    with pytest.raises(litellm.BadRequestError, match="You passed in model=missing. There are no healthy deployments"):
        await core.select("dangling")


# --- wildcard / pattern groups ---------------------------------------------------------------

WILDCARD_MODEL_LIST: Final = [
    deployment("*", "catch_all", model="openai/*"),
    deployment("anthropic/*", "anthropic_any", model="anthropic/*"),
    deployment("openai/gpt-4o", "exact"),
    deployment("openai/gpt-*", "gpt_any", model="openai/gpt-*"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "expected_id", "expected_group", "expected_model"),
    [
        ("openai/gpt-4o", "exact", "openai/gpt-4o", "openai/exact"),
        ("openai/gpt-4o-mini", "gpt_any", "openai/gpt-*", "openai/gpt-4o-mini"),
        ("anthropic/claude-x", "anthropic_any", "anthropic/*", "anthropic/claude-x"),
        ("gpt-5", "catch_all", "*", "openai/gpt-5"),
        ("whatever/foo", "catch_all", "*", "openai/whatever/foo"),
    ],
)
async def test_select_prefers_an_exact_group_then_the_most_specific_pattern(
    make_core: MakeCore, requested: str, expected_id: str, expected_group: str, expected_model: str
) -> None:
    core: Final = make_core(model_list=list(WILDCARD_MODEL_LIST))
    chosen: Final = await core.select(requested)
    assert chosen["model_info"]["id"] == expected_id  # pyright: ignore[reportIndexIssue]
    assert chosen["model_name"] == expected_group  # pyright: ignore[reportIndexIssue]
    assert chosen["litellm_params"]["model"] == expected_model  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_completion_through_a_wildcard_calls_the_requested_model(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("openai/*", "wild", model="openai/*")])
    response: Final = await core.acompletion(model="openai/gpt-x", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == ["wild"]
    assert host.calls[-1].model == "openai/gpt-x"
    assert response._hidden_params["additional_headers"]["x-litellm-model-group"] == "openai/gpt-x"


@pytest.mark.asyncio
async def test_a_pattern_that_does_not_match_is_an_unknown_model(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("anthropic/*", "anthropic_any", model="anthropic/*")])
    with pytest.raises(litellm.BadRequestError, match="model=openai/gpt-4o"):
        await core.select("openai/gpt-4o")


# --- selecting by deployment id or litellm model --------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_id", ["a", "b", "c"])
async def test_select_by_deployment_id_returns_that_deployment(make_core: MakeCore, deployment_id: str) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b"), deployment("h", "c")])
    assert await _picked_ids(core, deployment_id) == {deployment_id}


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment_id", ["a", "b"])
async def test_completion_by_deployment_id_calls_that_deployment(
    make_core: MakeCore, host: ScriptedHost, deployment_id: str
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")])
    for _ in range(5):
        await core.acompletion(model=deployment_id, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == [deployment_id] * 5


@pytest.mark.asyncio
async def test_select_by_litellm_model_name_finds_the_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")])
    assert await _picked_ids(core, "openai/b") == {"b"}


@pytest.mark.asyncio
async def test_specific_deployment_picks_by_litellm_model_across_groups(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g1", "c", model="openai/shared"),
            deployment("g2", "d", model="openai/shared"),
            deployment("g3", "e"),
        ]
    )
    assert await _picked_ids(core, "openai/shared", specific_deployment=True) <= {"c", "d"}
    assert await _picked_ids(core, "openai/e", specific_deployment=True) == {"e"}


@pytest.mark.asyncio
async def test_completion_with_specific_deployment_calls_the_named_litellm_model(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")])
    for _ in range(5):
        await core.acompletion(model="openai/b", messages=MESSAGES, specific_deployment=True)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == ["b"] * 5


# --- unknown models ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "request_payload"),
    [
        ("acompletion", {"messages": MESSAGES}),
        ("aembedding", {"input": "hi"}),
        ("atext_completion", {"prompt": "hi"}),
        ("completion", {"messages": MESSAGES}),
        ("embedding", {"input": "hi"}),
    ],
)
async def test_unknown_model_raises_bad_request_naming_the_model(
    make_core: MakeCore, host: ScriptedHost, op: str, request_payload: dict[str, object]
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.call(op, {"model": "nope", **request_payload})
    assert (
        str(raised.value)
        == "litellm.BadRequestError: You passed in model=nope. There are no healthy deployments for this model"
    )
    assert raised.value.status_code == 400
    assert host.calls == []


@pytest.mark.asyncio
async def test_select_of_an_unknown_model_raises_bad_request(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("nope")
    assert raised.value.status_code == 400
    assert raised.value.model == "nope"
    assert "There are no healthy deployments for this model" in str(raised.value)


# --- team-scoped deployments --------------------------------------------------------------------

TEAM_MODEL_LIST: Final = [
    deployment("g", "public"),
    deployment("team-x-group", "team_x", model_info={"team_id": "tx", "team_public_model_name": "g"}),
    deployment("team-y-group", "team_y", model_info={"team_id": "ty", "team_public_model_name": "g"}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, "public"),
        ({"user_api_key_team_id": "tx"}, "team_x"),
        ({"user_api_key_team_id": "ty"}, "team_y"),
        ({"user_api_key_team_id": "unknown-team"}, "public"),
    ],
)
async def test_select_serves_a_team_its_own_deployment_under_the_public_name(
    make_core: MakeCore, metadata: dict[str, str], expected: str
) -> None:
    core: Final = make_core(model_list=list(TEAM_MODEL_LIST))
    assert await _picked_ids(core, "g", request_kwargs={"metadata": metadata}) == {expected}


@pytest.mark.asyncio
@pytest.mark.parametrize(("team_id", "expected"), [("tx", "team_x"), ("ty", "team_y"), ("tz", "public")])
async def test_completion_for_a_team_calls_its_deployment_under_the_public_group(
    make_core: MakeCore, host: ScriptedHost, team_id: str, expected: str
) -> None:
    core: Final = make_core(model_list=list(TEAM_MODEL_LIST))
    await core.acompletion(model="g", messages=MESSAGES, metadata={"user_api_key_team_id": team_id})  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == [expected]
    assert host.calls[-1].model_group == "g"


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{}, {"user_api_key_team_id": "other"}])
async def test_team_only_public_name_is_unknown_to_other_callers(make_core: MakeCore, metadata: dict[str, str]) -> None:
    core: Final = make_core(
        model_list=[deployment("team-x-group", "team_x", model_info={"team_id": "tx", "team_public_model_name": "pub"})]
    )
    with pytest.raises(litellm.BadRequestError, match="model=pub"):
        await core.select("pub", {"metadata": metadata})
    assert await _picked_ids(core, "pub", request_kwargs={"metadata": {"user_api_key_team_id": "tx"}}) == {"team_x"}


@pytest.mark.parametrize(("team_id", "names"), [(None, []), ("tx", ["pub"])])
def test_team_public_names_are_routable_only_for_their_team(
    make_core: MakeCore, team_id: str | None, names: list[str]
) -> None:
    core: Final = make_core(
        model_list=[deployment("team-x-group", "team_x", model_info={"team_id": "tx", "team_public_model_name": "pub"})]
    )
    assert list(core.registry().names(kind="groups", team_id=team_id)) == names


# --- weights, rpm and tpm under simple-shuffle -----------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("metric", ["weight", "rpm", "tpm"])
async def test_simple_shuffle_never_picks_a_zero_weighted_deployment_beside_a_positive_one(
    make_core: MakeCore, metric: str
) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "unset"),
            deployment("g", "positive", **{metric: 5}),
            deployment("g", "zero", **{metric: 0}),
        ]
    )
    assert await _picked_ids(core, "g") == {"positive"}


@pytest.mark.asyncio
async def test_weight_takes_precedence_over_rpm_and_tpm(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "weighted", weight=1, rpm=0, tpm=0),
            deployment("g", "by_limits", weight=0, rpm=100, tpm=100),
        ]
    )
    assert await _picked_ids(core, "g") == {"weighted"}


@pytest.mark.asyncio
async def test_completion_calls_only_the_weighted_deployment(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", weight=0), deployment("g", "b", weight=3)])
    for _ in range(10):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert set(host.deployments_called("acompletion")) == {"b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("favoured", ["a", "b"])
async def test_request_router_weights_override_deployment_weights(make_core: MakeCore, favoured: str) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", weight=1), deployment("g", "b", weight=1)])
    picked: Final = await _picked_ids(core, "g", request_kwargs={"_router_weights": {"g": {favoured: 1.0}}})
    assert picked == {favoured}


# --- order --------------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_select_picks_only_the_lowest_order(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "second", order=2),
            deployment("g", "first_a", order=1),
            deployment("g", "first_b", order=1),
            deployment("g", "third", order=3),
        ]
    )
    assert await _picked_ids(core, "g") <= {"first_a", "first_b"}


@pytest.mark.asyncio
async def test_completion_calls_only_the_lowest_order(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "second", order=2), deployment("g", "first", order=1)])
    for _ in range(5):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == ["first"] * 5


@pytest.mark.asyncio
async def test_next_order_serves_while_the_lowest_order_cools_down(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "second", order=2), deployment("g", "first", order=1)],
        allowed_fails=0,
        num_retries=0,
        cooldown_time=60,
    )
    host.answer("openai/first", rate_limit())
    await _fail_through(core, "first")
    assert await _picked_ids(core, "g") == {"second"}


# --- routing strategies ----------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("routing_strategy", ["least-busy", "usage-based-routing"])
@pytest.mark.parametrize("ids", [("a", "b"), ("b", "a")])
async def test_strategy_without_traffic_picks_the_first_configured_deployment(
    make_core: MakeCore, routing_strategy: str, ids: tuple[str, str]
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", deployment_id) for deployment_id in ids], routing_strategy=routing_strategy
    )
    assert await _picked_ids(core, "g", messages=MESSAGES) == {ids[0]}


@pytest.mark.asyncio
@pytest.mark.parametrize("cheap_first", [True, False])
async def test_cost_based_routing_picks_the_cheapest_deployment(make_core: MakeCore, cheap_first: bool) -> None:
    cheap: Final = deployment("g", "cheap", input_cost_per_token=0.000001, output_cost_per_token=0.000002)
    pricey: Final = deployment("g", "pricey", input_cost_per_token=0.00001, output_cost_per_token=0.00002)
    core: Final = make_core(
        model_list=[cheap, pricey] if cheap_first else [pricey, cheap], routing_strategy="cost-based-routing"
    )
    assert await _picked_ids(core, "g", messages=MESSAGES) == {"cheap"}


@pytest.mark.asyncio
@pytest.mark.parametrize("ids", [("tight", "roomy"), ("roomy", "tight")])
async def test_usage_based_v2_skips_a_deployment_whose_tpm_the_request_exceeds(
    make_core: MakeCore, ids: tuple[str, str]
) -> None:
    limits: Final = {"tight": 1, "roomy": 100_000}
    core: Final = make_core(
        model_list=[deployment("g", deployment_id, tpm=limits[deployment_id]) for deployment_id in ids],
        routing_strategy="usage-based-routing-v2",
    )
    assert await _picked_ids(core, "g", messages=MESSAGES) == {"roomy"}


@pytest.mark.asyncio
async def test_usage_based_v2_rate_limits_when_the_request_exceeds_every_tpm(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a", tpm=1), deployment("g", "b", tpm=1)],
        routing_strategy="usage-based-routing-v2",
    )
    with pytest.raises(litellm.RateLimitError, match="No deployments available for selected model. Passed model=g"):
        await core.select("g", messages=MESSAGES)  # pyright: ignore[reportCallIssue]


# --- cooldowns ----------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [rate_limit, auth_error, server_error, timeout], ids=lambda e: e.__name__)
async def test_a_deployment_that_failed_past_allowed_fails_is_not_picked(
    make_core: MakeCore, host: ScriptedHost, error: Callable[[], Exception]
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0, num_retries=0, cooldown_time=60
    )
    host.answer("openai/a", error())
    await _fail_through(core, "a")
    assert await _picked_ids(core, "g") == {"b"}


@pytest.mark.asyncio
async def test_completion_avoids_a_cooling_deployment(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0, num_retries=0, cooldown_time=60
    )
    host.answer("openai/a", rate_limit())
    await _fail_through(core, "a")
    for _ in range(5):
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called("acompletion") == ["a", "b", "b", "b", "b", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [bad_request, context_window], ids=lambda e: e.__name__)
async def test_client_errors_do_not_cool_a_deployment_down(
    make_core: MakeCore, host: ScriptedHost, error: Callable[[], Exception]
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0, num_retries=0, cooldown_time=60
    )
    host.answer("openai/a", error())
    await _fail_through(core, "a", times=3)
    assert await core.select("a") is not None
    assert "a" in await _picked_ids(core, "g", times=60)


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed_fails", [1, 2, 3])
async def test_a_deployment_stays_pickable_until_it_fails_more_than_allowed_fails(
    make_core: MakeCore, host: ScriptedHost, allowed_fails: int
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")],
        allowed_fails=allowed_fails,
        num_retries=0,
        cooldown_time=60,
    )
    host.answer("openai/a", rate_limit())
    await _fail_through(core, "a", times=allowed_fails)
    assert "a" in await _picked_ids(core, "g", times=60)
    await _fail_through(core, "a")
    assert await _picked_ids(core, "g") == {"b"}


@pytest.mark.asyncio
async def test_a_single_deployment_group_is_never_cooled_down(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], allowed_fails=0, num_retries=0, cooldown_time=60)
    host.answer("openai/a", rate_limit())
    await _fail_through(core, "a", times=3)
    assert await _picked_ids(core, "g") == {"a"}


@pytest.mark.asyncio
async def test_selecting_a_cooling_deployment_by_id_still_returns_it(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0, num_retries=0, cooldown_time=60
    )
    host.answer("openai/a", rate_limit())
    await _fail_through(core, "a")
    assert await _picked_ids(core, "g") == {"b"}
    assert await _picked_ids(core, "a") == {"a"}


async def _all_cooling_core(make_core: MakeCore, host: ScriptedHost) -> RouterCore:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], allowed_fails=0, num_retries=0, cooldown_time=60
    )
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())
    await _fail_through(core, "a")
    await _fail_through(core, "b")
    return core


ALL_COOLING_MESSAGE: Final = (
    "No deployments available for selected model, Try again in 60 seconds. All deployments for selected "
    "model are in cooldown. Passed model=g. pre-call-checks=False, cooldown_list=['a', 'b']"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["select", "acompletion", "completion"])
async def test_every_deployment_cooling_raises_router_rate_limit_error(
    make_core: MakeCore, host: ScriptedHost, entry: str
) -> None:
    core: Final = await _all_cooling_core(make_core, host)
    calls_before: Final = len(host.calls)
    with pytest.raises(RouterRateLimitError) as raised:
        if entry == "select":
            await core.select("g")
        else:
            await core.call(entry, {"model": "g", "messages": MESSAGES})
    assert str(raised.value) == ALL_COOLING_MESSAGE
    assert raised.value.model == "g"
    assert raised.value.cooldown_time == 60
    assert sorted(raised.value.cooldown_list) == ["a", "b"]
    assert len(host.calls) == calls_before


@pytest.mark.asyncio
async def test_cooling_down_one_group_leaves_other_groups_pickable(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b"), deployment("h", "c"), deployment("h", "d")],
        allowed_fails=0,
        num_retries=0,
        cooldown_time=60,
    )
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())
    await _fail_through(core, "a")
    await _fail_through(core, "b")
    with pytest.raises(RouterRateLimitError):
        await core.select("g")
    assert await _picked_ids(core, "h") <= {"c", "d"}
