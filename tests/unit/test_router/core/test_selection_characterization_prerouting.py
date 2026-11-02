"""
Characterization of what runs before a router core's strategy picks: pre-routing strategies (the
complexity router, chosen among tag-scoped markers), the strategy marker deployments they are
registered from, tier ``litellm_params`` pinned onto the request, alias-marker params forwarded to
the routed call, the Claude Code session router binding, and the ``Router(plugins=[...])``
routing-plugin pipeline.
"""

import asyncio
import re
from collections.abc import Mapping
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.types.router import RoutingContext
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment
from tests.unit.test_router.core.selection_helpers import MESSAGES, MakeCore, picked_ids

HARD_MESSAGES: Final = [
    {
        "role": "user",
        "content": (
            "Design a distributed consensus algorithm, prove its correctness step by step, analyze the time "
            "complexity, and implement it in Rust with a kubernetes deployment and database sharding strategy. "
            "Think step by step and reason about edge cases."
        ),
    }
]

TIERS: Final = {"SIMPLE": "small", "MEDIUM": "big", "COMPLEX": "big", "REASONING": "big"}
ALL_BIG: Final = {"SIMPLE": "big", "MEDIUM": "big", "COMPLEX": "big", "REASONING": "big"}

ONLY_MARKERS_MESSAGE: Final = (
    "litellm.BadRequestError: You passed in model={model}. Every deployment for it is a strategy router marker "
    "(auto_router/...), which is not a callable model, and no pre-routing strategy selected a deployment for "
    "this request"
)

SESSION_ID: Final = "0f8fad5b-d9cb-469f-a165-70867728950e"


def _complexity_router(
    name: str = "smart",
    tiers: Mapping[str, object] = TIERS,
    deployment_id: str = "marker",
    **params: object,  # kwargs-ok: extra marker litellm_params
) -> dict[str, object]:  # mutable-ok: a model_list entry
    return {
        "model_name": name,
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": dict(tiers)},
            "complexity_router_default_model": "big",
            **params,
        },
        "model_info": {"id": deployment_id},
    }


def _tiered(
    **marker_params: object,
) -> list[dict[str, object]]:  # mutable-ok: a model_list  # kwargs-ok: marker litellm_params
    return [deployment("small", "s"), deployment("big", "b"), _complexity_router(**marker_params)]


def _sync_entry(core: RouterCore, entry: str, model: str) -> object:
    if entry == "completion":
        return core.completion(model=model, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    return core.get_available_deployment(model, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]


def _metadata(request: Mapping[str, object]) -> Mapping[str, object]:
    metadata: Final = request.get("metadata")
    assert isinstance(metadata, Mapping)
    return metadata


async def _routed_metadata(
    core: RouterCore,
    host: ScriptedHost,
    model: str,
    messages: list[dict[str, str]],
    **request: object,  # kwargs-ok: the request's own keyword payload
) -> Mapping[str, object]:
    """The metadata the routed call carried: ``select`` leaves its caller's request alone, so the
    routing stamps are read where they land, on the request the router sends."""
    await core.acompletion(model=model, messages=messages, **request)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    return _metadata(host.calls[-1].kwargs)


# --- the complexity router ------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("messages", "expected"), [(MESSAGES, "s"), (HARD_MESSAGES, "b")], ids=["simple", "hard"])
async def test_a_complexity_router_picks_the_tier_deployment_for_the_prompt(
    make_core: MakeCore, messages: list[dict[str, str]], expected: str
) -> None:
    core: Final = make_core(model_list=_tiered())
    chosen: Final = await core.select("smart", {}, messages=messages)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == expected  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_a_complexity_router_pick_stamps_the_routing_decision_on_the_routed_request(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_tiered())
    metadata: Final = await _routed_metadata(core, host, "smart", MESSAGES)
    decision: Final = metadata["routing_decision"]
    assert isinstance(decision, Mapping)
    assert decision["router_model_name"] == "smart"
    assert decision["router_type"] == "complexity"
    assert decision["router_config_id"] == "marker"
    assert decision["routed_model"] == "small"
    assert decision["tier"] == "SIMPLE"
    assert metadata["pre_routing_selected_model"] == "small"


@pytest.mark.asyncio
async def test_completion_through_a_complexity_router_reports_the_router_name_as_model_group(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_tiered())
    await core.acompletion(model="smart", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["s"]
    assert host.calls[-1].model == "openai/s"
    assert host.calls[-1].model_group == "smart"


@pytest.mark.asyncio
async def test_an_alias_to_a_complexity_router_routes_through_it(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=_tiered(), model_group_alias={"al": "smart"})
    chosen: Final = await core.select("al", {}, messages=MESSAGES)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
    assert chosen["model_info"]["id"] == "s"  # pyright: ignore[reportIndexIssue]
    metadata: Final = await _routed_metadata(core, host, "al", MESSAGES)
    assert metadata["routing_decision"]["router_model_name"] == "smart"  # pyright: ignore[reportIndexIssue]
    assert host.calls[-1].deployment_id == "s"
    assert host.calls[-1].model_group == "al"


@pytest.mark.asyncio
async def test_a_complexity_router_takes_precedence_over_plain_deployments_of_its_name(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[*_tiered(), deployment("smart", "plain")])
    assert await picked_ids(core, "smart", times=10, messages=MESSAGES) == {"s"}


@pytest.mark.asyncio
async def test_a_complexity_router_tier_naming_an_unknown_model_falls_back_to_the_default_model(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("big", "b"), _complexity_router(tiers={**TIERS, "SIMPLE": "missing"})]
    )
    chosen: Final = await core.select("smart", {}, messages=MESSAGES)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]
    decision: Final = (await _routed_metadata(core, host, "smart", MESSAGES))["routing_decision"]
    assert isinstance(decision, Mapping)
    assert decision["cause"] == "health_default_fallback"
    assert decision["routed_model"] == "big"


def test_a_complexity_router_without_tiers_or_a_default_model_is_rejected(make_core: MakeCore) -> None:
    with pytest.raises(ValueError, match="complexity_router_default_model is required for complexity-router"):
        make_core(model_list=[deployment("smart", "marker", model="auto_router/complexity_router")])


# --- strategy markers are never callable ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_healthy_listing_of_a_router_name_refuses_the_marker(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("smart", {}, kind="healthy")
    assert str(raised.value) == ONLY_MARKERS_MESSAGE.format(model="smart")


@pytest.mark.asyncio
async def test_pass_through_selection_of_a_router_name_refuses_the_marker(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("smart", {}, kind="pass_through", messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert str(raised.value) == ONLY_MARKERS_MESSAGE.format(model="smart")


@pytest.mark.asyncio
async def test_selecting_a_marker_by_its_id_refuses_it_naming_its_litellm_model(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    with pytest.raises(litellm.BadRequestError) as raised:
        await core.select("marker", {}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert str(raised.value) == ONLY_MARKERS_MESSAGE.format(model="auto_router/complexity_router")


@pytest.mark.parametrize("model", ["smart", "al"])  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]  # pytest and the cores' untyped members
def test_sync_get_available_deployment_runs_pre_routing_like_the_async_pick(make_core: MakeCore, model: str) -> None:
    core: Final = make_core(model_list=_tiered(), model_group_alias={"al": "smart"})
    request: Final[dict[str, object]] = {}
    chosen: Final = core.get_available_deployment(model, messages=MESSAGES, request_kwargs=request)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]
    assert chosen["model_info"]["id"] == "s"
    assert _metadata(request)["routing_decision"]["router_model_name"] == "smart"  # pyright: ignore[reportIndexIssue]


@pytest.mark.parametrize(("model", "messages", "expected"), [("smart", MESSAGES, "s"), ("al", HARD_MESSAGES, "b")])  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]  # pytest and the cores' untyped members
def test_sync_completion_routes_through_a_complexity_router_like_acompletion(
    make_core: MakeCore, host: ScriptedHost, model: str, messages: list[dict[str, str]], expected: str
) -> None:
    core: Final = make_core(model_list=_tiered(), model_group_alias={"al": "smart"})
    core.completion(model=model, messages=messages)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
    assert host.deployments_called() == [expected]
    assert host.calls[-1].model_group == model


# --- choosing among tag-scoped markers -----------------------------------------------------------------

TAG_SCOPED: Final = [
    deployment("small", "s"),
    deployment("big", "b"),
    _complexity_router(deployment_id="cheap_marker", tags=["cheap"]),
    _complexity_router(tiers=ALL_BIG, deployment_id="premium_marker", tags=["premium"]),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_kwargs", "expected"),
    [({"metadata": {"tags": ["cheap"]}}, "s"), ({"metadata": {"tags": ["premium"]}}, "b"), ({}, "s")],
    ids=["cheap-tag", "premium-tag", "untagged-takes-the-first"],
)
async def test_request_tags_pick_among_markers_of_one_name(
    make_core: MakeCore, request_kwargs: dict[str, object], expected: str
) -> None:
    core: Final = make_core(model_list=list(TAG_SCOPED))
    chosen: Final = await core.select("smart", request_kwargs, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == expected  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_an_untagged_request_takes_the_default_tagged_marker(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("small", "s"),
            deployment("big", "b"),
            _complexity_router(deployment_id="cheap_marker", tags=["cheap"]),
            _complexity_router(tiers=ALL_BIG, deployment_id="default_marker", tags=["default"]),
        ]
    )
    chosen: Final = await core.select("smart", {}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]


SCOPED_WITH_PLAIN: Final = [
    deployment("small", "s"),
    deployment("big", "b"),
    _complexity_router(deployment_id="cheap_marker", tags=["cheap"]),
    deployment("smart", "plain"),
]


@pytest.mark.asyncio
async def test_with_tag_filtering_an_unmatched_tag_skips_the_markers_for_plain_tag_selection(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(model_list=list(SCOPED_WITH_PLAIN), enable_tag_filtering=True)
    with pytest.raises(
        ValueError, match=re.escape("Not allowed to access model due to tags configuration. Passed model=smart")
    ):
        await core.select("smart", {"metadata": {"tags": ["other"]}}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    chosen: Final = await core.select("smart", {"metadata": {"tags": ["cheap"]}}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "s"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_without_tag_filtering_an_unmatched_tag_still_takes_the_first_marker(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(SCOPED_WITH_PLAIN))
    chosen: Final = await core.select("smart", {"metadata": {"tags": ["other"]}}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "s"  # pyright: ignore[reportIndexIssue]


# --- tier litellm_params and alias-marker params -------------------------------------------------------------

PINNED_TIERS: Final = {
    **TIERS,
    "SIMPLE": {"model_name": "small", "litellm_params": {"temperature": 0.1, "max_tokens": 7}},
}


@pytest.mark.asyncio
async def test_tier_litellm_params_override_the_callers_params_on_the_call(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_tiered(tiers=PINNED_TIERS))
    await core.acompletion(model="smart", messages=MESSAGES, temperature=0.9, max_completion_tokens=100)  # pyright: ignore[reportAttributeAccessIssue]
    sent: Final = host.calls[-1].kwargs
    assert host.deployments_called() == ["s"]
    assert sent["temperature"] == 0.1
    assert sent["max_tokens"] == 7
    assert "max_completion_tokens" not in sent


@pytest.mark.asyncio
async def test_select_leaves_the_callers_request_without_the_tier_pins(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered(tiers=PINNED_TIERS))
    request: Final[dict[str, object]] = {"temperature": 0.9, "max_completion_tokens": 100}
    picked: Final = await core.select("smart", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
    assert picked["model_info"]["id"] == "s"  # pyright: ignore[reportIndexIssue]  # a deployment dict
    assert request == {"temperature": 0.9, "max_completion_tokens": 100}


@pytest.mark.asyncio  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]  # untyped in the checker's view
async def test_select_leaves_the_callers_request_and_metadata_without_marker_params_or_stamps(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(model_list=_tiered(temperature=0.3))
    metadata: Final[dict[str, object]] = {"tags": ["t"]}
    request: Final[dict[str, object]] = {"metadata": metadata, "specific_deployment": False}
    await core.select("smart", request, messages=HARD_MESSAGES)  # pyright: ignore[reportCallIssue]
    assert request == {"metadata": {"tags": ["t"]}, "specific_deployment": False}
    assert request["metadata"] is metadata
    assert metadata == {"tags": ["t"]}


@pytest.mark.asyncio
async def test_tier_litellm_params_apply_only_to_their_tier(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=_tiered(tiers=PINNED_TIERS))
    await core.acompletion(model="smart", messages=HARD_MESSAGES, temperature=0.9)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["b"]
    assert host.calls[-1].kwargs["temperature"] == 0.9
    assert "max_tokens" not in host.calls[-1].kwargs


@pytest.mark.asyncio
async def test_marker_params_fill_gaps_on_the_routed_call_but_never_override_the_caller(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_tiered(temperature=0.3))
    await core.acompletion(model="smart", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls[-1].kwargs["temperature"] == 0.3
    await core.acompletion(model="smart", messages=MESSAGES, temperature=0.9)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls[-1].kwargs["temperature"] == 0.9


@pytest.mark.asyncio
async def test_marker_credentials_and_pricing_are_never_forwarded(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=_tiered(api_base="http://marker.example", input_cost_per_token=0.5))
    await core.acompletion(model="smart", messages=HARD_MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    sent: Final = host.calls[-1].kwargs
    assert host.deployments_called() == ["b"]
    assert sent.get("api_base") is None
    assert "input_cost_per_token" not in sent


@pytest.mark.asyncio
async def test_a_deployments_own_param_beats_the_forwarded_marker_param(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[
            deployment("small", "s", aws_region_name="deployment-region"),
            deployment("big", "b"),
            _complexity_router(aws_region_name="marker-region"),
        ]
    )
    await core.acompletion(model="smart", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls[-1].kwargs["aws_region_name"] == "deployment-region"
    await core.acompletion(model="smart", messages=HARD_MESSAGES)  # pyright: ignore[reportAttributeAccessIssue]
    assert host.calls[-1].kwargs["aws_region_name"] == "marker-region"


# --- the Claude Code session router binding --------------------------------------------------------------------


def _session_request(
    agent_id: str | None = None, caller: str = "caller-hash"
) -> dict[str, object]:  # mutable-ok: request kwargs
    headers: Final[dict[str, str]] = {"x-claude-code-session-id": SESSION_ID}
    if agent_id is not None:
        headers["x-claude-code-agent-id"] = agent_id
    return {"proxy_server_request": {"headers": headers}, "metadata": {"user_api_key_hash": caller}}


@pytest.mark.asyncio
async def test_a_subagent_request_follows_the_router_its_session_started_on(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=_tiered())
    await core.select("smart", _session_request(), messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    chosen: Final = await core.select("big", _session_request(agent_id="agent-1"), messages=MESSAGES)  # pyright: ignore[reportCallIssue, reportUnknownVariableType]
    assert chosen["model_info"]["id"] == "s"  # pyright: ignore[reportIndexIssue]
    metadata: Final = await _routed_metadata(core, host, "big", MESSAGES, **_session_request(agent_id="agent-1"))
    assert host.calls[-1].deployment_id == "s"
    assert metadata["model_group"] == "smart"


@pytest.mark.asyncio
async def test_a_session_binding_is_scoped_to_the_caller(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    await core.select("smart", _session_request(), messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    other_caller: Final = _session_request(agent_id="agent-1", caller="other-hash")
    chosen: Final = await core.select("big", other_caller, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_a_main_agent_request_is_not_redirected_by_its_session_binding(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    await core.select("smart", _session_request(), messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    chosen: Final = await core.select("big", _session_request(), messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_a_subagent_request_without_a_session_binding_keeps_its_model(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=_tiered())
    chosen: Final = await core.select("big", _session_request(agent_id="agent-1"), messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["short", "has spaces in it"])
async def test_an_invalid_session_id_never_binds(make_core: MakeCore, session_id: str) -> None:
    core: Final = make_core(model_list=_tiered())
    first: Final = _session_request()
    first["proxy_server_request"] = {"headers": {"x-claude-code-session-id": session_id}}
    await core.select("smart", first, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    subagent: Final = _session_request(agent_id="agent-1")
    subagent["proxy_server_request"] = {
        "headers": {"x-claude-code-session-id": session_id, "x-claude-code-agent-id": "agent-1"}
    }
    chosen: Final = await core.select("big", subagent, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "b"  # pyright: ignore[reportIndexIssue]


# --- the routing-plugin pipeline ------------------------------------------------------------------------------


class Narrowing:
    """Keeps only ``keep`` among the candidate models and records what it was handed."""

    def __init__(self, keep: frozenset[str]) -> None:
        self.keep: Final = keep
        self.seen: list[list[str]] = []  # mutable-ok: the candidate lists each run saw

    async def run(self, context: RoutingContext) -> RoutingContext:
        self.seen.append(list(context.candidate_models))
        context.candidate_models = [model for model in context.candidate_models if model in self.keep]
        context.signals["narrowing"] = {"kept": sorted(self.keep)}
        return context


TWO: Final = [deployment("g", "a"), deployment("g", "b")]


@pytest.mark.asyncio
async def test_a_routing_plugin_narrows_the_candidates_to_the_models_it_keeps(make_core: MakeCore) -> None:
    plugin: Final = Narrowing(frozenset({"openai/b"}))
    core: Final = make_core(model_list=list(TWO), plugins=[plugin])
    assert await picked_ids(core, "g", messages=MESSAGES) == {"b"}
    assert plugin.seen[0] == ["openai/a", "openai/b"]


@pytest.mark.asyncio
async def test_routing_plugin_signals_and_narrowed_candidates_are_stamped_on_the_routed_request(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(TWO), plugins=[Narrowing(frozenset({"openai/b"}))])
    metadata: Final = await _routed_metadata(core, host, "g", MESSAGES)
    assert metadata["routing_plugin_signals"] == {"narrowing": {"kept": ["openai/b"]}}
    assert metadata["_routing_plugin_candidate_models"] == ["openai/b"]


@pytest.mark.asyncio
async def test_a_routing_plugin_that_keeps_every_candidate_stamps_only_its_signals(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(TWO), plugins=[Narrowing(frozenset({"openai/a", "openai/b"}))])
    metadata: Final = await _routed_metadata(core, host, "g", MESSAGES)
    assert metadata["routing_plugin_signals"] == {"narrowing": {"kept": ["openai/a", "openai/b"]}}
    assert "_routing_plugin_candidate_models" not in metadata


@pytest.mark.asyncio  # pyright: ignore[reportUnknownMemberType, reportUntypedFunctionDecorator]  # untyped in the checker's view
async def test_select_leaves_the_callers_request_without_routing_plugin_stamps(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TWO), plugins=[Narrowing(frozenset({"openai/b"}))])
    request: Final[dict[str, object]] = {}
    await core.select("g", request, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert request == {}


@pytest.mark.asyncio
async def test_routing_plugins_run_in_order_each_seeing_the_previous_narrowing(make_core: MakeCore) -> None:
    first: Final = Narrowing(frozenset({"openai/b", "openai/c"}))
    second: Final = Narrowing(frozenset({"openai/c"}))
    core: Final = make_core(model_list=[*TWO, deployment("g", "c")], plugins=[first, second])
    assert await picked_ids(core, "g", times=1, messages=MESSAGES) == {"c"}
    assert second.seen == [["openai/b", "openai/c"]]


@pytest.mark.asyncio
async def test_a_routing_plugin_that_narrows_to_nothing_fails_the_pick(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=list(TWO), plugins=[Narrowing(frozenset())])
    with pytest.raises(ValueError, match="No deployments left") as raised:
        await core.select("g", {}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert str(raised.value) == "No deployments left after routing-plugin filtering. candidate_models=[]"


SYNC_PLUGIN_MESSAGE_START: Final = (
    "Router(plugins=[...]) is configured, but this call resolved to the synchronous deployment-selection path"
)


@pytest.mark.parametrize("entry", ["get_available_deployment", "completion"])
def test_routing_plugins_refuse_the_sync_selection_path(make_core: MakeCore, host: ScriptedHost, entry: str) -> None:
    core: Final = make_core(model_list=list(TWO), plugins=[Narrowing(frozenset({"openai/a"}))])
    with pytest.raises(ValueError, match=re.escape(SYNC_PLUGIN_MESSAGE_START)) as raised:
        _sync_entry(core, entry, "g")
    assert str(raised.value).startswith(SYNC_PLUGIN_MESSAGE_START)
    assert host.calls == []


@pytest.mark.asyncio
async def test_routing_plugins_refuse_usage_based_routing_v1_even_from_async_callers(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=list(TWO), plugins=[Narrowing(frozenset({"openai/a"}))], routing_strategy="usage-based-routing"
    )
    with pytest.raises(ValueError, match=re.escape(SYNC_PLUGIN_MESSAGE_START)) as raised:
        await core.select("g", {}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert str(raised.value).startswith(SYNC_PLUGIN_MESSAGE_START)


@pytest.mark.asyncio
async def test_a_routing_plugin_does_not_narrow_a_pick_by_deployment_id(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(TWO), plugins=[Narrowing(frozenset({"openai/b"}))])
    chosen: Final = await core.select("a", {}, messages=MESSAGES)  # pyright: ignore[reportCallIssue]
    assert chosen["model_info"]["id"] == "a"  # pyright: ignore[reportIndexIssue]


@pytest.mark.asyncio
async def test_completion_with_a_routing_plugin_calls_a_kept_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(TWO), plugins=[Narrowing(frozenset({"openai/a"}))])
    await asyncio.gather(*(core.acompletion(model="g", messages=MESSAGES) for _ in range(5)))  # pyright: ignore[reportAttributeAccessIssue]
    assert host.deployments_called() == ["a"] * 5
