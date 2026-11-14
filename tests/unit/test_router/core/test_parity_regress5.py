"""
Parity of the Rust core with ``router_python_core.py`` on the selection causes of the regress5 pack:
LAR-1 set after build, endpoint eligibility, web search, ``order`` in ``model_info``, ``priority``
through an alias, old-style custom strategies, wildcard filling and cache flags, ``specific_deployment``
read as a strict True, pass-through picks, affinity narrowing and a health scope given as a dict.
"""

import time
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

import litellm
from litellm.exceptions import EndpointNotSupportedByModelError
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.types.router import CustomRoutingStrategyBase, RouterRateLimitError
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment
from tests.unit.test_router.core.selection_helpers import MESSAGES, MakeCore, ids_of, picked_ids

SPREAD_PICKS: Final = 40
WEB_SEARCH_TOOLS: Final = [{"type": "web_search"}]
EMBEDDINGS_ONLY: Final = {"enforce_supported_endpoints": True, "supported_endpoints": ["/v1/embeddings"]}


def _embedding_group() -> Sequence[Mapping[str, object]]:
    return (
        deployment("g", "embonly", weight=1, model_info=dict(EMBEDDINGS_ONLY)),
        deployment("g", "chat", weight=0),
    )


def _web_group() -> Sequence[Mapping[str, object]]:
    return (
        deployment("g", "noweb", weight=1, model_info={"supports_web_search": False}),
        deployment("g", "web", weight=0),
    )


# --- V1: lar1 set after build ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lar1_set_after_build_serves_the_next_async_call(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("a", "a1")], num_retries=0)
    core.apply(ChangeSet(settings={"routing_strategy": "lar1"}))
    await core.acompletion(model="a", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("acompletion") == ["a1"]
    assert core.settings().routing_strategy == "lar1"


@pytest.mark.asyncio
async def test_lar1_set_after_build_picks_without_calling(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("a", "a1"), deployment("a", "a2")], num_retries=0)
    core.apply(ChangeSet(settings={"routing_strategy": "lar1"}))
    assert await picked_ids(core, "a", times=SPREAD_PICKS, request_kwargs={}) <= {"a1", "a2"}


def test_lar1_set_after_build_refuses_the_sync_path(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("a", "a1")], num_retries=0)
    core.apply(ChangeSet(settings={"routing_strategy": "lar1"}))
    with pytest.raises(NotImplementedError, match="LAR-1 routing only supports async routing"):
        core.completion(model="a", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint


@pytest.mark.asyncio
async def test_leaving_lar1_goes_back_to_the_router_strategy(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("a", "a1")], num_retries=0)
    core.apply(ChangeSet(settings={"routing_strategy": "lar1"}))
    core.apply(ChangeSet(settings={"routing_strategy": "simple-shuffle"}))
    core.completion(model="a", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("completion") == ["a1"]


# --- R4: endpoint eligibility ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_acompletion_skips_a_deployment_that_enforces_other_endpoints(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(_embedding_group()), num_retries=0)
    await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("acompletion") == ["chat"]


@pytest.mark.asyncio
async def test_atext_completion_skips_a_deployment_that_enforces_other_endpoints(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(_embedding_group()), num_retries=0)
    await core.atext_completion(model="g", prompt="hi")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("atext_completion") == ["chat"]


def test_sync_text_completion_skips_a_deployment_that_enforces_other_endpoints(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(_embedding_group()), num_retries=0)
    core.text_completion(model="g", prompt="hi")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    # The sync text path sends the group as ``model`` and no ``model_info``: the params name the deployment.
    assert [call.kwargs["weight"] for call in host.calls_to("text_completion")] == [0]


def test_sync_completion_skips_a_deployment_that_enforces_other_endpoints(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """``function_with_fallbacks`` is the async walk, whose ``endpoint_selection`` covers the sync attempt."""
    core: Final = make_core(model_list=list(_embedding_group()), num_retries=0)
    core.completion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("completion") == ["chat"]


@pytest.mark.asyncio
async def test_an_endpoint_outside_the_chat_table_keeps_every_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(_embedding_group()), num_retries=0)
    await core.aembedding(model="g", input="hi")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("aembedding") == ["embonly"]


@pytest.mark.asyncio
async def test_a_group_that_serves_no_chat_endpoint_names_what_it_serves(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "e1", model_info=dict(EMBEDDINGS_ONLY))],
        num_retries=0,
    )
    with pytest.raises(EndpointNotSupportedByModelError) as refused:
        await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert (
        refused.value.message
        == "/v1/chat/completions is not supported by model=g. Supported endpoints: /v1/embeddings"
    )


@pytest.mark.asyncio
async def test_the_refusal_falls_back_like_any_bad_request_naming_no_provider(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "e1", model_info=dict(EMBEDDINGS_ONLY)), deployment("h", "h1")],
        num_retries=0,
        fallbacks=[{"g": ["h"]}],
    )
    await core.acompletion(model="g", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("acompletion") == ["h1"]


@pytest.mark.asyncio
async def test_a_direct_pick_runs_no_endpoint_selection(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "e1", model_info=dict(EMBEDDINGS_ONLY))], num_retries=0)
    assert await picked_ids(core, "g", times=1) == {"e1"}


def test_routable_deployments_read_the_call_type(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(_embedding_group()), num_retries=0)
    routable: Final = core.registry().deployments("g", source="routable", call_type="completion")
    assert ids_of(list(routable)) == ["chat"]


# --- R5: web search -------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_web_search_request_skips_deployments_that_declare_no_support(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(_web_group()), num_retries=0)
    await core.acompletion(model="g", messages=MESSAGES, tools=WEB_SEARCH_TOOLS)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("acompletion") == ["web"]


def test_the_sync_path_runs_no_web_search_filter(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=list(_web_group()), num_retries=0)
    core.completion(model="g", messages=MESSAGES, tools=WEB_SEARCH_TOOLS)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("completion") == ["noweb"]


@pytest.mark.asyncio
async def test_no_deployment_with_web_search_is_a_no_deployments_error(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "noweb", model_info={"supports_web_search": False})],
        num_retries=0,
    )
    with pytest.raises(RouterRateLimitError):
        await core.acompletion(model="g", messages=MESSAGES, tools=WEB_SEARCH_TOOLS)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint


@pytest.mark.asyncio
async def test_the_healthy_listing_runs_the_web_search_filter(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=list(_web_group()), num_retries=0)
    listed: Final = await core.select("g", {"tools": WEB_SEARCH_TOOLS}, kind="healthy")  # pyright: ignore[reportCallIssue, reportArgumentType]  # PythonCore's select takes keyword options
    assert ids_of(listed) == ["web"]


# --- R6: order ------------------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first", "second"),
    [
        ({"model_info": {"order": 1}}, {"model_info": {"order": 2}}),
        ({"order": "1"}, {"order": "2"}),
        ({"model_info": {"order": "1"}}, {"model_info": {"order": "2"}}),
        ({"order": None, "model_info": {"order": 1}}, {"model_info": {"order": 2}}),
    ],
    ids=["model-info", "numeric-string", "model-info-numeric-string", "null-param-reads-model-info"],
)
async def test_order_is_read_where_get_deployment_order_reads_it(
    make_core: MakeCore, first: dict[str, object], second: dict[str, object]
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "first", weight=0, **first), deployment("g", "second", weight=1, **second)],
        num_retries=0,
    )
    assert await picked_ids(core, "g", times=SPREAD_PICKS) == {"first"}


# --- R10: priority through an alias ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_priority_through_an_alias_reads_the_target_group(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", drop_params=True)],
        num_retries=0,
        model_group_alias={"al": "g"},
    )
    await core.acompletion(model="al", messages=MESSAGES, priority="high")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("acompletion") == ["d1"]


# --- R11: old-style custom strategies -------------------------------------------------------------


class _OldStyleStrategy(CustomRoutingStrategyBase):
    """Written before picks took ``request_kwargs``."""

    def __init__(self, picked: Mapping[str, object]) -> None:
        self.picked: Final = picked

    async def async_get_available_deployment(  # pyright: ignore[reportIncompatibleMethodOverride]  # the old signature is the point
        self, model: str, messages: object = None, input: object = None, specific_deployment: bool | None = False
    ) -> Mapping[str, object]:
        return self.picked

    def get_available_deployment(  # pyright: ignore[reportIncompatibleMethodOverride]  # the old signature is the point
        self, model: str, messages: object = None, input: object = None, specific_deployment: bool | None = False
    ) -> Mapping[str, object]:
        return self.picked


def _with_old_style_strategy(make_core: MakeCore) -> RouterCore:
    picked: Final = deployment("b", "db")
    core: Final = make_core(model_list=[deployment("a", "da"), picked], num_retries=0)
    core.register_plugin(_OldStyleStrategy(picked))
    return core


@pytest.mark.asyncio
async def test_an_old_style_strategy_answers_a_pick_that_names_only_the_model(make_core: MakeCore) -> None:
    core: Final = _with_old_style_strategy(make_core)
    picked: Final = await core._core_async_get_available_deployment("b")  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportPrivateUsage]  # the member the deprecated pick returns
    assert ids_of(picked) == ["db"]


def test_an_old_style_strategy_answers_a_sync_pick_that_names_only_the_model(make_core: MakeCore) -> None:
    core: Final = _with_old_style_strategy(make_core)
    assert ids_of(core.get_available_deployment("b")) == ["db"]  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the sync pick


@pytest.mark.asyncio
async def test_an_endpoint_always_hands_the_strategy_its_request(make_core: MakeCore) -> None:
    core: Final = _with_old_style_strategy(make_core)
    with pytest.raises(TypeError, match="request_kwargs"):
        await core.acompletion(model="b", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint


# --- N1, N15: wildcards ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_alias_to_a_wildcard_fills_the_model_from_the_alias_target(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("anthropic/*", "w", model="anthropic/*")],
        num_retries=0,
        model_group_alias={"al": "anthropic/claude-3-haiku"},
    )
    await core.acompletion(model="al", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.calls_to("acompletion")[0].model == "anthropic/claude-3-haiku"


@pytest.mark.asyncio
async def test_a_name_no_pattern_matches_is_tried_under_its_provider(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("openai/*", "w0", model="openai/*")], num_retries=0)
    listed: Final = await core.select("gpt-4o", {"user": "u"}, kind="healthy")  # pyright: ignore[reportCallIssue, reportArgumentType]  # PythonCore's select takes keyword options
    assert ids_of(listed) == ["w0"]
    assert isinstance(listed, list)
    assert listed[0]["litellm_params"]["model"] == "openai/gpt-4o"


@pytest.mark.asyncio
async def test_a_call_through_the_provider_spelling_is_served(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("openai/*", "w0", model="openai/*")], num_retries=0)
    await core.acompletion(model="gpt-4o", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.calls_to("acompletion")[0].model == "openai/gpt-4o"


# --- #49685: wildcard cache flags -----------------------------------------------------------------

CACHE_FLAGS: Final = {"supports_prompt_cache_breakpoint": True, "supports_prompt_caching": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pattern", "request_model", "kept"),
    [
        ("openai/*", "openai/gpt-4o", {}),
        ("azure_ai/*", "azure_ai/gpt-4o", {"supports_prompt_cache_breakpoint": True}),
    ],
    ids=["any-wildcard", "azure-ai-keeps-the-breakpoint-flag"],
)
async def test_a_wildcard_leaves_the_cache_flags_to_the_catalog(
    make_core: MakeCore, host: ScriptedHost, pattern: str, request_model: str, kept: Mapping[str, object]
) -> None:
    core: Final = make_core(
        model_list=[deployment(pattern, "w", model=pattern, model_info=dict(CACHE_FLAGS))],
        num_retries=0,
    )
    await core.acompletion(model=request_model, messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    model_info: Final = host.calls_to("acompletion")[0].kwargs["model_info"]
    assert isinstance(model_info, Mapping)
    assert {flag: model_info[flag] for flag in CACHE_FLAGS if flag in model_info} == kept


@pytest.mark.asyncio
async def test_a_wildcard_group_with_a_fixed_model_keeps_its_cache_flags(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("openai/*", "w", model="openai/gpt-4o", model_info=dict(CACHE_FLAGS))],
        num_retries=0,
    )
    await core.acompletion(model="openai/whatever", messages=MESSAGES)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    model_info: Final = host.calls_to("acompletion")[0].kwargs["model_info"]
    assert isinstance(model_info, Mapping)
    assert model_info["supports_prompt_caching"] is True


def test_the_wildcard_route_lookup_leaves_the_cache_flags_to_the_catalog(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("openai/*", "w", model="openai/*", model_info=dict(CACHE_FLAGS))],
        num_retries=0,
    )
    routed: Final = core.registry().deployments("openai/gpt-4o", source="wildcard")
    assert not set(CACHE_FLAGS) & set(routed[0]["model_info"])


# --- N12: specific_deployment ---------------------------------------------------------------------


def test_only_true_names_a_litellm_model(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("gpt-4o", "d1")], num_retries=0)
    with pytest.raises(litellm.BadRequestError):
        core.get_available_deployment(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the sync pick
            model="gpt-4o ", specific_deployment="yes", request_kwargs={"input": "hi"}
        )


# --- N14: pass-through picks ----------------------------------------------------------------------

COMPLEXITY_GROUP: Final = [
    deployment("g1", "b-1", model="openai/gpt-4o", use_in_pass_through=True),
    deployment("g2", "b-2", model="openai/gpt-4o-mini", use_in_pass_through=True),
    {
        "model_name": "smart",
        "litellm_params": {
            "model": "auto_router/complexity_router",
            "complexity_router_config": {"tiers": {"SIMPLE": "g2", "REASONING": "g1"}},
            "complexity_router_default_model": "g2",
        },
    },
]


@pytest.mark.asyncio
async def test_a_listed_pass_through_pick_names_the_group_the_alias_resolved_to(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("team-gpt", "d1")],
        num_retries=0,
        model_group_alias={"alias-b": "team-gpt"},
    )
    with pytest.raises(litellm.BadRequestError, match="Model team-gpt has no deployment"):
        await core.select("alias-b", {}, kind="pass_through")  # pyright: ignore[reportCallIssue]  # PythonCore's select takes keyword options


@pytest.mark.asyncio
async def test_an_async_pass_through_pick_of_a_strategy_marker_group_takes_the_tier(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=list(COMPLEXITY_GROUP), num_retries=0, routing_strategy="usage-based-routing-v2"
    )
    picked: Final = await core.async_get_available_deployment_for_pass_through(  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # the async pass-through pick
        "smart", {}, messages=[{"role": "user", "content": "what is up"}]
    )
    assert ids_of(picked) == ["b-2"]


# --- N16: affinity --------------------------------------------------------------------------------

ENCRYPTED_INPUT: Final = [
    {"role": "user", "content": "hi"},
    {
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": "thought"}],
        "encrypted_content": "litellm_enc:bW9kZWxfaWQ6bTktMg==;gAAAAABenc",
    },
    {"type": "reasoning", "encrypted_content": "litellm_enc:bW9kZWxfaWQ6bTktMQ==;gAAAAABenc", "summary": []},
]


@pytest.mark.asyncio
async def test_encrypted_affinity_reads_the_ids_the_pick_excludes(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("h", "m9-0", order=1),
            deployment("h", "m9-1", order=1),
            deployment("g", "m9-2"),
        ],
        num_retries=0,
        model_group_affinity_config={"h": ["encrypted_content_affinity"]},
    )
    request: Final = {
        "input": list(ENCRYPTED_INPUT),
        "_excluded_deployment_ids": ["m9-1"],
        "_retry_skipped_deployment_ids": ["m9-2", "m9-0"],
    }
    assert ids_of(await core.select("h", request, input=list(ENCRYPTED_INPUT))) == ["m9-0"]  # pyright: ignore[reportCallIssue, reportArgumentType]  # PythonCore's select takes keyword options


@pytest.mark.asyncio
async def test_a_target_order_on_a_call_narrows_its_pick(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("gpt", "id-a"), deployment("gpt", "id-b")],
        num_retries=0,
        optional_pre_call_checks=["session_affinity", "deployment_affinity"],
    )
    with pytest.raises(RouterRateLimitError):
        await core.acompletion(model="gpt", messages=MESSAGES, _target_order=1)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint


@pytest.mark.asyncio
async def test_a_target_order_on_a_call_serves_that_level(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("gpt", "low", order=1), deployment("gpt", "high", order=2)],
        num_retries=0,
    )
    await core.acompletion(model="gpt", messages=MESSAGES, _target_order=2)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]  # an endpoint
    assert host.deployments_called("acompletion") == ["high"]
    assert "_target_order" not in host.calls_to("acompletion")[0].kwargs


# --- S7: a health scope as a dict -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_health_scope_given_as_a_dict_reads_its_keys(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", weight=1), deployment("g", "d2", weight=0), deployment("h", "h1")],
        num_retries=0,
        enable_health_check_routing=True,
        background_health_check_model_groups={"h": True},
    )
    assert core.settings().background_health_check_model_groups == frozenset({"h"})
    core.health().health_checks.set_deployment_health_states(  # pyright: ignore[reportArgumentType]  # health states
        {"d1": {"is_healthy": False, "timestamp": time.time(), "reason": "probe failed"}}
    )
    assert await picked_ids(core, "g", times=SPREAD_PICKS) == {"d1"}
