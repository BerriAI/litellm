"""
The fallback walk over what the Python router walks and a smaller walk would miss: order levels
behind any name ``_core_get_model_list`` lists, errors raised while picking, fallback checks that
raise, dict targets naming no group, weighted failover past an affinity pin, and the encrypted
reasoning a fallback hop strips.
"""

from collections.abc import Awaitable, Callable, Mapping
from typing import Final

import pytest

import litellm
from litellm.router_core import RouterCore
from litellm.router_utils.pre_call_checks.deployment_affinity_check import DeploymentAffinityCheck
from litellm.types.router import RoutingContext as PluginContext
from tests.unit.test_router.core.fallbacks_helpers import (
    MESSAGES,
    MakeCore,
    build,
    complete,
    fails_once_then_answers,
    headers,
    show_router_debug,
)
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    completion_response,
    deployment,
    rate_limit,
    server_error,
)

TEAM_METADATA: Final = {"user_api_key_team_id": "T1"}
T1_PUBLIC: Final = {"team_id": "T1", "team_public_model_name": "tp"}
T2_PUBLIC: Final = {"team_id": "T2", "team_public_model_name": "tp"}
INTERLEAVED_TEAM_TIERS: Final = [
    deployment("t1-first", "t1-first", order=1, model_info=T1_PUBLIC),
    deployment("t2-first", "t2-first", order=2, model_info=T2_PUBLIC),
    deployment("t1-second", "t1-second", order=3, model_info=T1_PUBLIC),
    deployment("t2-second", "t2-second", order=4, model_info=T2_PUBLIC),
]

# --- order levels behind a name -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_order_levels_walk_through_a_wildcard(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = build(
        make_core,
        model_list=[
            deployment("openai/*", "w1", model="openai/*", order=1),
            deployment("openai/*", "w2", model="openai/*", order=2),
        ],
    )
    host.answer("openai/gpt-4o", rate_limit(), completion_response())

    await complete(core, "openai/gpt-4o")

    assert host.deployments_called("acompletion") == ["w1", "w2"]


@pytest.mark.asyncio
async def test_order_levels_walk_through_an_alias(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = build(
        make_core,
        model_list=[deployment("gg", "d1", order=1), deployment("gg", "d2", order=2)],
        model_group_alias={"al": "gg"},
    )
    host.answer("openai/d1", rate_limit())

    await complete(core, "al")

    assert host.deployments_called("acompletion") == ["d1", "d2"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("buckets", "expected"),
    [
        ({"metadata": {"user_api_key_team_id": "T1"}}, ["t1-first", "t1-second"]),
        ({"litellm_metadata": {"user_api_key_team_id": "T1"}}, ["t1-first", "t1-second"]),
        ({"metadata": {}, "litellm_metadata": {"user_api_key_team_id": "T1"}}, ["t1-first", "t1-second"]),
        (
            {"metadata": {"user_api_key_team_id": None}, "litellm_metadata": {"user_api_key_team_id": "T1"}},
            ["t1-first", "t1-second"],
        ),
        (
            {"metadata": {"user_api_key_team_id": ""}, "litellm_metadata": {"user_api_key_team_id": "T1"}},
            ["t1-first", "t1-second"],
        ),
        (
            {"metadata": {"user_api_key_team_id": 7}, "litellm_metadata": {"user_api_key_team_id": "T1"}},
            ["t1-first", "t1-second"],
        ),
        (
            {"metadata": {"user_api_key_team_id": "T2"}, "litellm_metadata": {"user_api_key_team_id": "T1"}},
            ["t2-first", "t2-second"],
        ),
    ],
    ids=[
        "metadata",
        "litellm-metadata",
        "empty-metadata-beside-litellm-metadata",
        "teamless-metadata-beside-litellm-metadata",
        "empty-string-metadata-team-beside-litellm-metadata",
        "numeric-metadata-team-beside-litellm-metadata",
        "metadata-team-wins-over-litellm-metadata-team",
    ],
)
async def test_order_levels_walk_through_a_team_public_name_within_the_callers_team(
    make_core: MakeCore,
    host: ScriptedHost,
    buckets: Mapping[str, Mapping[str, object]],
    expected: list[str],
) -> None:
    core: Final = build(make_core, model_list=INTERLEAVED_TEAM_TIERS)
    host.answer("openai/t1-first", rate_limit())
    host.answer("openai/t2-first", rate_limit())

    await complete(core, "tp", **{name: dict(bucket) for name, bucket in buckets.items()})

    assert host.deployments_called("acompletion") == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("buckets", "expected"),
    [
        ({"metadata": {"user_api_key_team_id": ""}}, ["shared-first", "shared-third"]),
        (
            {"metadata": {"user_api_key_team_id": ""}, "litellm_metadata": {"user_api_key_team_id": "T1"}},
            ["shared-first", "t1-owned-second"],
        ),
    ],
    ids=["empty-metadata-team-alone-skips-team-owned-tiers", "litellm-metadata-team-walks-its-owned-tier"],
)
async def test_order_levels_of_a_shared_group_name_follow_the_listing_team(
    make_core: MakeCore,
    host: ScriptedHost,
    buckets: Mapping[str, Mapping[str, object]],
    expected: list[str],
) -> None:
    core: Final = build(
        make_core,
        model_list=[
            deployment("g", "shared-first", order=1),
            deployment("g", "t1-owned-second", order=2, model_info={"team_id": "T1"}),
            deployment("g", "shared-third", order=3),
        ],
    )
    host.answer("openai/shared-first", rate_limit())

    await complete(core, "g", **{name: dict(bucket) for name, bucket in buckets.items()})

    assert host.deployments_called("acompletion") == expected


async def _responses(core: RouterCore, litellm_metadata: Mapping[str, object]) -> object:
    return await core.aresponses(model="tp", input="hi", litellm_metadata=litellm_metadata)  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method


async def _anthropic_messages(core: RouterCore, litellm_metadata: Mapping[str, object]) -> object:
    return await core.aanthropic_messages(  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method
        model="tp", messages=MESSAGES, max_tokens=5, litellm_metadata=litellm_metadata
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("op", "call"),
    [("aresponses", _responses), ("anthropic_messages", _anthropic_messages)],
    ids=["responses", "anthropic-messages"],
)
async def test_order_levels_walk_through_a_team_public_name_on_litellm_metadata_endpoints(
    make_core: MakeCore,
    host: ScriptedHost,
    op: str,
    call: Callable[[RouterCore, Mapping[str, object]], Awaitable[object]],
) -> None:
    core: Final = build(make_core, model_list=INTERLEAVED_TEAM_TIERS)
    host.answer("openai/t1-first", rate_limit())

    await call(core, dict(TEAM_METADATA))

    assert host.deployments_called(op) == ["t1-first", "t1-second"]


# --- a team wildcard ------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_team_wildcard_public_name_routes_with_its_wildcard_filled_in(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(
        make_core,
        model_list=[
            deployment(
                "t1-w",
                "tw1",
                model="openai/*",
                model_info={"team_id": "T1", "team_public_model_name": "tw-*"},
            ),
            deployment("g", "gp"),
        ],
    )

    await complete(core, "tw-gpt-4o", metadata=dict(TEAM_METADATA))

    assert host.deployments_called("acompletion") == ["tw1"]
    assert host.calls[0].kwargs["model"] == "openai/gpt-4o"


# --- errors raised while picking ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_usage_refusal_takes_the_fallback(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = build(
        make_core,
        model_list=[deployment("one", "o", rpm=1), deployment("safe", "s")],
        routing_strategy="usage-based-routing-v2",
        fallbacks=[{"one": ["safe"]}],
    )
    await complete(core, "one")

    response: Final = await complete(core, "one")

    assert host.deployments_called("acompletion") == ["o", "s"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1


@pytest.mark.asyncio
async def test_a_usage_refusal_carries_the_routing_note(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", rpm=1)], routing_strategy="usage-based-routing-v2", num_retries=0
    )
    show_router_debug(host)
    await complete(core, "g")

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core, "g")

    assert "No fallback" in str(raised.value)


class _RaisingPlugin:
    """Refuses every pick of ``refused`` in the pre-routing hook."""

    def __init__(self, refused: str) -> None:
        self.refused: Final = refused
        self.runs: int = 0

    async def run(self, context: PluginContext) -> PluginContext:
        if self.refused in context.candidate_models:
            self.runs += 1
            raise rate_limit()
        return context


@pytest.mark.asyncio
async def test_a_pre_routing_error_runs_the_pipeline_once_per_retry(make_core: MakeCore, host: ScriptedHost) -> None:
    plugin: Final = _RaisingPlugin("openai/a")
    core: Final = build(make_core, num_retries=1, retry_after=0, plugins=[plugin])

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert plugin.runs == 2
    assert host.calls == []
    assert getattr(raised.value, "max_retries", None) == 1


@pytest.mark.asyncio
async def test_a_pre_routing_error_takes_the_fallback(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = build(make_core, plugins=[_RaisingPlugin("openai/a")], fallbacks=[{"a": ["b"]}])

    await complete(core)

    assert host.deployments_called("acompletion") == ["b"]


# --- fallback checks that raise --------------------------------------------------------------------


def _sync_check(*, model: str, request_kwargs: Mapping[str, object], ctx: object) -> bool:
    return True


@pytest.mark.asyncio
async def test_a_sync_access_check_ends_the_chain_with_the_levels_own_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b"]}], fallback_access_check=_sync_check)
    show_router_debug(host)
    host.answer("openai/a", server_error())

    with pytest.raises(litellm.InternalServerError) as raised:
        await complete(core)

    assert "also failed" in str(raised.value)
    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_a_sync_access_check_on_the_order_path_raises_its_own_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(
        make_core,
        model_list=[deployment("a", "a1", order=1), deployment("a", "a2", order=2), deployment("b", "b")],
        fallbacks=[{"a": ["b"]}],
        fallback_access_check=_sync_check,
    )
    host.answer("openai/a1", server_error())
    host.answer("openai/a2", server_error())

    with pytest.raises(TypeError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a1", "a2"]


# --- dict targets naming no group -------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("model", [5, None], ids=["int", "none"])
async def test_a_dict_target_with_a_non_string_model_is_a_hop_that_calls_nothing(
    make_core: MakeCore, host: ScriptedHost, model: object
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": [{"model": model}, "b"]}])
    host.answer("openai/a", server_error())

    response: Final = await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 2


@pytest.mark.asyncio
async def test_a_chain_ending_on_a_non_string_model_notes_the_hops_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": [{"model": 5}]}])
    show_router_debug(host)
    host.answer("openai/a", server_error())

    with pytest.raises(litellm.InternalServerError) as raised:
        await complete(core)

    assert "argument of type 'int'" in str(raised.value)
    assert host.deployments_called("acompletion") == ["a"]


# --- weighted failover ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket", ["metadata", "litellm_metadata"])
async def test_weighted_failover_moves_to_a_sibling_of_the_callers_team_only(
    make_core: MakeCore, host: ScriptedHost, bucket: str
) -> None:
    core: Final = build(
        make_core,
        model_list=[
            deployment("t1-a", "t1-a", model_info=T1_PUBLIC),
            deployment("t1-b", "t1-b", model_info=T1_PUBLIC),
            deployment("t2-a", "t2-a", model_info=T2_PUBLIC),
        ],
        enable_weighted_failover=True,
    )
    first_call_fails: Final = fails_once_then_answers(rate_limit())
    for deployment_id in ("t1-a", "t1-b", "t2-a"):
        host.answer(f"openai/{deployment_id}", first_call_fails)

    await complete(core, "tp", **{bucket: dict(TEAM_METADATA)})

    assert sorted(host.deployments_called("acompletion")) == ["t1-a", "t1-b"]


@pytest.mark.asyncio
@pytest.mark.parametrize("repin", [True, False])
async def test_weighted_failover_skips_an_affinity_pin_on_the_failed_deployment(
    make_core: MakeCore, host: ScriptedHost, repin: bool
) -> None:
    core: Final = build(
        make_core,
        model_list=[deployment("g", "a"), deployment("g", "b")],
        enable_weighted_failover=True,
        optional_pre_call_checks=["session_affinity"],
        session_affinity_repin_on_failover=repin,
    )
    check: Final = next(
        callback
        for callback in core.optional_callbacks or ()  # pyright: ignore[reportAttributeAccessIssue, reportUnknownVariableType, reportUnknownMemberType]
        if isinstance(callback, DeploymentAffinityCheck)
    )
    key: Final = DeploymentAffinityCheck.get_session_affinity_cache_key("g", "s", "c")
    await check.pins.set(key, {"model_id": "a"}, 3600)
    host.answer("openai/a", server_error())

    response: Final = await complete(core, "g", metadata={"session_id": "s", "user_api_key_hash": "c"})

    assert host.deployments_called("acompletion") == ["a", "b"]
    assert headers(response)["x-litellm-attempted-fallbacks"] == 1


def _fails_then_raises(raised: BaseException):  # noqa: ANN202  # a scripted outcome
    """One outcome both deployments of a group share: a provider failure, then ``raised``."""
    calls: Final = [0]

    def outcome(**_request: object) -> object:  # kwargs-ok: the scripted request
        calls[0] += 1
        return server_error() if calls[0] == 1 else raised

    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("hop_error", "error"),
    [(TypeError("a bug"), TypeError), (rate_limit(), litellm.InternalServerError)],
    ids=["a-bug-propagates", "a-provider-error-defers"],
)
async def test_weighted_failover_swallows_only_api_and_rate_limit_errors(
    make_core: MakeCore, host: ScriptedHost, hop_error: BaseException, error: type[Exception]
) -> None:
    core: Final = build(
        make_core,
        model_list=[deployment("a", "a1", model="openai/same"), deployment("a", "a2", model="openai/same")],
        enable_weighted_failover=True,
    )
    host.answer("openai/same", _fails_then_raises(hop_error))

    with pytest.raises(error):
        await complete(core)

    assert len(host.deployments_called("acompletion")) == 2


# --- refusals the walk falls back from ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_paused_deployment_named_by_id_falls_back(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = build(
        make_core,
        model_list=[deployment("a", "a1", model_info={"blocked": True}), deployment("b", "b")],
        fallbacks=[{"a1": ["b"]}],
    )

    await complete(core, "a1")

    assert host.deployments_called("acompletion") == ["b"]


# --- encrypted reasoning on a fallback hop --------------------------------------------------------


@pytest.mark.asyncio
async def test_a_fallback_hop_strips_the_reasoning_the_target_cannot_decrypt(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(
        make_core,
        model_list=[deployment("a", "a", api_key="k-a"), deployment("b", "b", api_key="k-b")],
        fallbacks=[{"a": ["b"]}],
    )
    host.answer("openai/a", server_error())
    reasoning: Final = {
        "type": "reasoning",
        "encrypted_content": "opaque",
        "summary": [{"type": "summary_text", "text": "thought"}],
    }

    await core.aresponses(model="a", input=[reasoning, {"role": "user", "content": "hi"}])  # pyright: ignore[reportAttributeAccessIssue]  # endpoint method

    sent: Final = host.calls_to("aresponses")[-1].kwargs["input"]
    assert isinstance(sent, list)
    assert not any(isinstance(item, dict) and item.get("encrypted_content") for item in sent)
