"""
Access and policy gaps differential fuzzing found between the cores: dict fallback targets that
may not be set from any list, which metadata bucket names the caller's team, the routing plugins on
the async pass-through pick, and a name several teams share falling back.
"""

from typing import Final

import pytest

import litellm
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import CustomRoutingStrategyBase, RouterRateLimitError
from tests.unit.test_router.core.fallbacks_helpers import (
    MakeCore,
    build,
    complete,
    complete_sync,
    show_router_debug,
)
from tests.unit.test_router.core.scripted_host import ScriptedHost, content_policy, deployment, rate_limit
from tests.unit.test_router.core.test_selection_characterization_prerouting import Narrowing

OIDC_TARGET: Final = {"model": "b", "api_key": "oidc/google/x"}

# --- dict fallback targets from a router's lists ---------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [OIDC_TARGET, {"model": "b", "anthropic_issuer_url": "https://issuer.example"}],
    ids=["oidc-api-key", "federation-field"],
)
async def test_a_router_level_dict_target_refuses_the_whole_chain(
    make_core: MakeCore, host: ScriptedHost, target: dict[str, str]
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": [target, "c"]}])
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_a_refused_router_level_chain_explains_itself_under_router_debug(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(make_core, fallbacks=[{"a": [OIDC_TARGET]}])
    show_router_debug(host)
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError) as raised:
        await complete(core)

    assert "also failed: Rejected Request: api_key holds an oidc/ secret reference" in str(raised.value)


@pytest.mark.asyncio
async def test_a_content_policy_list_with_a_refused_target_tries_nothing(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = build(make_core, content_policy_fallbacks=[{"a": [OIDC_TARGET]}])
    host.answer("openai/a", content_policy())

    with pytest.raises(litellm.ContentPolicyViolationError):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_a_refused_chain_on_the_order_path_raises_the_refusal_itself(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    # The order path's `run_async_fallback` sits outside the `try` that re-raises the original error.
    core: Final = make_core(
        model_list=[deployment("a", "a1", order=1), deployment("a", "a2", order=2), deployment("b", "b")],
        num_retries=0,
        fallbacks=[{"a": [OIDC_TARGET]}],
    )
    host.answer("openai/a1", rate_limit())

    with pytest.raises(ValueError, match="oidc/ secret reference"):
        await complete(core)

    assert host.deployments_called("acompletion") == ["a1"]


@pytest.mark.asyncio
async def test_a_refused_chain_fails_only_its_own_level(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = build(make_core, fallbacks=[{"a": ["b", "c"]}, {"b": [{"model": "c", "api_key": "oidc/google/x"}]}])
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())

    await complete(core)

    assert host.deployments_called("acompletion") == ["a", "b", "c"]


def test_a_refused_router_level_chain_is_refused_on_the_sync_path(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = build(make_core, fallbacks=[{"a": [OIDC_TARGET]}])
    host.answer("openai/a", rate_limit())

    with pytest.raises(litellm.RateLimitError):
        complete_sync(core)

    assert host.deployments_called("completion") == ["a"]


# --- which bucket names the team -------------------------------------------------------------------

_TEAM: Final = {"user_api_key_team_id": "t1"}
_OWN_OR_PUBLIC: Final = [
    deployment("g", "own", weight=1, model_info={"team_id": "t1"}),
    deployment("g", "public", weight=0),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("buckets", "expected"),
    [
        ({"litellm_metadata": _TEAM}, "own"),
        ({"metadata": {"trace": "x"}, "litellm_metadata": _TEAM}, "own"),
        ({"metadata": {"user_api_key_team_id": ""}, "litellm_metadata": _TEAM}, "own"),
        ({"metadata": {"user_api_key_team_id": "t2"}, "litellm_metadata": _TEAM}, "public"),
    ],
    ids=["only-litellm-metadata", "metadata-without-a-team", "metadata-with-an-empty-team", "metadata-wins"],
)
async def test_the_first_bucket_holding_a_team_routes_the_call(
    make_core: MakeCore, host: ScriptedHost, buckets: dict[str, dict[str, str]], expected: str
) -> None:
    core: Final = make_core(model_list=list(_OWN_OR_PUBLIC), num_retries=0)

    await complete(core, "g", **{name: dict(bucket) for name, bucket in buckets.items()})

    assert host.deployments_called("acompletion") == [expected]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bucket", "raised"),
    [("metadata", litellm.RateLimitError), ("litellm_metadata", RouterRateLimitError)],
    ids=["one-level-listed-so-the-level-raises-its-own-error", "no-team-listed-so-the-second-level-raises"],
)
async def test_the_order_levels_of_a_failed_group_read_only_the_metadata_bucket(
    make_core: MakeCore, host: ScriptedHost, bucket: str, raised: type[Exception]
) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "own", order=1, model_info={"team_id": "t1"}),
            deployment("g", "theirs", order=2, model_info={"team_id": "t2"}),
        ],
        num_retries=0,
    )
    host.answer("openai/own", rate_limit())

    with pytest.raises(raised):
        await complete(core, "g", **{bucket: dict(_TEAM)})

    assert host.deployments_called("acompletion") == ["own"]


# --- the routing plugins on the async pass-through pick --------------------------------------------

_PASS_THROUGH_PAIR: Final = [
    deployment("g", "a", use_in_pass_through=True, weight=1),
    deployment("g", "b", use_in_pass_through=True, weight=0),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["simple-shuffle", "usage-based-routing"])
async def test_the_async_pass_through_pick_runs_the_routing_plugins(make_core: MakeCore, strategy: str) -> None:
    plugin: Final = Narrowing(frozenset({"openai/b"}))
    core: Final = make_core(model_list=list(_PASS_THROUGH_PAIR), plugins=[plugin], routing_strategy=strategy)

    chosen: Final = await core.async_get_available_deployment_for_pass_through("g", {})  # pyright: ignore[reportAttributeAccessIssue]

    assert chosen["model_info"]["id"] == "b"
    assert plugin.seen == [["openai/a", "openai/b"]]


@pytest.mark.asyncio
async def test_a_custom_strategy_does_not_answer_the_async_pass_through_pick(make_core: MakeCore) -> None:
    class Refuses(CustomRoutingStrategyBase):
        async def async_get_available_deployment(self, *args: object, **kwargs: object) -> object:  # pyright: ignore[reportIncompatibleMethodOverride]  # kwargs-ok: never called
            raise AssertionError("a pass-through pick never reaches the custom strategy")

    core: Final = make_core(model_list=[deployment("g", "a", use_in_pass_through=True)])
    core.register_plugin(Refuses())

    chosen: Final = await core.async_get_available_deployment_for_pass_through("g", {})  # pyright: ignore[reportAttributeAccessIssue]

    assert chosen["model_info"]["id"] == "a"


# --- a name several teams share --------------------------------------------------------------------

_SHARED_NAME: Final = [
    deployment("g", "gp"),
    deployment("t1-pub", "p1", model_info={"team_id": "T1", "team_public_model_name": "pub"}),
    deployment("t2-pub", "p2", model_info={"team_id": "T2", "team_public_model_name": "pub"}),
]
_REFUSAL: Final = (
    "litellm.BadRequestError: Model name 'pub' matches deployments from multiple teams. "
    "Specify the deployment ID directly to disambiguate."
)


def _admin() -> dict[str, object]:  # mutable-ok: a request's metadata
    return {"user_api_key_auth": UserAPIKeyAuth(user_role="proxy_admin")}  # pyright: ignore[reportCallIssue]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("configured", "requested"),
    [({"fallbacks": [{"pub": ["g"]}]}, {}), ({}, {"fallbacks": ["g"]})],
    ids=["router-list", "request-list"],
)
async def test_a_name_several_teams_share_falls_back(
    make_core: MakeCore, host: ScriptedHost, configured: dict[str, object], requested: dict[str, object]
) -> None:
    core: Final = make_core(model_list=list(_SHARED_NAME), num_retries=0, **configured)

    await complete(core, "pub", metadata=_admin(), **requested)

    assert host.deployments_called("acompletion") == ["gp"]


def test_a_name_several_teams_share_falls_back_on_the_sync_path(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=list(_SHARED_NAME), num_retries=0)

    complete_sync(core, "pub", metadata=_admin(), fallbacks=["g"])

    assert host.deployments_called("completion") == ["gp"]


@pytest.mark.asyncio
async def test_a_shared_name_raises_its_refusal_when_the_fallback_fails(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(_SHARED_NAME), num_retries=0)
    host.answer("openai/gp", rate_limit())

    with pytest.raises(litellm.BadRequestError) as raised:
        await complete(core, "pub", metadata=_admin(), fallbacks=["g"])

    assert raised.value.status_code == 400
    assert str(raised.value) == _REFUSAL


@pytest.mark.asyncio
async def test_a_shared_name_explains_the_failed_fallback_under_router_debug(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=list(_SHARED_NAME), num_retries=0)
    show_router_debug(host)
    host.answer("openai/gp", rate_limit())

    with pytest.raises(litellm.BadRequestError) as raised:
        await complete(core, "pub", metadata=_admin(), fallbacks=["g"])

    assert str(raised.value).startswith(f"{_REFUSAL}\n\nLiteLLM: model group 'pub' failed with the error above.")
