"""
Gaps the Rust core closes inside router-core, pinned to ``PythonCore``'s behaviour (oddities included):
``default_deployment``, a ``model_group_retry_policy`` whose lookup raises, the deprecated registry
index writes, a raw ``routing_groups`` write against the stale member map, and ``cooldown_time=0``
assigned after construction.
"""

import warnings
from typing import Final

import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.router import Router
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.types.router import RoutingGroup
from tests.unit.test_router.core.registry_helpers import team_deployment
from tests.unit.test_router.core.retries_health_helpers import MakeCore, attempt, cooled_down, headers_of
from tests.unit.test_router.core.scripted_host import ScriptedHost, completion_response, deployment, rate_limit

DEFAULT: Final[dict[str, object]] = {  # mutable-ok: a deployment dict, copied per test
    "model_name": "dm",
    "litellm_params": {"model": "openai/other-model"},
    "model_info": {"id": "dd"},
}


def as_router(core: RouterCore) -> Router:
    router: Final = Router.__new__(Router)
    router._attach(core)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType]  # the facade's own attach, as from_config calls it
    return router


def write(router: Router, name: str, value: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        setattr(router, name, value)


async def picked(router: Router, model: str) -> str:
    """The id ``select`` picks, or the error's class and message."""
    try:
        chosen: Final[object] = await router.select(model, {})
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return str((as_str_mapping((as_str_mapping(chosen) or {}).get("model_info")) or {}).get("id"))


# --- default_deployment --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_unknown_model_routes_to_the_default_deployment_without_retries_or_cooldown(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: the default deployment skips cooldown and the strategy, and the retry count reads no healthy
    deployment, so ``num_retries`` is unused; its id is never cooled while it names no configured deployment."""
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=1, allowed_fails=0)
    _ = core.apply(ChangeSet(assign={"default_deployment": dict(DEFAULT)}))
    host.answer("other-model", rate_limit(), completion_response())

    failed: Final = await attempt(core, model="other-model")
    assert isinstance(failed, litellm.RateLimitError)
    assert cooled_down(core, ["dd", "a"]) == {}
    served: Final = await attempt(core, model="other-model")

    assert headers_of(served)["x-litellm-model-group"] == "other-model"
    assert [call.deployment_id for call in host.calls] == ["dd", "dd"]
    assert host.calls[0].kwargs["model"] == "other-model"
    assert host.sleeps == []
    assert core.settings().default_deployment == DEFAULT


@pytest.mark.asyncio
async def test_a_known_model_never_reaches_the_default_deployment(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    _ = core.apply(ChangeSet(assign={"default_deployment": dict(DEFAULT)}))

    _ = await attempt(core)

    assert host.deployments_called("acompletion") == ["a"]


# --- model_group_retry_policy lookups that raise -----------------------------------------------------


RAISING_POLICIES: Final = [
    (3, TypeError, "argument of type 'int' is not iterable"),
    ("g", TypeError, "string indices must be integers"),
    (["g"], TypeError, "list indices must be integers or slices"),
    ({"g": 3}, AttributeError, "'int' object has no attribute 'DefaultRetries'"),
    ({"g": {"DefaultRetries": "x"}}, ValidationError, "validation error for RetryPolicy"),
]
"""Each policy, the error its lookup raises for group ``g``, and a fragment of that error's message."""


@pytest.mark.asyncio
@pytest.mark.parametrize(("policy", "raised", "message"), RAISING_POLICIES)
async def test_a_raising_retry_policy_lookup_replaces_the_error_and_stops_the_hop(
    make_core: MakeCore, host: ScriptedHost, policy: object, raised: type[Exception], message: str
) -> None:
    """Oddity: the lookup runs on the hop's first failure; its error replaces the attempt's, and no retry
    follows. The error is raised while the attempt's is handled, so it chains to it, except pydantic's
    ``ValidationError``, which pydantic raises with no context."""
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=2, model_group_retry_policy=policy
    )
    host.answer("*", rate_limit())

    error: Final = await attempt(core)

    assert isinstance(error, raised)
    assert message in str(error)
    if raised is ValidationError:
        assert error.__context__ is None
    else:
        assert isinstance(error.__context__, litellm.RateLimitError)
    assert len(host.calls) == 1
    assert host.sleeps == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("policy", "raised", "message"), RAISING_POLICIES)
async def test_a_raising_retry_policy_lookup_takes_the_generic_fallbacks(
    make_core: MakeCore, host: ScriptedHost, policy: object, raised: type[Exception], message: str
) -> None:
    del raised, message
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b"), deployment("h", "c")],
        num_retries=2,
        model_group_retry_policy=policy,
        fallbacks=[{"g": ["h"]}],
    )
    host.answer("openai/a", rate_limit())
    host.answer("openai/b", rate_limit())

    served: Final = await attempt(core)

    assert headers_of(served)["x-litellm-attempted-fallbacks"] == 1
    assert host.deployments_called("acompletion")[-1] == "c"
    assert len(host.calls) == 2


@pytest.mark.asyncio
async def test_a_policy_naming_another_group_leaves_this_group_retrying(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=2, model_group_retry_policy={"h": 3}
    )
    host.answer("*", rate_limit())

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RateLimitError)
    assert len(host.calls) == 2


# --- deprecated registry index writes ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_written_model_name_index_decides_which_names_resolve(make_core: MakeCore) -> None:
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a"), deployment("claude", "c")]))
    write(router, "model_name_to_deployment_indices", {"gpt": [0], "claude": []})

    assert await picked(router, "gpt") == "a"
    assert (await picked(router, "claude")).startswith("BadRequestError")


@pytest.mark.asyncio
async def test_a_model_name_index_entry_pointing_at_another_name_resolves_nothing(make_core: MakeCore) -> None:
    """Each indexed deployment is still checked against the requested name."""
    router: Final = as_router(make_core(model_list=[deployment("gpt", "a"), deployment("claude", "c")]))
    write(router, "model_name_to_deployment_indices", {"gpt": [1], "claude": [1]})

    assert (await picked(router, "gpt")).startswith("BadRequestError")
    assert await picked(router, "claude") == "c"


@pytest.mark.asyncio
async def test_a_written_model_names_set_decides_which_names_reach_the_default_deployment(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(model_list=[deployment("gpt", "a"), deployment("claude", "c")])
    _ = core.apply(ChangeSet(assign={"default_deployment": dict(DEFAULT)}))
    router: Final = as_router(core)

    write(router, "model_names", set[str]())
    assert [await picked(router, name) for name in ("gpt", "claude", "other-model")] == ["dd", "dd", "dd"]

    write(router, "model_names", {"gpt", "claude", "other-model"})
    assert await picked(router, "gpt") == "a"
    assert (await picked(router, "other-model")).startswith("BadRequestError")


@pytest.mark.asyncio
async def test_an_emptied_team_index_still_finds_team_deployments_by_scanning(make_core: MakeCore) -> None:
    router: Final = as_router(
        make_core(model_list=[deployment("gpt", "a"), team_deployment("int-t", "t1", "TA", "pub")])
    )
    write(router, "team_model_to_deployment_indices", {})
    write(router, "model_name_to_deployment_indices", {"gpt": [0]})

    chosen: Final = await router.select("pub", {"metadata": {"user_api_key_team_id": "TA"}})
    assert chosen["model_info"]["id"] == "t1"
    assert (await picked(router, "int-t")).startswith("BadRequestError")


# --- a raw routing_groups write: the member map stays as constructed ------------------------------------


def _grouped(make_core: MakeCore) -> Router:
    return as_router(
        make_core(
            model_list=[deployment("gpt", "a"), deployment("gpt", "b"), deployment("claude", "c")],
            routing_groups=[{"group_name": "G", "models": ["gpt"], "routing_strategy": "least-busy"}],
        )
    )


@pytest.mark.asyncio
async def test_dropping_a_routing_group_raw_leaves_its_members_raising_a_key_error(make_core: MakeCore) -> None:
    """Oddity: the member-to-group map is built once, so a member of a dropped group looks it up and fails."""
    router: Final = _grouped(make_core)
    write(router, "routing_groups", {})

    assert await picked(router, "gpt") == "KeyError: 'G'"
    assert await picked(router, "claude") == "c"


@pytest.mark.asyncio
async def test_a_routing_group_added_raw_leaves_its_members_on_the_default_strategy(make_core: MakeCore) -> None:
    router: Final = _grouped(make_core)
    write(
        router,
        "routing_groups",
        {
            "G": RoutingGroup(group_name="G", models=["gpt"], routing_strategy="least-busy"),
            "K": RoutingGroup(group_name="K", models=["claude"], routing_strategy="latency-based-routing"),
        },
    )

    assert await picked(router, "gpt") == "a"
    assert await picked(router, "claude") == "c"
    assert (await picked(router, "K")).startswith("RouterRateLimitError: No deployments available")


# --- cooldown_time assigned after construction ----------------------------------------------------------


@pytest.mark.asyncio
async def test_an_assigned_zero_cooldown_time_never_keeps_a_deployment_cooled(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: an assigned 0 is a zero cooldown ttl (only the constructor reads 0 as the default)."""
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0, allowed_fails=0)
    _ = core.apply(ChangeSet(assign={"cooldown_time": 0}))
    host.answer("*", rate_limit())

    for _ in range(3):
        assert isinstance(await attempt(core), litellm.RateLimitError)
        assert cooled_down(core, ["a", "b"]) == {}

    assert core.settings().cooldown_time == 0
    assert len(host.calls) == 3


@pytest.mark.asyncio
async def test_an_assigned_none_cooldown_time_cools_for_the_default(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0, allowed_fails=0)
    _ = core.apply(ChangeSet(assign={"cooldown_time": None}))
    host.answer("*", rate_limit())

    _ = await attempt(core)
    _ = await attempt(core)

    assert sorted(cooled_down(core, ["a", "b"])) == ["a", "b"]
    assert {entry["cooldown_time"] for entry in cooled_down(core, ["a", "b"]).values()} == {5}
    assert isinstance(await attempt(core), litellm.RouterRateLimitError)
