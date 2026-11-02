"""
Pick errors the Rust core serves inside router-core, pinned to ``PythonCore``'s behaviour (oddities
included): a pick's error retried by the retry loop, a ``default_deployment`` Python cannot serve, a
written index position past the model list, a stale routing group at a deeper fallback level or in
weighted failover, and a strategy change through ``update_settings`` after a raw ``routing_groups`` write.
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
from tests.unit.test_router.core.retries_health_helpers import MakeCore, attempt, attempt_sync
from tests.unit.test_router.core.scripted_host import ScriptedHost, deployment, rate_limit
from tests.unit.test_router.core.test_fallbacks_characterization_outcomes import FallbackEvents

TEAM: Final = {"metadata": {"user_api_key_team_id": "TA"}}


def as_router(core: RouterCore) -> Router:
    router: Final = Router.__new__(Router)
    router._attach(core)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType]  # the facade's own attach, as from_config calls it
    return router


def write(router: Router, name: str, value: object) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        setattr(router, name, value)


async def selected(router: Router, model: str, **request: object) -> object:  # kwargs-ok: the request kwargs
    """The deployment ``select`` picks, or the error it raised."""
    try:
        return await router.select(model, dict(request))
    except Exception as error:
        return error


def picked_id(picked: object) -> object:
    return (as_str_mapping((as_str_mapping(picked) or {}).get("model_info")) or {}).get("id")


# --- a pick's error in the retry loop -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_retry_policy_retries_a_group_whose_deployments_all_cool_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """The pick's ``RouterRateLimitError`` is a plain ``ValueError``: ``DefaultRetries`` applies to it."""
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")],
        num_retries=2,
        allowed_fails=0,
        model_group_retry_policy={"g": {"DefaultRetries": 1}},
    )
    host.answer("*", rate_limit())
    for _ in range(2):
        _ = await attempt(core, num_retries=0)
    host.calls.clear()
    host.sleeps.clear()

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RouterRateLimitError)
    assert isinstance(error.__context__, litellm.RouterRateLimitError)
    assert host.calls == []
    assert len(host.sleeps) == 2
    assert all(sleep > 0 for sleep in host.sleeps)


@pytest.mark.asyncio
async def test_a_retry_policy_retries_an_unknown_model_and_stamps_the_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2, retry_policy={"DefaultRetries": 1})

    error: Final = await attempt(core, model="zz")

    assert isinstance(error, litellm.BadRequestError)
    assert (getattr(error, "num_retries", None), getattr(error, "max_retries", None)) == (1, 1)
    assert isinstance(error.__context__, litellm.BadRequestError)
    assert len(host.sleeps) == 2


@pytest.mark.asyncio
async def test_invalid_router_weights_are_retried_while_a_deployment_is_healthy(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """No policy: ``should_retry_this_error`` retries a plain exception while a deployment is healthy."""
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=2)

    error: Final = await attempt(core, _router_weights={"a": "x"})

    assert isinstance(error, ValidationError)
    assert isinstance(error.__context__, ValidationError)
    assert host.calls == []
    assert host.sleeps == [0, 0, 0]
    assert all(type(sleep) is int for sleep in host.sleeps)


@pytest.mark.asyncio
async def test_without_a_policy_an_unknown_model_is_raised_at_once(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2)

    error: Final = await attempt(core, model="zz")

    assert isinstance(error, litellm.BadRequestError)
    assert error.__context__ is None
    assert host.sleeps == []


@pytest.mark.asyncio
async def test_the_last_retry_failure_chains_to_the_hops_first_failure(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")],
        num_retries=2,
        model_group_retry_policy={"g": {"DefaultRetries": 1}},
    )
    host.answer("*", rate_limit())

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RateLimitError)
    assert isinstance(error.__context__, litellm.RateLimitError)
    assert error.__context__ is not error
    assert host.sleeps[0] == 0
    assert type(host.sleeps[0]) is int


# --- a default_deployment Python cannot serve ---------------------------------------------------------


UNCOPYABLE_DEFAULTS: Final[list[tuple[object, type[Exception], str]]] = [  # mutable-ok: pytest parameters
    ({}, KeyError, "'litellm_params'"),
    ({"model_name": "dm"}, KeyError, "'litellm_params'"),
    ({"model_name": "dm", "litellm_params": "s", "model_info": {"id": "dd"}}, AttributeError, "no attribute 'copy'"),
    ("str", AttributeError, "'str' object has no attribute 'copy'"),
    (5, AttributeError, "'int' object has no attribute 'copy'"),
    (["a"], TypeError, "list indices must be integers or slices, not str"),
]
"""Each value, and the error its copy raises for any name."""


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "raised", "message"), UNCOPYABLE_DEFAULTS)
async def test_a_default_deployment_python_cannot_copy_fails_each_pick_of_an_unknown_name(
    make_core: MakeCore, host: ScriptedHost, value: object, raised: type[Exception], message: str
) -> None:
    """Oddity: assigning it succeeds; every request for a name nothing else serves raises the copy's error."""
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=1)
    _ = core.apply(ChangeSet(assign={"default_deployment": value}))

    for error in (
        await attempt(core, model="zz"),
        attempt_sync(core, model="zz"),
        await selected(as_router(core), "zz"),
    ):
        assert isinstance(error, raised)
        assert message in str(error)
    assert host.deployments_called("acompletion") == []
    _ = await attempt(core)
    assert host.deployments_called("acompletion") == ["a"]


FAILING_DEFAULTS: Final[list[tuple[object, str, object]]] = [  # mutable-ok: pytest parameters
    ({"litellm_params": {"model": "x"}}, "'model_name'", None),
    ({"model_name": "dm", "litellm_params": {"model": "x"}}, "'model_info'", None),
    ({"model_name": "dm", "litellm_params": {"model": "x"}, "model_info": {}}, "'id'", {}),
]
"""Each value, the ``KeyError`` every call to it raises, and the ``model_info`` its pick returns."""


@pytest.mark.asyncio
@pytest.mark.parametrize(("value", "missing", "model_info"), FAILING_DEFAULTS)
async def test_a_default_deployment_missing_a_field_is_picked_but_every_call_fails(
    make_core: MakeCore, host: ScriptedHost, value: object, missing: str, model_info: object
) -> None:
    """Oddity: the pick returns Python's copy; the call's first reads of it raise ``KeyError``, with no retry."""
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=1)
    _ = core.apply(ChangeSet(assign={"default_deployment": value}))

    called: Final = await attempt(core, model="zz")
    called_sync: Final = attempt_sync(core, model="zz")
    chosen: Final = as_str_mapping(await selected(as_router(core), "zz")) or {}

    assert isinstance(called, KeyError)
    assert isinstance(called_sync, KeyError)
    assert str(called) == missing
    assert chosen.get("model_info") == model_info
    assert (as_str_mapping(chosen.get("litellm_params")) or {}).get("model") == "zz"
    assert host.calls == []
    assert host.sleeps == []


# --- a written index position past the model list -----------------------------------------------------


def _teams(make_core: MakeCore) -> tuple[RouterCore, Router]:
    core: Final = make_core(
        model_list=[deployment("gpt", "a"), deployment("claude", "c"), team_deployment("int-t", "t1", "TA", "pub")],
        num_retries=1,
        fallbacks=[{"gpt": ["claude"]}, {"pub": ["claude"]}],
    )
    return core, as_router(core)


@pytest.mark.asyncio
async def test_a_name_index_position_past_the_list_raises_index_error_for_that_name(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core, router = _teams(make_core)
    write(router, "model_name_to_deployment_indices", {"gpt": [0, 7], "claude": [1], "int-t": [2]})

    picked: Final = await selected(router, "gpt")
    called: Final = await attempt(core, model="gpt")

    assert isinstance(picked, IndexError)
    assert isinstance(called, IndexError)
    assert str(called) == "list index out of range"
    assert picked_id(await selected(router, "claude")) == "c"
    assert host.calls == []


@pytest.mark.asyncio
async def test_a_team_index_position_past_the_list_raises_from_the_fallback_handling(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: the level's fallback handling lists the group again and raises a second ``IndexError``
    (chained to the pick's), before any fallback."""
    core, router = _teams(make_core)
    write(router, "team_model_to_deployment_indices", {("TA", "pub"): [9]})

    picked: Final = await selected(router, "pub", **TEAM)
    called: Final = await attempt(core, model="pub", **TEAM)

    assert isinstance(picked, IndexError)
    assert isinstance(called, IndexError)
    assert isinstance(called.__context__, IndexError)
    assert host.calls == []
    assert picked_id(await selected(router, "gpt")) == "a"


# --- a stale routing group deeper in the walk ---------------------------------------------------------


def _stale(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: extra router arguments
    core: Final = make_core(
        model_list=[deployment("gpt", "a"), deployment("gpt", "b"), deployment("claude", "c"), deployment("x", "x1")],
        routing_groups=[{"group_name": "G", "models": ["gpt"], "routing_strategy": "simple-shuffle"}],
        num_retries=0,
        fallbacks=[{"x": ["gpt"]}, {"gpt": ["claude"]}],
        **arguments,
    )
    write(as_router(core), "routing_groups", {})
    return core


@pytest.mark.asyncio
async def test_a_stale_routing_group_at_a_fallback_level_is_that_levels_error(
    make_core: MakeCore, host: ScriptedHost, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: Final = FallbackEvents()
    monkeypatch.setattr(litellm, "callbacks", [events])
    core: Final = _stale(make_core)
    host.answer("*", rate_limit())

    error: Final = await attempt(core, model="x")

    assert isinstance(error, litellm.RateLimitError)
    assert events.events == [("failure", "gpt", "claude", "KeyError"), ("failure", "x", "gpt", "RateLimitError")]


@pytest.mark.asyncio
async def test_weighted_failover_raises_the_stale_groups_key_error_before_any_fallback(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """Oddity: ``_maybe_run_weighted_failover`` reads the routing context first, so the stale member map's
    ``KeyError`` escapes the fallback handling (chained to the pick's) and the regular fallback never runs."""
    core: Final = _stale(make_core, enable_weighted_failover=True)

    error: Final = await attempt(core, model="gpt")

    assert isinstance(error, KeyError)
    assert str(error) == "'G'"
    assert isinstance(error.__context__, KeyError)
    assert host.calls == []


# --- update_settings(routing_strategy) after a raw routing_groups write -----------------------------------


@pytest.mark.asyncio
async def test_a_strategy_change_re_inits_the_routing_groups_from_their_last_input(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("gpt", "a"), deployment("gpt", "b"), deployment("claude", "c")],
        routing_groups=[{"group_name": "G", "models": ["gpt"], "routing_strategy": "least-busy"}],
    )
    router: Final = as_router(core)
    write(
        router, "routing_groups", {"K": RoutingGroup(group_name="K", models=["claude"], routing_strategy="least-busy")}
    )
    assert isinstance(await selected(router, "gpt"), KeyError)

    _ = core.apply(ChangeSet(settings={"routing_strategy": "usage-based-routing-v2"}))

    assert sorted(core.registry().routing_groups()) == ["G"]
    assert picked_id(await selected(router, "G")) == "a"
    assert picked_id(await selected(router, "gpt")) == "a"
    assert isinstance(await selected(router, "K"), litellm.BadRequestError)


@pytest.mark.asyncio
async def test_an_unchanged_strategy_keeps_the_raw_routing_groups(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[deployment("gpt", "a"), deployment("claude", "c")],
        routing_groups=[{"group_name": "G", "models": ["gpt"], "routing_strategy": "least-busy"}],
    )
    write(
        as_router(core),
        "routing_groups",
        {"K": RoutingGroup(group_name="K", models=["claude"], routing_strategy="least-busy")},
    )

    _ = core.apply(ChangeSet(settings={"routing_strategy": "simple-shuffle"}))

    assert sorted(core.registry().routing_groups()) == ["K"]


@pytest.mark.asyncio
async def test_a_negative_index_position_counts_from_the_end_of_the_list(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    """As Python's ``model_list[index]`` reads it; one before the start raises ``IndexError``."""
    core, router = _teams(make_core)
    write(router, "model_name_to_deployment_indices", {"gpt": [-3], "claude": [-2], "int-t": [-4]})

    assert picked_id(await selected(router, "gpt")) == "a"
    assert picked_id(await selected(router, "claude")) == "c"
    before_the_start: Final = await attempt(core, model="int-t")
    assert isinstance(before_the_start, IndexError)
    assert str(before_the_start) == "list index out of range"
    assert host.calls == []


@pytest.mark.asyncio
async def test_a_default_deployment_whose_id_is_not_a_string_is_picked_and_called_as_python_copies_it(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    value: Final = {
        "model_name": "dm",
        "litellm_params": {"model": "openai/dm", "api_key": "k"},
        "model_info": {"id": 5},
    }
    _ = core.apply(ChangeSet(assign={"default_deployment": value}))

    served: Final = await attempt(core, model="zz")
    chosen: Final = await selected(as_router(core), "zz")

    assert isinstance(served, litellm.ModelResponse)
    assert host.calls[0].kwargs["model"] == "zz"
    assert chosen == {"model_name": "dm", "litellm_params": {"model": "zz", "api_key": "k"}, "model_info": {"id": 5}}
