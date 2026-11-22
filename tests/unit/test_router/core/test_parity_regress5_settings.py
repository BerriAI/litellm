"""
Parity of the Rust core with ``router_python_core.py`` on the settings causes of the regress5 pack:
attributes written after build, ``update_settings`` applied key by key, a request ``num_retries``
that is no count, calls after ``retire``, deleted attributes, and a default deployment that fails
every call. The refusals at the end are ruled to differ, so they run on the Rust core only.
"""

import inspect
from typing import Final

import pytest

import litellm
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.types.router import Deployment, LiteLLM_Params, RetryPolicy
from tests.unit.test_router.core.gaps_l11100_helpers import TIERS, complexity_marker
from tests.unit.test_router.core.retries_health_helpers import MakeCore, attempt
from tests.unit.test_router.core.scripted_host import (
    CORE_BUILDERS,
    ScriptedHost,
    deployment,
    rate_limit,
    server_error,
)

HEURISTIC_V2: Final = {"classifier_type": "heuristic_v2", "tiers": dict(TIERS)}


def _one_group(make_core: MakeCore, **arguments: object) -> RouterCore:  # kwargs-ok: the router's constructor arguments
    return make_core(model_list=[deployment("g", "d1")], **{"num_retries": 0, **arguments})


# --- U7: retry policies written after build ---------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_policy_assigned_after_build_governs_the_next_failure(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one_group(make_core, num_retries=1, retry_after=0, disable_cooldowns=True)
    core.retry_policy = RetryPolicy(RateLimitErrorRetries=0)  # pyright: ignore[reportAttributeAccessIssue]  # a plain attribute of both cores
    host.answer("*", lambda **_: rate_limit())

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RateLimitError)
    assert host.deployments_called("acompletion") == ["d1"]


def test_model_group_retry_policy_assigned_after_build_answers_the_lookup(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core)
    policies: Final = {"g": RetryPolicy(RateLimitErrorRetries=1)}
    core.model_group_retry_policy = policies  # pyright: ignore[reportAttributeAccessIssue]  # see above

    assert core.model_group_retry_policy == policies  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.get_num_retries_from_retry_policy(rate_limit(), "g") == 1  # pyright: ignore[reportAttributeAccessIssue]  # see above


@pytest.mark.asyncio
async def test_a_retry_policy_written_as_a_dict_stays_a_dict_and_applies(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one_group(make_core, num_retries=1, retry_after=0, disable_cooldowns=True)
    core.retry_policy = {"RateLimitErrorRetries": 0}  # pyright: ignore[reportAttributeAccessIssue]  # stored as given
    host.answer("*", lambda **_: rate_limit())

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RateLimitError)
    assert core.retry_policy == {"RateLimitErrorRetries": 0}  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert host.deployments_called("acompletion") == ["d1"]


@pytest.mark.asyncio
async def test_a_retry_policy_written_as_no_policy_raises_on_the_first_failure_only(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = _one_group(make_core, num_retries=1, retry_after=0, disable_cooldowns=True)
    core.retry_policy = 5  # pyright: ignore[reportAttributeAccessIssue]  # stored as given

    served: Final = await attempt(core)
    host.answer("*", lambda **_: rate_limit())
    error: Final = await attempt(core)

    assert not isinstance(served, Exception)
    assert isinstance(error, AttributeError)
    assert "DefaultRetries" in str(error)


@pytest.mark.parametrize(
    ("name", "value"), [("enable_weighted_failover", True), ("fallback_on_unsupported_params", False)]
)
def test_other_plain_settings_read_back_what_was_written(make_core: MakeCore, name: str, value: object) -> None:
    core: Final = _one_group(make_core)
    setattr(core, name, value)
    assert getattr(core, name) is value


# --- N8: update_settings applies its keys in order, with no rollback --------------------------------


def test_update_settings_keeps_the_keys_before_the_one_that_raises(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core)
    with pytest.raises(TypeError):
        core.update_settings(  # pyright: ignore[reportAttributeAccessIssue]  # the deprecated member
            context_window_fallbacks=[{"g": ["h"]}], allowed_fails=["not a number"], timeout=3
        )
    assert core.settings().context_window_fallbacks == [{"g": ["h"]}]
    assert core.timeout != 3  # pyright: ignore[reportAttributeAccessIssue]  # a plain attribute of both cores


@pytest.mark.parametrize("strategy", [1, "bogus"])
def test_update_settings_refuses_an_unknown_strategy(make_core: MakeCore, strategy: object) -> None:
    core: Final = _one_group(make_core)
    with pytest.raises(ValueError, match="Invalid routing_strategy"):
        core.update_settings(fallbacks=[{"g": ["h"]}], routing_strategy=strategy, timeout=3)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.settings().routing_strategy == "simple-shuffle"
    assert core.settings().fallbacks == [{"g": ["h"]}]


def test_update_settings_keeps_no_strategy_as_given(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core)
    core.update_settings(routing_strategy=None)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.settings().routing_strategy is None


def test_a_failed_setting_leaves_the_groups_passed_before_it_unread(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core)
    groups: Final = [{"group_name": "z", "models": ["g"], "routing_strategy": "least-busy"}]
    with pytest.raises(ValueError, match="invalid literal"):
        core.update_settings(routing_groups=groups, timeout="soon")  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.settings().as_dict()["routing_groups"] == []


def test_groups_that_do_not_parse_raise_after_every_other_key(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core)
    with pytest.raises(Exception, match="group_name"):
        core.update_settings(timeout=6, routing_groups=[{"bad": 1}], retry_after=5)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert (core.timeout, core.retry_after) == (6, 5)  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_optional_pre_call_checks_that_cannot_be_read_stop_the_keys_after_them(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core)
    with pytest.raises(TypeError, match="not iterable"):
        core.update_settings(timeout=5, optional_pre_call_checks=1, retry_after=4)  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.timeout == 5  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert core.retry_after != 4  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_a_changeset_stops_at_the_setting_that_raises(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1"), deployment("h", "d2")], num_retries=0)
    with pytest.raises(TypeError):
        core.apply(ChangeSet(settings={"fallbacks": [{"g": ["h"]}], "allowed_fails": [1]}, delete=["d1"]))
    assert core.settings().fallbacks == [{"g": ["h"]}]
    assert sorted(core.registry().names(kind="ids")) == ["d1", "d2"]


# --- N7: a request num_retries is compared, never cast ----------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ["2", 1.5, [1], -1, float("nan"), True, False, 0.0])
async def test_a_request_that_succeeds_never_reads_its_num_retries(make_core: MakeCore, value: object) -> None:
    served: Final = await attempt(_one_group(make_core), num_retries=value)
    assert not isinstance(served, Exception)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("2", "'>' not supported between instances of 'str' and 'int'"),
        ([1], "'>' not supported between instances of 'list' and 'int'"),
        (1.5, "'float' object cannot be interpreted as an integer"),
    ],
)
async def test_a_failure_reads_a_num_retries_that_is_no_count(
    make_core: MakeCore, host: ScriptedHost, value: object, message: str
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0)
    host.answer("*", lambda **_: server_error())

    error: Final = await attempt(core, num_retries=value)

    assert isinstance(error, TypeError)
    assert str(error) == message
    assert isinstance(error.__context__, litellm.InternalServerError)
    assert len(host.deployments_called("acompletion")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [-1, float("nan"), False, 0.0])
async def test_a_num_retries_below_one_leaves_the_failure_standing(
    make_core: MakeCore, host: ScriptedHost, value: object
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=3)
    host.answer("*", lambda **_: server_error())

    error: Final = await attempt(core, num_retries=value)

    assert isinstance(error, litellm.InternalServerError)
    assert len(host.deployments_called("acompletion")) == 1


@pytest.mark.asyncio
async def test_a_retry_policy_outranks_a_num_retries_that_is_no_count(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")],
        num_retries=0,
        retry_policy=RetryPolicy(InternalServerErrorRetries=1),
        retry_after=0,
    )
    host.answer("*", server_error(), lambda **_: None)

    served: Final = await attempt(core, num_retries="2")

    assert not isinstance(served, Exception)
    assert len(host.deployments_called("acompletion")) == 2


@pytest.mark.asyncio
async def test_num_retries_true_counts_as_one_retry(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("g", "b")], num_retries=0, retry_after=0, disable_cooldowns=True
    )
    host.answer("*", lambda **_: server_error())

    error: Final = await attempt(core, num_retries=True)

    assert isinstance(error, litellm.InternalServerError)
    assert len(host.deployments_called("acompletion")) == 2


# --- N17: retire releases registrations, not the router ----------------------------------------------


@pytest.mark.asyncio
async def test_a_retired_core_keeps_serving(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = _one_group(make_core)
    core.retire()

    served: Final = await attempt(core)

    assert not isinstance(served, Exception)
    assert host.deployments_called("acompletion") == ["d1"]


# --- N18: attributes the Python core keeps as plain attributes ---------------------------------------


@pytest.mark.asyncio
async def test_a_deleted_scheduler_is_gone_for_reads_and_for_prioritized_calls(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core)
    del core.scheduler  # pyright: ignore[reportAttributeAccessIssue]  # a plain attribute of both cores

    with pytest.raises(AttributeError, match="scheduler"):
        _ = core.scheduler  # pyright: ignore[reportAttributeAccessIssue]  # see above
    served: Final = await attempt(core)
    refused: Final = await attempt(core, priority=1)

    assert not isinstance(served, Exception)
    assert isinstance(refused, AttributeError)
    assert "scheduler" in str(refused)


@pytest.mark.parametrize("name", ["total_calls", "success_calls", "fail_calls"])
def test_call_counters_are_replaced_and_deleted_like_attributes(make_core: MakeCore, name: str) -> None:
    core: Final = _one_group(make_core)
    replacement: Final[dict[str, int]] = {"openai/d1": 7}  # mutable-ok: the value under test

    setattr(core, name, replacement)
    assert getattr(core, name) is replacement

    delattr(core, name)
    with pytest.raises(AttributeError, match=name):
        getattr(core, name)


@pytest.mark.asyncio
async def test_a_success_counter_replaced_with_a_plain_dict_fails_the_attempt(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core)
    core.success_calls = {}  # pyright: ignore[reportAttributeAccessIssue]  # see above

    error: Final = await attempt(core)

    assert isinstance(error, KeyError)
    assert getattr(error, "failed_deployment_id", None) == "d1"
    assert core.fail_calls == {"openai/d1": 1}  # pyright: ignore[reportAttributeAccessIssue]  # see above


def test_embedding_names_its_parameters(make_core: MakeCore) -> None:
    signature: Final = inspect.signature(_one_group(make_core).embedding)  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint
    assert list(signature.parameters) == ["model", "input", "is_async", "kwargs"]
    assert signature.parameters["is_async"].default is False


def test_embedding_takes_its_arguments_by_position(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = _one_group(make_core)
    core.embedding("g", "hi")  # pyright: ignore[reportAttributeAccessIssue]  # an endpoint
    assert host.deployments_called("embedding") == ["d1"]


# --- O2: deleted aliases and retry budget -------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_deleted_alias_map_fails_every_read_until_it_is_assigned_again(make_core: MakeCore) -> None:
    core: Final = _one_group(make_core, model_group_alias={"al": "g"})
    del core.model_group_alias  # pyright: ignore[reportAttributeAccessIssue]  # the deprecated attribute

    assert "model_group_alias" not in core.settings().as_dict()
    with pytest.raises(AttributeError, match="model_group_alias"):
        core.registry().aliases()
    with pytest.raises(AttributeError, match="model_group_alias"):
        core.registry().names()
    for model in ("al", "g"):
        refused = await attempt(core, model=model)
        assert isinstance(refused, AttributeError)
        assert "model_group_alias" in str(refused)
    with pytest.raises(AttributeError, match="model_group_alias"):
        del core.model_group_alias  # pyright: ignore[reportAttributeAccessIssue]  # see above

    core.apply(ChangeSet(assign={"model_group_alias": {"x": "g"}}))

    assert not isinstance(await attempt(core, model="x"), Exception)


@pytest.mark.asyncio
async def test_a_deleted_num_retries_fails_calls_that_name_none(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "d1"), deployment("h", "d2")], fallbacks=[{"g": ["h"]}])
    del core.num_retries  # pyright: ignore[reportAttributeAccessIssue]  # a plain attribute of both cores

    assert "num_retries" not in core.settings().as_dict()
    with pytest.raises(AttributeError, match="num_retries"):
        _ = core.num_retries  # pyright: ignore[reportAttributeAccessIssue]  # see above
    refused: Final = await attempt(core)
    assert isinstance(refused, AttributeError)
    assert "num_retries" in str(refused)
    assert host.deployments_called("acompletion") == []

    assert not isinstance(await attempt(core, num_retries=0), Exception)
    with pytest.raises(AttributeError, match="num_retries"):
        del core.num_retries  # pyright: ignore[reportAttributeAccessIssue]  # see above

    core.num_retries = 1  # pyright: ignore[reportAttributeAccessIssue]  # see above

    assert core.num_retries == 1  # pyright: ignore[reportAttributeAccessIssue]  # see above
    assert not isinstance(await attempt(core), Exception)


# --- T5: a default deployment every call to fails ------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_info", "stamped"), [(None, None), ({"id": "dd"}, "dd")], ids=["no_model_info", "its_own_id"]
)
async def test_a_failing_default_deployment_stamps_only_its_own_id(
    make_core: MakeCore, model_info: dict[str, object] | None, stamped: str | None
) -> None:
    core: Final = _one_group(make_core)
    assigned: Final = {
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "k"},
        **({"model_info": model_info} if model_info is not None else {}),
    }
    core.apply(ChangeSet(assign={"default_deployment": assigned}))

    error: Final = await attempt(core, model="unknown")

    assert isinstance(error, KeyError)
    assert error.args == ("model_name",)
    assert getattr(error, "failed_deployment_id", None) == stamped


# --- S10: the capability limit is called as given --------------------------------------------------------


def test_a_capability_limit_that_cannot_be_called_fails_the_build(make_core: MakeCore) -> None:
    with pytest.raises(TypeError, match="not callable"):
        make_core(
            model_list=[
                deployment("big", "big"),
                deployment("small", "small"),
                complexity_marker("smart", "m1", config=HEURISTIC_V2),
            ],
            auto_router_capability_limit=0,
        )


# --- ruled to differ: the Rust core only ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("params", "named"),
    [
        ({"model_info": {"input_cost_per_token": "inf"}}, "model_info.input_cost_per_token"),
        ({"model_info": {"output_cost_per_token": "nan"}}, "model_info.output_cost_per_token"),
        ({"model_info": {"input_cost_per_token": "1e400"}}, "model_info.input_cost_per_token"),
        ({"model_info": {"input_cost_per_token": float("inf")}}, "model_info.input_cost_per_token"),
        ({"input_cost_per_token": "inf"}, "litellm_params.input_cost_per_token"),
    ],
    ids=["inf", "nan", "overflow", "float", "litellm_params"],
)
def test_the_rust_core_refuses_a_cost_that_is_not_finite(
    host: ScriptedHost, params: dict[str, object], named: str
) -> None:
    with pytest.raises(ValueError, match=f"{named} is .*Set a finite cost"):
        CORE_BUILDERS["rust"](host, model_list=[deployment("g", "m", model="openai/gpt-4o", **params)])


def test_the_rust_core_refuses_a_cost_that_is_not_finite_on_add(host: ScriptedHost) -> None:
    core: Final = CORE_BUILDERS["rust"](host, model_list=[deployment("g", "m")])
    added: Final = Deployment(
        model_name="g",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o", api_key="k"),
        model_info={"id": "late", "input_cost_per_token": float("inf")},
    )
    with pytest.raises(ValueError, match="Set a finite cost"):
        core.add_deployment(added)  # pyright: ignore[reportAttributeAccessIssue]  # the deprecated member
    assert core.registry().names(kind="ids") == ["m"]


def test_the_rust_core_leaves_out_a_bad_cost_under_ignore_invalid_deployments(host: ScriptedHost) -> None:
    core: Final = CORE_BUILDERS["rust"](
        host,
        model_list=[deployment("g", "m", model_info={"input_cost_per_token": "inf"}), deployment("g", "n")],
        ignore_invalid_deployments=True,
    )
    assert core.registry().names(kind="ids") == ["n"]


@pytest.mark.parametrize("share", [10**20, 1e20], ids=["int", "float"])
def test_the_rust_core_refuses_a_share_wider_than_it_can_hold(host: ScriptedHost, share: object) -> None:
    listed: Final = deployment("g", "sh", model_info={"db_model": True, "ptu_shares": {"T1": share}})
    with pytest.raises(ValueError, match=r"model_info\.ptu_shares\.T1 is 100000000000000000000, outside"):
        CORE_BUILDERS["rust"](host, model_list=[listed])


def test_the_rust_core_refuses_a_share_wider_than_it_can_hold_on_add(host: ScriptedHost) -> None:
    core: Final = CORE_BUILDERS["rust"](host, model_list=[deployment("g", "free")])
    added: Final = Deployment(
        model_name="g",
        litellm_params=LiteLLM_Params(model="openai/gpt-4o", api_key="k"),
        model_info={"id": "late", "db_model": True, "ptu_shares": {"T1": 10**20}},
    )
    with pytest.raises(ValueError, match=r"model_info\.ptu_shares\.T1 is 100000000000000000000, outside"):
        core.add_deployment(added)  # pyright: ignore[reportAttributeAccessIssue]  # the deprecated member
    assert core.registry().names(kind="ids") == ["free"]


# --- S5: in-place edits never reach the Rust walk, or its notes -------------------------------------------


@pytest.mark.asyncio
async def test_the_rust_error_note_reads_the_fallbacks_the_walk_used(host: ScriptedHost) -> None:
    host.set_dynamic(expose_router_debug_in_errors=True)
    core: Final = CORE_BUILDERS["rust"](
        host,
        model_list=[deployment("g", "d1"), deployment("h", "h1")],
        num_retries=0,
        fallbacks=[{"x": ["h"]}],
    )
    core.settings().fallbacks.append({"g": ["h"]})  # pyright: ignore[reportOptionalMemberAccess]  # the live list
    host.answer("*", lambda **_: server_error())

    error: Final = await attempt(core)

    assert isinstance(error, litellm.InternalServerError)
    assert "configured: h" not in error.message
    assert host.deployments_called("acompletion") == ["d1"]
