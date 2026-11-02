"""
Characterization of deployment health in a router core: when a failure cools a deployment down (by status,
group size, allowed_fails and allowed_fails_policy at router and deployment level), what the failure and
success callbacks count, blocked deployments and health-check-driven filtering.
"""

import time
from typing import Final

import pytest

import litellm
from litellm.constants import DEFAULT_COOLDOWN_TIME_SECONDS, DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS
from litellm.router_utils.cooldown_handlers import mark_advisor_orchestration_failure
from litellm.types.router import AllowedFailsPolicy, Deployment
from tests.unit.test_router.core.retries_health_helpers import (
    BackgroundSuccessLoggingHost,
    ErrorBuilder,
    MakeCore,
    api_connection,
    api_status,
    attempt,
    cooled_down,
    fresh,
    not_found,
    permission_denied,
    service_unavailable,
)
from tests.unit.test_router.core.scripted_host import (
    ScriptedHost,
    auth_error,
    bad_request,
    deployment,
    rate_limit,
    server_error,
    timeout,
)


def group(
    *ids: str, **params: object
) -> list[dict[str, object]]:  # kwargs-ok: litellm_params  # mutable-ok: a model_list
    return [deployment("g", deployment_id, **params) for deployment_id in ids]


def unhealthy(*ids: str, age: float = 0.0) -> dict[str, dict[str, object]]:  # mutable-ok: health states
    return {i: {"is_healthy": False, "timestamp": time.time() - age, "reason": "probe failed"} for i in ids}


def failure_kwargs(
    model_info: object, exception: Exception, **extra: object
) -> dict[str, object]:  # kwargs-ok: logging kwargs  # mutable-ok: litellm's logging kwargs
    litellm_params: Final = {"metadata": {"model_group": "g"}, "model_info": model_info}
    litellm_params.update(extra.pop("litellm_params", {}))  # pyright: ignore[reportCallIssue, reportArgumentType]  # test extras
    return {"exception": exception, "litellm_params": litellm_params, **extra}


# --- which failures cool a deployment down -------------------------------------------------------


@pytest.mark.parametrize("builder", [auth_error, not_found])
@pytest.mark.asyncio
async def test_a_lone_deployment_cools_down_on_a_401_or_404(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder
) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=0)
    host.answer("openai/a", fresh(builder))

    first: Final = await attempt(core)
    second: Final = await attempt(core)

    assert type(first) is type(builder())
    assert set(cooled_down(core, ["a"])) == {"a"}
    assert isinstance(second, litellm.RouterRateLimitError)
    assert host.deployments_called("acompletion") == ["a"]


@pytest.mark.asyncio
async def test_a_404_cools_a_group_deployment_down_at_once(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0)
    host.answer("*", fresh(not_found))

    await attempt(core)

    assert list(cooled_down(core, ["a", "b"])) == host.deployments_called("acompletion")


@pytest.mark.parametrize("builder", [permission_denied, bad_request, api_status(422), api_status(409)])
@pytest.mark.asyncio
async def test_other_4xx_errors_never_cool_down(make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, allowed_fails=0)
    host.answer("*", fresh(builder))

    for _ in range(4):
        await attempt(core)

    assert cooled_down(core, ["a", "b"]) == {}


@pytest.mark.parametrize("builder", [server_error, timeout, service_unavailable])
@pytest.mark.asyncio
async def test_by_default_a_group_deployment_cools_down_on_its_fifth_failure_this_minute(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0)
    host.answer("*", fresh(builder))

    while not cooled_down(core, ["a", "b"]):
        await attempt(core)

    called: Final = host.deployments_called("acompletion")
    (cooled,) = cooled_down(core, ["a", "b"])
    assert called[-1] == cooled
    assert called.count(cooled) == DEFAULT_FAILURE_THRESHOLD_MINIMUM_REQUESTS


@pytest.mark.asyncio
async def test_successes_this_minute_keep_the_failure_rate_under_the_default_threshold(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"))
    for _ in range(5):
        core.sync_deployment_callback_on_success(  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
            {"litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": "a"}}}, None, 0.0, 0.0
        )

    outcomes: Final = [core.health().record_failure("a", server_error(), 500) for _ in range(6)]

    assert outcomes == [False] * 5 + [True]


@pytest.mark.parametrize("allowed_fails", [3, 1])
@pytest.mark.asyncio
async def test_an_explicit_allowed_fails_counts_failures_even_at_the_default_value(
    make_core: MakeCore, allowed_fails: int
) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=allowed_fails)

    outcomes: Final = [core.health().record_failure("a", server_error(), 500) for _ in range(allowed_fails + 1)]

    assert outcomes == [False] * allowed_fails + [True]


@pytest.mark.asyncio
async def test_a_connection_error_never_cools_down(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, allowed_fails=0)
    host.answer("*", fresh(api_connection))

    for _ in range(3):
        await attempt(core)

    assert cooled_down(core, ["a", "b"]) == {}


@pytest.mark.parametrize(("ids", "cools"), [(("a", "b"), True), (("a",), False)], ids=["group", "lone"])
@pytest.mark.asyncio
async def test_a_402_cools_a_group_deployment_but_not_a_lone_one(
    make_core: MakeCore, host: ScriptedHost, ids: tuple[str, ...], cools: bool
) -> None:
    core: Final = make_core(model_list=group(*ids), num_retries=0)
    host.answer("*", fresh(api_status(402)))

    await attempt(core)

    assert bool(cooled_down(core, list(ids))) is cools


@pytest.mark.asyncio
async def test_a_lone_deployment_with_a_fallback_group_cools_down_under_explicit_allowed_fails(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("f", "z")],
        num_retries=0,
        allowed_fails=0,
        fallbacks=[{"g": ["f"]}],
    )
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert set(cooled_down(core, ["a", "z"])) == {"a"}
    assert host.deployments_called("acompletion") == ["a", "z"]


@pytest.mark.asyncio
async def test_a_lone_deployment_with_a_fallback_group_is_not_cooled_by_a_429_under_defaults(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("f", "z")], num_retries=0, fallbacks=[{"g": ["f"]}]
    )
    host.answer("openai/a", fresh(rate_limit))

    await attempt(core)

    assert cooled_down(core, ["a", "z"]) == {}


@pytest.mark.asyncio
async def test_record_failure_never_cools_down_when_cooldowns_are_disabled(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0, disable_cooldowns=True)

    cooled: Final = core.health().record_failure("a", rate_limit(), 429)

    assert cooled is False
    assert cooled_down(core, ["a", "b"]) == {}
    assert core.health().failures_this_minute("a") == 1


@pytest.mark.parametrize(
    ("deployment_id", "status"), [("missing", 429), ("a", ""), ("a", 400)], ids=["unknown", "empty", "400"]
)
@pytest.mark.asyncio
async def test_record_failure_without_a_cooldown(make_core: MakeCore, deployment_id: str, status: object) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0)

    assert core.health().record_failure(deployment_id, rate_limit(), status) is False  # pyright: ignore[reportArgumentType]  # status as given
    assert core.health().failures_this_minute(deployment_id) == 1


@pytest.mark.asyncio
async def test_a_router_cooldown_time_of_zero_falls_back_to_the_default(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0, cooldown_time=0)
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    assert core.settings().cooldown_time == DEFAULT_COOLDOWN_TIME_SECONDS
    (entry,) = cooled_down(core, ["a", "b"]).values()
    assert entry["cooldown_time"] == DEFAULT_COOLDOWN_TIME_SECONDS


# --- allowed_fails_policy ------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_router_allowed_fails_policy_counts_every_error_type_in_one_counter(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=group("a", "b"),
        allowed_fails=5,
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=1),
    )

    outcomes: Final = [
        core.health().record_failure("a", server_error(), 500),
        core.health().record_failure("a", server_error(), 500),
        core.health().record_failure("a", rate_limit(), 429),
    ]

    assert outcomes == [False, False, True]


@pytest.mark.parametrize(
    ("field", "error", "status"),
    [
        ("AuthenticationErrorAllowedFails", auth_error, 401),
        ("TimeoutErrorAllowedFails", timeout, 408),
        ("RateLimitErrorAllowedFails", rate_limit, 429),
        ("InternalServerErrorAllowedFails", server_error, 500),
        ("ServiceUnavailableErrorAllowedFails", service_unavailable, 503),
        ("NotFoundErrorAllowedFails", not_found, 404),
    ],
)
@pytest.mark.asyncio
async def test_router_allowed_fails_policy_threshold_per_error_type(
    make_core: MakeCore, field: str, error: ErrorBuilder, status: int
) -> None:
    core: Final = make_core(
        model_list=group("a", "b"), allowed_fails=10, allowed_fails_policy=AllowedFailsPolicy(**{field: 2})
    )

    outcomes: Final = [core.health().record_failure("a", error(), status) for _ in range(3)]

    assert outcomes == [False, False, True]


@pytest.mark.asyncio
async def test_router_allowed_fails_policy_for_bad_requests_does_not_cool_down(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=group("a", "b"), allowed_fails_policy=AllowedFailsPolicy(BadRequestErrorAllowedFails=0)
    )

    outcomes: Final = [core.health().record_failure("a", bad_request(), 400) for _ in range(3)]

    assert outcomes == [False, False, False]


@pytest.mark.asyncio
async def test_deployment_allowed_fails_policy_counts_each_error_type_separately(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=group(
            "a",
            "b",
            model_info={
                "allowed_fails_policy": {"RateLimitErrorAllowedFails": 1, "InternalServerErrorAllowedFails": 1}
            },
        )
    )

    outcomes: Final = [
        core.health().record_failure("a", rate_limit(), 429),
        core.health().record_failure("a", server_error(), 500),
        core.health().record_failure("a", rate_limit(), 429),
    ]

    assert outcomes == [False, False, True]


@pytest.mark.asyncio
async def test_deployment_allowed_fails_policy_wins_over_the_router_policy(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=[
            deployment("g", "a", model_info={"allowed_fails_policy": {"RateLimitErrorAllowedFails": 2}}),
            deployment("g", "b"),
        ],
        allowed_fails_policy=AllowedFailsPolicy(RateLimitErrorAllowedFails=0),
    )

    a_outcomes: Final = [core.health().record_failure("a", rate_limit(), 429) for _ in range(3)]
    b_outcome: Final = core.health().record_failure("b", rate_limit(), 429)

    assert a_outcomes == [False, False, True]
    assert b_outcome is True


@pytest.mark.asyncio
async def test_deployment_policy_not_naming_the_error_falls_back_to_router_allowed_fails(make_core: MakeCore) -> None:
    core: Final = make_core(
        model_list=group("a", "b", model_info={"allowed_fails_policy": {"RateLimitErrorAllowedFails": 9}})
    )

    outcomes: Final = [core.health().record_failure("a", server_error(), 500) for _ in range(4)]

    assert outcomes == [False, False, False, True]


@pytest.mark.parametrize(("ids", "cools"), [(("a", "b"), True), (("a",), False)], ids=["group", "lone"])
@pytest.mark.asyncio
async def test_deployment_allowed_fails_in_model_info(make_core: MakeCore, ids: tuple[str, ...], cools: bool) -> None:
    core: Final = make_core(model_list=group(*ids, model_info={"allowed_fails": 1}))

    outcomes: Final = [core.health().record_failure("a", server_error(), 500) for _ in range(2)]

    assert outcomes == [False, cools]


@pytest.mark.asyncio
async def test_deployment_allowed_fails_policy_cools_down_a_lone_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", model_info={"allowed_fails_policy": {"TimeoutErrorAllowedFails": 0}}))

    assert core.health().record_failure("a", timeout(), 408) is True


@pytest.mark.asyncio
async def test_deployment_cooldown_time_in_model_info_sets_the_cooldown(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=group(
            "a", "b", model_info={"cooldown_time": 17, "allowed_fails_policy": {"RateLimitErrorAllowedFails": 0}}
        ),
        num_retries=0,
    )
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    (entry,) = cooled_down(core, ["a", "b"]).values()
    assert entry["cooldown_time"] == 17


# --- deployment_callback_on_failure ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_failure_callback_counts_the_failure_and_cools_down(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0)

    cooled: Final = core.deployment_callback_on_failure(failure_kwargs({"id": "a"}, server_error()), None, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member

    assert cooled is True
    assert core.health().failures_this_minute("a") == 1
    assert set(cooled_down(core, ["a", "b"])) == {"a"}


@pytest.mark.parametrize("model_info", [None, "a", {}, {"name": "no id"}], ids=["none", "not-a-dict", "empty", "no-id"])
@pytest.mark.asyncio
async def test_failure_callback_without_a_deployment_id_does_nothing(make_core: MakeCore, model_info: object) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0)

    cooled: Final = core.deployment_callback_on_failure(failure_kwargs(model_info, server_error()), None, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member

    assert cooled is False
    assert core.health().failures_this_minute("a") == 0


@pytest.mark.asyncio
async def test_failure_callback_ignores_an_advisor_orchestration_failure(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0)
    error: Final = server_error()
    mark_advisor_orchestration_failure(error)

    cooled: Final = core.deployment_callback_on_failure(failure_kwargs({"id": "a"}, error), None, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member

    assert cooled is False
    assert core.health().failures_this_minute("a") == 0


@pytest.mark.parametrize(
    "extra",
    [
        {
            "litellm_params": {
                "metadata": {"model_group": "g", "internal_call_origin": "background_response_status_poll"}
            }
        },
        {"call_type": "avideo_status"},
    ],
    ids=["background-poll", "video-lookup"],
)
@pytest.mark.asyncio
async def test_failure_callback_ignores_a_404_that_is_not_about_the_deployment(
    make_core: MakeCore, extra: dict[str, object]
) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0)

    cooled: Final = core.deployment_callback_on_failure(  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
        failure_kwargs({"id": "a"}, not_found(), **extra), None, 0.0, 0.0
    )

    assert cooled is False
    assert cooled_down(core, ["a", "b"]) == {}


@pytest.mark.asyncio
async def test_failure_callback_ignores_a_408_from_the_callers_own_timeout(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0)
    started: Final = time.time()
    from datetime import datetime, timedelta

    start: Final = datetime.fromtimestamp(started)
    kwargs: Final = failure_kwargs(
        {"id": "a"},
        timeout(),
        litellm_params={"client_side_timeout": True, "timeout": 1.0},
        start_time=start,
        end_time=start + timedelta(seconds=2),
    )

    cooled: Final = core.deployment_callback_on_failure(kwargs, None, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member

    assert cooled is False
    assert core.health().failures_this_minute("a") == 0


@pytest.mark.parametrize(
    ("model_info", "litellm_params", "expected"),
    [
        ({"id": "a", "cooldown_time": 3}, {"cooldown_time": 4}, 3),
        ({"id": "a"}, {"cooldown_time": 4}, 4),
        ({"id": "a", "cooldown_time": -1}, {}, 60),
    ],
    ids=["model-info-first", "litellm-params", "negative-ignored"],
)
@pytest.mark.asyncio
async def test_failure_callback_cooldown_time_resolution(
    make_core: MakeCore, model_info: dict[str, object], litellm_params: dict[str, object], expected: float
) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0, cooldown_time=60)

    core.deployment_callback_on_failure(  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
        failure_kwargs(model_info, server_error(), litellm_params=litellm_params), None, 0.0, 0.0
    )

    (entry,) = cooled_down(core, ["a", "b"]).values()
    assert entry["cooldown_time"] == expected


# --- success callbacks and usage counters --------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_success_callback_counts_a_success_and_returns_its_key(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"))
    kwargs: Final = {"litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": "a"}}}

    key: Final = core.sync_deployment_callback_on_success(kwargs, None, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
    core.sync_deployment_callback_on_success(kwargs, None, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member

    assert key == "a:successes"
    assert core.health().store.get_cache("a:successes", local_only=True) == 2


@pytest.mark.asyncio
async def test_sync_success_callback_turns_an_integer_id_into_a_string_key(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a"))

    key: Final = core.sync_deployment_callback_on_success(  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
        {"litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": 7}}}, None, 0.0, 0.0
    )

    assert key == "7:successes"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"litellm_params": {"metadata": None, "model_info": {"id": "a"}}},
        {"litellm_params": {"metadata": {}, "model_info": {"id": "a"}}},
        {"litellm_params": {"metadata": {"model_group": "g"}, "model_info": None}},
        {
            "call_type": "aretrieve_batch",
            "litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": "a"}},
        },
    ],
    ids=["no-metadata", "no-group", "no-model-info", "batch-retrieve"],
)
@pytest.mark.asyncio
async def test_sync_success_callback_counts_nothing_without_a_group_and_id(
    make_core: MakeCore, kwargs: dict[str, object]
) -> None:
    core: Final = make_core(model_list=group("a"))

    assert core.sync_deployment_callback_on_success(kwargs, None, 0.0, 0.0) is None  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
    assert core.health().store.get_cache("a:successes", local_only=True) is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"call_type": "_arealtime"},
        {"call_type": "aretrieve_batch"},
        {"litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": "a"}}},
    ],
    ids=["realtime", "batch-retrieve", "no-standard-logging-object"],
)
@pytest.mark.asyncio
async def test_async_success_callback_skips_calls_it_does_not_track(
    make_core: MakeCore, kwargs: dict[str, object]
) -> None:
    core: Final = make_core(model_list=group("a", rpm=100))
    full: Final = {
        "litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": "a"}},
        "standard_logging_object": {"total_tokens": 10},
        **kwargs,
    }
    if "litellm_params" in kwargs:
        del full["standard_logging_object"]

    result: Final = await core.deployment_callback_on_success(full, None, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member

    assert result is None
    assert core.health().store.get_cache("a:successes", local_only=True) is None
    assert await core.health().usage("g") == (None, None)


@pytest.mark.asyncio
async def test_async_success_callback_for_an_unknown_deployment_counts_nothing(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", rpm=100))

    await core.deployment_callback_on_success(  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
        {
            "litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": "zz"}},
            "standard_logging_object": {"total_tokens": 10},
        },
        None,
        0.0,
        0.0,
    )

    assert core.health().store.get_cache("zz:successes", local_only=True) is None


@pytest.mark.asyncio
async def test_async_success_callback_writes_tokens_and_a_request_for_a_tracked_deployment(
    make_core: MakeCore,
) -> None:
    core: Final = make_core(model_list=group("a", rpm=100))

    await core.deployment_callback_on_success(  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
        {
            "litellm_params": {
                "metadata": {"model_group": "g", "deployment": "openai/a"},
                "model_info": {"id": "a"},
            },
            "standard_logging_object": {"total_tokens": 12},
        },
        None,
        0.0,
        0.0,
    )

    assert core.health().store.get_cache("a:successes", local_only=True) == 1
    assert await core.health().usage("g") == (12, 1)


@pytest.mark.asyncio
async def test_async_success_callback_adds_only_tokens_not_yet_counted(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", rpm=100))

    await core.deployment_callback_on_success(  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
        {
            "litellm_params": {
                "metadata": {
                    "model_group": "g",
                    "deployment": "openai/a",
                    "_litellm_router_usage_counted_tokens": 8,
                },
                "model_info": {"id": "a"},
            },
            "standard_logging_object": {"total_tokens": 12},
        },
        None,
        0.0,
        0.0,
    )

    assert await core.health().usage("g") == (4, 0)


@pytest.mark.asyncio
async def test_async_success_callback_counts_successes_for_an_untracked_deployment(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a"))

    await core.deployment_callback_on_success(  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
        {
            "litellm_params": {"metadata": {"model_group": "g"}, "model_info": {"id": "a"}},
            "standard_logging_object": {"total_tokens": 12},
        },
        None,
        0.0,
        0.0,
    )

    assert core.health().store.get_cache("a:successes", local_only=True) == 1
    assert await core.health().usage("g") == (None, None)


@pytest.mark.parametrize("limit", [{"rpm": 100}, {"tpm": 1000}], ids=["rpm", "tpm"])
@pytest.mark.asyncio
async def test_a_successful_request_counts_its_tokens_and_one_request_once(
    make_core: MakeCore, host: ScriptedHost, limit: dict[str, int]
) -> None:
    core: Final = make_core(model_list=group("a", **limit))

    await attempt(core)
    if isinstance(host, BackgroundSuccessLoggingHost):
        await host.drain()

    assert await core.health().usage("g") == (10, 1)


@pytest.fixture
def host(request: pytest.FixtureRequest) -> ScriptedHost:
    """Background success logging for the usage test above, the plain host for the rest."""
    if request.node.name.startswith("test_a_successful_request_counts_its_tokens"):  # pyright: ignore[reportAny]  # pytest node
        return BackgroundSuccessLoggingHost()
    return ScriptedHost()


@pytest.mark.asyncio
async def test_a_failed_request_counts_one_request_and_no_tokens(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a", rpm=100), num_retries=0)
    host.answer("openai/a", fresh(bad_request))

    await attempt(core)

    assert await core.health().usage("g") == (None, 1)
    assert core.health().failures_this_minute("a") == 1


@pytest.mark.asyncio
async def test_an_untracked_deployment_reports_no_usage(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"))

    await attempt(core)

    assert await core.health().usage("g") == (None, None)


@pytest.mark.asyncio
async def test_a_successful_request_counts_one_success_this_minute(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"))

    await attempt(core)
    await attempt(core)

    assert core.health().store.get_cache("a:successes", local_only=True) == 2
    assert core.health().failures_this_minute("a") == 0


# --- blocked deployments -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_blocked_deployment_is_never_called(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", model_info={"blocked": True}), deployment("g", "b")])

    for _ in range(4):
        await attempt(core)

    assert host.deployments_called("acompletion") == ["b"] * 4
    assert core.health().fully_blocked_model_names() == set()


@pytest.mark.asyncio
async def test_a_group_with_every_deployment_blocked_raises_no_deployments_available(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b", model_info={"blocked": True}))

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RouterRateLimitError)
    assert "No deployments available for selected model" in str(error)
    assert host.calls == []
    assert core.health().fully_blocked_model_names() == {"g"}


def test_is_blocked_and_all_blocked_read_model_info(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a"))
    blocked: Final = Deployment(**deployment("g", "a", model_info={"blocked": True}))  # pyright: ignore[reportArgumentType]  # a model_list entry
    open_: Final = Deployment(**deployment("g", "b"))  # pyright: ignore[reportArgumentType]  # a model_list entry

    assert core.health().is_blocked(blocked) is True
    assert core.health().is_blocked(open_) is False
    assert core.health().all_blocked([blocked.model_dump()]) is True  # pyright: ignore[reportArgumentType]  # deployment dicts
    assert core.health().all_blocked([blocked.model_dump(), open_.model_dump()]) is False  # pyright: ignore[reportArgumentType]  # deployment dicts


# --- health-check-driven filtering ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_check_routing_skips_deployments_marked_unhealthy(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a", "b"), enable_health_check_routing=True)
    core.health().health_checks.set_deployment_health_states(unhealthy("a"))  # pyright: ignore[reportArgumentType]  # health states

    for _ in range(4):
        await attempt(core)

    assert host.deployments_called("acompletion") == ["b"] * 4
    assert await core.health().fully_unhealthy_model_names() == set()


@pytest.mark.asyncio
async def test_health_check_routing_uses_every_deployment_when_all_are_unhealthy(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b"), enable_health_check_routing=True)
    core.health().health_checks.set_deployment_health_states(unhealthy("a", "b"))  # pyright: ignore[reportArgumentType]  # health states

    for _ in range(12):
        await attempt(core)

    assert set(host.deployments_called("acompletion")) == {"a", "b"}
    assert await core.health().fully_unhealthy_model_names() == {"g"}


@pytest.mark.parametrize(
    ("arguments", "age"),
    [
        ({}, 0.0),
        (
            {
                "enable_health_check_routing": True,
                "allowed_fails_policy": AllowedFailsPolicy(TimeoutErrorAllowedFails=1),
            },
            0.0,
        ),
        ({"enable_health_check_routing": True, "health_check_staleness_threshold": 10}, 30.0),
        ({"enable_health_check_routing": True, "background_health_check_model_groups": ["other"]}, 0.0),
    ],
    ids=["routing-off", "allowed-fails-policy", "stale", "group-not-listed"],
)
@pytest.mark.asyncio
async def test_health_check_state_is_ignored(
    make_core: MakeCore, host: ScriptedHost, arguments: dict[str, object], age: float
) -> None:
    core: Final = make_core(model_list=group("a", "b"), **arguments)
    core.health().health_checks.set_deployment_health_states(unhealthy("a", age=age))  # pyright: ignore[reportArgumentType]  # health states

    for _ in range(12):
        await attempt(core)

    assert set(host.deployments_called("acompletion")) == {"a", "b"}


@pytest.mark.asyncio
async def test_listed_health_check_group_is_filtered_even_with_an_allowed_fails_policy(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=group("a", "b"),
        enable_health_check_routing=True,
        background_health_check_model_groups=["g"],
        allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=1),
    )
    core.health().health_checks.set_deployment_health_states(unhealthy("a"))  # pyright: ignore[reportArgumentType]  # health states

    for _ in range(4):
        await attempt(core)

    assert host.deployments_called("acompletion") == ["b"] * 4


@pytest.mark.asyncio
async def test_an_unhealthy_deployment_stays_excluded_when_the_healthy_one_is_cooling_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b"), enable_health_check_routing=True, num_retries=0)
    core.health().health_checks.set_deployment_health_states(unhealthy("a"))  # pyright: ignore[reportArgumentType]  # health states
    host.answer("openai/b", fresh(rate_limit))

    await attempt(core)
    error: Final = await attempt(core)

    assert set(cooled_down(core, ["a", "b"])) == {"b"}
    assert isinstance(error, litellm.RouterRateLimitError)
    assert "cooldown_list=['b']" in str(error)
    assert host.deployments_called("acompletion") == ["b"]


def test_a_cooldown_recorded_outside_an_event_loop_is_written_and_reported(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0)

    cooled: Final = core.health().record_failure("a", server_error(), 500)

    assert cooled is True
    assert set(cooled_down(core, ["a", "b"])) == {"a"}


def test_a_failure_below_the_threshold_outside_an_event_loop_is_only_counted(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=3)

    cooled: Final = core.health().record_failure("a", server_error(), 500)

    assert cooled is False
    assert core.health().failures_this_minute("a") == 1
    assert cooled_down(core, ["a", "b"]) == {}


def test_failure_callback_outside_an_event_loop_cools_down(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a", "b"), allowed_fails=0)

    cooled: Final = core.deployment_callback_on_failure(failure_kwargs({"id": "a"}, server_error()), None, 0.0, 0.0)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownVariableType]  # PythonCore's public member

    assert cooled is True
    assert core.health().failures_this_minute("a") == 1
    assert set(cooled_down(core, ["a", "b"])) == {"a"}


@pytest.mark.asyncio
async def test_a_request_cooldown_time_reaches_the_call_and_sets_the_cooldown(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=0)
    host.answer("*", fresh(rate_limit))

    await attempt(core, cooldown_time=11)

    assert host.calls[0].kwargs["cooldown_time"] == 11
    (entry,) = cooled_down(core, ["a", "b"]).values()
    assert entry["cooldown_time"] == 11
