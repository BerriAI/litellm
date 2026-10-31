"""
Retries and cooldowns of a router core, driven through its public surface on a ``ScriptedHost``.

Backoff values carry litellm's random jitter (up to ``JITTER`` seconds), so sleeps are asserted
inside ``[base, base + JITTER)`` rather than exactly.
"""

from collections.abc import Callable, Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.constants import DEFAULT_COOLDOWN_TIME_SECONDS, INITIAL_RETRY_DELAY, JITTER
from litellm.router_api import ChangeSet
from litellm.router_core import RouterCore
from litellm.types.router import AllowedFailsPolicy, RetryPolicy
from tests.unit.test_router.core.scripted_host import (
    Call,
    ScriptedHost,
    auth_error,
    bad_request,
    content_policy,
    context_window,
    deployment,
    rate_limit,
    server_error,
    timeout,
)

MESSAGES: Final = [{"role": "user", "content": "hi"}]

MakeCore = Callable[..., RouterCore]
ErrorBuilder = Callable[[], Exception]


class LitellmParamsLoggingHost(ScriptedHost):
    """``ScriptedHost`` whose failure logging carries the call's ``cooldown_time`` in ``litellm_params``, as
    litellm's own logging does (``get_litellm_params``)."""

    def _logging_kwargs(
        self, call: Call, error: BaseException | None, response: object
    ) -> dict[str, object]:  # mutable-ok: litellm's logging kwargs
        logged: Final = super()._logging_kwargs(call, error, response)
        litellm_params: Final = logged["litellm_params"]
        if isinstance(litellm_params, dict) and "cooldown_time" in call.kwargs:
            litellm_params["cooldown_time"] = call.kwargs["cooldown_time"]
        return logged


@pytest.fixture
def host() -> ScriptedHost:
    return LitellmParamsLoggingHost()


def fresh(builder: ErrorBuilder) -> Callable[..., Exception]:
    """A scripted outcome that raises a new error per call, so stamps from one call never leak into the next."""
    return lambda **_: builder()


def rate_limited(retry_after: str | None = None) -> litellm.RateLimitError:
    headers: Final = {"retry-after": retry_after} if retry_after is not None else {}
    return litellm.RateLimitError(
        message="slow down", llm_provider="openai", model="gpt-4o", response=httpx.Response(429, headers=headers)
    )


async def attempt(core: RouterCore, model: str = "g", **kwargs: object) -> object:  # kwargs-ok: request kwargs
    """The response, or the error the core raised."""
    try:
        return await core.acompletion(model=model, messages=MESSAGES, **kwargs)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
    except Exception as error:
        return error


def cooled_down(core: RouterCore, ids: list[str]) -> dict[str, Mapping[str, object]]:  # mutable-ok: a fresh dict
    return dict(core.health().cooldowns.get_active_cooldowns(ids, None))


def headers_of(response: object) -> Mapping[str, object]:
    hidden: Final = getattr(response, "_hidden_params", None) or {}
    return hidden.get("additional_headers") or {}


def two_deployments(
    **params: object,
) -> list[dict[str, object]]:  # kwargs-ok: litellm_params  # mutable-ok: a model_list
    return [deployment("g", "a", **params), deployment("g", "b", **params)]


# --- how many retries ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("constructor", "request_value", "process", "expected_retries"),
    [
        (None, None, None, 2),
        (3, None, None, 3),
        (None, None, 4, 4),
        (1, None, 4, 1),
        (3, 1, None, 1),
        (3, 0, None, 0),
        (None, 2, 5, 2),
        (0, 2, None, 2),
    ],
)
@pytest.mark.asyncio
async def test_num_retries_comes_from_request_then_constructor_then_process_then_default(
    make_core: MakeCore,
    host: ScriptedHost,
    constructor: int | None,
    request_value: int | None,
    process: int | None,
    expected_retries: int,
) -> None:
    if process is not None:
        host.set_dynamic(num_retries=process)
    arguments: Final = {"num_retries": constructor} if constructor is not None else {}
    core: Final = make_core(model_list=[deployment("g", "a")], **arguments)
    host.answer("openai/a", fresh(server_error))
    request: Final = {"num_retries": request_value} if request_value is not None else {}

    error: Final = await attempt(core, **request)

    assert isinstance(error, litellm.InternalServerError)
    assert host.deployments_called("acompletion") == ["a"] * (expected_retries + 1)


@pytest.mark.asyncio
async def test_process_num_retries_is_read_once_when_the_core_is_built(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")])
    host.set_dynamic(num_retries=5)
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert len(host.calls_to("acompletion")) == 3


@pytest.mark.asyncio
async def test_num_retries_applied_through_settings_changes_later_requests(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=3)
    core.apply(ChangeSet(settings={"num_retries": 1}))
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert len(host.calls_to("acompletion")) == 2


# --- which errors retry -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "expected_calls"),
    [
        (bad_request, 1),
        (auth_error, 1),
        (context_window, 1),
        (content_policy, 1),
        (rate_limit, 3),
        (server_error, 3),
        (timeout, 3),
    ],
)
@pytest.mark.asyncio
async def test_lone_deployment_retries_only_retryable_errors(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder, expected_calls: int
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2)
    host.answer("openai/a", fresh(builder))

    error: Final = await attempt(core)

    assert type(error) is type(builder())
    assert len(host.calls_to("acompletion")) == expected_calls


@pytest.mark.parametrize(
    ("builder", "expected_calls"),
    [(bad_request, 1), (context_window, 1), (content_policy, 1), (server_error, 3), (timeout, 3)],
)
@pytest.mark.asyncio
async def test_two_deployment_group_retries_only_retryable_errors(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder, expected_calls: int
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=2)
    host.answer("*", fresh(builder))

    error: Final = await attempt(core)

    assert type(error) is type(builder())
    assert len(host.calls_to("acompletion")) == expected_calls


@pytest.mark.asyncio
async def test_auth_error_retries_on_the_other_deployment_of_the_group(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=2)
    host.answer("*", fresh(auth_error))

    error: Final = await attempt(core)

    assert isinstance(error, litellm.AuthenticationError)
    assert sorted(host.deployments_called("acompletion")) == ["a", "b"]


@pytest.mark.asyncio
async def test_rate_limits_stop_retrying_once_every_deployment_is_cooling_down(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=5)
    host.answer("*", fresh(rate_limit))

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RateLimitError)
    assert sorted(host.deployments_called("acompletion")) == ["a", "b"]
    assert set(cooled_down(core, ["a", "b"])) == {"a", "b"}


@pytest.mark.parametrize("builder", [bad_request, auth_error, context_window, content_policy])
@pytest.mark.asyncio
async def test_an_error_raised_without_retrying_carries_no_retry_counts(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2)
    host.answer("openai/a", fresh(builder))

    error: Final = await attempt(core)

    assert getattr(error, "num_retries", None) is None
    assert getattr(error, "max_retries", None) is None


@pytest.mark.parametrize("builder", [rate_limit, server_error, timeout])
@pytest.mark.parametrize("num_retries", [1, 3])
@pytest.mark.asyncio
async def test_exhausted_retries_stamp_num_retries_and_max_retries_on_the_final_error(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder, num_retries: int
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=num_retries)
    host.answer("openai/a", fresh(builder))

    error: Final = await attempt(core)

    assert getattr(error, "num_retries", None) == num_retries
    assert getattr(error, "max_retries", None) == num_retries
    assert f"LiteLLM Retried: {num_retries} times" in str(error)


@pytest.mark.asyncio
async def test_the_final_error_is_the_latest_one_not_the_first(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2)
    first: Final = server_error()
    last: Final = timeout()
    host.answer("openai/a", first, fresh(server_error), last)

    error: Final = await attempt(core)

    assert error is host.raised[-1]
    assert type(error) is type(last)


# --- retry policies -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "builder", "retries"),
    [
        ("BadRequestErrorRetries", bad_request, 2),
        ("BadRequestErrorRetries", context_window, 1),
        ("ContentPolicyViolationErrorRetries", content_policy, 1),
        ("RateLimitErrorRetries", rate_limit, 0),
        ("TimeoutErrorRetries", timeout, 1),
        ("InternalServerErrorRetries", server_error, 4),
        ("DefaultRetries", server_error, 1),
    ],
)
@pytest.mark.asyncio
async def test_retry_policy_sets_retries_per_error_type(
    make_core: MakeCore, host: ScriptedHost, field: str, builder: ErrorBuilder, retries: int
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], num_retries=3, retry_policy=RetryPolicy(**{field: retries})
    )
    host.answer("openai/a", fresh(builder))

    error: Final = await attempt(core)

    assert type(error) is type(builder())
    assert len(host.calls_to("acompletion")) == retries + 1


@pytest.mark.asyncio
async def test_retry_policy_given_as_a_dict_works_like_a_retry_policy(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], num_retries=3, retry_policy={"BadRequestErrorRetries": 1}
    )
    host.answer("openai/a", fresh(bad_request))

    await attempt(core)

    assert len(host.calls_to("acompletion")) == 2


@pytest.mark.asyncio
async def test_retry_policy_for_one_error_type_leaves_others_to_num_retries(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], num_retries=1, retry_policy=RetryPolicy(RateLimitErrorRetries=4)
    )
    host.answer("openai/a", fresh(timeout))

    await attempt(core)

    assert len(host.calls_to("acompletion")) == 2


@pytest.mark.asyncio
async def test_model_group_retry_policy_applies_only_to_its_group(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a"), deployment("h", "c")],
        num_retries=2,
        model_group_retry_policy={"g": RetryPolicy(InternalServerErrorRetries=0)},
    )
    host.answer("*", fresh(server_error))

    await attempt(core, model="g")
    await attempt(core, model="h")

    assert host.deployments_called("acompletion") == ["a", "c", "c", "c"]


@pytest.mark.asyncio
async def test_model_group_retry_policy_wins_over_retry_policy(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")],
        retry_policy=RetryPolicy(InternalServerErrorRetries=4),
        model_group_retry_policy={"g": RetryPolicy(InternalServerErrorRetries=1)},
    )
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert len(host.calls_to("acompletion")) == 2


@pytest.mark.parametrize(("request_value", "expected_calls"), [(4, 2), (0, 1)])
@pytest.mark.asyncio
async def test_retry_policy_wins_over_request_num_retries_unless_the_request_opts_out(
    make_core: MakeCore, host: ScriptedHost, request_value: int, expected_calls: int
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], retry_policy=RetryPolicy(InternalServerErrorRetries=1))
    host.answer("openai/a", fresh(server_error))

    await attempt(core, num_retries=request_value)

    assert len(host.calls_to("acompletion")) == expected_calls


# --- per-deployment num_retries -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("deployment_value", "request_kwargs", "expected_calls"),
    [
        (1, {}, 2),
        ("1", {}, 2),
        (0, {}, 1),
        (4, {}, 5),
        (4, {"num_retries": 0}, 1),
        (0, {"num_retries": 2}, 3),
    ],
)
@pytest.mark.asyncio
async def test_deployment_num_retries_beats_the_router_but_not_the_request(
    make_core: MakeCore,
    host: ScriptedHost,
    deployment_value: object,
    request_kwargs: dict[str, object],
    expected_calls: int,
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a", num_retries=deployment_value)], num_retries=3)
    host.answer("openai/a", fresh(server_error))

    error: Final = await attempt(core, **request_kwargs)

    assert isinstance(error, litellm.InternalServerError)
    assert len(host.calls_to("acompletion")) == expected_calls


# --- backoff ------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lone_deployment_backs_off_exponentially(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=3)
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    bases: Final = [INITIAL_RETRY_DELAY * 2**n for n in range(3)]
    assert len(host.sleeps) >= 3
    assert all(base <= slept < base + JITTER for base, slept in zip(bases, host.sleeps))


@pytest.mark.asyncio
async def test_retry_after_setting_is_the_minimum_backoff(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2, retry_after=3)
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert len(host.sleeps) >= 2
    assert all(3 <= slept < 3 + JITTER for slept in host.sleeps[:2])


@pytest.mark.asyncio
async def test_lone_deployment_waits_the_retry_after_header(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2)
    host.answer("openai/a", lambda **_: rate_limited(retry_after="7"))

    await attempt(core)

    assert len(host.sleeps) >= 2
    assert all(7 <= slept < 7 + JITTER for slept in host.sleeps[:2])


@pytest.mark.asyncio
async def test_lone_deployment_ignores_a_retry_after_header_over_a_minute(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2)
    host.answer("openai/a", lambda **_: rate_limited(retry_after="120"))

    await attempt(core)

    assert INITIAL_RETRY_DELAY <= host.sleeps[0] < INITIAL_RETRY_DELAY + JITTER


@pytest.mark.parametrize("failure", [fresh(server_error), fresh(timeout), lambda **_: rate_limited(retry_after="7")])
@pytest.mark.asyncio
async def test_a_group_with_siblings_retries_immediately(
    make_core: MakeCore, host: ScriptedHost, failure: Callable[..., Exception]
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=2)
    host.answer("*", failure, None)

    response: Final = await attempt(core)

    assert not isinstance(response, Exception)
    assert host.sleeps == [0]


@pytest.mark.asyncio
async def test_no_sleep_when_the_error_is_not_retried(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=3)
    host.answer("openai/a", fresh(bad_request))

    await attempt(core)

    assert host.sleeps == []


# --- retry headers on success -------------------------------------------------------------------


@pytest.mark.parametrize("failures_first", [1, 2, 3])
@pytest.mark.asyncio
async def test_success_after_retries_reports_attempted_and_max_retries(
    make_core: MakeCore, host: ScriptedHost, failures_first: int
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=3)
    host.answer("openai/a", *[fresh(server_error)] * failures_first, None)

    response: Final = await attempt(core)

    headers: Final = headers_of(response)
    assert headers["x-litellm-attempted-retries"] == failures_first
    assert headers["x-litellm-max-retries"] == 3
    assert headers["x-litellm-model-group"] == "g"


@pytest.mark.asyncio
async def test_first_try_success_reports_zero_retries_and_no_max(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=3)

    response: Final = await attempt(core)

    headers: Final = headers_of(response)
    assert headers["x-litellm-attempted-retries"] == 0
    assert "x-litellm-max-retries" not in headers


@pytest.mark.asyncio
async def test_success_headers_report_the_retry_policy_budget(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")], num_retries=1, retry_policy=RetryPolicy(InternalServerErrorRetries=4)
    )
    host.answer("openai/a", fresh(server_error), fresh(server_error), None)

    response: Final = await attempt(core)

    assert headers_of(response)["x-litellm-attempted-retries"] == 2
    assert headers_of(response)["x-litellm-max-retries"] == 4


# --- allowed fails and cooldowns ----------------------------------------------------------------


@pytest.mark.parametrize("allowed_fails", [0, 1, 2])
@pytest.mark.asyncio
async def test_a_deployment_cools_down_once_its_failures_pass_allowed_fails(
    make_core: MakeCore, host: ScriptedHost, allowed_fails: int
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0, allowed_fails=allowed_fails)
    host.answer("*", fresh(server_error))

    for _ in range(2 * allowed_fails + 2):
        await attempt(core)

    called: Final = host.deployments_called("acompletion")
    cooled: Final = cooled_down(core, ["a", "b"])
    assert cooled
    assert {d for d in ("a", "b") if called.count(d) > allowed_fails} == set(cooled)


@pytest.mark.asyncio
async def test_process_allowed_fails_is_the_default(make_core: MakeCore, host: ScriptedHost) -> None:
    host.set_dynamic(allowed_fails=0)
    core: Final = make_core(model_list=two_deployments(), num_retries=0)
    host.answer("*", fresh(server_error))

    await attempt(core)

    assert core.settings().allowed_fails == 0
    assert list(cooled_down(core, ["a", "b"])) == host.deployments_called("acompletion")


@pytest.mark.asyncio
async def test_a_cooled_deployment_is_skipped_by_later_picks(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0, allowed_fails=0)
    host.answer("*", fresh(server_error))

    await attempt(core)
    first: Final = host.deployments_called("acompletion")[0]
    healthy: Final = await core.select("g", {}, kind="healthy")  # pyright: ignore[reportAttributeAccessIssue]  # every core serves select
    await attempt(core)

    assert [d["model_info"]["id"] for d in healthy] == [{"a": "b", "b": "a"}[first]]
    assert host.deployments_called("acompletion")[1] != first


@pytest.mark.asyncio
async def test_a_group_with_every_deployment_cooling_down_raises_without_calling(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0, allowed_fails=0)
    host.answer("*", fresh(server_error))
    await attempt(core)
    await attempt(core)

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RouterRateLimitError)
    assert len(host.calls_to("acompletion")) == 2


@pytest.mark.asyncio
async def test_rate_limit_cools_down_a_deployment_with_siblings_at_once(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0)
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    assert list(cooled_down(core, ["a", "b"])) == host.deployments_called("acompletion")


@pytest.mark.parametrize(
    "policy",
    [AllowedFailsPolicy(RateLimitErrorAllowedFails=2), {"RateLimitErrorAllowedFails": 2}],
    ids=["AllowedFailsPolicy", "dict"],
)
@pytest.mark.asyncio
async def test_allowed_fails_policy_lets_rate_limits_through_until_its_count(
    make_core: MakeCore, host: ScriptedHost, policy: object
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0, allowed_fails_policy=policy)
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    assert cooled_down(core, ["a", "b"]) == {}


@pytest.mark.asyncio
async def test_allowed_fails_policy_of_zero_cools_down_on_the_first_failure_of_that_type(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=two_deployments(),
        num_retries=0,
        allowed_fails=5,
        allowed_fails_policy=AllowedFailsPolicy(TimeoutErrorAllowedFails=0),
    )
    host.answer("*", fresh(server_error), fresh(timeout))

    await attempt(core)
    after_server_error: Final = cooled_down(core, ["a", "b"])
    await attempt(core)

    assert after_server_error == {}
    assert list(cooled_down(core, ["a", "b"])) == host.deployments_called("acompletion")[1:]


@pytest.mark.asyncio
async def test_deployment_allowed_fails_policy_in_model_info_cools_down_on_a_bad_request(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=two_deployments(model_info={"allowed_fails_policy": {"BadRequestErrorAllowedFails": 0}}),
        num_retries=0,
    )
    host.answer("*", fresh(bad_request))

    await attempt(core)

    assert list(cooled_down(core, ["a", "b"])) == host.deployments_called("acompletion")


@pytest.mark.asyncio
async def test_bad_request_does_not_cool_down_without_a_deployment_policy(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0, allowed_fails=0)
    host.answer("*", fresh(bad_request))

    await attempt(core)

    assert cooled_down(core, ["a", "b"]) == {}


# --- cooldown time ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("arguments", "params", "failure", "expected_seconds"),
    [
        ({}, {}, fresh(rate_limit), DEFAULT_COOLDOWN_TIME_SECONDS),
        ({"cooldown_time": 42}, {}, fresh(rate_limit), 42),
        ({}, {"cooldown_time": 30}, fresh(rate_limit), 30),
        ({}, {"model_info": {"cooldown_time": 30}}, fresh(rate_limit), 30),
        ({}, {}, lambda **_: rate_limited(retry_after="9"), 9),
        ({"cooldown_time": 42}, {}, lambda **_: rate_limited(retry_after="9"), 9),
        ({}, {"cooldown_time": 30}, lambda **_: rate_limited(retry_after="9"), 30),
    ],
    ids=[
        "default",
        "constructor",
        "litellm_params",
        "model_info",
        "retry-after",
        "header-over-router",
        "deployment-over-header",
    ],
)
@pytest.mark.asyncio
async def test_cooldown_time_comes_from_deployment_then_retry_after_then_router(
    make_core: MakeCore,
    host: ScriptedHost,
    arguments: dict[str, object],
    params: dict[str, object],
    failure: Callable[..., Exception],
    expected_seconds: float,
) -> None:
    core: Final = make_core(model_list=two_deployments(**params), num_retries=0, **arguments)
    host.answer("*", failure)

    await attempt(core)

    (entry,) = cooled_down(core, ["a", "b"]).values()
    assert entry["cooldown_time"] == expected_seconds


@pytest.mark.parametrize("change", [{"settings": {"cooldown_time": 20}}, {"assign": {"cooldown_time": 20}}])
@pytest.mark.asyncio
async def test_cooldown_time_applied_at_runtime_is_used_for_later_cooldowns(
    make_core: MakeCore, host: ScriptedHost, change: dict[str, Mapping[str, object]]
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0)
    core.apply(ChangeSet(**change))  # pyright: ignore[reportArgumentType]  # ChangeSet's mapping fields
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    assert core.settings().cooldown_time == 20
    (entry,) = cooled_down(core, ["a", "b"]).values()
    assert entry["cooldown_time"] == 20


@pytest.mark.asyncio
async def test_a_deployment_cooldown_time_of_zero_skips_the_cooldown(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=two_deployments(model_info={"cooldown_time": 0}), num_retries=0)
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    assert cooled_down(core, ["a", "b"]) == {}


# --- disable_cooldowns --------------------------------------------------------------------------


@pytest.mark.parametrize("failure", [fresh(rate_limit), fresh(server_error)])
@pytest.mark.asyncio
async def test_disable_cooldowns_keeps_failing_deployments_routable(
    make_core: MakeCore, host: ScriptedHost, failure: Callable[..., Exception]
) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0, allowed_fails=0, disable_cooldowns=True)
    host.answer("*", failure)

    for _ in range(4):
        await attempt(core)

    assert cooled_down(core, ["a", "b"]) == {}
    assert len(host.calls_to("acompletion")) == 4


@pytest.mark.asyncio
async def test_disable_cooldowns_assigned_at_runtime_stops_cooldowns(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0)
    core.apply(ChangeSet(assign={"disable_cooldowns": True}))
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    assert core.settings().disable_cooldowns is True
    assert cooled_down(core, ["a", "b"]) == {}


# --- single-deployment groups -------------------------------------------------------------------


@pytest.mark.parametrize("arguments", [{}, {"allowed_fails": 0}], ids=["default", "allowed_fails=0"])
@pytest.mark.parametrize("builder", [rate_limit, server_error, timeout])
@pytest.mark.asyncio
async def test_a_single_deployment_group_is_not_cooled_down(
    make_core: MakeCore, host: ScriptedHost, arguments: dict[str, object], builder: ErrorBuilder
) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=0, **arguments)
    host.answer("openai/a", fresh(builder))

    for _ in range(3):
        await attempt(core)

    assert cooled_down(core, ["a"]) == {}
    assert host.deployments_called("acompletion") == ["a", "a", "a"]


@pytest.mark.asyncio
async def test_an_allowed_fails_policy_naming_the_error_cools_down_a_single_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "a")],
        num_retries=0,
        allowed_fails_policy=AllowedFailsPolicy(InternalServerErrorAllowedFails=0),
    )
    host.answer("openai/a", fresh(server_error))

    await attempt(core)
    error: Final = await attempt(core)

    assert set(cooled_down(core, ["a"])) == {"a"}
    assert isinstance(error, litellm.RouterRateLimitError)
    assert host.deployments_called("acompletion") == ["a"]


# --- health() -----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_cooldown_entry_describes_the_failure(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=two_deployments(), num_retries=0)
    host.answer("*", fresh(rate_limit))

    await attempt(core)

    (entry,) = cooled_down(core, ["a", "b"]).values()
    assert entry["status_code"] == "429"
    assert "slow down" in str(entry["exception_received"])


@pytest.mark.parametrize("failures", [1, 3])
@pytest.mark.asyncio
async def test_health_counts_failures_this_minute(make_core: MakeCore, host: ScriptedHost, failures: int) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=0)
    host.answer("openai/a", fresh(server_error))

    for _ in range(failures):
        await attempt(core)

    assert core.health().failures_this_minute("a") == failures
    assert core.health().failures_this_minute("missing") == 0


@pytest.mark.asyncio
async def test_health_counts_every_retried_attempt_as_a_failure(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=[deployment("g", "a")], num_retries=2)
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert core.health().failures_this_minute("a") == 3


@pytest.mark.asyncio
async def test_health_record_failure_cools_down_past_allowed_fails(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=two_deployments(), allowed_fails=1)

    first: Final = core.health().record_failure("a", server_error(), 500)
    second: Final = core.health().record_failure("a", server_error(), 500)

    assert (first, second) == (False, True)
    assert set(cooled_down(core, ["a", "b"])) == {"a"}
    assert core.health().failures_this_minute("a") == 2
