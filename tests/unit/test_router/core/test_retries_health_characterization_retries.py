"""
Characterization of a router core's retry loop: which errors retry, how many times, how long it sleeps,
what it records on the request and the final error, and the public retry helpers.
"""

from typing import Final

import pytest

import litellm
from litellm.constants import INITIAL_RETRY_DELAY, JITTER, MAX_RETRY_DELAY
from litellm.exceptions import GuardrailRaisedException
from litellm.types.router import RetryPolicy
from tests.unit.test_router.core.retries_health_helpers import (
    ErrorBuilder,
    MakeCore,
    MetadataRecorder,
    api_connection,
    api_status,
    attempt,
    attempt_sync,
    bad_gateway,
    cooled_down,
    fresh,
    headers_of,
    http_response,
    not_found,
    permission_denied,
    plain_error,
    rate_limited,
    service_unavailable,
    unprocessable,
)
from tests.unit.test_router.core.scripted_host import (
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


def group(
    *ids: str, **params: object
) -> list[dict[str, object]]:  # kwargs-ok: litellm_params  # mutable-ok: a model_list
    return [deployment("g", deployment_id, **params) for deployment_id in ids]


def within(slept: float, base: float) -> bool:
    return base <= slept < base + JITTER


# --- which errors retry, by error class and status ------------------------------------------------


@pytest.mark.parametrize(
    ("builder", "lone_calls", "pair_calls"),
    [
        (not_found, 1, 1),
        (permission_denied, 1, 3),
        (unprocessable, 1, 1),
        (service_unavailable, 3, 3),
        (bad_gateway, 3, 3),
        (api_connection, 3, 3),
        (api_status(409), 3, 3),
        (plain_error, 3, 3),
    ],
    ids=["404", "403", "422", "503", "502", "connection", "409", "no-status"],
)
@pytest.mark.parametrize("size", ["lone", "pair"])
@pytest.mark.asyncio
async def test_retry_decision_per_error_class_and_group_size(
    make_core: MakeCore, host: ScriptedHost, builder: ErrorBuilder, lone_calls: int, pair_calls: int, size: str
) -> None:
    core: Final = make_core(model_list=group("a") if size == "lone" else group("a", "b"), num_retries=2)
    host.answer("*", fresh(builder))

    error: Final = await attempt(core)

    assert type(error) is type(builder())
    assert len(host.calls_to("acompletion")) == (lone_calls if size == "lone" else pair_calls)


@pytest.mark.asyncio
async def test_an_error_without_a_litellm_type_is_retried_but_not_stamped(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=2)
    host.answer("openai/a", fresh(plain_error))

    error: Final = await attempt(core)

    assert isinstance(error, ValueError)
    assert getattr(error, "num_retries", None) is None
    assert getattr(error, "max_retries", None) is None
    assert "LiteLLM Retried" not in str(error)


@pytest.mark.asyncio
async def test_a_403_retry_moves_to_a_deployment_that_has_not_refused_yet(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b", "c"), num_retries=4)
    host.answer("*", fresh(permission_denied))

    error: Final = await attempt(core)

    called: Final = host.deployments_called("acompletion")
    assert isinstance(error, litellm.PermissionDeniedError)
    assert len(called) == 5
    assert sorted(called[:3]) == ["a", "b", "c"]
    assert cooled_down(core, ["a", "b", "c"]) == {}


@pytest.mark.asyncio
async def test_a_401_on_each_deployment_cools_each_down_and_stops_when_none_is_left(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b", "c"), num_retries=4)
    host.answer("*", fresh(auth_error))

    error: Final = await attempt(core)

    assert isinstance(error, litellm.AuthenticationError)
    assert sorted(host.deployments_called("acompletion")) == ["a", "b", "c"]
    assert set(cooled_down(core, ["a", "b", "c"])) == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_a_non_retryable_error_mid_loop_stops_the_retries(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=4)
    host.answer("openai/a", fresh(server_error), fresh(bad_request))

    error: Final = await attempt(core)

    assert isinstance(error, litellm.BadRequestError)
    assert len(host.calls_to("acompletion")) == 2
    assert getattr(error, "num_retries", None) is None


@pytest.mark.asyncio
async def test_a_retry_policy_keeps_retrying_through_a_later_non_retryable_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a"), retry_policy=RetryPolicy(InternalServerErrorRetries=3))
    host.answer("openai/a", fresh(server_error), fresh(bad_request))

    error: Final = await attempt(core)

    assert isinstance(error, litellm.BadRequestError)
    assert len(host.calls_to("acompletion")) == 4
    assert getattr(error, "num_retries", None) == 3


@pytest.mark.asyncio
async def test_a_guardrail_block_is_raised_without_retrying(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=3)
    host.answer("*", lambda **_: GuardrailRaisedException(guardrail_name="pii", message="blocked by guardrail"))

    error: Final = await attempt(core)

    assert isinstance(error, GuardrailRaisedException)
    assert len(host.calls_to("acompletion")) == 1
    assert host.sleeps == []


# --- retry policies, more error types --------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "builder", "expected_calls"),
    [
        (RetryPolicy(ServiceUnavailableErrorRetries=2), service_unavailable, 3),
        (RetryPolicy(InternalServerErrorRetries=2), service_unavailable, 1 + 2),
        (RetryPolicy(DefaultRetries=0), api_status(409), 1),
        (RetryPolicy(DefaultRetries=2), unprocessable, 3),
        (RetryPolicy(BadRequestErrorRetries=1), content_policy, 2),
        (RetryPolicy(BadRequestErrorRetries=1, ContentPolicyViolationErrorRetries=0), content_policy, 1),
    ],
    ids=[
        "service-unavailable",
        "503-falls-back-to-internal-server-error",
        "default-for-409",
        "default-for-422",
        "content-policy-inherits-bad-request",
        "content-policy-specific-wins",
    ],
)
@pytest.mark.asyncio
async def test_retry_policy_field_chosen_for_each_error(
    make_core: MakeCore, host: ScriptedHost, policy: RetryPolicy, builder: ErrorBuilder, expected_calls: int
) -> None:
    core: Final = make_core(model_list=group("a"), retry_policy=policy)
    host.answer("openai/a", fresh(builder))

    error: Final = await attempt(core)

    assert type(error) is type(builder())
    assert len(host.calls_to("acompletion")) == expected_calls


@pytest.mark.parametrize(
    ("policy", "builder"),
    [
        (RetryPolicy(AuthenticationErrorRetries=2), auth_error),
        (RetryPolicy(NotFoundErrorRetries=2), not_found),
        (RetryPolicy(DefaultRetries=1), not_found),
    ],
    ids=["auth", "not-found", "default-for-404"],
)
@pytest.mark.asyncio
async def test_retry_policy_retries_a_401_or_404_into_the_cooled_lone_deployment(
    make_core: MakeCore, host: ScriptedHost, policy: RetryPolicy, builder: ErrorBuilder
) -> None:
    core: Final = make_core(model_list=group("a"), retry_policy=policy)
    host.answer("openai/a", fresh(builder))

    error: Final = await attempt(core)

    assert isinstance(error, litellm.RouterRateLimitError)
    assert "All deployments for selected model are in cooldown" in str(error)
    assert len(host.calls_to("acompletion")) == 1


@pytest.mark.parametrize(
    ("policy", "builder"),
    [(RetryPolicy(AuthenticationErrorRetries=2), auth_error), (RetryPolicy(NotFoundErrorRetries=2), not_found)],
    ids=["auth", "not-found"],
)
@pytest.mark.asyncio
async def test_retry_policy_retries_a_401_or_404_when_cooldowns_are_off(
    make_core: MakeCore, host: ScriptedHost, policy: RetryPolicy, builder: ErrorBuilder
) -> None:
    core: Final = make_core(model_list=group("a"), retry_policy=policy, disable_cooldowns=True)
    host.answer("openai/a", fresh(builder))

    error: Final = await attempt(core)

    assert type(error) is type(builder())
    assert len(host.calls_to("acompletion")) == 3


@pytest.mark.asyncio
async def test_retry_policy_wins_over_the_deployment_num_retries(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=group("a", num_retries=4), retry_policy=RetryPolicy(InternalServerErrorRetries=1)
    )
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert len(host.calls_to("acompletion")) == 2


@pytest.mark.asyncio
async def test_request_model_group_retry_policy_overrides_the_router_one(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(
        model_list=group("a"), model_group_retry_policy={"g": RetryPolicy(InternalServerErrorRetries=0)}
    )
    host.answer("openai/a", fresh(server_error))

    await attempt(core, model_group_retry_policy={"g": RetryPolicy(InternalServerErrorRetries=3)})

    assert len(host.calls_to("acompletion")) == 4


@pytest.mark.asyncio
async def test_model_group_retry_policy_given_as_a_dict(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(
        model_list=group("a"), num_retries=0, model_group_retry_policy={"g": {"InternalServerErrorRetries": 2}}
    )
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert len(host.calls_to("acompletion")) == 3


@pytest.mark.parametrize(
    ("error", "model_group", "expected"),
    [
        (server_error(), "g", 1),
        (server_error(), "h", 4),
        (server_error(), None, 4),
        (rate_limit(), "g", None),
        (rate_limit(), "h", 2),
        (bad_request(), "h", 7),
        (context_window(), "h", 7),
        (not_found(), "h", 7),
    ],
)
def test_get_num_retries_from_retry_policy(
    make_core: MakeCore, error: Exception, model_group: str | None, expected: int | None
) -> None:
    core: Final = make_core(
        model_list=group("a"),
        retry_policy=RetryPolicy(InternalServerErrorRetries=4, RateLimitErrorRetries=2, DefaultRetries=7),
        model_group_retry_policy={"g": RetryPolicy(InternalServerErrorRetries=1)},
    )

    assert core.get_num_retries_from_retry_policy(error, model_group=model_group) == expected  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member


def test_get_num_retries_from_retry_policy_without_any_policy_is_none(make_core: MakeCore) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=3)

    assert core.get_num_retries_from_retry_policy(server_error(), model_group="g") is None  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member


# --- should_retry_this_error ---------------------------------------------------------------------

HEALTHY_ONE: Final = [{"model_info": {"id": "a"}}]
ALL_TWO: Final = [{"model_info": {"id": "a"}}, {"model_info": {"id": "b"}}]


@pytest.mark.parametrize(
    ("error", "arguments", "retries"),
    [
        (server_error(), {"healthy_deployments": HEALTHY_ONE, "all_deployments": HEALTHY_ONE}, True),
        (server_error(), {"healthy_deployments": [], "all_deployments": HEALTHY_ONE}, False),
        (server_error(), {}, False),
        (timeout(), {"healthy_deployments": HEALTHY_ONE}, True),
        (plain_error(), {"healthy_deployments": HEALTHY_ONE}, True),
        (plain_error(), {"healthy_deployments": None}, False),
        (bad_request(), {"healthy_deployments": HEALTHY_ONE}, False),
        (context_window(), {"healthy_deployments": HEALTHY_ONE, "context_window_fallbacks": []}, False),
        (content_policy(), {"healthy_deployments": HEALTHY_ONE, "content_policy_fallbacks": []}, False),
        (not_found(), {"healthy_deployments": ALL_TWO, "all_deployments": ALL_TWO}, False),
        (unprocessable(), {"healthy_deployments": ALL_TWO, "all_deployments": ALL_TWO}, False),
        (auth_error(), {"healthy_deployments": HEALTHY_ONE, "all_deployments": HEALTHY_ONE}, False),
        (auth_error(), {"healthy_deployments": HEALTHY_ONE, "all_deployments": ALL_TWO}, True),
        (auth_error(), {"healthy_deployments": [], "all_deployments": ALL_TWO}, False),
        (permission_denied(), {"healthy_deployments": HEALTHY_ONE, "all_deployments": HEALTHY_ONE}, False),
        (permission_denied(), {"healthy_deployments": HEALTHY_ONE, "all_deployments": ALL_TWO}, True),
        (rate_limit(), {"healthy_deployments": HEALTHY_ONE, "regular_fallbacks": [{"g": ["h"]}]}, True),
        (rate_limit(), {"healthy_deployments": [], "regular_fallbacks": [{"g": ["h"]}]}, False),
        (rate_limit(), {"healthy_deployments": [], "regular_fallbacks": []}, False),
        (api_status(409)(), {"healthy_deployments": HEALTHY_ONE}, True),
        (service_unavailable(), {"healthy_deployments": HEALTHY_ONE}, True),
    ],
)
def test_should_retry_this_error_returns_true_or_raises_the_error(
    make_core: MakeCore, error: Exception, arguments: dict[str, object], retries: bool
) -> None:
    core: Final = make_core(model_list=group("a", "b"))

    if retries:
        assert core.should_retry_this_error(error=error, **arguments) is True  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
    else:
        with pytest.raises(type(error)) as raised:
            core.should_retry_this_error(error=error, **arguments)  # pyright: ignore[reportAttributeAccessIssue]  # PythonCore's public member
        assert raised.value is error


# --- mock rate limit errors ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_rate_limit_error_raises_without_calling_the_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=0)

    error: Final = await attempt(core, mock_testing_rate_limit_error=True)

    assert isinstance(error, litellm.RateLimitError)
    assert "This is a mock exception for model=g, to trigger a rate limit error." in str(error)
    assert host.calls == []


@pytest.mark.asyncio
async def test_mock_rate_limit_error_is_retried_and_the_retry_reaches_the_deployment(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=2)

    response: Final = await attempt(core, mock_testing_rate_limit_error=True)

    assert not isinstance(response, Exception)
    assert host.deployments_called("acompletion") == ["a"]
    assert "mock_testing_rate_limit_error" not in host.calls[0].kwargs
    assert headers_of(response)["x-litellm-attempted-retries"] == 1
    assert headers_of(response)["x-litellm-max-retries"] == 2


@pytest.mark.asyncio
async def test_mock_rate_limit_error_carries_a_lone_deployments_num_retries(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", num_retries=1), num_retries=5)

    response: Final = await attempt(core, mock_testing_rate_limit_error=True)

    assert headers_of(response)["x-litellm-max-retries"] == 1


@pytest.mark.parametrize("value", [False, None])
@pytest.mark.asyncio
async def test_mock_rate_limit_error_only_fires_when_true(
    make_core: MakeCore, host: ScriptedHost, value: object
) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=0)

    response: Final = await attempt(core, mock_testing_rate_limit_error=value)

    assert not isinstance(response, Exception)
    assert host.deployments_called("acompletion") == ["a"]


# --- sleeps --------------------------------------------------------------------------------------


@pytest.mark.parametrize("num_retries", [1, 2, 3])
@pytest.mark.asyncio
async def test_lone_deployment_sleeps_once_more_than_it_retries(
    make_core: MakeCore, host: ScriptedHost, num_retries: int
) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=num_retries)
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    bases: Final = [INITIAL_RETRY_DELAY * 2**n for n in range(num_retries + 1)]
    assert len(host.sleeps) == num_retries + 1
    assert all(within(slept, base) for slept, base in zip(host.sleeps, bases))


@pytest.mark.asyncio
async def test_backoff_is_capped_at_max_retry_delay(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=6)
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    expected: Final = [min(INITIAL_RETRY_DELAY * 2**n, MAX_RETRY_DELAY) for n in range(7)]
    assert all(within(slept, base) for slept, base in zip(host.sleeps, expected))
    assert within(host.sleeps[-1], MAX_RETRY_DELAY)


@pytest.mark.asyncio
async def test_retry_after_setting_above_max_retry_delay_is_capped(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=1, retry_after=20)
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert all(within(slept, MAX_RETRY_DELAY) for slept in host.sleeps)


@pytest.mark.asyncio
async def test_a_retry_after_header_of_zero_falls_back_to_backoff(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=1)
    host.answer("openai/a", lambda **_: rate_limited(retry_after="0"))

    await attempt(core)

    assert within(host.sleeps[0], INITIAL_RETRY_DELAY)


@pytest.mark.asyncio
async def test_retry_after_header_on_any_error_type_sets_the_sleep(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=1)
    host.answer(
        "openai/a",
        lambda **_: litellm.InternalServerError(
            message="boom",
            llm_provider="openai",
            model="gpt-4o",
            response=http_response(500, {"retry-after": "4"}),
        ),
    )

    await attempt(core)

    assert all(within(slept, 4) for slept in host.sleeps)


@pytest.mark.asyncio
async def test_litellm_response_headers_on_the_error_set_the_sleep(make_core: MakeCore, host: ScriptedHost) -> None:
    def failure(**_: object) -> Exception:
        error: Final = server_error()
        error.litellm_response_headers = {"retry-after": "6"}  # pyright: ignore[reportAttributeAccessIssue]  # what provider handlers attach
        return error

    core: Final = make_core(model_list=group("a"), num_retries=1)
    host.answer("openai/a", failure)

    await attempt(core)

    assert all(within(slept, 6) for slept in host.sleeps)


@pytest.mark.asyncio
async def test_retry_after_header_is_ignored_while_siblings_are_healthy(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", "b"), num_retries=2, allowed_fails=10)
    host.answer("*", fresh(server_error))

    await attempt(core)

    assert host.sleeps == [0, 0, 0]


# --- what the retry loop writes on the request -------------------------------------------------


@pytest.mark.asyncio
async def test_each_attempt_carries_its_retry_count_and_the_failures_before_it(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    recorder: Final = MetadataRecorder()
    core: Final = make_core(model_list=group("a", "b"), num_retries=2, allowed_fails=10)
    host.answer("*", recorder.then(server_error), recorder.then(server_error), recorder.then(None))

    response: Final = await attempt(core)

    assert not isinstance(response, Exception)
    snapshots: Final = recorder.snapshots
    assert [s["attempted_retries"] for s in snapshots] == [0, 1, 2]
    assert [s["max_retries"] for s in snapshots] == [2, 2, 2]
    assert [s["model_group_size"] for s in snapshots] == [2, 2, 2]
    assert [s["request_retry_count"] for s in snapshots] == [None, 1, 2]
    assert snapshots[0]["previous_models"] is None
    trail: Final = snapshots[2]["previous_models"]
    assert isinstance(trail, tuple)
    assert [entry["attempted_retries"] for entry in trail] == [0, 1]
    assert [entry["exception_type"] for entry in trail] == ["InternalServerError"] * 2
    assert all(entry["model_group"] == "g" for entry in trail)
    called: Final = host.deployments_called("acompletion")
    assert [entry["deployment_id"] for entry in trail] == called[:2]
    assert [entry["model"] for entry in trail] == [f"openai/{d}" for d in called[:2]]
    assert all(entry["exception_string"] == "litellm.InternalServerError: boom" for entry in trail)


@pytest.mark.asyncio
async def test_max_retries_on_the_request_follows_the_deployment_num_retries(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    recorder: Final = MetadataRecorder()
    core: Final = make_core(model_list=group("a", num_retries=1), num_retries=3)
    host.answer("openai/a", recorder.then(server_error), recorder.then(None))

    response: Final = await attempt(core)

    assert [s["max_retries"] for s in recorder.snapshots] == [3, 1]
    assert headers_of(response)["x-litellm-max-retries"] == 1


@pytest.mark.asyncio
async def test_exhausted_deployment_num_retries_are_stamped_on_the_error(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a", num_retries=2), num_retries=5)
    host.answer("openai/a", fresh(server_error))

    error: Final = await attempt(core)

    assert getattr(error, "num_retries", None) == 2
    assert getattr(error, "max_retries", None) == 2


@pytest.mark.parametrize("bad_value", ["many", None])
@pytest.mark.asyncio
async def test_a_deployment_num_retries_that_is_not_a_number_falls_back_to_the_router(
    make_core: MakeCore, host: ScriptedHost, bad_value: object
) -> None:
    core: Final = make_core(model_list=group("a", num_retries=bad_value), num_retries=1)
    host.answer("openai/a", fresh(server_error))

    await attempt(core)

    assert len(host.calls_to("acompletion")) == 2


# --- sync and other endpoints ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_completion_retries_like_the_async_one(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=2)
    host.answer("openai/a", fresh(server_error))

    error: Final = attempt_sync(core)

    assert isinstance(error, litellm.InternalServerError)
    assert host.deployments_called("completion") == ["a", "a", "a"]
    assert len(host.sleeps) == 3
    assert getattr(error, "num_retries", None) == 2


@pytest.mark.asyncio
async def test_sync_completion_success_after_a_retry_reports_retry_headers(
    make_core: MakeCore, host: ScriptedHost
) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=2)
    host.answer("openai/a", fresh(server_error), None)

    response: Final = attempt_sync(core)

    assert headers_of(response)["x-litellm-attempted-retries"] == 1
    assert headers_of(response)["x-litellm-max-retries"] == 2


@pytest.mark.parametrize("op", ["aembedding", "embedding"])
@pytest.mark.asyncio
async def test_embeddings_retry_with_the_router_num_retries(make_core: MakeCore, host: ScriptedHost, op: str) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=2)
    host.answer("openai/a", fresh(server_error))

    async def embed() -> object:
        if op == "aembedding":
            return await core.aembedding(model="g", input=["x"])  # pyright: ignore[reportAttributeAccessIssue]  # every core serves aembedding
        return core.embedding(model="g", input=["x"])  # pyright: ignore[reportAttributeAccessIssue]  # every core serves embedding

    with pytest.raises(litellm.InternalServerError):
        await embed()

    assert host.deployments_called(op) == ["a", "a", "a"]


@pytest.mark.asyncio
async def test_embedding_success_after_a_retry_reports_retry_headers(make_core: MakeCore, host: ScriptedHost) -> None:
    core: Final = make_core(model_list=group("a"), num_retries=2)
    host.answer("openai/a", fresh(timeout), None)

    response: Final = await core.aembedding(model="g", input=["x"])  # pyright: ignore[reportAttributeAccessIssue]  # every core serves aembedding

    assert headers_of(response)["x-litellm-attempted-retries"] == 1
    assert headers_of(response)["x-litellm-max-retries"] == 2
