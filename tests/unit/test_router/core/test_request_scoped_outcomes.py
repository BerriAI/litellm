"""
What a router records once its outcome callbacks left the process lists, on both cores:
``observe(request_scoped=True)``, the old ``serve_through_request_logging()`` and ``retire()`` take
them off, and ``observe(logging_obj)`` puts them back on one request's logging object. What the
Python router writes outside those callbacks goes on: ``make_call``'s usage, a refused pre-call
check's cooldown and a failed fallback hop's cooldown.
"""

import asyncio
from collections.abc import Callable, Iterable, Mapping
from typing import Final

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.router import Router
from litellm.router_core import RouterCore
from tests.unit.test_router.core.endpoints_helpers import EndpointHost
from tests.unit.test_router.core.retries_health_helpers import (
    MESSAGES,
    MakeCore,
    attempt,
    attempt_sync,
    cooled_down,
    fresh,
)
from tests.unit.test_router.core.scripted_host import Call, deployment, rate_limit, server_error
from tests.unit.test_router.core.streaming_helpers import chat_stream, collect_texts, request_logging

USAGE: Final = {"prompt_tokens": 4, "completion_tokens": 8, "total_tokens": 12}


class RequestLoggingHost(EndpointHost):
    """``EndpointHost`` that logs on a request's logging object as litellm does: its first failure
    alone runs the failure callbacks, and what ``_core_bind_to_request_logging`` put on the object
    runs beside the process lists. A stream logs a failure through its own copy of the object, so
    only an attempt's failure marks it."""

    __slots__ = ()

    async def _answer_async(self, op: str, args: tuple[object, ...], kwargs: Mapping[str, object]) -> object:
        logging_obj: Final = kwargs.get("litellm_logging_obj")
        if not isinstance(logging_obj, LiteLLMLogging):
            return await super()._answer_async(op, args, kwargs)
        details: Final = logging_obj.model_call_details
        details["current_attempt_api_call_start_time"] = object()
        try:
            return await super()._answer_async(op, args, kwargs)
        except Exception:
            if details.get("has_logged_async_failure") is not True:
                details["has_logged_async_failure"] = True
                details["async_failure_logged_attempt"] = details.get("current_attempt_api_call_start_time")
            raise

    async def log_async(self, call: Call, error: BaseException | None, response: object) -> None:
        logging_obj: Final = call.kwargs.get("litellm_logging_obj")
        if not isinstance(logging_obj, LiteLLMLogging):
            await super().log_async(call, error, response)
            return
        if error is not None and logging_obj.model_call_details.get("has_logged_async_failure") is True:
            return
        await super().log_async(call, error, response)
        logged: Final = self._logging_kwargs(call, error, response)
        if error is not None:
            for callback in _functions(logging_obj.dynamic_failure_callbacks):
                _ = callback(logged, None, 0.0, 0.0)
            for callback in _functions(logging_obj.dynamic_async_failure_callbacks):
                await callback(logged, None, 0.0, 0.0)  # pyright: ignore[reportGeneralTypeIssues]  # an async callback
            return
        for callback in _functions(logging_obj.dynamic_async_success_callbacks):
            await callback(logged, response, 0.0, 0.0)  # pyright: ignore[reportGeneralTypeIssues]  # an async callback

    def log_sync(self, call: Call, error: BaseException | None, response: object) -> None:
        super().log_sync(call, error, response)
        logging_obj: Final = call.kwargs.get("litellm_logging_obj")
        if error is None or not isinstance(logging_obj, LiteLLMLogging):
            return
        logged: Final = self._logging_kwargs(call, error, response)
        for callback in _functions(logging_obj.dynamic_failure_callbacks):
            _ = callback(logged, None, 0.0, 0.0)


def _functions(callbacks: Iterable[object] | None) -> tuple[Callable[..., object], ...]:
    """The function callbacks of a logging object's list; a ``CustomLogger`` logs on its own hooks."""
    return tuple(
        callback for callback in callbacks or () if callable(callback) and not isinstance(callback, CustomLogger)
    )


@pytest.fixture
def host() -> RequestLoggingHost:
    return RequestLoggingHost()


def observed(core: RouterCore) -> None:
    _ = core.observe(request_scoped=True)  # pyright: ignore[reportCallIssue]  # every core takes it


def through_the_old_member(core: RouterCore) -> None:
    router: Final = Router.__new__(Router)
    router._attach(core)  # pyright: ignore[reportPrivateUsage, reportAttributeAccessIssue, reportUnknownMemberType]  # as Router.from_config attaches it
    with pytest.warns(DeprecationWarning, match=r"Router\.serve_through_request_logging is deprecated"):
        router.serve_through_request_logging()


def retired(core: RouterCore) -> None:
    core.retire()


RELEASES: Final = pytest.mark.parametrize(
    "release", [observed, through_the_old_member, retired], ids=["observe", "old-member", "retire"]
)


def weighted_pair(make_core: MakeCore, **limits: object) -> RouterCore:  # kwargs-ok: both deployments' params
    """``d1`` takes every pick while it is not cooling down, and one failure cools it."""
    return make_core(
        model_list=[deployment("g", "d1", weight=1, **limits), deployment("g", "d2", weight=0, **limits)],
        num_retries=0,
        allowed_fails=0,
    )


def successes(core: RouterCore, deployment_id: str) -> object:
    return core.health().store.get_cache(f"{deployment_id}:successes", local_only=True)


async def settle() -> None:
    """Wait for the logging and recording tasks a finished stream scheduled, and any tasks they schedule."""
    while pending := asyncio.all_tasks() - {asyncio.current_task()}:
        _, still_running = await asyncio.wait(pending, timeout=5)
        assert not still_running, still_running


# --- a router still on the process lists ------------------------------------------------------


async def test_a_router_on_the_lists_cools_a_failing_deployment_and_moves_on(
    make_core: MakeCore, host: RequestLoggingHost
) -> None:
    core: Final = weighted_pair(make_core)
    host.answer("openai/d1", fresh(server_error))

    first: Final = await attempt(core, disable_fallbacks=True)
    second: Final = await attempt(core, disable_fallbacks=True)

    assert isinstance(first, litellm.InternalServerError)
    assert not isinstance(second, Exception)
    assert host.deployments_called() == ["d1", "d2"]
    assert list(cooled_down(core, ["d1", "d2"])) == ["d1"]
    assert core.health().failures_this_minute("d1") == 1
    assert successes(core, "d2") == 1


async def test_a_binding_changes_nothing_on_a_router_on_the_lists(
    make_core: MakeCore, host: RequestLoggingHost
) -> None:
    core: Final = weighted_pair(make_core)
    bound: Final = request_logging()
    _ = core.observe(bound)  # pyright: ignore[reportCallIssue]  # every core takes it

    _ = attempt_sync(core, litellm_logging_obj=bound)

    assert successes(core, "d1") == 1


# --- released, with no binding ------------------------------------------------------------------


@RELEASES
async def test_a_failing_deployment_never_cools_and_the_next_call_returns_to_it(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    core: Final = weighted_pair(make_core)
    host.answer("openai/d1", fresh(server_error))
    release(core)

    first: Final = await attempt(core, disable_fallbacks=True)
    second: Final = await attempt(core, disable_fallbacks=True)

    assert isinstance(first, litellm.InternalServerError)
    assert isinstance(second, litellm.InternalServerError)
    assert host.deployments_called() == ["d1", "d1"]
    assert cooled_down(core, ["d1", "d2"]) == {}
    assert core.health().failures_this_minute("d1") == 0
    assert await core.health().usage("g") == (None, None)


@RELEASES
async def test_a_retry_goes_back_to_the_failing_deployment(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", weight=1), deployment("g", "d2", weight=0)], num_retries=1
    )
    host.answer("openai/d1", fresh(rate_limit))
    release(core)

    answered: Final = await attempt(core)

    assert isinstance(answered, litellm.RateLimitError)
    assert host.deployments_called() == ["d1", "d1"]
    assert cooled_down(core, ["d1", "d2"]) == {}


@RELEASES
def test_a_sync_failure_records_nothing(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    core: Final = weighted_pair(make_core)
    host.answer("openai/d1", fresh(server_error))
    release(core)

    answered: Final = attempt_sync(core)

    assert isinstance(answered, litellm.InternalServerError)
    assert cooled_down(core, ["d1", "d2"]) == {}
    assert core.health().failures_this_minute("d1") == 0


@RELEASES
async def test_a_success_counts_its_usage_and_no_success(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    core: Final = weighted_pair(make_core, rpm=100)
    release(core)

    answered: Final = await attempt(core)

    assert not isinstance(answered, Exception)
    assert successes(core, "d1") is None
    assert await core.health().usage("g") == (10, 1)


@RELEASES
async def test_a_sync_success_counts_its_usage_and_no_success(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    core: Final = weighted_pair(make_core, rpm=100)
    release(core)

    answered: Final = attempt_sync(core)

    assert not isinstance(answered, Exception)
    assert successes(core, "d1") is None
    assert await core.health().usage("g") == (10, 1)


async def test_an_endpoint_outside_make_call_counts_no_usage(make_core: MakeCore, host: RequestLoggingHost) -> None:
    """``amoderation`` makes one attempt, outside ``make_call``: only the success callback counts it."""
    core: Final = make_core(model_list=[deployment("g", "d1", rpm=100)], num_retries=0)
    observed(core)

    _ = await core.amoderation(model="g", input="x")  # pyright: ignore[reportAttributeAccessIssue]  # every core serves it

    assert await core.health().usage("g") == (None, None)


@RELEASES
async def test_a_stream_counts_its_request_when_it_opens_and_nothing_when_it_ends(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    host.answer("openai/d1", chat_stream("hello", usage=USAGE, logged=True))
    core: Final = weighted_pair(make_core, rpm=100)
    release(core)

    stream: Final = await core.acompletion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
    _ = await collect_texts(stream)
    await settle()

    assert successes(core, "d1") is None
    assert await core.health().usage("g") == (0, 1)


@RELEASES
async def test_a_stream_failing_after_content_records_nothing(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    host.answer("openai/d1", chat_stream("partial", error=rate_limit(message="mid stream"), logged=True))
    core: Final = weighted_pair(make_core)
    release(core)

    stream: Final = await core.acompletion(model="g", messages=MESSAGES, stream=True)  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        _ = await collect_texts(stream)
    await settle()

    assert cooled_down(core, ["d1", "d2"]) == {}
    assert core.health().failures_this_minute("d1") == 0


# --- what goes on outside the callbacks ------------------------------------------------------------


class RefusesEveryDeployment(CustomLogger):
    async def async_pre_call_check(self, deployment: object, parent_otel_span: object) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]  # any deployment
        raise rate_limit()


@RELEASES
async def test_a_refused_pre_call_check_still_cools_the_deployment(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    core: Final = weighted_pair(make_core)
    release(core)
    host.callback_list("callbacks").append(RefusesEveryDeployment())

    _ = await attempt(core, disable_fallbacks=True)

    assert list(cooled_down(core, ["d1", "d2"])) == ["d1"]
    assert core.health().failures_this_minute("d1") == 0


async def test_a_failed_fallback_hop_on_a_shared_logging_object_is_still_counted(
    make_core: MakeCore, host: RequestLoggingHost
) -> None:
    """The first failure logs through no callback, yet blocks the object for the hops, and a blocked
    hop records through ``_trigger_cooldown_for_failed_deployment``."""
    core: Final = make_core(
        model_list=[deployment("a", "a"), deployment("b", "b"), deployment("c", "c")],
        fallbacks=[{"a": ["b"]}, {"b": ["c"]}],
        num_retries=0,
    )
    observed(core)
    host.answer("*", fresh(server_error))

    _ = await attempt(core, "a", litellm_logging_obj=request_logging())

    assert core.health().failures_this_minute("a") == 0
    assert core.health().failures_this_minute("b") == 1
    assert core.health().failures_this_minute("c") == 1


async def test_failed_fallback_hops_with_no_logging_object_are_not_counted(
    make_core: MakeCore, host: RequestLoggingHost
) -> None:
    core: Final = make_core(
        model_list=[deployment("a", "a"), deployment("b", "b")], fallbacks=[{"a": ["b"]}], num_retries=0
    )
    observed(core)
    host.answer("*", fresh(server_error))

    _ = await attempt(core, "a")

    assert core.health().failures_this_minute("a") == 0
    assert core.health().failures_this_minute("b") == 0


# --- a binding, for its request alone -----------------------------------------------------------


@pytest.mark.parametrize("release", [observed, retired], ids=["observe", "retire"])
async def test_a_binding_records_a_failure_for_that_request_only(
    make_core: MakeCore, host: RequestLoggingHost, release: Callable[[RouterCore], None]
) -> None:
    core: Final = weighted_pair(make_core)
    host.answer("openai/d1", fresh(server_error))
    release(core)

    _ = await attempt(core, disable_fallbacks=True, litellm_logging_obj=request_logging())
    assert cooled_down(core, ["d1", "d2"]) == {}
    assert core.health().failures_this_minute("d1") == 0

    bound: Final = request_logging()
    _ = core.observe(bound)  # pyright: ignore[reportCallIssue]  # every core takes it
    _ = await attempt(core, disable_fallbacks=True, litellm_logging_obj=bound)

    assert list(cooled_down(core, ["d1", "d2"])) == ["d1"]
    assert core.health().failures_this_minute("d1") == 1
    assert await core.health().usage("g") == (None, 1)


async def test_a_binding_records_an_async_success(make_core: MakeCore, host: RequestLoggingHost) -> None:
    core: Final = weighted_pair(make_core)
    observed(core)
    bound: Final = request_logging()
    _ = core.observe(bound)  # pyright: ignore[reportCallIssue]  # every core takes it

    _ = await attempt(core, litellm_logging_obj=bound)

    assert successes(core, "d1") == 1


async def test_a_binding_leaves_a_sync_success_uncounted(make_core: MakeCore, host: RequestLoggingHost) -> None:
    """``_core_bind_to_request_logging`` does not bind ``sync_deployment_callback_on_success``."""
    core: Final = weighted_pair(make_core, rpm=100)
    observed(core)
    bound: Final = request_logging()
    _ = core.observe(bound)  # pyright: ignore[reportCallIssue]  # every core takes it

    _ = attempt_sync(core, litellm_logging_obj=bound)

    assert successes(core, "d1") is None
    assert await core.health().usage("g") == (10, 1)


async def test_a_binding_records_a_sync_failure_with_no_request(make_core: MakeCore, host: RequestLoggingHost) -> None:
    core: Final = weighted_pair(make_core)
    host.answer("openai/d1", fresh(server_error))
    observed(core)
    bound: Final = request_logging()
    _ = core.observe(bound)  # pyright: ignore[reportCallIssue]  # every core takes it

    _ = attempt_sync(core, litellm_logging_obj=bound)

    assert list(cooled_down(core, ["d1", "d2"])) == ["d1"]
    assert core.health().failures_this_minute("d1") == 1
    assert await core.health().usage("g") == (None, None)


async def test_a_binding_records_a_stream_that_ends(make_core: MakeCore, host: RequestLoggingHost) -> None:
    host.answer("openai/d1", chat_stream("hello", usage=USAGE, logged=True))
    core: Final = weighted_pair(make_core, rpm=100)
    observed(core)
    bound: Final = request_logging()
    _ = core.observe(bound)  # pyright: ignore[reportCallIssue]  # every core takes it

    stream: Final = await core.acompletion(  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
        model="g", messages=MESSAGES, stream=True, litellm_logging_obj=bound
    )
    _ = await collect_texts(stream)
    await settle()

    assert successes(core, "d1") == 1
    assert await core.health().usage("g") == (12, 1)


async def test_a_binding_records_a_stream_failing_after_content(make_core: MakeCore, host: RequestLoggingHost) -> None:
    host.answer("openai/d1", chat_stream("partial", error=rate_limit(message="mid stream"), logged=True))
    core: Final = weighted_pair(make_core)
    observed(core)
    bound: Final = request_logging()
    _ = core.observe(bound)  # pyright: ignore[reportCallIssue]  # every core takes it

    stream: Final = await core.acompletion(  # pyright: ignore[reportAttributeAccessIssue]  # every core serves acompletion
        model="g", messages=MESSAGES, stream=True, litellm_logging_obj=bound
    )
    with pytest.raises(litellm.RateLimitError, match="mid stream"):
        _ = await collect_texts(stream)
    await settle()

    assert list(cooled_down(core, ["d1", "d2"])) == ["d1"]
    assert core.health().failures_this_minute("d1") == 1
    # No limit on the deployment, so the open counted nothing: the failure is the one request.
    assert await core.health().usage("g") == (None, 1)


# --- retire ----------------------------------------------------------------------------------------


@pytest.mark.parametrize("strategy", ["latency-based-routing", "least-busy", "usage-based-routing-v2"])
async def test_a_retired_router_still_picks_with_its_strategy(
    make_core: MakeCore, host: RequestLoggingHost, strategy: str
) -> None:
    core: Final = make_core(
        model_list=[deployment("g", "d1", rpm=100), deployment("g", "d2", rpm=100)],
        routing_strategy=strategy,
        num_retries=0,
    )
    core.retire()

    answered: Final = await attempt(core)

    assert not isinstance(answered, Exception)
    assert len(host.deployments_called()) == 1
