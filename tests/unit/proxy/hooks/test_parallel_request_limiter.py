"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

import asyncio
import itertools
from collections.abc import Callable, Coroutine, Iterator, Mapping
from datetime import datetime
from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter import (
    PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import _request_stash
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token
from litellm.types.utils import CallTypes, EmbeddingResponse, ModelResponse, TextCompletionResponse, Usage

FROZEN_INSTANT: Final = datetime(2026, 1, 31, 23, 59, 30)
LAST_MICROSECOND_OF_JANUARY: Final = datetime(2026, 1, 31, 23, 59, 59, 999999)
FIRST_MICROSECOND_OF_FEBRUARY: Final = datetime(2026, 2, 1, 0, 0, 0, 1)
LAST_MINUTE_OF_JANUARY: Final = "2026-01-31-23-59"
FIRST_MINUTE_OF_FEBRUARY: Final = "2026-02-01-00-00"
TORN_MINUTE_OF_JANUARY: Final = "2026-01-31-00-00"


def _frozen_clock() -> datetime:
    return FROZEN_INSTANT


def _clock_reading(instants: Iterator[datetime]) -> Callable[[], datetime]:
    return lambda: next(instants)


def _clock_rolling_over_after_first_read() -> Callable[[], datetime]:
    return _clock_reading(
        itertools.chain([LAST_MICROSECOND_OF_JANUARY], itertools.repeat(FIRST_MICROSECOND_OF_FEBRUARY))
    )


@pytest.fixture(autouse=True)
def _isolated_request_stash():
    token = _request_stash.set(None)
    yield
    _request_stash.reset(token)


def _frozen_handler() -> PROXY_MaxParallelRequestsHandler:
    return PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()), clock=_frozen_clock)


async def _in_flight(
    handler: PROXY_MaxParallelRequestsHandler, api_key: str, minute: str = FROZEN_INSTANT.strftime("%Y-%m-%d-%H-%M")
) -> int:
    await asyncio.sleep(0)
    bucket: Final = await handler.internal_usage_cache.async_get_cache(
        key=f"{api_key}::{minute}::request_count", litellm_parent_otel_span=None
    )
    return bucket["current_requests"]


async def _admit(
    handler: PROXY_MaxParallelRequestsHandler,
    auth: UserAPIKeyAuth,
    call_type: str = "completion",
    litellm_call_id: str = "call-main",
) -> None:
    await handler.async_pre_call_hook(
        user_api_key_dict=auth,
        cache=DualCache(),
        data={"model": "gpt-4o-mini", "litellm_call_id": litellm_call_id},
        call_type=call_type,
    )
    await asyncio.sleep(0)


def _as_its_own_request(coro: Coroutine[object, object, None]) -> asyncio.Task[None]:
    return asyncio.create_task(coro)


@pytest.mark.parametrize(
    "current,rpm_limit",
    [
        (None, 0),
        ({"current_requests": 0, "current_tpm": 0, "current_rpm": 1}, 1),
    ],
)
@pytest.mark.asyncio
async def test_model_per_key_rate_limit_error_carries_descriptor_key(current: Mapping[str, int] | None, rpm_limit: int):
    handler: Final = PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

    with pytest.raises(ProxyRateLimitError) as exc_info:
        await handler.check_key_in_limits(
            user_api_key_dict=UserAPIKeyAuth(),
            cache=DualCache(),
            data={"model": "gpt-4o-mini"},
            call_type="completion",
            max_parallel_requests=10,
            tpm_limit=100,
            rpm_limit=rpm_limit,
            current=dict(current) if current is not None else None,
            request_count_api_key="test-key:model_per_key",
            rate_limit_type="model_per_key",
            values_to_update_in_cache=[],
        )

    assert exc_info.value.descriptor_key == "model_per_key"


@pytest.mark.asyncio
async def test_pre_call_hook_counts_a_cli_session_under_the_per_user_alias_not_the_login_token():
    handler = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=_frozen_clock
    )
    session = UserAPIKeyAuth(
        api_key="cli-session-Qm7xJ2kP9sLw4vT1nR8yAa",
        user_id="alice",
        key_alias="cli-session-alice",
        is_session_token=True,
        max_parallel_requests=5,
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=session, cache=DualCache(), data={"model": "gpt-4o-mini"}, call_type="completion"
    )

    precise_minute = FROZEN_INSTANT.strftime("%Y-%m-%d-%H-%M")
    counted = await handler.internal_usage_cache.async_get_cache(
        key=f"cli-session-alice::{precise_minute}::request_count", litellm_parent_otel_span=None
    )
    assert counted == {"current_requests": 1, "current_tpm": 0, "current_rpm": 1}, counted


@pytest.mark.parametrize(
    "response_obj",
    [
        EmbeddingResponse(
            model="text-embedding-3-small",
            usage=Usage(prompt_tokens=50, completion_tokens=0, total_tokens=50),
        ),
        TextCompletionResponse(
            model="gpt-3.5-turbo-instruct",
            usage=Usage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        ),
    ],
)
@pytest.mark.parametrize("opened_this_minute", [True, False], ids=["opened-this-minute", "opened-last-minute"])
@pytest.mark.asyncio
async def test_async_log_success_event_counts_non_chat_response_tokens(response_obj, opened_this_minute):
    """
    Embedding and text completion responses must increment the per key, user,
    team, and end user TPM counters, not just chat completion ModelResponse
    objects.
    """
    _api_key = hash_token("sk-98765")
    user_id = "ishaan"
    team_id = "litellm-team"
    end_user_id = "customer-1"

    parallel_request_handler = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=_frozen_clock
    )

    precise_minute = FROZEN_INSTANT.strftime("%Y-%m-%d-%H-%M")

    scope_ids = [_api_key, user_id, team_id, end_user_id]
    for scope_id in scope_ids if opened_this_minute else ():
        await parallel_request_handler.internal_usage_cache.async_set_cache(
            key=f"{scope_id}::{precise_minute}::request_count",
            value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
            litellm_parent_otel_span=None,
        )

    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key": _api_key,
                "user_api_key_user_id": user_id,
                "user_api_key_team_id": team_id,
                "user_api_key_model_max_budget": {},
            }
        },
        "user": end_user_id,
    }

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=response_obj,
        start_time=FROZEN_INSTANT,
        end_time=FROZEN_INSTANT,
    )

    for scope_id in scope_ids:
        current = await parallel_request_handler.internal_usage_cache.async_get_cache(
            key=f"{scope_id}::{precise_minute}::request_count",
            litellm_parent_otel_span=None,
        )
        assert current["current_tpm"] == 50, f"expected 50 tokens counted for {scope_id}, got {current['current_tpm']}"


def _socket_turn_events(input_tokens: int, output_tokens: int) -> list[dict[str, object]]:
    usage: Final = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }
    return [
        {"type": "response.created", "response": {"id": "resp_ws_turn", "status": "in_progress", "output": []}},
        {"type": "response.completed", "response": {"id": "resp_ws_turn", "status": "completed", "usage": usage}},
    ]


@pytest.mark.parametrize(
    "call_type, events, tokens",
    [
        (CallTypes.aresponses_websocket.value, _socket_turn_events(input_tokens=590, output_tokens=10), 600),
        (CallTypes.aresponses.value, _socket_turn_events(input_tokens=590, output_tokens=10), 0),
    ],
    ids=["responses-websocket-turn", "events-of-a-non-socket-call"],
)
@pytest.mark.parametrize("opened_this_minute", [True, False], ids=["opened-this-minute", "opened-last-minute"])
@pytest.mark.asyncio
async def test_async_log_success_event_counts_socket_session_tokens(
    call_type: str, events: list[dict[str, object]], tokens: int, opened_this_minute: bool
):
    api_key: Final = hash_token("sk-socket-session")
    handler: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=_frozen_clock
    )
    precise_minute: Final = FROZEN_INSTANT.strftime("%Y-%m-%d-%H-%M")
    scope_ids: Final = (api_key, "socket-user", "socket-team", "socket-end-user")
    for scope_id in scope_ids if opened_this_minute else ():
        await handler.internal_usage_cache.async_set_cache(
            key=f"{scope_id}::{precise_minute}::request_count",
            value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
            litellm_parent_otel_span=None,
        )

    await handler.async_log_success_event(
        kwargs={
            "call_type": call_type,
            "litellm_params": {
                "metadata": {
                    "user_api_key": api_key,
                    "user_api_key_user_id": "socket-user",
                    "user_api_key_team_id": "socket-team",
                    "user_api_key_model_max_budget": {},
                }
            },
            "user": "socket-end-user",
        },
        response_obj=events,
        start_time=FROZEN_INSTANT,
        end_time=FROZEN_INSTANT,
    )

    counted: Final = [
        (
            await handler.internal_usage_cache.async_get_cache(
                key=f"{scope_id}::{precise_minute}::request_count", litellm_parent_otel_span=None
            )
        )["current_tpm"]
        for scope_id in scope_ids
    ]
    assert counted == [tokens, tokens, tokens, tokens]


@pytest.mark.asyncio
async def test_a_pre_call_across_a_minute_rollover_lands_in_the_bucket_of_its_first_clock_read():
    internal_usage_cache: Final = InternalUsageCache(DualCache())
    handler: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=internal_usage_cache, clock=_clock_rolling_over_after_first_read()
    )
    session: Final = UserAPIKeyAuth(api_key="sk-torn-pre", max_parallel_requests=5)
    api_key: Final = session.api_key

    await handler.async_pre_call_hook(
        user_api_key_dict=session, cache=DualCache(), data={"model": "gpt-4o-mini"}, call_type="completion"
    )

    assert await internal_usage_cache.async_get_cache(
        key=f"{api_key}::{LAST_MINUTE_OF_JANUARY}::request_count", litellm_parent_otel_span=None
    ) == {"current_requests": 1, "current_tpm": 0, "current_rpm": 1}
    for torn_minute in (FIRST_MINUTE_OF_FEBRUARY, TORN_MINUTE_OF_JANUARY):
        assert (
            await internal_usage_cache.async_get_cache(
                key=f"{api_key}::{torn_minute}::request_count", litellm_parent_otel_span=None
            )
            is None
        )


@pytest.mark.asyncio
async def test_a_success_event_across_a_minute_rollover_lands_in_the_bucket_of_its_first_clock_read():
    internal_usage_cache: Final = InternalUsageCache(DualCache())
    handler: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=internal_usage_cache, clock=_clock_rolling_over_after_first_read()
    )
    api_key: Final = hash_token("sk-torn-success")
    await internal_usage_cache.async_set_cache(
        key=f"{api_key}::{LAST_MINUTE_OF_JANUARY}::request_count",
        value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )

    await handler.async_log_success_event(
        kwargs={"litellm_params": {"metadata": {"user_api_key": api_key, "user_api_key_model_max_budget": {}}}},
        response_obj=ModelResponse(usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7)),
        start_time=LAST_MICROSECOND_OF_JANUARY,
        end_time=FIRST_MICROSECOND_OF_FEBRUARY,
    )

    assert await internal_usage_cache.async_get_cache(
        key=f"{api_key}::{LAST_MINUTE_OF_JANUARY}::request_count", litellm_parent_otel_span=None
    ) == {"current_requests": 0, "current_tpm": 7, "current_rpm": 1}
    for torn_minute in (FIRST_MINUTE_OF_FEBRUARY, TORN_MINUTE_OF_JANUARY):
        assert (
            await internal_usage_cache.async_get_cache(
                key=f"{api_key}::{torn_minute}::request_count", litellm_parent_otel_span=None
            )
            is None
        )


@pytest.mark.asyncio
async def test_a_failure_event_across_a_minute_rollover_lands_in_the_bucket_of_its_first_clock_read():
    internal_usage_cache: Final = InternalUsageCache(DualCache())
    handler: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=internal_usage_cache, clock=_clock_rolling_over_after_first_read()
    )
    api_key: Final = hash_token("sk-torn-failure")
    await internal_usage_cache.async_set_cache(
        key=f"{api_key}::{LAST_MINUTE_OF_JANUARY}::request_count",
        value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )

    await handler.async_log_failure_event(
        kwargs={
            "litellm_params": {"metadata": {"user_api_key": api_key}},
            "exception": Exception("upstream boom"),
        },
        response_obj=None,
        start_time=LAST_MICROSECOND_OF_JANUARY,
        end_time=FIRST_MICROSECOND_OF_FEBRUARY,
    )

    assert await internal_usage_cache.async_get_cache(
        key=f"{api_key}::{LAST_MINUTE_OF_JANUARY}::request_count", litellm_parent_otel_span=None
    ) == {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}
    for torn_minute in (FIRST_MINUTE_OF_FEBRUARY, TORN_MINUTE_OF_JANUARY):
        assert (
            await internal_usage_cache.async_get_cache(
                key=f"{api_key}::{torn_minute}::request_count", litellm_parent_otel_span=None
            )
            is None
        )


@pytest.mark.asyncio
async def test_a_post_call_headers_read_across_a_minute_rollover_uses_the_bucket_of_its_first_clock_read():
    internal_usage_cache: Final = InternalUsageCache(DualCache())
    handler: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=internal_usage_cache, clock=_clock_rolling_over_after_first_read()
    )
    user_api_key_dict: Final = UserAPIKeyAuth(api_key="sk-torn-post", rpm_limit=5, tpm_limit=100)
    api_key: Final = user_api_key_dict.api_key
    await internal_usage_cache.async_set_cache(
        key=f"{api_key}::{LAST_MINUTE_OF_JANUARY}::request_count",
        value={"current_requests": 1, "current_tpm": 10, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )
    response: Final = ModelResponse()
    response._hidden_params = {}

    await handler.async_post_call_success_hook(
        data={"model": "gpt-4o-mini"},
        user_api_key_dict=user_api_key_dict,
        response=response,
    )

    assert response._hidden_params["additional_headers"] == {
        "x-ratelimit-remaining-requests": 4,
        "x-ratelimit-limit-requests": 5,
        "x-ratelimit-remaining-tokens": 90,
        "x-ratelimit-limit-tokens": 100,
    }


@pytest.mark.asyncio
async def test_a_request_in_one_minute_is_not_counted_by_a_pre_call_in_the_next_minute():
    internal_usage_cache: Final = InternalUsageCache(DualCache())
    handler: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=internal_usage_cache,
        clock=_clock_reading(iter([datetime(2026, 3, 10, 12, 0, 0), datetime(2026, 3, 10, 12, 1, 0)])),
    )
    session: Final = UserAPIKeyAuth(api_key="sk-minute-reset", rpm_limit=1)
    api_key: Final = session.api_key

    await handler.async_pre_call_hook(
        user_api_key_dict=session, cache=DualCache(), data={"model": "gpt-4o-mini"}, call_type="completion"
    )
    await handler.async_pre_call_hook(
        user_api_key_dict=session, cache=DualCache(), data={"model": "gpt-4o-mini"}, call_type="completion"
    )

    assert await internal_usage_cache.async_get_cache(
        key=f"{api_key}::2026-03-10-12-01::request_count", litellm_parent_otel_span=None
    ) == {"current_requests": 1, "current_tpm": 0, "current_rpm": 1}


@pytest.mark.asyncio
async def test_retry_after_is_the_seconds_until_the_next_minute_of_the_injected_clock():
    internal_usage_cache: Final = InternalUsageCache(DualCache())
    handler: Final = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=internal_usage_cache,
        clock=_clock_reading(itertools.repeat(datetime(2026, 3, 10, 12, 0, 45, 500000))),
    )
    session: Final = UserAPIKeyAuth(api_key="sk-retry-after", rpm_limit=1)

    await handler.async_pre_call_hook(
        user_api_key_dict=session, cache=DualCache(), data={"model": "gpt-4o-mini"}, call_type="completion"
    )
    with pytest.raises(ProxyRateLimitError) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=session, cache=DualCache(), data={"model": "gpt-4o-mini"}, call_type="completion"
        )

    assert exc_info.value.headers["retry-after"] == "14.5"


@pytest.mark.asyncio
async def test_websocket_turn_callbacks_keep_the_sockets_slot_until_the_disconnect_release():
    api_key = hash_token("sk-websocket")
    handler = _frozen_handler()
    auth = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=1)
    await _admit(handler, auth, call_type=CallTypes.aresponses_websocket.value)
    assert await _in_flight(handler, api_key) == 1
    turn_kwargs = {
        "call_type": CallTypes.aresponses_websocket.value,
        "litellm_params": {"metadata": {"user_api_key": api_key}},
        "exception": RuntimeError("provider rejected the turn"),
    }

    await handler.async_log_success_event(
        kwargs=turn_kwargs, response_obj=None, start_time=FROZEN_INSTANT, end_time=FROZEN_INSTANT
    )
    await handler.async_log_failure_event(
        kwargs=turn_kwargs, response_obj=None, start_time=FROZEN_INSTANT, end_time=FROZEN_INSTANT
    )
    assert await _in_flight(handler, api_key) == 1

    await handler.async_release_max_parallel_requests_on_disconnect(auth)
    assert await _in_flight(handler, api_key) == 0


@pytest.mark.asyncio
async def test_a_rejected_realtime_attempt_does_not_free_a_slot_it_never_took():
    api_key = hash_token("sk-at-limit")
    handler = _frozen_handler()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = handler
    auth = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=1)
    await _as_its_own_request(_admit(handler, auth, call_type=CallTypes.aresponses_websocket.value))
    with pytest.raises(ProxyRateLimitError):
        await _admit(handler, auth, call_type=CallTypes.aresponses_websocket.value)
    assert await _in_flight(handler, api_key) == 1

    await proxy_logging_obj.arelease_max_parallel_requests_on_disconnect(auth)
    await proxy_logging_obj.arelease_responses_websocket_slot(auth)
    assert await _in_flight(handler, api_key) == 1


def _completed(api_key: str, litellm_call_id: str = "call-main") -> dict[str, object]:
    return {
        "litellm_call_id": litellm_call_id,
        "litellm_params": {"metadata": {"user_api_key": api_key, "user_api_key_model_max_budget": {}}},
    }


def _failed(api_key: str, litellm_call_id: str = "call-main") -> dict[str, object]:
    return {
        "litellm_call_id": litellm_call_id,
        "litellm_params": {"metadata": {"user_api_key": api_key}},
        "exception": RuntimeError("upstream failed"),
    }


@pytest.mark.parametrize(
    "callbacks_before_cleanup",
    [
        pytest.param((), id="client hung up before any callback"),
        pytest.param(("success",), id="stream completed"),
        pytest.param(("failure",), id="upstream failed"),
    ],
)
@pytest.mark.asyncio
async def test_an_ended_request_frees_its_slot_exactly_once_however_often_cleanup_runs(
    callbacks_before_cleanup: tuple[str, ...],
):
    api_key = hash_token("sk-stream")
    handler = _frozen_handler()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = handler
    auth = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=2)

    async def ended_request() -> None:
        await _admit(handler, auth)
        for callback in callbacks_before_cleanup:
            if callback == "success":
                await handler.async_log_success_event(
                    kwargs=_completed(api_key), response_obj=None, start_time=FROZEN_INSTANT, end_time=FROZEN_INSTANT
                )
            else:
                await handler.async_log_failure_event(
                    kwargs=_failed(api_key), response_obj=None, start_time=FROZEN_INSTANT, end_time=FROZEN_INSTANT
                )
        await proxy_logging_obj.arelease_max_parallel_requests(auth)
        await proxy_logging_obj.arelease_max_parallel_requests_on_disconnect(auth)
        assert await _in_flight(handler, api_key) == 1
        await proxy_logging_obj.arelease_max_parallel_requests_on_disconnect(auth)
        await handler.async_log_success_event(
            kwargs=_completed(api_key), response_obj=None, start_time=FROZEN_INSTANT, end_time=FROZEN_INSTANT
        )

    await _as_its_own_request(_admit(handler, auth))
    assert await _in_flight(handler, api_key) == 1
    await _as_its_own_request(ended_request())
    assert await _in_flight(handler, api_key) == 1


@pytest.mark.asyncio
async def test_a_request_dropped_after_the_minute_rolled_over_frees_the_slot_it_took_in_its_own_minute():
    api_key = hash_token("sk-straddles-the-minute")
    handler = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=_clock_rolling_over_after_first_read()
    )
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = handler
    auth = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=2)

    first_admitted = asyncio.Event()
    second_admitted = asyncio.Event()

    async def admitted_in_january_dropped_in_february() -> None:
        await _admit(handler, auth)
        first_admitted.set()
        await second_admitted.wait()
        await proxy_logging_obj.arelease_max_parallel_requests_on_disconnect(auth)

    january = _as_its_own_request(admitted_in_january_dropped_in_february())
    await first_admitted.wait()
    await _as_its_own_request(_admit(handler, auth))
    second_admitted.set()
    await january
    assert await _in_flight(handler, api_key, LAST_MINUTE_OF_JANUARY) == 0
    assert await _in_flight(handler, api_key, FIRST_MINUTE_OF_FEBRUARY) == 1


@pytest.mark.asyncio
async def test_a_request_that_straddles_a_minute_frees_its_slot_in_its_own_minute_and_counts_tokens_in_the_new_one():
    api_key = hash_token("sk-tokens-straddle")
    handler = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=_clock_rolling_over_after_first_read()
    )
    auth = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=2)

    async def admitted_in_january_finished_in_february() -> None:
        await _admit(handler, auth)
        await handler.async_log_success_event(
            kwargs=_completed(api_key),
            response_obj=ModelResponse(usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7)),
            start_time=LAST_MICROSECOND_OF_JANUARY,
            end_time=FIRST_MICROSECOND_OF_FEBRUARY,
        )

    await _as_its_own_request(admitted_in_january_finished_in_february())
    january = await handler.internal_usage_cache.async_get_cache(
        key=f"{api_key}::{LAST_MINUTE_OF_JANUARY}::request_count", litellm_parent_otel_span=None
    )
    february = await handler.internal_usage_cache.async_get_cache(
        key=f"{api_key}::{FIRST_MINUTE_OF_FEBRUARY}::request_count", litellm_parent_otel_span=None
    )
    assert january["current_requests"] == 0 and january["current_tpm"] == 0, january
    assert february["current_requests"] == 0 and february["current_tpm"] == 7, february


@pytest.mark.asyncio
async def test_a_nested_calls_callbacks_leave_the_requests_slot_to_the_request():
    api_key = hash_token("sk-guardrail")
    handler = _frozen_handler()
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = handler
    auth = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=2)

    async def request_with_a_guardrail_call() -> None:
        await _admit(handler, auth)
        await handler.async_log_success_event(
            kwargs=_completed(api_key, "call-guardrail"),
            response_obj=None,
            start_time=FROZEN_INSTANT,
            end_time=FROZEN_INSTANT,
        )
        await handler.async_log_failure_event(
            kwargs=_failed(api_key, "call-guardrail"),
            response_obj=None,
            start_time=FROZEN_INSTANT,
            end_time=FROZEN_INSTANT,
        )
        assert await _in_flight(handler, api_key) == 2
        await proxy_logging_obj.arelease_max_parallel_requests_on_disconnect(auth)

    await _as_its_own_request(_admit(handler, auth))
    await _as_its_own_request(request_with_a_guardrail_call())
    assert await _in_flight(handler, api_key) == 1
