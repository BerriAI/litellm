"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

import itertools
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Final

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter import (
    PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.utils import EmbeddingResponse, ModelResponse, TextCompletionResponse, Usage

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
@pytest.mark.asyncio
async def test_async_log_success_event_counts_non_chat_response_tokens(response_obj):
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
    for scope_id in scope_ids:
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
        assert current["current_tpm"] == 50, (
            f"expected 50 tokens counted for {scope_id}, "
            f"got {current['current_tpm']}"
        )


@pytest.mark.asyncio
async def test_async_log_failure_event_skips_batch_line_item_events():
    """Failed line children were never admitted by the limiter, so the failure
    hook must not decrement request counters they never incremented."""
    parallel_request_handler = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )
    local_cache = parallel_request_handler.internal_usage_cache.dual_cache.in_memory_cache

    await parallel_request_handler.async_log_failure_event(
        kwargs={
            "exception": "litellm.APIError: upstream 500",
            "litellm_params": {
                "batch_parent_id": "batch_1",
                "metadata": {"user_api_key": hash_token("sk-line-item")},
            },
            "model": "gpt-3.5-turbo",
        },
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert local_cache.cache_dict == {}


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
        assert await internal_usage_cache.async_get_cache(
            key=f"{api_key}::{torn_minute}::request_count", litellm_parent_otel_span=None
        ) is None


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
        kwargs={
            "litellm_params": {
                "metadata": {"user_api_key": api_key, "user_api_key_model_max_budget": {}}
            }
        },
        response_obj=ModelResponse(usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7)),
        start_time=LAST_MICROSECOND_OF_JANUARY,
        end_time=FIRST_MICROSECOND_OF_FEBRUARY,
    )

    assert await internal_usage_cache.async_get_cache(
        key=f"{api_key}::{LAST_MINUTE_OF_JANUARY}::request_count", litellm_parent_otel_span=None
    ) == {"current_requests": 0, "current_tpm": 7, "current_rpm": 1}
    for torn_minute in (FIRST_MINUTE_OF_FEBRUARY, TORN_MINUTE_OF_JANUARY):
        assert await internal_usage_cache.async_get_cache(
            key=f"{api_key}::{torn_minute}::request_count", litellm_parent_otel_span=None
        ) is None


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
        assert await internal_usage_cache.async_get_cache(
            key=f"{api_key}::{torn_minute}::request_count", litellm_parent_otel_span=None
        ) is None


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
