"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

from datetime import datetime

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.utils import EmbeddingResponse, TextCompletionResponse, Usage


@pytest.mark.asyncio
async def test_pre_call_hook_counts_a_cli_session_under_the_per_user_alias_not_the_login_token():
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
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

    precise_minute = datetime.now().strftime("%Y-%m-%d-%H-%M")
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
    _api_key = hash_token("sk-12345")
    user_id = "ishaan"
    team_id = "litellm-team"
    end_user_id = "customer-1"

    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache())
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"

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
        start_time=datetime.now(),
        end_time=datetime.now(),
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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
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
