"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.utils import EmbeddingResponse, TextCompletionResponse, Usage


@pytest.mark.asyncio
async def test_realtime_release_preserves_newer_local_admission_while_redis_finishes():
    started, finish = asyncio.Event(), asyncio.Event()

    async def release(**kwargs):
        started.set()
        await finish.wait()

    remote = MagicMock(spec=RedisCache)
    remote.async_register_script.return_value = AsyncMock(side_effect=release)
    cache = DualCache(redis_cache=remote)
    handler = _PROXY_MaxParallelRequestsHandler(InternalUsageCache(cache))
    await cache.async_set_cache("key", {"current_requests": 1, "current_rpm": 1, "current_tpm": 7}, local_only=True)
    task = asyncio.create_task(handler._release_realtime_counter("key"))
    await started.wait()
    next_admission = {"current_requests": 1, "current_rpm": 2, "current_tpm": 7}
    await cache.async_set_cache("key", next_admission, local_only=True)
    finish.set()
    await task
    assert await cache.async_get_cache("key", local_only=True) == next_admission


@pytest.mark.asyncio
@pytest.mark.parametrize("reject_team", [False, True])
async def test_realtime_attachment_releases_only_acquired_legacy_slots(reject_team):
    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(InternalUsageCache(cache))
    auth = UserAPIKeyAuth(
        api_key="attachment-key",
        user_id="attachment-user",
        team_id="attachment-team",
        team_rpm_limit=0 if reject_team else 100,
        max_parallel_requests=1,
        end_user_id="attachment-end-user",
        metadata={"model_rpm_limit": {"test-model": 100}},
    )
    data = {"model": "test-model", "metadata": {"global_max_parallel_requests": 10}}
    minute = datetime.now().strftime("%Y-%m-%d-%H-%M")
    team_key = f"attachment-team::{minute}::request_count"
    await cache.async_set_cache(team_key, {"current_requests": 3, "current_tpm": 7, "current_rpm": 4})
    handler.begin_realtime_attachment(data)
    if reject_team:
        with pytest.raises(ProxyRateLimitError, match="Rate Limit Handler"):
            await handler.async_pre_call_hook(auth, cache, data, "_arealtime")
    else:
        await handler.async_pre_call_hook(auth, cache, data, "_arealtime")
    await handler.async_release_realtime_attachment(data, auth)
    await handler.async_release_realtime_attachment(data, auth)
    assert await cache.async_get_cache("global_max_parallel_requests") == 0
    assert await cache.async_get_cache(f"attachment-key::{minute}::request_count") == {
        "current_requests": 0,
        "current_tpm": 0,
        "current_rpm": 1,
    }
    assert await cache.async_get_cache(f"attachment-user::{minute}::request_count") == {
        "current_requests": 0,
        "current_tpm": 0,
        "current_rpm": 1,
    }
    assert await cache.async_get_cache(team_key) == {
        "current_requests": 3,
        "current_tpm": 7,
        "current_rpm": 4 if reject_team else 5,
    }
    assert await cache.async_get_cache(f"attachment-key::test-model::{minute}::request_count") == {
        "current_requests": 0,
        "current_tpm": 0,
        "current_rpm": 1,
    }
    end_user = await cache.async_get_cache(f"attachment-end-user::{minute}::request_count")
    assert end_user == (None if reject_team else {"current_requests": 0, "current_tpm": 0, "current_rpm": 1})
    if not reject_team:
        handler.begin_realtime_attachment(data)
        await handler.async_pre_call_hook(auth, cache, data, "_arealtime")
        await handler.async_release_realtime_attachment(data, auth)


@pytest.mark.asyncio
async def test_realtime_attachment_rejected_before_acquisition_preserves_other_slot():
    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(InternalUsageCache(cache))
    auth = UserAPIKeyAuth(api_key="busy-key", max_parallel_requests=1)
    minute = datetime.now().strftime("%Y-%m-%d-%H-%M")
    key = f"busy-key::{minute}::request_count"
    current = {"current_requests": 1, "current_tpm": 13, "current_rpm": 2}
    await cache.async_set_cache(key, current)
    data = {"model": "test-model"}
    handler.begin_realtime_attachment(data)
    with pytest.raises(ProxyRateLimitError, match="Rate Limit Handler"):
        await handler.async_pre_call_hook(auth, cache, data, "_arealtime")
    await handler.async_release_realtime_attachment(data, auth)
    assert await cache.async_get_cache(key) == current


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

    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))

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
        assert current["current_tpm"] == 50, f"expected 50 tokens counted for {scope_id}, got {current['current_tpm']}"
