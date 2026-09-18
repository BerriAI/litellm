"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

from datetime import datetime

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.utils import EmbeddingResponse, TextCompletionResponse, Usage


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
async def test_team_max_parallel_requests_is_enforced_across_keys_in_the_team():
    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth

    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(cache)
    )
    team_id = "legacy-team"

    def team_key(raw: str) -> UserAPIKeyAuth:
        return UserAPIKeyAuth(
            api_key=hash_token(raw), team_id=team_id, team_max_parallel_requests=2
        )

    for raw in ("sk-a", "sk-b"):
        await handler.async_pre_call_hook(
            user_api_key_dict=team_key(raw),
            cache=cache,
            data={"model": "gpt-4o-mini"},
            call_type="",
        )

    with pytest.raises(HTTPException) as exc_info:
        await handler.async_pre_call_hook(
            user_api_key_dict=team_key("sk-a"),
            cache=cache,
            data={"model": "gpt-4o-mini"},
            call_type="",
        )
    assert exc_info.value.status_code == 429
    assert "rate limit type = team" in exc_info.value.detail
    assert "max_parallel_requests: 2" in exc_info.value.detail

    await handler.async_pre_call_hook(
        user_api_key_dict=UserAPIKeyAuth(
            api_key=hash_token("sk-c"),
            team_id="other-team",
            team_max_parallel_requests=2,
        ),
        cache=cache,
        data={"model": "gpt-4o-mini"},
        call_type="",
    )


@pytest.mark.asyncio
async def test_failed_request_releases_team_parallel_slot():
    from fastapi import HTTPException

    from litellm.proxy._types import UserAPIKeyAuth

    cache = DualCache()
    handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(cache)
    )
    key = UserAPIKeyAuth(
        api_key=hash_token("sk-a"), team_id="legacy-team", team_max_parallel_requests=1
    )
    data = {"model": "gpt-4o-mini"}

    await handler.async_pre_call_hook(
        user_api_key_dict=key, cache=cache, data=data, call_type=""
    )
    with pytest.raises(HTTPException):
        await handler.async_pre_call_hook(
            user_api_key_dict=key, cache=cache, data=data, call_type=""
        )

    await handler.async_log_failure_event(
        kwargs={
            "litellm_params": {
                "metadata": {
                    "user_api_key": key.api_key,
                    "user_api_key_team_id": "legacy-team",
                }
            },
            "exception": Exception("upstream 500"),
        },
        response_obj=None,
        start_time=None,
        end_time=None,
    )

    await handler.async_pre_call_hook(
        user_api_key_dict=key, cache=cache, data=data, call_type=""
    )
