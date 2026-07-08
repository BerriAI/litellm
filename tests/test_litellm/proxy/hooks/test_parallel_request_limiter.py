"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

from datetime import datetime
from typing import Optional

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
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
                "user_api_key_auth": UserAPIKeyAuth(
                    api_key=_api_key,
                    rpm_limit=100,
                    user_rpm_limit=100,
                    team_rpm_limit=100,
                    end_user_rpm_limit=100,
                ),
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


def _build_kwargs(
    *,
    api_key: str,
    user_id: Optional[str] = None,
    team_id: Optional[str] = None,
    end_user_id: Optional[str] = None,
    auth_object: Optional[UserAPIKeyAuth] = None,
    user_api_key_metadata: Optional[dict] = None,
    user_api_key_team_metadata: Optional[dict] = None,
    user_api_key_model_max_budget: Optional[dict] = None,
) -> dict:
    metadata = {
        "user_api_key": api_key,
        "user_api_key_user_id": user_id,
        "user_api_key_team_id": team_id,
        "user_api_key_model_max_budget": user_api_key_model_max_budget or {},
    }
    if auth_object is not None:
        metadata["user_api_key_auth"] = auth_object
    if user_api_key_metadata is not None:
        metadata["user_api_key_metadata"] = user_api_key_metadata
    if user_api_key_team_metadata is not None:
        metadata["user_api_key_team_metadata"] = user_api_key_team_metadata
    kwargs: dict = {"litellm_params": {"metadata": metadata}}
    if end_user_id is not None:
        kwargs["user"] = end_user_id
    return kwargs


def _make_response(total_tokens: int = 10) -> EmbeddingResponse:
    return EmbeddingResponse(
        model="text-embedding-3-small",
        usage=Usage(prompt_tokens=total_tokens, completion_tokens=0, total_tokens=total_tokens),
    )


@pytest.mark.asyncio
async def test_async_log_success_event_skips_write_when_no_key_limits_configured():
    _api_key = hash_token("sk-no-limits")
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    kwargs = _build_kwargs(
        api_key=_api_key,
        auth_object=UserAPIKeyAuth(api_key=_api_key),
    )
    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=_make_response(10),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current = await handler.internal_usage_cache.async_get_cache(
        key=f"{_api_key}::{datetime.now().strftime('%Y-%m-%d-%H-%M')}::request_count",
        litellm_parent_otel_span=None,
    )
    assert current is None, "cache write should be skipped when no key limits are set"


@pytest.mark.asyncio
async def test_async_log_success_event_writes_when_key_rpm_limit_configured():
    _api_key = hash_token("sk-with-rpm")
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    kwargs = _build_kwargs(
        api_key=_api_key,
        auth_object=UserAPIKeyAuth(api_key=_api_key, rpm_limit=100),
    )
    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=_make_response(10),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    current = await handler.internal_usage_cache.async_get_cache(
        key=f"{_api_key}::{precise_minute}::request_count",
        litellm_parent_otel_span=None,
    )
    assert current is not None, "cache write should happen when rpm_limit is configured"
    assert current["current_tpm"] == 10


@pytest.mark.asyncio
async def test_async_log_success_event_skips_write_when_no_user_limits_configured():
    _api_key = hash_token("sk-user-no-limits")
    user_id = "user-no-limits"
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    kwargs = _build_kwargs(
        api_key=_api_key,
        user_id=user_id,
        auth_object=UserAPIKeyAuth(api_key=_api_key),
    )
    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=_make_response(10),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current = await handler.internal_usage_cache.async_get_cache(
        key=f"{user_id}::{datetime.now().strftime('%Y-%m-%d-%H-%M')}::request_count",
        litellm_parent_otel_span=None,
    )
    assert current is None, "user cache write should be skipped when no user limits are set"


@pytest.mark.asyncio
async def test_async_log_success_event_skips_write_when_no_team_limits_configured():
    _api_key = hash_token("sk-team-no-limits")
    team_id = "team-no-limits"
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    kwargs = _build_kwargs(
        api_key=_api_key,
        team_id=team_id,
        auth_object=UserAPIKeyAuth(api_key=_api_key),
    )
    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=_make_response(10),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current = await handler.internal_usage_cache.async_get_cache(
        key=f"{team_id}::{datetime.now().strftime('%Y-%m-%d-%H-%M')}::request_count",
        litellm_parent_otel_span=None,
    )
    assert current is None, "team cache write should be skipped when no team limits are set"


@pytest.mark.asyncio
async def test_async_log_success_event_skips_write_when_no_end_user_limits_configured():
    _api_key = hash_token("sk-enduser-no-limits")
    end_user_id = "enduser-no-limits"
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    kwargs = _build_kwargs(
        api_key=_api_key,
        end_user_id=end_user_id,
        auth_object=UserAPIKeyAuth(api_key=_api_key),
    )
    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=_make_response(10),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current = await handler.internal_usage_cache.async_get_cache(
        key=f"{end_user_id}::{datetime.now().strftime('%Y-%m-%d-%H-%M')}::request_count",
        litellm_parent_otel_span=None,
    )
    assert current is None, "end user cache write should be skipped when no end user limits are set"


@pytest.mark.asyncio
async def test_async_log_success_event_writes_when_auth_object_missing_from_metadata():
    _api_key = hash_token("sk-no-auth-obj")
    user_id = "user-fallback"
    team_id = "team-fallback"
    end_user_id = "enduser-fallback"
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    kwargs = _build_kwargs(
        api_key=_api_key,
        user_id=user_id,
        team_id=team_id,
        end_user_id=end_user_id,
    )
    assert "user_api_key_auth" not in kwargs["litellm_params"]["metadata"]
    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=_make_response(10),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    for scope_id in [_api_key, user_id, team_id, end_user_id]:
        current = await handler.internal_usage_cache.async_get_cache(
            key=f"{scope_id}::{precise_minute}::request_count",
            litellm_parent_otel_span=None,
        )
        assert current is not None, f"cache write for {scope_id} must happen when auth object is missing"
        assert current["current_tpm"] > 0


@pytest.mark.asyncio
async def test_async_log_success_event_model_key_block_unaffected():
    _api_key = hash_token("sk-model-key")
    model_rpm_limit = {"gpt-4": 100}
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    await handler.internal_usage_cache.async_set_cache(
        key=f"{_api_key}::gpt-4::{precise_minute}::request_count",
        value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )
    kwargs = _build_kwargs(
        api_key=_api_key,
        auth_object=UserAPIKeyAuth(api_key=_api_key),
        user_api_key_metadata={"model_rpm_limit": model_rpm_limit},
    )
    kwargs["litellm_params"]["metadata"]["model_group"] = "gpt-4"
    await handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=_make_response(20),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current = await handler.internal_usage_cache.async_get_cache(
        key=f"{_api_key}::gpt-4::{precise_minute}::request_count",
        litellm_parent_otel_span=None,
    )
    assert current is not None, "model+key block should still write when model_rpm_limit is set"
    assert current["current_tpm"] == 20


def test_entity_has_any_limit_returns_true_when_auth_object_missing():
    assert _PROXY_MaxParallelRequestsHandler._entity_has_any_limit(None, ["rpm_limit"]) is True


def test_entity_has_any_limit_returns_false_when_all_fields_none():
    auth = UserAPIKeyAuth(api_key="sk-test")
    assert auth.rpm_limit is None
    assert auth.tpm_limit is None
    assert _PROXY_MaxParallelRequestsHandler._entity_has_any_limit(auth, ["rpm_limit", "tpm_limit"]) is False


def test_entity_has_any_limit_returns_true_when_one_field_set():
    auth = UserAPIKeyAuth(api_key="sk-test", rpm_limit=100)
    assert _PROXY_MaxParallelRequestsHandler._entity_has_any_limit(auth, ["rpm_limit", "tpm_limit"]) is True


@pytest.mark.asyncio
async def test_async_log_failure_event_skips_decrement_when_no_limits_configured():
    _api_key = hash_token("sk-fail-no-limits")
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    await handler.internal_usage_cache.async_set_cache(
        key=f"{_api_key}::{precise_minute}::request_count",
        value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )
    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key": _api_key,
                "user_api_key_auth": UserAPIKeyAuth(api_key=_api_key),
            }
        },
        "exception": ValueError("test error"),
    }
    await handler.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current = await handler.internal_usage_cache.async_get_cache(
        key=f"{_api_key}::{precise_minute}::request_count",
        litellm_parent_otel_span=None,
    )
    assert current["current_requests"] == 1, "failure decrement should be skipped when no limits are configured"


@pytest.mark.asyncio
async def test_async_log_failure_event_decrements_when_limit_configured():
    _api_key = hash_token("sk-fail-with-limit")
    handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(DualCache()))
    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    await handler.internal_usage_cache.async_set_cache(
        key=f"{_api_key}::{precise_minute}::request_count",
        value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )
    kwargs = {
        "litellm_params": {
            "metadata": {
                "user_api_key": _api_key,
                "user_api_key_auth": UserAPIKeyAuth(api_key=_api_key, rpm_limit=100),
            }
        },
        "exception": ValueError("test error"),
    }
    await handler.async_log_failure_event(
        kwargs=kwargs,
        response_obj=None,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    current = await handler.internal_usage_cache.async_get_cache(
        key=f"{_api_key}::{precise_minute}::request_count",
        litellm_parent_otel_span=None,
    )
    assert current["current_requests"] == 0, "failure decrement should happen when limit is configured"
