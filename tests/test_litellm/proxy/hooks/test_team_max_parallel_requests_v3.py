"""
Team-level ``max_parallel_requests`` on the v3 limiter: one concurrency
gauge per team, shared by every key in the team, enforced alongside the
per-key gauge and released when the request settles.
"""

from datetime import datetime

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RequestRateLimiterStash,
    _PROXY_MaxParallelRequestsHandler_v3,
    _request_stash,
    get_or_create_request_stash,
    get_request_stash,
)
from litellm.proxy.utils import InternalUsageCache, hash_token
from litellm.types.utils import ModelResponse, Usage

TEAM_ID = "team-parallel"
TEAM_COUNTER_KEY = f"{{team:{TEAM_ID}}}:max_parallel_requests"


@pytest.fixture(autouse=True)
def _isolated_request_stash():
    token = _request_stash.set(None)
    yield
    _request_stash.reset(token)


def _handler() -> tuple[_PROXY_MaxParallelRequestsHandler_v3, DualCache]:
    cache = DualCache()
    return _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(cache)), cache


def _team_key(raw_key: str, team_max_parallel_requests: int, **extra) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key=hash_token(raw_key),
        team_id=TEAM_ID,
        team_max_parallel_requests=team_max_parallel_requests,
        **extra,
    )


async def _admit(
    handler: _PROXY_MaxParallelRequestsHandler_v3, cache: DualCache, key: UserAPIKeyAuth
) -> RequestRateLimiterStash:
    """Run the pre-call hook in a fresh per-request stash, as the proxy does for every request."""
    _request_stash.set(None)
    await handler.async_pre_call_hook(
        user_api_key_dict=key,
        cache=cache,
        data={"model": "gpt-4o-mini"},
        call_type="",
    )
    return get_or_create_request_stash()


async def _team_in_flight(handler: _PROXY_MaxParallelRequestsHandler_v3, cache: DualCache) -> int:
    return handler._gauge_in_flight_from_cache_value(await cache.async_get_cache(key=TEAM_COUNTER_KEY))


def test_team_descriptor_carries_max_parallel_requests_without_rpm_or_tpm():
    handler, _ = _handler()
    descriptors = handler._create_rate_limit_descriptors(
        user_api_key_dict=_team_key("sk-a", team_max_parallel_requests=3),
        data={"model": "gpt-4o-mini"},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )
    team_descriptors = [d for d in descriptors if d["key"] == "team"]
    assert len(team_descriptors) == 1
    assert team_descriptors[0]["value"] == TEAM_ID
    rate_limit = team_descriptors[0]["rate_limit"]
    assert rate_limit["max_parallel_requests"] == 3
    assert rate_limit["requests_per_unit"] is None
    assert rate_limit["tokens_per_unit"] is None


def test_team_without_any_limit_creates_no_team_descriptor():
    handler, _ = _handler()
    descriptors = handler._create_rate_limit_descriptors(
        user_api_key_dict=UserAPIKeyAuth(api_key=hash_token("sk-a"), team_id=TEAM_ID),
        data={"model": "gpt-4o-mini"},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )
    assert all(d["key"] != "team" for d in descriptors)


@pytest.mark.asyncio
async def test_team_gauge_is_shared_across_keys_in_the_team():
    handler, cache = _handler()
    key_a = _team_key("sk-a", team_max_parallel_requests=2)
    key_b = _team_key("sk-b", team_max_parallel_requests=2)
    key_other_team = UserAPIKeyAuth(api_key=hash_token("sk-c"), team_id="other-team", team_max_parallel_requests=2)

    await _admit(handler, cache, key_a)
    await _admit(handler, cache, key_b)
    assert await _team_in_flight(handler, cache) == 2
    assert get_request_stash().parallel_slot["counter_keys"] == [TEAM_COUNTER_KEY]

    with pytest.raises(HTTPException) as exc_info:
        await _admit(handler, cache, key_a)
    assert exc_info.value.status_code == 429
    assert f"team: {TEAM_ID}" in exc_info.value.detail
    assert "max_parallel_requests" in exc_info.value.detail
    assert await _team_in_flight(handler, cache) == 2

    await _admit(handler, cache, key_other_team)


@pytest.mark.asyncio
async def test_team_slot_released_on_success_readmits_next_request():
    handler, cache = _handler()
    key = _team_key("sk-a", team_max_parallel_requests=1)

    admitted = await _admit(handler, cache, key)
    with pytest.raises(HTTPException):
        await _admit(handler, cache, _team_key("sk-b", team_max_parallel_requests=1))

    _request_stash.set(admitted)
    await handler.async_log_success_event(
        kwargs={"standard_logging_object": {"metadata": {"user_api_key_hash": key.api_key}}},
        response_obj=ModelResponse(usage=Usage(prompt_tokens=1, completion_tokens=1, total_tokens=2)),
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert await _team_in_flight(handler, cache) == 0

    await _admit(handler, cache, _team_key("sk-b", team_max_parallel_requests=1))


@pytest.mark.asyncio
async def test_team_slot_released_on_failure():
    handler, cache = _handler()
    key = _team_key("sk-a", team_max_parallel_requests=1)

    await _admit(handler, cache, key)
    await handler.async_log_failure_event(
        kwargs={"standard_logging_object": {"metadata": {"user_api_key_hash": key.api_key}}},
        response_obj=None,
        start_time=None,
        end_time=None,
    )
    assert await _team_in_flight(handler, cache) == 0
    await _admit(handler, cache, key)


@pytest.mark.asyncio
async def test_key_and_team_gauges_are_enforced_together():
    handler, cache = _handler()
    key_a = _team_key("sk-a", team_max_parallel_requests=5, max_parallel_requests=1)
    key_b = _team_key("sk-b", team_max_parallel_requests=5, max_parallel_requests=1)

    await _admit(handler, cache, key_a)
    assert set(get_request_stash().parallel_slot["counter_keys"]) == {
        f"{{api_key:{key_a.api_key}}}:max_parallel_requests",
        TEAM_COUNTER_KEY,
    }

    with pytest.raises(HTTPException) as exc_info:
        await _admit(handler, cache, key_a)
    assert "api_key" in exc_info.value.detail
    assert await _team_in_flight(handler, cache) == 1

    await _admit(handler, cache, key_b)
    assert await _team_in_flight(handler, cache) == 2


@pytest.mark.asyncio
async def test_team_gauge_rejection_does_not_consume_key_slot():
    handler, cache = _handler()
    key_a = _team_key("sk-a", team_max_parallel_requests=1, max_parallel_requests=3)
    key_b = _team_key("sk-b", team_max_parallel_requests=1, max_parallel_requests=3)
    key_b_counter = f"{{api_key:{key_b.api_key}}}:max_parallel_requests"

    await _admit(handler, cache, key_a)
    with pytest.raises(HTTPException):
        await _admit(handler, cache, key_b)

    assert handler._gauge_in_flight_from_cache_value(await cache.async_get_cache(key=key_b_counter)) == 0
    assert get_request_stash().parallel_slot is None
