from collections.abc import Iterator, Sequence
from datetime import datetime

import pytest
from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    PARALLEL_REQUEST_SLOT_TTL_SECONDS,
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
def _isolated_request_stash() -> Iterator[None]:
    token = _request_stash.set(None)
    yield
    _request_stash.reset(token)


def _handler() -> tuple[_PROXY_MaxParallelRequestsHandler_v3, DualCache]:
    cache = DualCache()
    return _PROXY_MaxParallelRequestsHandler_v3(internal_usage_cache=InternalUsageCache(cache)), cache


def _team_key(
    raw_key: str, team_max_parallel_requests: int, max_parallel_requests: int | None = None
) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key=hash_token(raw_key),
        team_id=TEAM_ID,
        team_max_parallel_requests=team_max_parallel_requests,
        max_parallel_requests=max_parallel_requests,
    )


async def _admit(
    handler: _PROXY_MaxParallelRequestsHandler_v3, cache: DualCache, key: UserAPIKeyAuth
) -> RequestRateLimiterStash:
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


KEY_A = hash_token("sk-a")
KEY_A_COUNTER_KEY = f"{{api_key:{KEY_A}}}:max_parallel_requests"


class _FakeRedisGauges:
    def __init__(self, reject_key: str | None = None, raise_on_key: str | None = None) -> None:
        self.reject_key = reject_key
        self.raise_on_key = raise_on_key
        self.acquire_calls: list[tuple[tuple[str, ...], tuple[object, ...]]] = []
        self.release_calls: list[tuple[tuple[str, ...], tuple[object, ...]]] = []
        self.count_calls: list[tuple[str, ...]] = []

    async def acquire(self, keys: Sequence[str], args: Sequence[object]) -> list[int]:
        self.acquire_calls.append((tuple(keys), tuple(args)))
        if keys[0] == self.raise_on_key:
            raise ConnectionError("CROSSSLOT Keys in request don't hash to the same slot")
        if keys[0] == self.reject_key:
            return [1, 1, 1, 1]
        return [0, 1]

    async def release(self, keys: Sequence[str], args: Sequence[object]) -> list[int]:
        self.release_calls.append((tuple(keys), tuple(args)))
        return [0]

    async def count(self, keys: Sequence[str], args: Sequence[object]) -> list[int]:
        self.count_calls.append(tuple(keys))
        return [0]


def _handler_with_fake_redis(fake: _FakeRedisGauges) -> tuple[_PROXY_MaxParallelRequestsHandler_v3, DualCache]:
    handler, cache = _handler()
    handler.parallel_acquire_script = fake.acquire
    handler.parallel_release_script = fake.release
    handler.parallel_count_script = fake.count
    return handler, cache


@pytest.mark.asyncio
async def test_key_and_team_gauges_use_one_redis_call_per_key():
    fake = _FakeRedisGauges()
    handler, cache = _handler_with_fake_redis(fake)

    stash = await _admit(handler, cache, _team_key("sk-a", team_max_parallel_requests=2, max_parallel_requests=3))

    slot_id = stash.parallel_slot["slot_id"]
    assert fake.acquire_calls == [
        ((KEY_A_COUNTER_KEY,), (3, PARALLEL_REQUEST_SLOT_TTL_SECONDS, slot_id)),
        ((TEAM_COUNTER_KEY,), (2, PARALLEL_REQUEST_SLOT_TTL_SECONDS, slot_id)),
    ]

    await handler.async_release_max_parallel_requests_on_disconnect(user_api_key_dict=UserAPIKeyAuth())
    assert fake.release_calls == [((KEY_A_COUNTER_KEY,), (slot_id,)), ((TEAM_COUNTER_KEY,), (slot_id,))]


@pytest.mark.asyncio
async def test_team_rejection_in_redis_releases_key_slot_taken_first():
    fake = _FakeRedisGauges(reject_key=TEAM_COUNTER_KEY)
    handler, cache = _handler_with_fake_redis(fake)

    with pytest.raises(HTTPException) as exc_info:
        await _admit(handler, cache, _team_key("sk-a", team_max_parallel_requests=1, max_parallel_requests=3))

    assert f"team: {TEAM_ID}" in exc_info.value.detail
    slot_id = fake.acquire_calls[0][1][2]
    assert fake.release_calls == [((KEY_A_COUNTER_KEY,), (slot_id,))]
    assert get_request_stash().parallel_slot is None


@pytest.mark.asyncio
async def test_redis_failure_on_second_gauge_releases_first_before_falling_back():
    fake = _FakeRedisGauges(raise_on_key=TEAM_COUNTER_KEY)
    handler, cache = _handler_with_fake_redis(fake)

    stash = await _admit(handler, cache, _team_key("sk-a", team_max_parallel_requests=2, max_parallel_requests=3))

    slot_id = stash.parallel_slot["slot_id"]
    assert fake.release_calls == [((KEY_A_COUNTER_KEY,), (slot_id,))]
    assert await _team_in_flight(handler, cache) == 1
    assert handler._gauge_in_flight_from_cache_value(await cache.async_get_cache(key=KEY_A_COUNTER_KEY)) == 1


@pytest.mark.asyncio
async def test_read_only_count_uses_one_redis_call_per_key():
    fake = _FakeRedisGauges()
    handler, cache = _handler_with_fake_redis(fake)
    descriptors = handler._create_rate_limit_descriptors(
        user_api_key_dict=_team_key("sk-a", team_max_parallel_requests=2, max_parallel_requests=3),
        data={"model": "gpt-4o-mini"},
        rpm_limit_type=None,
        tpm_limit_type=None,
        model_has_failures=False,
    )
    _, _, gauges = handler._collect_windowed_keys_and_gauges(descriptors, skip_tpm_check=False)

    response = await handler._check_parallel_request_gauges(gauges, slot_id="unused", read_only=True)

    assert response["overall_code"] == "OK"
    assert fake.count_calls == [(KEY_A_COUNTER_KEY,), (TEAM_COUNTER_KEY,)]
