"""
Unit Tests for the max parallel request limiter v1 for the proxy
"""

import asyncio
import itertools
from collections.abc import Callable, Iterator, Mapping
from datetime import datetime
from functools import partial
from typing import Final

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.parallel_request_limiter import (
    PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token
from litellm.redis.stores.legacy_request_limits import MinuteLimit
from litellm.types.utils import CallTypes, EmbeddingResponse, ModelResponse, TextCompletionResponse, Usage
from tests._support.fake_redis import fake_redis_cache

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
            request_count_api_key=MinuteLimit("test-key", LAST_MINUTE_OF_JANUARY, "model_per_key"),
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
        assert current["current_tpm"] == 50, f"expected 50 tokens counted for {scope_id}, got {current['current_tpm']}"


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
    handler = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=_frozen_clock
    )
    bucket = f"{api_key}::{FROZEN_INSTANT.strftime('%Y-%m-%d-%H-%M')}::request_count"
    await handler.internal_usage_cache.async_set_cache(
        key=bucket,
        value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )
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
    after_two_turns = await handler.internal_usage_cache.async_get_cache(key=bucket, litellm_parent_otel_span=None)
    assert after_two_turns["current_requests"] == 1, after_two_turns

    await handler.async_release_max_parallel_requests_on_disconnect(UserAPIKeyAuth(api_key=api_key))
    after_disconnect = await handler.internal_usage_cache.async_get_cache(key=bucket, litellm_parent_otel_span=None)
    assert after_disconnect["current_requests"] == 0, after_disconnect


@pytest.mark.asyncio
async def test_a_rejected_realtime_attempt_does_not_free_a_slot_it_never_took():
    api_key = hash_token("sk-at-limit")
    handler = PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(DualCache()), clock=_frozen_clock
    )
    bucket = f"{api_key}::{FROZEN_INSTANT.strftime('%Y-%m-%d-%H-%M')}::request_count"
    await handler.internal_usage_cache.async_set_cache(
        key=bucket,
        value={"current_requests": 1, "current_tpm": 0, "current_rpm": 1},
        litellm_parent_otel_span=None,
    )
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    proxy_logging_obj.proxy_hook_mapping["parallel_request_limiter"] = handler
    auth = UserAPIKeyAuth(api_key=api_key, max_parallel_requests=1)

    await proxy_logging_obj.arelease_max_parallel_requests_on_disconnect(auth)
    after_rejection = await handler.internal_usage_cache.async_get_cache(key=bucket, litellm_parent_otel_span=None)
    assert after_rejection["current_requests"] == 1, after_rejection

    await proxy_logging_obj.arelease_responses_websocket_slot(auth)
    after_socket_close = await handler.internal_usage_cache.async_get_cache(key=bucket, litellm_parent_otel_span=None)
    assert after_socket_close["current_requests"] == 0, after_socket_close


@pytest.mark.asyncio
async def test_remote_snapshot_blocks_a_cold_worker_and_supplies_response_headers() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="legacy")
    auth: Final = UserAPIKeyAuth(api_key="legacy-key", rpm_limit=3, tpm_limit=100)
    key: Final = f"legacy-key::{LAST_MINUTE_OF_JANUARY}::request_count"
    await redis.async_set_cache(key, {"current_requests": 0, "current_tpm": 7, "current_rpm": 3}, ttl=60)
    local: Final = DualCache()
    handler: Final = PROXY_MaxParallelRequestsHandler(InternalUsageCache(local), clock=_frozen_clock)
    local.attach_redis_cache(redis)
    with pytest.raises(ProxyRateLimitError):
        await handler.async_pre_call_hook(auth, DualCache(), {"model": "group"}, "completion")
    response: Final = ModelResponse()
    response._hidden_params = {}
    await handler.async_post_call_success_hook({}, auth, response)
    assert response._hidden_params["additional_headers"] == {
        "x-ratelimit-remaining-requests": 0,
        "x-ratelimit-limit-requests": 3,
        "x-ratelimit-remaining-tokens": 93,
        "x-ratelimit-limit-tokens": 100,
    }


@pytest.mark.asyncio
async def test_duplicate_identity_pre_call_remains_non_atomic_and_global_gauge_stays_local() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="legacy")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    local: Final = DualCache(redis_cache=redis)
    handler: Final = PROXY_MaxParallelRequestsHandler(InternalUsageCache(local), clock=_frozen_clock)
    auth: Final = UserAPIKeyAuth(
        api_key="shared", team_id="shared", team_rpm_limit=10, max_parallel_requests=10, rpm_limit=10
    )
    await handler.async_pre_call_hook(
        auth, DualCache(), {"model": "group", "metadata": {"global_max_parallel_requests": 5}}, "completion"
    )
    await asyncio.sleep(0)
    assert await redis.async_get_cache(f"shared::{LAST_MINUTE_OF_JANUARY}::request_count") == {
        "current_requests": 1,
        "current_tpm": 0,
        "current_rpm": 1,
    }
    assert await raw.get("legacy:global_max_parallel_requests") is None
    assert local.in_memory_cache.get_cache("global_max_parallel_requests") == 1


@pytest.mark.asyncio
async def test_success_settles_all_minute_identities_including_model_and_failure_settles_key() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="legacy")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    identities: Final = ("key", "key::group", "user", "team", "end-user")
    for identity in identities:
        await redis.async_set_cache(
            f"{identity}::{LAST_MINUTE_OF_JANUARY}::request_count",
            {"current_requests": 2, "current_tpm": 10, "current_rpm": 4},
            ttl=60,
        )
    handler: Final = PROXY_MaxParallelRequestsHandler(
        InternalUsageCache(DualCache(redis_cache=redis)), clock=_frozen_clock
    )
    kwargs: Final = {
        "call_type": "acompletion",
        "exception": ValueError("provider failed"),
        "user": "end-user",
        "litellm_params": {
            "metadata": {
                "user_api_key": "key",
                "user_api_key_user_id": "user",
                "user_api_key_team_id": "team",
                "user_api_key_metadata": {"model_rpm_limit": {"group": 10}},
                "user_api_key_model_max_budget": {},
                "model_group": "group",
            }
        },
    }
    response: Final = ModelResponse(usage=Usage(total_tokens=5))
    await handler.async_log_success_event(kwargs, response, FROZEN_INSTANT, FROZEN_INSTANT)
    for identity in identities:
        key: Final = f"{identity}::{LAST_MINUTE_OF_JANUARY}::request_count"
        assert await redis.async_get_cache(key) == {"current_requests": 1, "current_tpm": 15, "current_rpm": 4}, (
            identity
        )
        assert 0 < await raw.pttl(f"legacy:{key}") <= 60000
    await handler.async_log_failure_event(kwargs, None, FROZEN_INSTANT, FROZEN_INSTANT)
    assert await redis.async_get_cache(f"key::{LAST_MINUTE_OF_JANUARY}::request_count") == {
        "current_requests": 0,
        "current_tpm": 15,
        "current_rpm": 4,
    }


@pytest.mark.asyncio
async def test_redis_outage_keeps_pre_call_local_admission_and_minute_rollover() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="legacy")
    local: Final = DualCache(redis_cache=redis)
    instants: Final = itertools.chain(
        itertools.repeat(LAST_MICROSECOND_OF_JANUARY, 3), itertools.repeat(FIRST_MICROSECOND_OF_FEBRUARY)
    )
    handler: Final = PROXY_MaxParallelRequestsHandler(InternalUsageCache(local), clock=_clock_reading(instants))
    auth: Final = UserAPIKeyAuth(api_key="key", rpm_limit=1)
    server.connected = False
    await handler.async_pre_call_hook(auth, DualCache(), {"model": "group"}, "completion")
    await asyncio.sleep(0)
    with pytest.raises(ProxyRateLimitError):
        await handler.async_pre_call_hook(auth, DualCache(), {"model": "group"}, "completion")
    await handler.async_pre_call_hook(auth, DualCache(), {"model": "group"}, "completion")
    await asyncio.sleep(0)
    assert local.in_memory_cache.get_cache(f"key::{FIRST_MINUTE_OF_FEBRUARY}::request_count") == {
        "current_requests": 1,
        "current_tpm": 0,
        "current_rpm": 1,
    }


def _lost_set_reply(reply: object, *, applied: asyncio.Event, **kwargs: object) -> bool:
    applied.set()
    raise RedisTimeoutError("SET applied but reply lost")


@pytest.mark.asyncio
async def test_unacknowledged_background_set_keeps_local_limit_and_remote_write() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="legacy")
    client: Final = redis.init_async_client()
    assert isinstance(client, fakeredis.FakeAsyncRedis)
    applied: Final = asyncio.Event()
    client.set_response_callback("SET", partial(_lost_set_reply, applied=applied))
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    handler: Final = PROXY_MaxParallelRequestsHandler(
        InternalUsageCache(DualCache(redis_cache=redis)), clock=_frozen_clock
    )
    auth: Final = UserAPIKeyAuth(api_key="key", rpm_limit=1)
    await handler.async_pre_call_hook(auth, DualCache(), {"model": "group"}, "completion")
    await asyncio.wait_for(applied.wait(), timeout=5)
    assert await raw.get(f"legacy:key::{LAST_MINUTE_OF_JANUARY}::request_count") == (
        b'{"current_requests": 1, "current_tpm": 0, "current_rpm": 1}'
    )
    with pytest.raises(ProxyRateLimitError):
        await handler.async_pre_call_hook(auth, DualCache(), {"model": "group"}, "completion")


@pytest.mark.parametrize("payload", [b"broken{", b"[]", b"0", b"null"])
@pytest.mark.asyncio
async def test_failure_settlement_preserves_missing_and_falsy_remote_default(payload: bytes) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="legacy")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    key: Final = f"key::{LAST_MINUTE_OF_JANUARY}::request_count"
    await raw.set(f"legacy:{key}", payload)
    handler: Final = PROXY_MaxParallelRequestsHandler(
        InternalUsageCache(DualCache(redis_cache=redis)), clock=_frozen_clock
    )
    await handler.async_log_failure_event(
        {"exception": ValueError("provider failed"), "litellm_params": {"metadata": {"user_api_key": "key"}}},
        None,
        FROZEN_INSTANT,
        FROZEN_INSTANT,
    )
    assert await redis.async_get_cache(key) == {"current_requests": 0, "current_tpm": 0, "current_rpm": 0}


@pytest.mark.asyncio
async def test_bad_batch_payload_keeps_pre_call_failure_after_local_backfill() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="legacy")
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    key: Final = f"key::{LAST_MINUTE_OF_JANUARY}::request_count"
    await redis.async_set_cache(key, {"current_requests": 0, "current_tpm": 0, "current_rpm": 0}, ttl=60)
    await raw.set(f"legacy:key::group::{LAST_MINUTE_OF_JANUARY}::request_count", "broken{")
    handler: Final = PROXY_MaxParallelRequestsHandler(
        InternalUsageCache(DualCache(redis_cache=redis)), clock=_frozen_clock
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="invalid legacy request limit snapshot"):
            await handler.async_pre_call_hook(
                UserAPIKeyAuth(api_key="key"), DualCache(), {"model": "group"}, "completion"
            )
    assert await redis.async_get_cache(key) == {"current_requests": 0, "current_tpm": 0, "current_rpm": 0}
