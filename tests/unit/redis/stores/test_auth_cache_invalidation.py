import json
from collections.abc import Sequence
from typing import Final

import pytest
from pydantic import TypeAdapter, ValidationError

from litellm._internal_context import current_service_target
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import AUTH_OBJECTS_TARGET
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthCacheStore
from litellm.redis.stores.auth_cache_invalidation import AuthCacheInvalidation, PythonCaches, connect_invalidation
from litellm.redis.stores.connection import connect
from litellm.redis.stores.spend_counters import SpendCounterStore
from tests.unit.redis.stores.auth_cache_double import shared_cache
from tests.unit.redis.stores.auth_cache_invalidation_double import FakeChannel, Message
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

_PAYLOAD: Final = TypeAdapter(dict[str, object])


def _mains_json(message: Message) -> str:
    return json.dumps(
        {field: value for field, value in zip(("cache_key", "new_value", "ttl"), message) if value is not None}
    )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _as_mains_subscriber_applies(payload: bytes) -> Message | None:
    try:
        decoded: Final = _PAYLOAD.validate_json(payload.decode("utf-8", errors="replace"))
    except ValidationError:
        return None
    cache_key: Final = decoded.get("cache_key")
    if not isinstance(cache_key, str):
        return None
    new_value: Final = _number(decoded.get("new_value"))
    return (cache_key, new_value, None if new_value is None else _number(decoded.get("ttl")))


@pytest.mark.parametrize(
    ("message", "payload"),
    [
        (("team_id:t1", None, None), r'{"cache_key": "team_id:t1"}'),
        (("t\u00e9", None, None), r'{"cache_key": "t\u00e9"}'),
        (("spend:key:abc", 1.5, 60), r'{"cache_key": "spend:key:abc", "new_value": 1.5, "ttl": 60}'),
        (
            ("spend_db_floor:spend:team:t", 0.1 + 0.2, 5),
            r'{"cache_key": "spend_db_floor:spend:team:t", "new_value": 0.30000000000000004, "ttl": 5}',
        ),
        (("spend:user:u", 0.0, None), r'{"cache_key": "spend:user:u", "new_value": 0.0}'),
    ],
)
def test_a_message_is_mains_json(message: Message, payload: str) -> None:
    assert _mains_json(message) == payload


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (rb'{"cache_key": "team_id:t1"}', ("team_id:t1", None, None)),
        (rb'{"cache_key": "k", "ttl": 60}', ("k", None, None)),
        (rb'{"cache_key": "k", "new_value": 2, "ttl": 60}', ("k", 2.0, 60.0)),
        (rb'{"cache_key": "k", "new_value": true}', ("k", None, None)),
        (rb'{"cache_key": "k", "new_value": "1.5"}', ("k", None, None)),
        (rb'{"cache_key": "k", "new_value": 1.5, "ttl": "60"}', ("k", 1.5, None)),
        (b'{"cache_key": "k\xff"}', ("k\ufffd", None, None)),
        (rb'{"cache_key": 7}', None),
        (rb'["team_id:t1"]', None),
        (b"not json", None),
    ],
)
def test_a_message_reads_as_mains_subscriber_reads_it(payload: bytes, expected: Message | None) -> None:
    applied: Final = _as_mains_subscriber_applies(payload)

    assert applied == expected
    assert applied is None or all(value is None or type(value) is float for value in applied[1:])


def test_a_value_is_written_into_the_copies_and_leaves_the_evicted_caches_alone() -> None:
    copy: Final = InMemoryCache()
    copy.set_cache("spend:key:k-1", 9.0)
    evicted: Final = InMemoryCache()
    evicted.set_cache("spend:key:k-1", "hook state")

    PythonCaches(copies=(copy,), evicted=(evicted,)).apply("spend:key:k-1", 0.0, 60.0)

    assert copy.get_cache("spend:key:k-1") == 0.0
    assert evicted.get_cache("spend:key:k-1") == "hook state"


def test_an_eviction_drops_the_key_from_the_copies_and_the_evicted_caches() -> None:
    copy: Final = InMemoryCache()
    evicted: Final = InMemoryCache()
    for cache in (copy, evicted):
        cache.set_cache("team_id:t-1", "stale")
        cache.set_cache("team_id:t-2", "kept")

    PythonCaches(copies=(copy,), evicted=(evicted,)).apply("team_id:t-1", None, 60.0)

    assert [cache.get_cache("team_id:t-1") for cache in (copy, evicted)] == [None, None]
    assert [cache.get_cache("team_id:t-2") for cache in (copy, evicted)] == ["kept", "kept"]


def _native_spend_counters() -> SpendCounterStore:
    return SpendCounterStore(connect(None).spend_counters())


def test_connect_invalidation_is_none_without_a_redis_client() -> None:
    assert (
        connect_invalidation(connect(None), None, UserApiKeyCache().store, _native_spend_counters(), PythonCaches())
        is None
    )


@pytest.mark.parametrize(
    ("auth_cache", "spend_counters"),
    [
        (shared_cache()[0].store, _native_spend_counters()),
        (UserApiKeyCache().store, FakeSpendCounters().store()),
    ],
    ids=["auth-cache-not-native", "spend-counters-not-native"],
)
def test_connect_invalidation_refuses_a_store_that_cannot_take_a_native_eviction(
    auth_cache: AuthCacheStore, spend_counters: SpendCounterStore
) -> None:
    redis: Final = connect(RedisCache(host="127.0.0.1", port=1))

    with pytest.raises(TypeError, match="not the native store"):
        _ = connect_invalidation(redis, None, auth_cache, spend_counters, PythonCaches())


def test_connect_invalidation_publishes_in_the_background_over_native_stores() -> None:
    invalidation: Final = connect_invalidation(
        connect(RedisCache(host="127.0.0.1", port=1)),
        None,
        UserApiKeyCache().store,
        _native_spend_counters(),
        PythonCaches(),
    )

    assert invalidation is not None
    assert invalidation.publish("team_id:t-1") is True


class _TargetRecordingChannel(FakeChannel):
    targets: tuple[str | None, ...] = ()

    async def broadcast_evictions(self, cache_keys: Sequence[str]) -> None:
        self.targets = (*self.targets, current_service_target())
        await super().broadcast_evictions(cache_keys)


@pytest.mark.asyncio
async def test_broadcast_evictions_skips_an_empty_batch_and_sends_the_rest_in_one() -> None:
    handle: Final = _TargetRecordingChannel()
    invalidation: Final = AuthCacheInvalidation(handle)

    await invalidation.broadcast_evictions([])
    await invalidation.broadcast_evictions(["team_id:t-1", "user-1"])

    assert handle.bulk == [("team_id:t-1", "user-1")]
    assert handle.targets == (AUTH_OBJECTS_TARGET,)
