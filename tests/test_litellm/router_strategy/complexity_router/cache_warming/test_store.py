import json
import logging

import pytest

from litellm.router_strategy.complexity_router.cache_warming.store import (
    CacheWarmingStore,
    _warn_redis_missing,
    _warn_session_cap_reached,
)
from litellm.router_strategy.complexity_router.cache_warming.types import (
    CACHE_WARMING_RECORD_SCHEMA_VERSION,
    CacheWarmingAttribution,
    CacheWarmingRecord,
)


class FakeRedisCache:
    def __init__(self, namespace: str | None = None) -> None:
        self.namespace = namespace
        self.data: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}
        self.sets: dict[str, set[str]] = {}
        self.expire_calls: list[str] = []

    def _namespaced(self, key: str) -> str:
        if self.namespace and not key.startswith(self.namespace):
            return f"{self.namespace}:{key}"
        return key

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        raw = self.data.get(self._namespaced(key))
        return json.loads(raw) if raw is not None else None

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        namespaced = self._namespaced(key)
        self.data[namespaced] = json.dumps(value)
        ttl = kwargs.get("ttl")
        self.ttls[namespaced] = ttl if isinstance(ttl, int) else None

    def async_register_script(self, script: str):
        if 'redis.call("expire"' in script:

            async def compare_and_expire(keys: list, args: list) -> int:
                key = self._namespaced(keys[0])
                raw = self.data.get(key)
                if raw is not None and raw == str(args[0]):
                    self.ttls[key] = int(args[1])
                    self.expire_calls.append(key)
                    return 1
                return 0

            return compare_and_expire
        if 'redis.call("del"' in script:

            async def compare_and_delete(keys: list, args: list) -> int:
                key = self._namespaced(keys[0])
                raw = self.data.get(key)
                if raw is not None and raw == str(args[0]):
                    del self.data[key]
                    return 1
                return 0

            return compare_and_delete
        if "HGET" in script:

            async def get_session(keys: list, args: list) -> list:
                record = self.hashes.get(self._namespaced(keys[0]), {}).get(str(args[0]))
                touched = sorted(self.sets.get(self._namespaced(keys[1]), set()))
                return [record, [member.encode("utf-8") for member in touched]]

            return get_session
        if "HSET" in script:

            async def capture(keys: list, args: list) -> int:
                sessions = self.hashes.setdefault(self._namespaced(keys[0]), {})
                index = self.zsets.setdefault(self._namespaced(keys[1]), {})
                touched = self.sets.setdefault(self._namespaced(keys[2]), set())
                member, record_json = str(args[0]), str(args[1])
                now, expires_at, max_sessions = float(args[2]), float(args[3]), int(args[4])
                for stale in [m for m, score in index.items() if score <= now]:
                    del index[stale]
                    sessions.pop(stale, None)
                if member not in index and len(index) >= max_sessions:
                    return 0
                sessions[member] = record_json
                index[member] = expires_at
                touched.add(str(args[5]))
                return 1

            return capture

        async def list_live(keys: list, args: list) -> list:
            index = self.zsets.get(self._namespaced(keys[0]), {})
            now, limit = float(args[0]), int(args[1])
            live = sorted((score, member) for member, score in index.items() if score > now)
            return [member.encode("utf-8") for _, member in live[:limit]]

        return list_live


def _record_json(**overrides: object) -> str:
    base = CacheWarmingRecord(
        schema_version=CACHE_WARMING_RECORD_SCHEMA_VERSION,
        payload_compressed="blob",
        payload_sha256="sha",
        token_estimate=2048,
        last_activity=1000.0,
        served_model="sonnet",
        tags=(),
        attribution=CacheWarmingAttribution(user_api_key="hashed"),
        auto_router_model_name="smart-router",
    ).model_dump()
    return json.dumps({**base, **overrides})


def _store(redis: FakeRedisCache | None) -> CacheWarmingStore:
    return CacheWarmingStore(redis_cache=redis, auto_router_model_name="smart-router")


async def _upsert(store: CacheWarmingStore, session_id: str = "s1", max_sessions: int = 100) -> None:
    await store.upsert_session(
        caller_scope="scope",
        session_id=session_id,
        payload_compressed="blob2",
        payload_sha256="sha2",
        token_estimate=4096,
        served_model="sonnet",
        tags=(),
        attribution=CacheWarmingAttribution(),
        ttl_seconds=1800,
        max_sessions=max_sessions,
    )


def test_key_shapes_are_scoped_and_hash_tagged():
    """Every key a session owns carries the auto-router name and shares one Cluster hash tag with the others,
    so two warming auto-routers on one Redis cannot read each other's warmth and a session's record, index
    entry and warmth stamps stay on one node."""
    record = CacheWarmingStore.record_key("smart-router", "keyhash", "session-1")
    other_router = CacheWarmingStore.record_key("other-router", "keyhash", "session-1")
    assert CacheWarmingStore.warmth_key(record, "opus") != CacheWarmingStore.warmth_key(other_router, "opus")
    store = _store(None)
    slot = "{cache_warm:v1:smart-router}"
    assert store.sessions_key() == f"{slot}:sessions"
    assert store.index_key() == f"{slot}:index"
    assert CacheWarmingStore.warmth_key(record, "opus").startswith(f"{slot}:")

    # the session id is caller-controlled and reaches this key, the index member, the touched key and every
    # warmth key, none of which max_payload_bytes bounds, so it is hashed rather than embedded
    huge = CacheWarmingStore.record_key("smart-router", "keyhash", "s" * 10_000)
    assert len(huge) == len(record) and "s" * 100 not in huge
    assert record != CacheWarmingStore.record_key("smart-router", "keyhash", "session-2")


@pytest.mark.asyncio
async def test_cap_enforced_atomically_with_the_record_write():
    _warn_session_cap_reached.cache_clear()
    redis = FakeRedisCache()
    store = _store(redis)
    await _upsert(store, session_id="s1", max_sessions=2)
    await _upsert(store, session_id="s2", max_sessions=2)
    await _upsert(store, session_id="s3", max_sessions=2)
    assert await store.get_record(store.record_key("smart-router", "scope", "s3")) is None
    assert len(await store.list_session_keys(max_sessions=10)) == 2


@pytest.mark.asyncio
async def test_get_record_returns_none_on_schema_version_mismatch():
    redis = FakeRedisCache()
    store = _store(redis)
    key = store.record_key("smart-router", "scope", "s1")
    redis.hashes[store.sessions_key()] = {key: _record_json(schema_version=CACHE_WARMING_RECORD_SCHEMA_VERSION + 1)}
    assert await store.get_record(key) is None
