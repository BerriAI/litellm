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
        if "HGET" in script:

            async def get_record(keys: list, args: list) -> str | None:
                return self.hashes.get(self._namespaced(keys[0]), {}).get(str(args[0]))

            return get_record
        if "HSET" in script:

            async def capture(keys: list, args: list) -> int:
                sessions = self.hashes.setdefault(self._namespaced(keys[0]), {})
                index = self.zsets.setdefault(self._namespaced(keys[1]), {})
                member, record_json = str(args[0]), str(args[1])
                now, expires_at, max_sessions = float(args[2]), float(args[3]), int(args[4])
                for stale in [m for m, score in index.items() if score <= now]:
                    del index[stale]
                    sessions.pop(stale, None)
                if member not in index and len(index) >= max_sessions:
                    return 0
                sessions[member] = record_json
                index[member] = expires_at
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
        attribution=CacheWarmingAttribution(),
        ttl_seconds=1800,
        max_sessions=max_sessions,
    )


def test_key_shapes_are_scoped_and_hash_tagged():
    assert CacheWarmingStore.record_key("smart-router", "keyhash", "session-1") == "keyhash:session-1"
    assert (
        CacheWarmingStore.warmth_key("keyhash:session-1", "opus")
        == "complexity_router_cache_warmth:v1:keyhash:session-1:opus"
    )
    store = _store(None)
    assert store.sessions_key() == "{cache_warm:v1:smart-router}:sessions"
    assert store.index_key() == "{cache_warm:v1:smart-router}:index"
    assert store.sessions_key().split("}")[0] == store.index_key().split("}")[0]


@pytest.mark.asyncio
async def test_upsert_writes_record_and_stamps_served_model_warmth():
    redis = FakeRedisCache()
    store = _store(redis)
    await _upsert(store)
    key = store.record_key("smart-router", "scope", "s1")
    stored = await store.get_record(key)
    assert stored is not None
    assert stored.payload_compressed == "blob2"
    assert stored.last_activity > 0
    warmth = await store.get_warmth(key, ("sonnet", "opus"))
    assert set(warmth) == {"sonnet"}
    assert redis.ttls[store.warmth_key(key, "sonnet")] == 1800


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
async def test_existing_session_updates_even_at_cap():
    redis = FakeRedisCache()
    store = _store(redis)
    await _upsert(store, session_id="s1", max_sessions=1)
    await _upsert(store, session_id="s1", max_sessions=1)
    stored = await store.get_record(store.record_key("smart-router", "scope", "s1"))
    assert stored is not None and stored.payload_compressed == "blob2"


@pytest.mark.asyncio
async def test_expired_session_frees_slot_and_record_together():
    redis = FakeRedisCache()
    store = _store(redis)
    await _upsert(store, session_id="s1", max_sessions=1)
    key1 = store.record_key("smart-router", "scope", "s1")
    redis.zsets[store.index_key()][key1] = 1.0
    await _upsert(store, session_id="s2", max_sessions=1)
    assert await store.get_record(store.record_key("smart-router", "scope", "s2")) is not None
    assert await store.get_record(key1) is None


@pytest.mark.asyncio
async def test_capture_fault_fails_closed_not_uncapped():
    redis = FakeRedisCache()
    store = _store(redis)

    async def exploding_capture(keys: list, args: list) -> int:
        raise RuntimeError("cluster moved slot")

    store._capture = exploding_capture
    await _upsert(store)
    assert redis.hashes == {} and redis.data == {}


@pytest.mark.asyncio
async def test_mark_warm_attempt_never_touches_the_record():
    redis = FakeRedisCache()
    store = _store(redis)
    key = store.record_key("smart-router", "scope", "s1")
    redis.hashes[store.sessions_key()] = {key: _record_json()}
    record_before = redis.hashes[store.sessions_key()][key]
    await store.mark_warm_attempt(key, "opus", attempted_at=999.0, ttl_seconds=3600)
    assert redis.hashes[store.sessions_key()][key] == record_before
    assert await store.get_warmth(key, ("opus", "sonnet")) == {"opus": 999.0}


@pytest.mark.asyncio
async def test_get_record_returns_none_on_schema_version_mismatch():
    redis = FakeRedisCache()
    store = _store(redis)
    key = store.record_key("smart-router", "scope", "s1")
    redis.hashes[store.sessions_key()] = {key: _record_json(schema_version=CACHE_WARMING_RECORD_SCHEMA_VERSION + 1)}
    assert await store.get_record(key) is None


@pytest.mark.asyncio
async def test_get_record_returns_none_on_validation_error():
    redis = FakeRedisCache()
    store = _store(redis)
    key = store.record_key("smart-router", "scope", "s1")
    redis.hashes[store.sessions_key()] = {key: json.dumps({"schema_version": "not-a-record"})}
    assert await store.get_record(key) is None


@pytest.mark.asyncio
async def test_get_warmth_skips_corrupt_stamps():
    redis = FakeRedisCache()
    store = _store(redis)
    key = store.record_key("smart-router", "scope", "s1")
    redis.data[store.warmth_key(key, "opus")] = json.dumps("not-a-float")
    redis.data[store.warmth_key(key, "sonnet")] = json.dumps(123.5)
    assert await store.get_warmth(key, ("opus", "sonnet")) == {"sonnet": 123.5}


@pytest.mark.asyncio
async def test_list_session_keys_returns_live_members_decoded():
    redis = FakeRedisCache(namespace="litellm")
    store = _store(redis)
    await _upsert(store, session_id="s1")
    await _upsert(store, session_id="s2")
    assert await store.list_session_keys(max_sessions=10) == ("scope:s1", "scope:s2")


@pytest.mark.asyncio
async def test_store_noops_without_redis_and_warns_once(caplog):
    _warn_redis_missing.cache_clear()
    store = CacheWarmingStore(redis_cache=None, auto_router_model_name="warnless-router")
    with caplog.at_level(logging.WARNING, logger="LiteLLM Router"):
        assert await store.get_record("k") is None
        assert await store.list_session_keys(max_sessions=5) == ()
        await store.mark_warm_attempt("k", "m", attempted_at=1.0, ttl_seconds=60)
        assert await store.get_warmth("k", ("m",)) == {}
    warnings = [r for r in caplog.records if "cache warming is inactive" in r.getMessage()]
    assert len(warnings) == 1
