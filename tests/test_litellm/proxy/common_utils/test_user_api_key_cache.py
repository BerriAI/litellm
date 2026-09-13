import hashlib
import json
from typing import Any

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    end_user_cache_key,
    get_management_object_ttl,
    is_user_key_cache_key,
)
from litellm.proxy.proxy_server import UserAPIKeyCacheTTLEnum

HASHED_TOKEN = hashlib.sha256(b"sk-lit7563-hot-key").hexdigest()


class CapturingInMemoryCache(InMemoryCache):
    """Records ``ttl`` passed into ``set_cache`` (what DualCache injects)."""

    def __init__(self) -> None:
        super().__init__()
        self.last_ttl: Any = None

    def set_cache(self, key, value, **kwargs):  # type: ignore[override]
        self.last_ttl = kwargs.get("ttl")
        super().set_cache(key, value, **kwargs)


class FakeRedisCache(RedisCache):
    """
    In-memory fake that enforces the UserApiKeyCache Redis payload contract.

    For user_api_key_cache entries we expect Redis to store a JSON object (dict)
    produced by `CacheCodec.serialize(..., model_type=...)`.

    This fake:
    - raises TypeError if the value is not a dict
    - raises TypeError if the dict is not JSON-serializable

    Records the ``ttl`` kwarg DualCache forwards on each Redis write for tests.
    """

    def __init__(self):  # noqa: super().__init__ skipped intentionally
        self._store: dict[str, str] = {}
        self.last_ttl: Any = None

    def set_cache(self, key: str, value: Any, **kwargs):  # type: ignore[override]
        if not isinstance(value, dict):
            raise TypeError("FakeRedisCache only accepts dict payloads")
        self.last_ttl = kwargs.get("ttl")
        self._store[key] = json.dumps(value)
        return True

    def get_cache(self, key: str, **kwargs):  # type: ignore[override]
        raw = self._store.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    async def async_set_cache(self, key: str, value: Any, **kwargs):  # type: ignore[override]
        if not isinstance(value, dict):
            raise TypeError("FakeRedisCache only accepts dict payloads")
        self.last_ttl = kwargs.get("ttl")
        self._store[key] = json.dumps(value)
        return True

    async def async_get_cache(self, key: str, **kwargs):  # type: ignore[override]
        raw = self._store.get(key)
        if raw is None:
            return None
        return json.loads(raw)

    def delete_cache(self, key: str):  # type: ignore[override]
        self._store.pop(key, None)

    async def async_delete_cache(self, key: str):  # type: ignore[override]
        self._store.pop(key, None)


def _make_key_obj(token: str = "tok") -> UserAPIKeyAuth:
    # Minimal object (UserAPIKeyAuth inherits token from base view).
    return UserAPIKeyAuth(token=token)


class TestUserApiKeyCache:
    @pytest.mark.asyncio
    async def test_async_set_in_memory_gets_enum_default_when_user_api_key_cache_ttl_omitted(
        self,
    ):
        """
        If ``general_settings.user_api_key_cache_ttl`` is absent, the proxy never
        calls ``update_cache_ttl``; ``user_api_key_cache`` keeps
        ``default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl``.
        DualCache must forward that as the in-memory ``ttl`` kwarg on each set.
        """
        mem = CapturingInMemoryCache()
        cache = UserApiKeyCache(
            in_memory_cache=mem,
            redis_cache=FakeRedisCache(),
            default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value,
        )
        await cache.async_set_cache(
            "k",
            _make_key_obj("t"),
            model_type=UserAPIKeyAuth,
        )
        expected = UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value
        assert mem.last_ttl == expected

    def test_sync_set_in_memory_gets_enum_default_when_user_api_key_cache_ttl_omitted(
        self,
    ):
        mem = CapturingInMemoryCache()
        cache = UserApiKeyCache(
            in_memory_cache=mem,
            redis_cache=FakeRedisCache(),
            default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value,
        )
        cache.set_cache("sk", _make_key_obj("s"), model_type=UserAPIKeyAuth)
        assert mem.last_ttl == UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value

    @pytest.mark.asyncio
    async def test_async_set_forwards_default_in_memory_ttl_to_redis_layer(self):
        """
        DualCache injects missing ``ttl`` from ``default_in_memory_ttl`` into kwargs
        before calling ``redis_cache.async_set_cache`` — Redis should receive the same
        TTL as memory (matches proxy defaults: enum 60s).
        """
        fake = FakeRedisCache()
        cache = UserApiKeyCache(
            redis_cache=fake,
            default_in_memory_ttl=60,
        )

        await cache.async_set_cache(
            key="ttl-key",
            value=_make_key_obj("ttl-tok"),
            model_type=UserAPIKeyAuth,
        )

        assert fake.last_ttl == 60

    @pytest.mark.asyncio
    async def test_async_set_explicit_ttl_override_reaches_redis(self):
        fake = FakeRedisCache()
        cache = UserApiKeyCache(
            redis_cache=fake,
            default_in_memory_ttl=60,
        )

        await cache.async_set_cache(
            key="k",
            value=_make_key_obj("x"),
            model_type=UserAPIKeyAuth,
            ttl=900,
        )

        assert fake.last_ttl == 900

    def test_sync_set_forwards_default_in_memory_ttl_to_redis_layer(self):
        fake = FakeRedisCache()
        cache = UserApiKeyCache(
            redis_cache=fake,
            default_in_memory_ttl=45,
        )
        cache.set_cache(
            "sk",
            _make_key_obj("sync"),
            model_type=UserAPIKeyAuth,
        )
        assert fake.last_ttl == 45

    @pytest.mark.asyncio
    async def test_async_set_typed_stores_serialized_payload_in_memory_and_redis(self):
        cache = UserApiKeyCache(redis_cache=FakeRedisCache())
        obj = _make_key_obj("abc")

        await cache.async_set_cache("k", obj, model_type=UserAPIKeyAuth)

        # In-memory hit should still be raw dict (not BaseModel) because wrapper
        # stores the serialized payload into both layers.
        raw = await cache.in_memory_cache.async_get_cache("k")  # type: ignore[union-attr]
        assert isinstance(raw, dict)
        assert raw["token"] == "abc"

        # Redis should also hold the same serialized dict
        redis_raw = await cache.redis_cache.async_get_cache("k")  # type: ignore[union-attr]
        assert redis_raw == raw

    @pytest.mark.asyncio
    async def test_async_get_typed_returns_model_on_valid_hit(self):
        cache = UserApiKeyCache(redis_cache=FakeRedisCache())
        await cache.async_set_cache("k", {"token": "abc"}, model_type=UserAPIKeyAuth)

        value = await cache.async_get_cache("k", model_type=UserAPIKeyAuth)
        assert value is not None
        assert isinstance(value, UserAPIKeyAuth)
        assert value.token == "abc"

    @pytest.mark.asyncio
    async def test_async_get_typed_returns_none_on_validation_failure_after_hit(self):
        cache = UserApiKeyCache(redis_cache=FakeRedisCache())

        # Bypass UserApiKeyCache.serialize: CacheCodec rejects non-dict cached values
        # for dict-based models (deserialize returns None).
        await cache.in_memory_cache.async_set_cache(key="k", value="invalid-payload-not-a-dict")

        value = await cache.async_get_cache("k", model_type=UserAPIKeyAuth)
        assert value is None

    def test_fake_redis_cache_rejects_non_json_serializable_values(self):
        fake = FakeRedisCache()

        class NotSerializable:
            pass

        with pytest.raises(TypeError):
            fake.set_cache("k", NotSerializable())

        with pytest.raises(TypeError):
            fake.set_cache("k2", {"ok": NotSerializable()})


class TestUserKeyObjectPartition:
    """
    Regression for LIT-7563: user-key objects share one 200-entry ``InMemoryCache`` with
    every other management object, so end-user / team / tag churn evicts hot keys and
    forces a ``LiteLLM_VerificationToken`` lookup on the next request.
    """

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            (HASHED_TOKEN, True),
            (HASHED_TOKEN.upper(), False),
            (f"team_id:{HASHED_TOKEN}", False),
            (end_user_cache_key("u1"), False),
            ("sk-lit7563-hot-key", False),
        ],
    )
    def test_is_user_key_cache_key(self, key: str, expected: bool):
        assert is_user_key_cache_key(key) is expected

    @pytest.mark.asyncio
    async def test_management_object_churn_does_not_evict_key_object(self):
        cache = UserApiKeyCache(in_memory_cache=InMemoryCache(max_size_in_memory=2))
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth, ttl=100)
        for i in range(2):
            await cache.async_set_cache(end_user_cache_key(f"u{i}"), {"user_id": f"u{i}"}, ttl=200)

        key_obj = await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN
        assert cache.get_cache(end_user_cache_key("u1")) == {"user_id": "u1"}
        assert HASHED_TOKEN not in cache.in_memory_cache.cache_dict

    def test_sync_write_and_read_route_to_key_object_partition(self):
        cache = UserApiKeyCache(in_memory_cache=InMemoryCache(max_size_in_memory=2))
        cache.set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth, ttl=100)
        for i in range(2):
            cache.set_cache(end_user_cache_key(f"u{i}"), {"user_id": f"u{i}"}, ttl=200)

        key_obj = cache.get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN

    @pytest.mark.asyncio
    async def test_redis_hit_backfills_key_object_partition_with_configured_ttl(self):
        redis = FakeRedisCache()
        writer = UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=30)
        await writer.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        key_partition = CapturingInMemoryCache()
        reader = UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=30, key_object_in_memory_cache=key_partition)
        key_obj = await reader.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)

        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN
        assert key_partition.last_ttl == 30
        assert HASHED_TOKEN not in reader.in_memory_cache.cache_dict

    @pytest.mark.asyncio
    async def test_update_cache_ttl_applies_to_key_object_partition(self):
        key_partition = CapturingInMemoryCache()
        cache = UserApiKeyCache(default_in_memory_ttl=60, key_object_in_memory_cache=key_partition)
        cache.update_cache_ttl(default_in_memory_ttl=7, default_redis_ttl=7)

        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        assert key_partition.last_ttl == 7

    @pytest.mark.asyncio
    async def test_attach_redis_cache_applies_to_key_object_partition(self):
        redis = FakeRedisCache()
        cache = UserApiKeyCache()
        cache.attach_redis_cache(redis)

        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        other_worker = UserApiKeyCache(redis_cache=redis)
        key_obj = await other_worker.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN

    @pytest.mark.asyncio
    async def test_delete_removes_key_object_from_partition_and_redis(self):
        redis = FakeRedisCache()
        cache = UserApiKeyCache(redis_cache=redis)
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        assert await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is not None

        cache.delete_cache(HASHED_TOKEN)

        assert await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is None
        assert await redis.async_get_cache(HASHED_TOKEN) is None

    @pytest.mark.asyncio
    async def test_async_delete_removes_key_object_from_partition_and_redis(self):
        redis = FakeRedisCache()
        cache = UserApiKeyCache(redis_cache=redis)
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        await cache.async_delete_cache(HASHED_TOKEN)

        assert await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is None
        assert await redis.async_get_cache(HASHED_TOKEN) is None

    @pytest.mark.asyncio
    async def test_pipeline_write_routes_each_entry_to_its_partition(self):
        cache = UserApiKeyCache(in_memory_cache=InMemoryCache(max_size_in_memory=2))
        await cache.async_set_cache_pipeline(
            [(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN))]
            + [(end_user_cache_key(f"u{i}"), {"user_id": f"u{i}"}) for i in range(2)],
            ttl=100,
        )

        key_obj = await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN
        assert HASHED_TOKEN not in cache.in_memory_cache.cache_dict
        assert cache.get_cache(end_user_cache_key("u1")) == {"user_id": "u1"}

    def test_flush_clears_key_object_partition(self):
        cache = UserApiKeyCache()
        cache.set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        cache.set_cache(end_user_cache_key("u1"), {"user_id": "u1"})

        cache.flush_cache()

        assert cache.get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is None
        assert cache.get_cache(end_user_cache_key("u1")) is None

    def test_in_memory_cache_for_routes_by_key(self):
        cache = UserApiKeyCache()
        assert cache.in_memory_cache_for(HASHED_TOKEN) is cache.key_object_cache.in_memory_cache
        assert cache.in_memory_cache_for(end_user_cache_key("u1")) is cache.in_memory_cache


class TestManagementObjectTTL:
    """
    Regression for LIT-3338: ``general_settings.user_api_key_cache_ttl`` (which the
    proxy propagates onto ``default_in_memory_ttl``) must win over the hardcoded
    ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` for management-object writes.
    """

    def test_returns_configured_default_in_memory_ttl(self):
        cache = UserApiKeyCache(default_in_memory_ttl=300)
        assert get_management_object_ttl(cache) == 300

    def test_falls_back_to_constant_when_no_default_configured(self):
        cache = UserApiKeyCache()
        assert cache.default_in_memory_ttl is None
        assert get_management_object_ttl(cache) == DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL

    def test_resolves_on_a_plain_dual_cache(self):
        # Many call sites are typed UserApiKeyCache but exercised in tests with a
        # bare DualCache; the resolver must work on the base type, not just the subclass.
        assert get_management_object_ttl(DualCache(default_in_memory_ttl=300)) == 300
        assert get_management_object_ttl(DualCache()) == DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL

    @pytest.mark.asyncio
    async def test_management_write_uses_configured_ttl_over_constant(self):
        mem = CapturingInMemoryCache()
        cache = UserApiKeyCache(
            in_memory_cache=mem,
            redis_cache=FakeRedisCache(),
            default_in_memory_ttl=300,
        )
        assert get_management_object_ttl(cache) != (DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)

        await cache.async_set_cache(
            "team_id:abc",
            _make_key_obj("t"),
            model_type=UserAPIKeyAuth,
            ttl=get_management_object_ttl(cache),
        )

        assert mem.last_ttl == 300
