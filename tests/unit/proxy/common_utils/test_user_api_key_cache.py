import datetime
import hashlib
from collections.abc import Awaitable, Callable
from typing import Any, Final, TypeAlias
from unittest.mock import MagicMock

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    budget_windows_absent_cache_key,
    end_user_cache_key,
    get_management_object_ttl,
    is_user_key_cache_key,
)
from litellm.proxy.proxy_server import UserAPIKeyCacheTTLEnum
from litellm.redis.stores.auth_cache import AuthCacheStore
from litellm.rust_bridge._native import NativeAuthCache
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache, shared_cache

HASHED_TOKEN = hashlib.sha256(b"sk-lit7563-hot-key").hexdigest()


class CapturingInMemoryCache(InMemoryCache):
    """Records ``ttl`` passed into ``set_cache`` (what DualCache injects)."""

    def __init__(self) -> None:
        super().__init__()
        self.last_ttl: Any = None

    def set_cache(self, key, value, **kwargs):  # type: ignore[override]
        self.last_ttl = kwargs.get("ttl")
        super().set_cache(key, value, **kwargs)


class _ManualClock:
    def __init__(self) -> None:
        self.seconds = 1000.0

    def __call__(self) -> float:
        return self.seconds

    def advance(self, seconds: float) -> None:
        self.seconds += seconds


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
            default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value,
        )
        cache.set_cache("sk", _make_key_obj("s"), model_type=UserAPIKeyAuth)
        assert mem.last_ttl == UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value

    @pytest.mark.asyncio
    async def test_the_shared_store_gets_default_in_memory_ttl_and_writes_without_a_ttl(self):
        """The native store applies ``default_in_memory_ttl`` to a write that names no TTL, in memory and Redis,
        as DualCache does; the Rust side pins the resulting Redis TTL."""
        cache, fake = shared_cache(default_in_memory_ttl=60)

        await cache.async_set_cache(key="ttl-key", value=_make_key_obj("ttl-tok"), model_type=UserAPIKeyAuth)

        assert [settings.memory_ttl for settings in fake.settings] == [60]
        assert fake.commands == [("SET", "ttl-key", None)]

    @pytest.mark.asyncio
    async def test_async_set_explicit_ttl_override_reaches_redis(self):
        cache, fake = shared_cache(default_in_memory_ttl=60)

        await cache.async_set_cache(key="k", value=_make_key_obj("x"), model_type=UserAPIKeyAuth, ttl=900)

        assert fake.commands == [("SET", "k", 900)]

    def test_sync_set_writes_through_the_shared_store_as_str(self):
        cache, fake = shared_cache(default_in_memory_ttl=45)

        cache.set_cache("sk", _make_key_obj("sync"), model_type=UserAPIKeyAuth, ttl=45)

        assert fake.commands == [("SET_STR", "sk", 45)]
        assert fake.redis["sk"]["token"] == "sync"

    @pytest.mark.asyncio
    async def test_async_set_typed_stores_serialized_payload_in_memory_and_redis(self):
        cache, fake = shared_cache()
        obj = _make_key_obj("abc")

        await cache.async_set_cache("k", obj, model_type=UserAPIKeyAuth)

        raw = cache.get_local("k")
        assert isinstance(raw, dict)
        assert raw["token"] == "abc"
        assert fake.redis["k"] == raw
        assert cache.in_memory_cache.get_cache("k") is None, "redis-py's DualCache tier is never written"

    @pytest.mark.asyncio
    async def test_async_get_typed_returns_model_on_valid_hit(self):
        cache, fake = shared_cache(FakeAuthCache(redis={"k": {"token": "abc"}}))

        value = await cache.async_get_cache("k", model_type=UserAPIKeyAuth)
        assert value is not None
        assert isinstance(value, UserAPIKeyAuth)
        assert value.token == "abc"
        assert fake.commands == [("GET", "k")]

    @pytest.mark.asyncio
    async def test_async_get_typed_returns_none_on_validation_failure_after_hit(self):
        cache, _fake = shared_cache()
        cache.set_local("k", "invalid-payload-not-a-dict", None)

        value = await cache.async_get_cache("k", model_type=UserAPIKeyAuth)
        assert value is None

    @pytest.mark.asyncio
    async def test_a_failed_shared_read_is_a_miss_and_a_failed_write_keeps_the_request_going(self):
        cache, fake = shared_cache(FakeAuthCache(fail=ConnectionError("redis down")))

        assert await cache.async_get_cache("team_id:t1") is None
        await cache.async_set_cache("team_id:t1", {"team_id": "t1"}, ttl=5)
        assert cache.get_cache("team_id:t2") is None

        assert [command[0] for command in fake.commands] == ["GET", "SET", "GET"]
        assert cache.get_local("team_id:t1") == {"team_id": "t1"}

    @pytest.mark.asyncio
    async def test_a_value_the_bridge_refuses_is_logged_not_raised(self, caplog: pytest.LogCaptureFixture):
        cache = UserApiKeyCache(
            redis_cache=MagicMock(spec=RedisCache),
            auth_cache=lambda _redis, _settings: AuthCacheStore(_refusing()),
        )

        await cache.async_set_cache("team_id:t1", {"team_id": "t1"})
        cache.set_cache("team_id:t1", {"team_id": "t1"})

        assert any("not a plain data value" in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_model_holding_a_datetime_is_cached_as_its_json_dump(self):
        cache = UserApiKeyCache(
            redis_cache=MagicMock(spec=RedisCache),
            auth_cache=lambda _redis, settings: AuthCacheStore(
                NativeAuthCache(None, memory_entries=settings.memory_entries)
            ),
        )
        key_obj = UserAPIKeyAuth(token="abc", expires=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc))

        await cache.async_set_cache("k", key_obj, model_type=UserAPIKeyAuth, local_only=True)

        assert cache.get_local("k") == key_obj.model_dump(mode="json")
        cached = await cache.async_get_cache("k", model_type=UserAPIKeyAuth)
        assert cached is not None
        assert cached.model_dump(mode="json") == key_obj.model_dump(mode="json")

    @pytest.mark.asyncio
    async def test_keyspace_operations_redis_py_cannot_reach_are_refused_once_shared(self):
        cache, _fake = shared_cache()

        with pytest.raises(NotImplementedError):
            await cache.async_increment_cache("team_id:t1", 1)
        with pytest.raises(NotImplementedError):
            await cache.async_batch_get_cache(["team_id:t1"])
        assert cache.redis_cache is None

    @pytest.mark.asyncio
    async def test_global_spend_increments_only_an_entry_redis_still_holds(self):
        cache, fake = shared_cache(FakeAuthCache(redis={"litellm-proxy-budget:spend": 1.5}))
        memory_only = UserApiKeyCache()

        assert await cache.increment_if_present("litellm-proxy-budget:spend", 2.0) == 3.5
        assert await cache.increment_if_present("absent", 2.0) is None
        assert await memory_only.increment_if_present("litellm-proxy-budget:spend", 2.0) is None
        assert [command[:2] for command in fake.commands] == [
            ("INCRBYFLOAT_IF_PRESENT", "litellm-proxy-budget:spend"),
            ("INCRBYFLOAT_IF_PRESENT", "absent"),
        ]


def _refusing() -> FakeAuthCache:
    class Refusing(FakeAuthCache):
        def set_local(self, key: str, value: object, ttl_seconds: float | None = None) -> None:
            raise TypeError("not a plain data value")

    return Refusing()


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
    async def test_redis_hit_backfills_this_workers_shared_memory(self):
        writer, fake = shared_cache(default_in_memory_ttl=30)
        await writer.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        fake.flush_local()

        key_obj = await writer.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth)

        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN
        assert writer.get_local(HASHED_TOKEN)["token"] == HASHED_TOKEN

    @pytest.mark.asyncio
    async def test_update_cache_ttl_applies_to_key_object_partition(self):
        key_partition = CapturingInMemoryCache()
        cache = UserApiKeyCache(default_in_memory_ttl=60, key_object_in_memory_cache=key_partition)
        cache.update_cache_ttl(default_in_memory_ttl=7, default_redis_ttl=7)

        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        assert key_partition.last_ttl == 7

    @pytest.mark.asyncio
    async def test_attach_redis_cache_moves_both_partitions_onto_the_shared_store(self):
        fake = FakeAuthCache()
        cache = UserApiKeyCache(auth_cache=fake.store)
        cache.set_cache(end_user_cache_key("before"), {"user_id": "before"})
        assert not cache.has_redis

        cache.attach_redis_cache(MagicMock(spec=RedisCache))
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        await cache.async_set_cache(end_user_cache_key("u1"), {"user_id": "u1"})

        assert cache.has_redis
        assert set(fake.redis) == {HASHED_TOKEN, end_user_cache_key("u1")}
        assert HASHED_TOKEN not in cache.key_object_cache.in_memory_cache.cache_dict

    @pytest.mark.asyncio
    async def test_delete_removes_key_object_from_memory_and_redis(self):
        cache, fake = shared_cache()
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        cache.delete_cache(HASHED_TOKEN)

        assert cache.get_local(HASHED_TOKEN) is None
        assert HASHED_TOKEN not in fake.redis

    @pytest.mark.asyncio
    async def test_async_delete_removes_key_object_from_memory_and_redis(self):
        cache, fake = shared_cache()
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)

        await cache.async_delete_cache(HASHED_TOKEN)

        assert await cache.async_get_cache(HASHED_TOKEN, model_type=UserAPIKeyAuth) is None
        assert HASHED_TOKEN not in fake.redis

    @pytest.mark.asyncio
    async def test_batch_delete_clears_every_key_in_one_round_trip(self):
        cache, fake = shared_cache()
        await cache.async_set_cache(HASHED_TOKEN, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        await cache.async_set_cache(end_user_cache_key("u1"), {"user_id": "u1"})

        await cache.async_delete_cache_keys([HASHED_TOKEN, end_user_cache_key("u1")])

        assert cache.get_local(HASHED_TOKEN) is None
        assert cache.get_local(end_user_cache_key("u1")) is None
        assert fake.redis == {}
        assert fake.commands[-1] == ("DEL", (HASHED_TOKEN, end_user_cache_key("u1")))

    @pytest.mark.asyncio
    async def test_a_failed_batch_delete_still_clears_memory_and_raises(self):
        """A caller batching these has already committed the rows they cache, so memory is dropped first and the
        Redis failure is still raised for the caller to report."""
        cache, fake = shared_cache()
        await cache.async_set_cache(end_user_cache_key("u1"), {"user_id": "u1"})
        fake.fail = ConnectionError("redis unavailable")

        with pytest.raises(ConnectionError):
            await cache.async_delete_cache_keys([HASHED_TOKEN, end_user_cache_key("u1")])

        assert cache.get_local(end_user_cache_key("u1")) is None

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

    def test_local_access_routes_by_key_without_redis(self):
        cache = UserApiKeyCache(default_in_memory_ttl=60)
        cache.set_local(HASHED_TOKEN, {"token": HASHED_TOKEN}, None)
        cache.set_local(end_user_cache_key("u1"), {"user_id": "u1"}, 5)

        assert cache.key_object_cache.in_memory_cache.get_cache(HASHED_TOKEN) == {"token": HASHED_TOKEN}
        assert cache.in_memory_cache.get_cache(end_user_cache_key("u1")) == {"user_id": "u1"}
        cache.delete_local(HASHED_TOKEN)
        assert cache.get_local(HASHED_TOKEN) is None

    def test_a_resized_memory_tier_reaches_the_shared_store(self):
        cache, fake = shared_cache()
        cache.update_in_memory_max_size(500)

        assert [settings.memory_entries for settings in fake.settings] == [200, 500]

    def test_update_in_memory_max_size_applies_to_key_object_partition(self):
        cache = UserApiKeyCache(
            in_memory_cache=InMemoryCache(max_size_in_memory=2),
            key_object_in_memory_cache=InMemoryCache(max_size_in_memory=2),
        )
        cache.update_in_memory_max_size(3)

        tokens = tuple(hashlib.sha256(f"sk-key-{i}".encode()).hexdigest() for i in range(3))
        for token in tokens:
            cache.set_cache(token, _make_key_obj(token), model_type=UserAPIKeyAuth, ttl=100)
        for i in range(3):
            cache.set_cache(end_user_cache_key(f"u{i}"), {"user_id": f"u{i}"}, ttl=100)

        first_key = cache.get_cache(tokens[0], model_type=UserAPIKeyAuth)
        assert first_key is not None, "key partition still evicts at its old capacity"
        assert first_key.token == tokens[0]
        assert cache.get_cache(end_user_cache_key("u0")) == {"user_id": "u0"}

    def test_update_in_memory_max_size_applies_to_no_window_notes(self):
        cache = UserApiKeyCache()
        cache.update_in_memory_max_size(1)

        cache.budget_windows_absent_cache.set_cache(key=budget_windows_absent_cache_key("key", "first"), value=True)
        cache.budget_windows_absent_cache.set_cache(key=budget_windows_absent_cache_key("key", "second"), value=True)

        assert cache.budget_windows_absent_cache.get_cache(key=budget_windows_absent_cache_key("key", "first")) is None
        assert cache.budget_windows_absent_cache.get_cache(key=budget_windows_absent_cache_key("key", "second")) is True

    def test_update_cache_ttl_applies_to_no_window_notes(self):
        clock: Final = _ManualClock()
        cache: Final = UserApiKeyCache(
            default_in_memory_ttl=60, budget_windows_absent_in_memory_cache=InMemoryCache(clock=clock)
        )
        cache.update_cache_ttl(default_in_memory_ttl=5, default_redis_ttl=5)
        note: Final = budget_windows_absent_cache_key("key", "no-windows")

        cache.budget_windows_absent_cache.set_cache(key=note, value=True)
        clock.advance(6)

        assert cache.budget_windows_absent_cache.get_cache(key=note) is None

    def test_update_in_memory_max_size_none_resets_key_object_partition_to_default(self):
        cache = UserApiKeyCache(key_object_in_memory_cache=InMemoryCache(max_size_in_memory=1))
        cache.update_in_memory_max_size(None)

        tokens = tuple(hashlib.sha256(f"sk-key-{i}".encode()).hexdigest() for i in range(2))
        for token in tokens:
            cache.set_cache(token, _make_key_obj(token), model_type=UserAPIKeyAuth, ttl=100)

        first_key = cache.get_cache(tokens[0], model_type=UserAPIKeyAuth)
        assert first_key is not None
        assert first_key.token == tokens[0]


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


async def _evict_by_delete(cache: UserApiKeyCache, key: str) -> None:
    await cache.async_delete_cache(key)


async def _evict_by_bulk_delete(cache: UserApiKeyCache, key: str) -> None:
    await cache.async_delete_cache_keys((key,))


async def _evict_by_broadcast(cache: UserApiKeyCache, key: str) -> None:
    cache.delete_local(key)


async def _evict_by_sync_delete(cache: UserApiKeyCache, key: str) -> None:
    cache.delete_cache(key)


async def _evict_by_flush(cache: UserApiKeyCache, key: str) -> None:
    cache.flush_cache()


def _memory_only() -> UserApiKeyCache:
    return UserApiKeyCache()


def _with_redis_auth_cache() -> UserApiKeyCache:
    return shared_cache()[0]


_NewCache: TypeAlias = Callable[[], UserApiKeyCache]
_Evict: TypeAlias = Callable[[UserApiKeyCache, str], Awaitable[None]]

_EVICTIONS: Final = (
    _evict_by_delete,
    _evict_by_bulk_delete,
    _evict_by_broadcast,
    _evict_by_sync_delete,
    _evict_by_flush,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("new_cache", [_memory_only, _with_redis_auth_cache])
@pytest.mark.parametrize("evict", _EVICTIONS)
async def test_a_load_that_began_before_an_eviction_cannot_cache_what_it_read(
    new_cache: _NewCache, evict: _Evict
) -> None:
    cache: Final = new_cache()
    lease: Final = await cache.lease("customer_member_id:m")
    await evict(cache, "customer_member_id:m")

    assert await cache.async_set_leased(lease, {"spend": 5.0}, ttl=60) is False
    assert await cache.async_get_cache("customer_member_id:m") is None

    assert await cache.async_set_leased(await cache.lease("customer_member_id:m"), {"spend": 0.0}, ttl=60) is True
    assert await cache.async_get_cache("customer_member_id:m") == {"spend": 0.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("new_cache", [_memory_only, _with_redis_auth_cache])
async def test_a_load_caches_what_it_read_when_only_another_key_was_evicted(new_cache: _NewCache) -> None:
    cache: Final = new_cache()
    lease: Final = await cache.lease("customer_member_id:m")
    await cache.async_delete_cache("customer_member_id:other")

    assert await cache.async_set_leased(lease, {"spend": 5.0}, ttl=60) is True
    assert await cache.async_get_cache("customer_member_id:m") == {"spend": 5.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("evict", [_evict_by_delete, _evict_by_bulk_delete, _evict_by_sync_delete])
async def test_a_load_cannot_cache_what_it_read_before_another_worker_evicted_the_key(evict: _Evict) -> None:
    """The eviction broadcast has not reached the loading worker, so only Redis can refuse its write."""
    resetter, shared = shared_cache()
    loader, _ = shared_cache(FakeAuthCache(redis=shared.redis, generations=shared.generations))
    lease: Final = await loader.lease("customer_member_id:m")
    await evict(resetter, "customer_member_id:m")

    assert await loader.async_set_leased(lease, {"spend": 5.0}, ttl=60) is False
    assert await loader.async_get_cache("customer_member_id:m") is None
    assert await resetter.async_get_cache("customer_member_id:m") is None

    assert await loader.async_set_leased(await loader.lease("customer_member_id:m"), {"spend": 0.0}, ttl=60) is True
    assert await resetter.async_get_cache("customer_member_id:m") == {"spend": 0.0}


class _EvictedMidWrite(FakeAuthCache):
    """Redis takes the leased write, and this worker evicts the key before the reply is handled."""

    cache: UserApiKeyCache | None = None

    async def set_leased(self, key: str, lease: str | None, value: object, ttl_seconds: float | None = None) -> bool:
        written: Final = await super().set_leased(key, lease, value, ttl_seconds)
        assert self.cache is not None
        await self.cache.async_delete_cache(key)
        return written


@pytest.mark.asyncio
async def test_an_eviction_that_lands_while_redis_takes_the_write_keeps_it_out_of_memory() -> None:
    double: Final = _EvictedMidWrite()
    cache, _ = shared_cache(double)
    double.cache = cache
    lease: Final = await cache.lease("customer_member_id:m")

    assert await cache.async_set_leased(lease, {"spend": 5.0}, ttl=60) is False
    assert await cache.async_get_cache("customer_member_id:m") is None
