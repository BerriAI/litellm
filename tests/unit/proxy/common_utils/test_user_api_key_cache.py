import asyncio
import datetime
import hashlib
from collections.abc import Awaitable, Callable
from typing import ClassVar, Final, TypeAlias

import pytest
from pydantic import BaseModel, model_validator

from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy._types import LiteLLM_TeamTableCachedObj, LiteLLM_UserTable, UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache, get_management_object_ttl
from litellm.proxy.proxy_server import UserAPIKeyCacheTTLEnum
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore, AuthObject
from litellm.rust_bridge._native import NativeRedisStoreError
from tests.unit.caching.redis_batch_double import FakeNatives, FakeRedisCache, fake_send
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache, shared_cache

HASHED_TOKEN: Final = hashlib.sha256(b"sk-lit7563-hot-key").hexdigest()
HEX_USER_ID: Final = hashlib.sha256(b"user-whose-id-looks-like-a-key").hexdigest()


def _make_key_obj(token: str = "tok") -> UserAPIKeyAuth:
    return UserAPIKeyAuth(token=token)


class _Refusing(FakeAuthCache):
    def set_local(self, key: AuthObject, value: object, ttl_seconds: float | None = None) -> None:
        raise TypeError("not a plain data value")


class TestReadsAndWrites:
    @pytest.mark.asyncio
    async def test_a_write_without_a_ttl_takes_the_default_in_memory_ttl(self):
        cache: Final = UserApiKeyCache(default_in_memory_ttl=0.05)
        expiring: Final = AuthObject.team("expiring")
        pinned: Final = AuthObject.team("pinned")

        await cache.async_set_cache(expiring, {"team_id": "expiring"})
        await cache.async_set_cache(pinned, {"team_id": "pinned"}, ttl=60)
        await asyncio.sleep(0.15)

        assert cache.get_local(expiring) is None
        assert cache.get_local(pinned) == {"team_id": "pinned"}

    @pytest.mark.asyncio
    async def test_the_proxy_default_ttl_reaches_the_store_and_writes_name_no_ttl(self):
        cache, fake = shared_cache(default_in_memory_ttl=UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value)
        key: Final = AuthObject.virtual_key(HASHED_TOKEN)

        await cache.async_set_cache(key, _make_key_obj("ttl-tok"), model_type=UserAPIKeyAuth)
        cache.set_cache(key, _make_key_obj("ttl-tok"), model_type=UserAPIKeyAuth)

        assert [settings.memory_ttl for settings in fake.settings] == [UserAPIKeyCacheTTLEnum.in_memory_cache_ttl.value]
        assert fake.commands == [("SET", HASHED_TOKEN, None), ("SET_STR", HASHED_TOKEN, None)]

    @pytest.mark.asyncio
    async def test_an_explicit_ttl_reaches_redis(self):
        cache, fake = shared_cache(default_in_memory_ttl=60)

        await cache.async_set_cache(AuthObject.team("t1"), {"team_id": "t1"}, ttl=900)

        assert fake.commands == [("SET", "team_id:t1", 900)]

    def test_a_sync_write_goes_through_the_store_as_its_serialized_payload(self):
        cache, fake = shared_cache(default_in_memory_ttl=45)

        cache.set_cache(AuthObject.virtual_key(HASHED_TOKEN), _make_key_obj("sync"), model_type=UserAPIKeyAuth, ttl=45)

        assert fake.commands == [("SET_STR", HASHED_TOKEN, 45)]
        assert fake.redis[HASHED_TOKEN]["token"] == "sync"

    @pytest.mark.asyncio
    async def test_a_typed_write_stores_the_same_json_payload_in_memory_and_redis(self):
        cache, fake = shared_cache()
        key: Final = AuthObject.virtual_key(HASHED_TOKEN)

        await cache.async_set_cache(key, _make_key_obj("abc"), model_type=UserAPIKeyAuth)

        raw: Final = cache.get_local(key)
        assert isinstance(raw, dict)
        assert raw["token"] == "abc"
        assert fake.redis[HASHED_TOKEN] == raw

    @pytest.mark.asyncio
    async def test_a_redis_hit_decodes_to_the_model_and_backfills_memory(self):
        cache, fake = shared_cache(FakeAuthCache(redis={HASHED_TOKEN: {"token": "abc"}}))
        key: Final = AuthObject.virtual_key(HASHED_TOKEN)

        value: Final = await cache.async_get_cache(key, model_type=UserAPIKeyAuth)

        assert isinstance(value, UserAPIKeyAuth)
        assert value.token == "abc"
        assert fake.commands == [("GET", HASHED_TOKEN)]
        assert cache.get_local(key) == {"token": "abc"}

    @pytest.mark.asyncio
    async def test_a_cached_payload_that_no_longer_validates_is_a_logged_miss(self, caplog: pytest.LogCaptureFixture):
        cache: Final = UserApiKeyCache()
        key: Final = AuthObject.virtual_key(HASHED_TOKEN)
        cache.set_local(key, "invalid-payload-not-a-dict", None)

        assert await cache.async_get_cache(key, model_type=UserAPIKeyAuth) is None
        assert cache.get_cache(key, model_type=UserAPIKeyAuth) is None
        assert cache.get_cache(key) == "invalid-payload-not-a-dict"
        drift: Final = [
            record.getMessage() for record in caplog.records if "failed to deserialize" in record.getMessage()
        ]
        assert len(drift) == 2
        assert all(HASHED_TOKEN in message and "UserAPIKeyAuth" in message for message in drift)

    @pytest.mark.asyncio
    async def test_a_failed_read_is_a_miss_and_a_failed_write_keeps_the_request_going(self):
        cache, fake = shared_cache(FakeAuthCache(fail=NativeRedisStoreError("transport", "redis down")))

        assert await cache.async_get_cache(AuthObject.team("t1")) is None
        await cache.async_set_cache(AuthObject.team("t1"), {"team_id": "t1"}, ttl=5)
        cache.set_cache(AuthObject.team("t3"), {"team_id": "t3"})
        assert cache.get_cache(AuthObject.team("t2")) is None

        assert [command[0] for command in fake.commands] == ["GET", "SET", "SET_STR", "GET"]
        assert cache.get_local(AuthObject.team("t1")) == {"team_id": "t1"}

    @pytest.mark.asyncio
    async def test_a_value_the_bridge_refuses_is_logged_not_raised(self, caplog: pytest.LogCaptureFixture):
        cache: Final = UserApiKeyCache(auth_cache=lambda _settings: AuthCacheStore(_Refusing()))

        await cache.async_set_cache(AuthObject.team("t1"), {"team_id": "t1"})
        cache.set_cache(AuthObject.team("t1"), {"team_id": "t1"})

        refused: Final = [
            record.getMessage() for record in caplog.records if "not a plain data value" in record.getMessage()
        ]
        assert len(refused) == 2

    @pytest.mark.asyncio
    async def test_a_model_holding_a_datetime_is_cached_as_its_json_dump(self):
        cache: Final = UserApiKeyCache()
        key: Final = AuthObject.virtual_key(HASHED_TOKEN)
        key_obj: Final = UserAPIKeyAuth(
            token="abc", expires=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
        )

        await cache.async_set_cache(key, key_obj, model_type=UserAPIKeyAuth, local_only=True)

        assert cache.get_local(key) == key_obj.model_dump(mode="json")
        cached: Final = await cache.async_get_cache(key, model_type=UserAPIKeyAuth)
        assert cached is not None
        assert cached.model_dump(mode="json") == key_obj.model_dump(mode="json")

    @pytest.mark.asyncio
    async def test_local_only_reads_and_writes_never_reach_redis(self):
        cache, fake = shared_cache(FakeAuthCache(redis={"team_id:t1": {"team_id": "t1"}}))

        assert await cache.async_get_cache(AuthObject.team("t1"), local_only=True) is None
        assert cache.get_cache(AuthObject.team("t1"), local_only=True) is None
        await cache.async_set_cache(AuthObject.team("t2"), {"team_id": "t2"}, local_only=True)
        cache.set_cache(AuthObject.team("t3"), {"team_id": "t3"}, local_only=True)

        assert fake.commands == []
        assert set(fake.redis) == {"team_id:t1"}
        assert cache.get_local(AuthObject.team("t3")) == {"team_id": "t3"}

    @pytest.mark.asyncio
    async def test_a_pipeline_write_stores_every_entry_in_one_round_trip(self):
        cache, fake = shared_cache()

        await cache.async_set_cache_pipeline(
            [
                (AuthObject.virtual_key(HASHED_TOKEN), _make_key_obj(HASHED_TOKEN)),
                (AuthObject.team("t1"), {"team_id": "t1"}),
            ],
            ttl=100,
        )

        assert fake.commands == [("SET_MANY", (HASHED_TOKEN, "team_id:t1"), 100)]
        key_obj: Final = await cache.async_get_cache(AuthObject.virtual_key(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN

    @pytest.mark.asyncio
    async def test_global_spend_increments_only_an_entry_redis_still_holds(self):
        cache, fake = shared_cache(FakeAuthCache(redis={"default_user_id:spend": 1.5}))
        memory_only: Final = UserApiKeyCache()

        assert await cache.increment_if_present(AuthObject.global_proxy_spend(), 2.0) == 3.5
        assert await cache.increment_if_present(AuthObject.team("absent"), 2.0) is None
        assert await memory_only.increment_if_present(AuthObject.global_proxy_spend(), 2.0) is None
        assert [command[:2] for command in fake.commands] == [
            ("INCRBYFLOAT_IF_PRESENT", "default_user_id:spend"),
            ("INCRBYFLOAT_IF_PRESENT", "team_id:absent"),
        ]


class TestInsideARequest:
    @pytest.mark.asyncio
    async def test_a_write_rides_the_pre_call_batch_and_a_write_outside_goes_through(self):
        cache, fake = shared_cache()
        natives: Final = FakeNatives()

        with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
            await cache.async_set_cache(AuthObject.team("t1"), {"team_id": "t1"}, ttl=60)
            assert cache.get_local(AuthObject.team("t1")) == {"team_id": "t1"}
            assert natives.built[0].events == ["declare set team_id:t1 ttl=60"]
            await request.flush_pre_call()
        await cache.async_set_cache(AuthObject.team("t2"), {"team_id": "t2"}, ttl=60)

        assert fake.commands == [("SET", "team_id:t2", 60.0)]

    @pytest.mark.asyncio
    async def test_a_declared_delete_drops_memory_now_and_rides_the_batch_only_inside_a_request(self):
        cache, fake = shared_cache()
        cache.set_local(AuthObject.team("t1"), {"team_id": "t1"}, None)
        natives: Final = FakeNatives()

        assert cache.declare_delete("team_id:t1") is None
        with request_redis_batch_scope(native_factory=natives, native_send_factory=fake_send) as request:  # pyright: ignore[reportArgumentType]  # test double
            pending: Final = cache.declare_delete("team_id:t1")
            assert pending is not None
            assert cache.get_local(AuthObject.team("t1")) is None
            await pending
            await request.flush_pre_call()

        assert natives.built[0].events[0] == "declare del team_id:t1"
        assert fake.commands == []


class TestKeyObjectPartition:
    """
    Regression for LIT-7563: user-key objects keep a memory partition of their own, so end-user / team / tag churn
    cannot evict hot keys and force a ``LiteLLM_VerificationToken`` lookup on the next request.
    """

    @pytest.mark.asyncio
    async def test_management_object_churn_does_not_evict_a_virtual_key(self):
        cache: Final = UserApiKeyCache(max_size_in_memory=2)
        key: Final = AuthObject.virtual_key(HASHED_TOKEN)
        await cache.async_set_cache(key, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth, ttl=100)
        for i in range(3):
            await cache.async_set_cache(AuthObject.end_user(f"u{i}"), {"user_id": f"u{i}"}, ttl=200)

        key_obj: Final = await cache.async_get_cache(key, model_type=UserAPIKeyAuth)
        assert key_obj is not None
        assert key_obj.token == HASHED_TOKEN
        assert cache.get_cache(AuthObject.end_user("u2")) == {"user_id": "u2"}
        sizes: Final = cache.store.len_local()
        assert (sizes.objects, sizes.key_objects) == (2, 1)

    def test_a_user_whose_id_looks_like_a_key_is_a_management_object(self):
        cache: Final = UserApiKeyCache(max_size_in_memory=1)
        cache.set_cache(AuthObject.user(HEX_USER_ID), {"user_id": HEX_USER_ID})
        cache.set_cache(AuthObject.virtual_key(HASHED_TOKEN), {"token": HASHED_TOKEN})
        cache.set_cache(AuthObject.team("t1"), {"team_id": "t1"})

        assert cache.get_local(AuthObject.user(HEX_USER_ID)) is None
        assert cache.get_local(AuthObject.virtual_key(HASHED_TOKEN)) == {"token": HASHED_TOKEN}

    def test_the_same_wire_key_is_held_apart_per_partition(self):
        cache: Final = UserApiKeyCache()
        cache.set_local(AuthObject.virtual_key(HASHED_TOKEN), {"token": HASHED_TOKEN}, None)
        cache.set_local(AuthObject.user(HASHED_TOKEN), {"user_id": HASHED_TOKEN}, None)

        assert cache.get_local(AuthObject.virtual_key(HASHED_TOKEN)) == {"token": HASHED_TOKEN}
        assert cache.get_local(AuthObject.user(HASHED_TOKEN)) == {"user_id": HASHED_TOKEN}

    @pytest.mark.parametrize(
        "evict",
        [
            pytest.param(lambda cache, key: _done(cache.delete_local(key)), id="delete_local"),
            pytest.param(lambda cache, key: _done(cache.delete_cache(key)), id="delete_cache"),
            pytest.param(lambda cache, key: cache.async_delete_cache(key), id="async_delete_cache"),
            pytest.param(lambda cache, key: cache.async_delete_cache_keys([key]), id="async_delete_cache_keys"),
        ],
    )
    @pytest.mark.asyncio
    async def test_an_eviction_by_wire_key_drops_it_from_both_partitions(
        self, evict: Callable[[UserApiKeyCache, str], Awaitable[None]]
    ):
        cache: Final = UserApiKeyCache()
        cache.set_cache(AuthObject.virtual_key(HASHED_TOKEN), {"token": HASHED_TOKEN})
        cache.set_cache(AuthObject.user(HASHED_TOKEN), {"user_id": HASHED_TOKEN})
        cache.set_cache(AuthObject.user(HEX_USER_ID), {"user_id": HEX_USER_ID})

        await evict(cache, HASHED_TOKEN)
        await evict(cache, HEX_USER_ID)

        assert cache.get_local(AuthObject.virtual_key(HASHED_TOKEN)) is None
        assert cache.get_local(AuthObject.user(HASHED_TOKEN)) is None
        assert cache.get_local(AuthObject.user(HEX_USER_ID)) is None
        sizes: Final = cache.store.len_local()
        assert (sizes.objects, sizes.key_objects) == (0, 0)

    @pytest.mark.asyncio
    async def test_a_delete_drops_memory_and_redis(self):
        cache, fake = shared_cache()
        await cache.async_set_cache(
            AuthObject.virtual_key(HASHED_TOKEN), _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth
        )
        await cache.async_set_cache(AuthObject.end_user("u1"), {"user_id": "u1"})

        cache.delete_cache(HASHED_TOKEN)
        await cache.async_delete_cache("end_user_id:u1")

        assert await cache.async_get_cache(AuthObject.virtual_key(HASHED_TOKEN), model_type=UserAPIKeyAuth) is None
        assert await cache.async_get_cache(AuthObject.end_user("u1")) is None
        assert fake.redis == {}

    @pytest.mark.asyncio
    async def test_a_batch_delete_clears_every_key_in_one_round_trip(self):
        cache, fake = shared_cache()
        await cache.async_set_cache(
            AuthObject.virtual_key(HASHED_TOKEN), _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth
        )
        await cache.async_set_cache(AuthObject.end_user("u1"), {"user_id": "u1"})

        await cache.async_delete_cache_keys([HASHED_TOKEN, "end_user_id:u1"])
        await cache.async_delete_cache_keys([])

        assert cache.get_local(AuthObject.virtual_key(HASHED_TOKEN)) is None
        assert cache.get_local(AuthObject.end_user("u1")) is None
        assert fake.redis == {}
        assert fake.commands[-1] == ("DEL", (HASHED_TOKEN, "end_user_id:u1"))

    @pytest.mark.asyncio
    async def test_a_failed_batch_delete_still_clears_memory_and_raises(self):
        """A caller batching these has already committed the rows they cache, so memory is dropped first and the
        Redis failure is still raised for the caller to report."""
        cache, fake = shared_cache()
        await cache.async_set_cache(AuthObject.end_user("u1"), {"user_id": "u1"})
        fake.fail = NativeRedisStoreError("transport", "redis unavailable")

        with pytest.raises(NativeRedisStoreError):
            await cache.async_delete_cache_keys([HASHED_TOKEN, "end_user_id:u1"])

        assert cache.get_local(AuthObject.end_user("u1")) is None

    def test_flush_clears_both_partitions(self):
        cache: Final = UserApiKeyCache()
        cache.set_cache(AuthObject.virtual_key(HASHED_TOKEN), _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        cache.set_cache(AuthObject.end_user("u1"), {"user_id": "u1"})

        cache.flush_cache()

        assert cache.get_cache(AuthObject.virtual_key(HASHED_TOKEN), model_type=UserAPIKeyAuth) is None
        assert cache.get_cache(AuthObject.end_user("u1")) is None


class TestReconfigure:
    def test_attach_redis_cache_attaches_once_and_keeps_memory(self):
        fake: Final = FakeAuthCache(attached=False)
        cache: Final = UserApiKeyCache(auth_cache=fake.store)
        cache.set_cache(AuthObject.end_user("before"), {"user_id": "before"})
        first: Final = FakeRedisCache()
        second: Final = FakeRedisCache()
        assert not cache.has_redis and cache.redis_connection is None

        cache.attach_redis_cache(first)
        fake.attached = False
        cache.attach_redis_cache(second)

        assert cache.redis_connection is first
        assert not fake.attached, "a second attach must not reach the store"
        assert cache.get_local(AuthObject.end_user("before")) == {"user_id": "before"}

    @pytest.mark.asyncio
    async def test_update_cache_ttl_applies_to_writes_without_a_ttl(self):
        cache: Final = UserApiKeyCache(default_in_memory_ttl=60)
        cache.update_cache_ttl(default_in_memory_ttl=0.05)
        key: Final = AuthObject.virtual_key(HASHED_TOKEN)

        await cache.async_set_cache(key, _make_key_obj(HASHED_TOKEN), model_type=UserAPIKeyAuth)
        await asyncio.sleep(0.15)

        assert cache.default_in_memory_ttl == 0.05
        assert cache.get_local(key) is None

    def test_update_in_memory_max_size_grows_both_partitions(self):
        cache: Final = UserApiKeyCache(max_size_in_memory=2)
        cache.update_in_memory_max_size(3)

        tokens: Final = tuple(hashlib.sha256(f"sk-key-{i}".encode()).hexdigest() for i in range(3))
        for token in tokens:
            cache.set_cache(AuthObject.virtual_key(token), _make_key_obj(token), model_type=UserAPIKeyAuth, ttl=100)
        for i in range(3):
            cache.set_cache(AuthObject.end_user(f"u{i}"), {"user_id": f"u{i}"}, ttl=100)

        first_key: Final = cache.get_cache(AuthObject.virtual_key(tokens[0]), model_type=UserAPIKeyAuth)
        assert first_key is not None, "key partition still evicts at its old capacity"
        assert first_key.token == tokens[0]
        assert cache.get_cache(AuthObject.end_user("u0")) == {"user_id": "u0"}
        assert cache.max_size_in_memory == 3

    def test_reconfiguring_reaches_the_store(self):
        cache, fake = shared_cache()

        cache.update_cache_ttl(default_in_memory_ttl=7)
        cache.update_in_memory_max_size(500)

        assert fake.settings == [
            AuthCacheSettings(memory_entries=200, memory_ttl=None),
            AuthCacheSettings(memory_entries=200, memory_ttl=7),
            AuthCacheSettings(memory_entries=500, memory_ttl=7),
        ]

    def test_a_cleared_max_size_restores_the_default_capacity(self):
        """Blanking ``user_api_key_cache_max_size`` in the dashboard reaches the cache as None."""
        cache: Final = UserApiKeyCache(max_size_in_memory=1)
        cache.update_in_memory_max_size(None)

        tokens: Final = tuple(hashlib.sha256(f"sk-key-{i}".encode()).hexdigest() for i in range(2))
        for token in tokens:
            cache.set_cache(AuthObject.virtual_key(token), _make_key_obj(token), model_type=UserAPIKeyAuth, ttl=100)

        first_key: Final = cache.get_cache(AuthObject.virtual_key(tokens[0]), model_type=UserAPIKeyAuth)
        assert first_key is not None
        assert first_key.token == tokens[0]


class TestManagementObjectTTL:
    """
    Regression for LIT-3338: ``general_settings.user_api_key_cache_ttl`` (which the
    proxy propagates onto ``default_in_memory_ttl``) must win over the hardcoded
    ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` for management-object writes.
    """

    def test_returns_configured_default_in_memory_ttl(self):
        cache: Final = UserApiKeyCache(default_in_memory_ttl=300)
        assert get_management_object_ttl(cache) == 300

    def test_falls_back_to_constant_when_no_default_configured(self):
        cache: Final = UserApiKeyCache()
        assert cache.default_in_memory_ttl is None
        assert get_management_object_ttl(cache) == DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL

    @pytest.mark.asyncio
    async def test_management_write_uses_configured_ttl_over_constant(self):
        cache, fake = shared_cache(default_in_memory_ttl=300)
        assert get_management_object_ttl(cache) != DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL

        await cache.async_set_cache(
            AuthObject.team("abc"), _make_key_obj("t"), model_type=UserAPIKeyAuth, ttl=get_management_object_ttl(cache)
        )

        assert fake.memory["team_id:abc"][1] == 300
        assert fake.commands == [("SET", "team_id:abc", 300)]


async def _done(_result: None) -> None:
    return None


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

_MEMBER: Final = AuthObject.customer_member("m")

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
    lease: Final = await cache.lease(_MEMBER)
    await evict(cache, _MEMBER.key)

    assert await cache.async_set_leased(lease, {"spend": 5.0}, ttl=60) is False
    assert await cache.async_get_cache(_MEMBER) is None

    assert await cache.async_set_leased(await cache.lease(_MEMBER), {"spend": 0.0}, ttl=60) is True
    assert await cache.async_get_cache(_MEMBER) == {"spend": 0.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("new_cache", [_memory_only, _with_redis_auth_cache])
async def test_a_load_caches_what_it_read_when_only_another_key_was_evicted(new_cache: _NewCache) -> None:
    cache: Final = new_cache()
    lease: Final = await cache.lease(_MEMBER)
    await cache.async_delete_cache(AuthObject.customer_member("other").key)

    assert await cache.async_set_leased(lease, {"spend": 5.0}, ttl=60) is True
    assert await cache.async_get_cache(_MEMBER) == {"spend": 5.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("evict", [_evict_by_delete, _evict_by_bulk_delete, _evict_by_sync_delete])
async def test_a_load_cannot_cache_what_it_read_before_another_worker_evicted_the_key(evict: _Evict) -> None:
    """The eviction broadcast has not reached the loading worker, so only Redis can refuse its write."""
    resetter, shared = shared_cache()
    loader, _ = shared_cache(FakeAuthCache(redis=shared.redis, generations=shared.generations))
    lease: Final = await loader.lease(_MEMBER)
    await evict(resetter, _MEMBER.key)

    assert await loader.async_set_leased(lease, {"spend": 5.0}, ttl=60) is False
    assert await loader.async_get_cache(_MEMBER) is None
    assert await resetter.async_get_cache(_MEMBER) is None

    assert await loader.async_set_leased(await loader.lease(_MEMBER), {"spend": 0.0}, ttl=60) is True
    assert await resetter.async_get_cache(_MEMBER) == {"spend": 0.0}


class _EvictedMidWrite(FakeAuthCache):
    """Redis takes the leased write, and this worker evicts the key before the reply is handled."""

    cache: UserApiKeyCache | None = None

    async def set_leased(
        self, key: AuthObject, lease: str | None, value: object, ttl_seconds: float | None, request: object
    ) -> bool:
        written: Final = await super().set_leased(key, lease, value, ttl_seconds, request)
        assert self.cache is not None
        await self.cache.async_delete_cache(key.key)
        return written


@pytest.mark.asyncio
async def test_an_eviction_that_lands_while_redis_takes_the_write_keeps_it_out_of_memory() -> None:
    double: Final = _EvictedMidWrite()
    cache, _ = shared_cache(double)
    double.cache = cache
    lease: Final = await cache.lease(_MEMBER)

    assert await cache.async_set_leased(lease, {"spend": 5.0}, ttl=60) is False
    assert await cache.async_get_cache(_MEMBER) is None


class _CountedValidation(BaseModel):
    validations: ClassVar[int] = 0

    spend: float

    @model_validator(mode="before")
    @classmethod
    def _count(cls, data: object) -> object:
        _CountedValidation.validations += 1
        return data


_TEAM_KEY: Final = AuthObject.team("t1")


def _auth_objects() -> tuple[tuple[AuthObject, BaseModel], ...]:
    return (
        (AuthObject.virtual_key(HASHED_TOKEN), UserAPIKeyAuth(token=HASHED_TOKEN, spend=1.5, models=["gpt-4o"], metadata={"a": 1})),
        (_TEAM_KEY, LiteLLM_TeamTableCachedObj(team_id="t1", spend=2.0, models=["gpt-4o"], metadata={"a": 1})),
        (AuthObject.user("u1"), LiteLLM_UserTable(user_id="u1", spend=3.0, models=["gpt-4o"], teams=["t1"])),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("new_cache", [_memory_only, _with_redis_auth_cache])
@pytest.mark.parametrize(("key", "stored"), _auth_objects())
async def test_every_hit_returns_an_object_equal_to_the_stored_one(
    new_cache: _NewCache, key: AuthObject, stored: BaseModel
) -> None:
    cache: Final = new_cache()
    await cache.async_set_cache(key, stored, model_type=type(stored))

    reads: Final = (
        await cache.async_get_cache(key, model_type=type(stored)),
        await cache.async_get_cache(key, model_type=type(stored)),
        cache.get_cache(key, model_type=type(stored)),
    )

    assert [type(read) for read in reads] == [type(stored)] * 3
    assert list(reads) == [stored] * 3


@pytest.mark.asyncio
@pytest.mark.parametrize("new_cache", [_memory_only, _with_redis_auth_cache])
@pytest.mark.parametrize(("key", "stored"), _auth_objects())
async def test_a_field_set_on_a_read_object_does_not_reach_the_next_read(
    new_cache: _NewCache, key: AuthObject, stored: BaseModel
) -> None:
    cache: Final = new_cache()
    await cache.async_set_cache(key, stored, model_type=type(stored))
    reads: Final = [await cache.async_get_cache(key, model_type=type(stored)) for _ in range(2)]
    for read in reads:
        assert read is not None
        read.spend = 99.0
        read.models = ["mutated"]

    after_mutation: Final = await cache.async_get_cache(key, model_type=type(stored))

    assert after_mutation == stored


@pytest.mark.asyncio
async def test_reads_of_an_unchanged_entry_validate_it_once() -> None:
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(_MEMBER, {"spend": 1.0})
    before: Final = _CountedValidation.validations

    reads: Final = [await cache.async_get_cache(_MEMBER, model_type=_CountedValidation) for _ in range(5)]

    assert _CountedValidation.validations - before == 1
    assert reads == [_CountedValidation(spend=1.0)] * 5


def test_a_read_without_redis_answers_without_suspending_the_caller() -> None:
    cache: Final = UserApiKeyCache()
    cache.set_cache(_MEMBER, {"spend": 1.0})
    read: Final = cache.async_get_cache(_MEMBER)

    with pytest.raises(StopIteration) as answered:
        read.send(None)

    assert answered.value.value == {"spend": 1.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("new_cache", [_memory_only, _with_redis_auth_cache])
async def test_each_write_changes_what_the_next_read_returns(new_cache: _NewCache) -> None:
    cache: Final = new_cache()
    await cache.async_set_cache(_TEAM_KEY, LiteLLM_TeamTableCachedObj(team_id="t1", spend=1.0))
    assert await cache.async_get_cache(_TEAM_KEY, model_type=LiteLLM_TeamTableCachedObj) is not None

    await cache.async_set_cache(_TEAM_KEY, LiteLLM_TeamTableCachedObj(team_id="t1", spend=2.0))
    after_async_set: Final = await cache.async_get_cache(_TEAM_KEY, model_type=LiteLLM_TeamTableCachedObj)
    cache.set_cache(_TEAM_KEY, LiteLLM_TeamTableCachedObj(team_id="t1", spend=3.0), local_only=True)
    after_local_set: Final = cache.get_cache(_TEAM_KEY, local_only=True, model_type=LiteLLM_TeamTableCachedObj)
    await cache.async_set_cache_pipeline([(_TEAM_KEY, LiteLLM_TeamTableCachedObj(team_id="t1", spend=4.0))])
    after_pipeline: Final = await cache.async_get_cache(_TEAM_KEY, model_type=LiteLLM_TeamTableCachedObj)

    assert [after_async_set, after_local_set, after_pipeline] == [
        LiteLLM_TeamTableCachedObj(team_id="t1", spend=2.0),
        LiteLLM_TeamTableCachedObj(team_id="t1", spend=3.0),
        LiteLLM_TeamTableCachedObj(team_id="t1", spend=4.0),
    ]


@pytest.mark.asyncio
async def test_a_payload_rewritten_in_place_and_set_again_is_read_back() -> None:
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(_MEMBER, {"spend": 1.0})
    assert await cache.async_get_cache(_MEMBER, model_type=_CountedValidation) == _CountedValidation(spend=1.0)
    payload: Final = await cache.async_get_cache(_MEMBER)
    payload["spend"] = 2.0

    await cache.async_set_cache(_MEMBER, payload)

    assert await cache.async_get_cache(_MEMBER, model_type=_CountedValidation) == _CountedValidation(spend=2.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("new_cache", [_memory_only, _with_redis_auth_cache])
@pytest.mark.parametrize("evict", _EVICTIONS)
async def test_a_typed_read_after_an_eviction_is_a_miss(new_cache: _NewCache, evict: _Evict) -> None:
    cache: Final = new_cache()
    key: Final = AuthObject.virtual_key(HASHED_TOKEN)
    await cache.async_set_cache(key, UserAPIKeyAuth(token=HASHED_TOKEN), model_type=UserAPIKeyAuth)
    assert await cache.async_get_cache(key, model_type=UserAPIKeyAuth) is not None

    await evict(cache, HASHED_TOKEN)

    assert await cache.async_get_cache(key, local_only=True, model_type=UserAPIKeyAuth) is None


@pytest.mark.asyncio
async def test_one_entry_read_as_two_models_returns_each_model() -> None:
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(_TEAM_KEY, LiteLLM_TeamTableCachedObj(team_id="t1", spend=1.0))

    as_team: Final = await cache.async_get_cache(_TEAM_KEY, model_type=LiteLLM_TeamTableCachedObj)
    as_counted: Final = await cache.async_get_cache(_TEAM_KEY, model_type=_CountedValidation)

    assert type(as_team) is LiteLLM_TeamTableCachedObj
    assert as_counted == _CountedValidation(spend=1.0)
