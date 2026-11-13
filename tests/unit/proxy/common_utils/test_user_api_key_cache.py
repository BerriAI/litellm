import asyncio
import datetime
import hashlib
from collections.abc import Awaitable, Callable
from typing import Final

import pytest

from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy._types import UserAPIKeyAuth
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
