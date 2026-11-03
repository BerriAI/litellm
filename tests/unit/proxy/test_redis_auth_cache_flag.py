"""
Tests for the enable_redis_auth_cache litellm_settings flag.

Verifies that _init_cache attaches Redis to user_api_key_cache only when
the flag is explicitly set to True, and leaves it in-memory-only otherwise.
"""

import json
from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

import pytest

import litellm
import litellm.proxy.proxy_server as ps
from litellm.caching.caching import RedisCache
from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.native_settings import NativeRedisSettings
from litellm.redis.stores.spend_counters import SpendCounterStore
from tests.unit.redis.stores.auth_cache_double import FakeAuthCache
from tests.unit.redis.stores.spend_counters_double import FakeSpendCounters

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRedisCache(RedisCache):
    """
    Minimal RedisCache subclass that passes isinstance checks without
    requiring a real Redis connection.  __init__ is bypassed so no
    network calls are made.
    """

    def __init__(self):  # super().__init__ skipped intentionally
        self._store = {}
        self.default_ttl = 60
        self.native_settings = NativeRedisSettings(
            redis_kwargs={"host": "127.0.0.1", "port": 1, "socket_timeout": 0.2}, namespace=None, default_ttl=60
        )

    def set_cache(self, key, value, **kwargs):  # type: ignore[override]
        # Enforce Redis JSON-serializable payload contract.
        self._store[key] = json.dumps(value)
        return True

    def get_cache(self, key, **kwargs):  # type: ignore[override]
        raw = self._store.get(key)
        if raw is None:
            return None
        return json.loads(raw)


def _spend_counter_store() -> SpendCounterStore:
    return SpendCounterStore(None, handle=lambda client, _: FakeSpendCounters(redis=None if client is None else {}))


@contextmanager
def _patched_init_cache(litellm_settings: dict, cache_params: dict, user_cache: DualCache | None = None):
    """
    Context manager that:
    1. Replaces the module-level globals with fresh DualCache instances.
    2. Patches ``litellm.Cache`` (locally imported inside _init_cache) so
       it returns a fake cache whose ``.cache`` attribute is a
       _FakeRedisCache (passes the isinstance guard in _init_cache).
    3. Extracts enable_redis_auth_cache from litellm_settings and passes it
       as the second argument to _init_cache (matching production behaviour).
    4. Yields (user_api_key_cache, spend_counter_store, cli_sso_session_cache)
       after calling _init_cache, then restores everything.
    """
    fake_redis = _FakeRedisCache()

    mock_litellm_cache = MagicMock()
    mock_litellm_cache.cache = fake_redis

    fresh_user_cache = DualCache() if user_cache is None else user_cache
    fresh_spend_store = _spend_counter_store()
    fresh_cli_sso_cache = DualCache()
    fresh_config_cache = DualCache()

    enable_redis_auth_cache = litellm_settings.get("enable_redis_auth_cache", False)

    with (
        patch.object(ps, "user_api_key_cache", fresh_user_cache),
        patch.object(ps, "spend_counter_cache", DualCache()),
        patch.object(ps, "spend_counter_store", fresh_spend_store),
        patch.object(ps, "cli_sso_session_cache", fresh_cli_sso_cache),
        patch.object(ps, "saml_sso_cache", DualCache()),
        patch.object(ps, "litellm_config_cache", fresh_config_cache),
        patch.object(ps, "llm_router", None),
        # Cache is locally imported inside _init_cache: patch it at source.
        patch("litellm.Cache", return_value=mock_litellm_cache),
    ):
        litellm.cache = None
        ps.ProxyConfig()._init_cache(cache_params, enable_redis_auth_cache)
        yield fresh_user_cache, fresh_spend_store, fresh_cli_sso_cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRedisAuthCacheFlag:
    @pytest.mark.asyncio
    async def test_flag_true_moves_the_whole_auth_keyspace_onto_the_native_store(self):
        store = FakeAuthCache()
        with _patched_init_cache(
            litellm_settings={"enable_redis_auth_cache": True},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
            user_cache=UserApiKeyCache(
                default_in_memory_ttl=60, auth_cache_handle=store.factory, redis_client=lambda _: None
            ),
        ) as (user_cache, _, _cli_sso_cache):
            assert isinstance(user_cache, UserApiKeyCache)
            await user_cache.async_set_cache("team_id:t1", {"team_id": "t1"})
            assert await user_cache.async_get_cache("team_id:t1") == {"team_id": "t1"}

            assert user_cache.redis_cache is None, "redis-py must never read or write the auth keyspace"
            assert isinstance(user_cache.redis_connection, _FakeRedisCache)
            assert store.redis == {"team_id:t1": {"team_id": "t1"}}
            assert [settings.memory_ttl for settings in store.settings] == [60]

    @pytest.mark.asyncio
    async def test_flag_false_keeps_the_auth_cache_in_python_memory_only(self):
        store = FakeAuthCache()
        with _patched_init_cache(
            litellm_settings={"enable_redis_auth_cache": False},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
            user_cache=UserApiKeyCache(
                default_in_memory_ttl=60, auth_cache_handle=store.factory, redis_client=lambda _: None
            ),
        ) as (user_cache, _, _cli_sso_cache):
            assert isinstance(user_cache, UserApiKeyCache)
            await user_cache.async_set_cache("team_id:t1", {"team_id": "t1"})

            assert not user_cache.has_redis and user_cache.redis_connection is None
            assert user_cache.in_memory_cache.get_cache("team_id:t1") == {"team_id": "t1"}
            assert store.settings == [] and store.redis == {}

    def test_flag_true_attaches_redis_to_user_api_key_cache(self):
        """When enable_redis_auth_cache=True, user_api_key_cache.redis_cache must be set."""
        with _patched_init_cache(
            litellm_settings={"enable_redis_auth_cache": True},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
        ) as (user_cache, _, _cli_sso_cache):
            assert user_cache.redis_cache is not None, (
                "Redis should be attached to user_api_key_cache when enable_redis_auth_cache=True"
            )

    def test_flag_false_leaves_user_api_key_cache_in_memory_only(self):
        """When enable_redis_auth_cache=False, user_api_key_cache must stay in-memory."""
        with _patched_init_cache(
            litellm_settings={"enable_redis_auth_cache": False},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
        ) as (user_cache, _, _cli_sso_cache):
            assert user_cache.redis_cache is None, (
                "user_api_key_cache must remain in-memory-only when enable_redis_auth_cache=False"
            )

    def test_flag_absent_leaves_user_api_key_cache_in_memory_only(self):
        """When enable_redis_auth_cache is not set at all, default is in-memory-only."""
        with _patched_init_cache(
            litellm_settings={},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
        ) as (user_cache, _, _cli_sso_cache):
            assert user_cache.redis_cache is None, (
                "user_api_key_cache must remain in-memory-only when "
                "enable_redis_auth_cache is absent from litellm_settings"
            )

    def test_spend_counter_store_always_gets_redis_regardless_of_flag(self):
        """The spend counter store must receive Redis regardless of the auth-cache flag."""
        for flag_value in (True, False, None):
            ls = {"enable_redis_auth_cache": flag_value} if flag_value is not None else {}
            with _patched_init_cache(
                litellm_settings=ls,
                cache_params={"type": "redis", "host": "localhost", "port": 6379},
            ) as (_, spend_store, _cli_sso_cache):
                assert spend_store.has_redis, (
                    f"spend_counter_store must always get Redis (enable_redis_auth_cache={flag_value!r})"
                )

    def test_flag_false_spend_gets_redis_but_user_cache_does_not(self):
        """Explicit False: spend cache wired, auth cache left in-memory."""
        with _patched_init_cache(
            litellm_settings={"enable_redis_auth_cache": False},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
        ) as (user_cache, spend_store, _cli_sso_cache):
            assert spend_store.has_redis
            assert user_cache.redis_cache is None

    def test_cli_sso_session_cache_always_gets_redis_regardless_of_flag(self):
        """
        cli_sso_session_cache must receive Redis regardless of the auth-cache
        flag so that `lite login` works on multi-worker deployments without
        enable_redis_auth_cache (regression for the CLI SSO "Invalid CLI login
        session" bug)
        """
        for flag_value in (True, False, None):
            ls = {"enable_redis_auth_cache": flag_value} if flag_value is not None else {}
            with _patched_init_cache(
                litellm_settings=ls,
                cache_params={"type": "redis", "host": "localhost", "port": 6379},
            ) as (_, _, cli_sso_cache):
                assert cli_sso_cache.redis_cache is not None, (
                    f"cli_sso_session_cache must always get Redis (enable_redis_auth_cache={flag_value!r})"
                )

    def test_flag_absent_still_shares_the_model_budget_counters_over_redis(self):
        """
        Per-model budget counters are spend counters: the limiter must be able to
        push and read them through Redis without the auth-cache opt-in, or every
        worker enforces and reports its own share of a key's spend
        """
        fake_redis = _FakeRedisCache()
        limiter_cache = ps.model_max_budget_limiter.dual_cache
        touched_caches = (
            limiter_cache,
            ps.spend_counter_cache,
            ps.cli_sso_session_cache,
            ps.saml_sso_cache,
            ps.user_api_key_cache,
            ps.litellm_config_cache,
        )
        with ExitStack() as detached:
            detached.enter_context(patch.object(ps, "spend_counter_store", _spend_counter_store()))
            for cache in touched_caches:
                detached.enter_context(patch.object(cache, "redis_cache", None))
            ps._attach_redis_usage_cache(fake_redis, enable_redis_auth_cache=False)
            assert limiter_cache.redis_cache is fake_redis
            assert ps.user_api_key_cache.redis_cache is None


@pytest.mark.parametrize(
    "litellm_settings",
    [{"enable_redis_auth_cache": True}, {"enable_redis_auth_cache": False}, {}],
    ids=["flag-true", "flag-false", "flag-absent"],
)
def test_saml_sso_cache_always_gets_redis_regardless_of_flag(litellm_settings: dict[str, bool]) -> None:
    with _patched_init_cache(
        litellm_settings=litellm_settings,
        cache_params={"type": "redis", "host": "localhost", "port": 6379},
    ):
        assert ps.saml_sso_cache.redis_cache is not None
