"""
Tests for the enable_redis_auth_cache litellm_settings flag.

Verifies that _init_cache attaches Redis to user_api_key_cache and the proxy state caches only when
the flag is explicitly set to True, and leaves them in-memory-only otherwise.
"""

import json
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Final
from unittest.mock import MagicMock, patch

import pytest

import litellm
import litellm.proxy.proxy_server as ps
from litellm.caching.caching import RedisCache
from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.store_settings import RedisStoreSettings
from litellm.redis.stores.auth_cache import AuthObject
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
        self.store_settings = RedisStoreSettings(
            redis_kwargs={"host": "127.0.0.1", "port": 1, "socket_timeout": 0.2},
            namespace=None,
            default_ttl=60,
            breaker=None,
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
    return FakeSpendCounters(redis=None).store()


_STATE_CACHES: Final = ("hook_cache", "mcp_state_cache", "jwks_cache", "ui_settings_cache")


@dataclass(frozen=True, slots=True)
class _Wired:
    redis: _FakeRedisCache
    user_cache: UserApiKeyCache
    auth_store: FakeAuthCache
    spend_store: SpendCounterStore
    cli_sso_cache: DualCache
    state_caches: tuple[DualCache, ...]


@contextmanager
def _patched_init_cache(
    litellm_settings: dict[str, object], cache_params: dict[str, object], auth_store: FakeAuthCache | None = None
) -> Iterator[_Wired]:
    """
    Replaces the module-level caches with fresh ones, patches ``litellm.Cache`` (locally imported inside
    _init_cache) so it returns a fake cache whose ``.cache`` is a _FakeRedisCache, and calls _init_cache with the
    flag the way production does.
    """
    fake_redis = _FakeRedisCache()

    mock_litellm_cache = MagicMock()
    mock_litellm_cache.cache = fake_redis

    store: Final = auth_store if auth_store is not None else FakeAuthCache(attached=False)
    wired: Final = _Wired(
        redis=fake_redis,
        user_cache=UserApiKeyCache(default_in_memory_ttl=60, auth_cache=store.store),
        auth_store=store,
        spend_store=_spend_counter_store(),
        cli_sso_cache=DualCache(),
        state_caches=tuple(DualCache() for _ in _STATE_CACHES),
    )

    enable_redis_auth_cache = litellm_settings.get("enable_redis_auth_cache", False)

    with ExitStack() as patched:
        for name, cache in zip(_STATE_CACHES, wired.state_caches, strict=True):
            patched.enter_context(patch.object(ps, name, cache))
        patched.enter_context(patch.object(ps, "user_api_key_cache", wired.user_cache))
        patched.enter_context(patch.object(ps, "spend_counter_cache", DualCache()))
        patched.enter_context(patch.object(ps, "spend_counter_store", wired.spend_store))
        patched.enter_context(patch.object(ps, "cli_sso_session_cache", wired.cli_sso_cache))
        patched.enter_context(patch.object(ps, "saml_sso_cache", DualCache()))
        patched.enter_context(patch.object(ps, "litellm_config_cache", DualCache()))
        patched.enter_context(patch.object(ps, "llm_router", None))
        patched.enter_context(patch("litellm.Cache", return_value=mock_litellm_cache))
        litellm.cache = None
        ps.ProxyConfig()._init_cache(cache_params, enable_redis_auth_cache)
        yield wired


_REDIS_PARAMS: Final = {"type": "redis", "host": "localhost", "port": 6379}


class TestRedisAuthCacheFlag:
    @pytest.mark.asyncio
    async def test_flag_true_moves_the_whole_auth_keyspace_onto_the_native_store(self):
        with _patched_init_cache({"enable_redis_auth_cache": True}, _REDIS_PARAMS) as wired:
            await wired.user_cache.async_set_cache(AuthObject.team("t1"), {"team_id": "t1"})
            assert await wired.user_cache.async_get_cache(AuthObject.team("t1")) == {"team_id": "t1"}

            assert wired.user_cache.has_redis
            assert wired.user_cache.redis_connection is wired.redis
            assert wired.auth_store.redis == {"team_id:t1": {"team_id": "t1"}}
            assert [settings.memory_ttl for settings in wired.auth_store.settings] == [60]

    @pytest.mark.parametrize(
        "litellm_settings",
        [{"enable_redis_auth_cache": False}, {}],
        ids=["flag-false", "flag-absent"],
    )
    @pytest.mark.asyncio
    async def test_without_the_flag_the_auth_cache_stays_in_this_workers_memory(
        self, litellm_settings: dict[str, object]
    ):
        with _patched_init_cache(litellm_settings, _REDIS_PARAMS) as wired:
            await wired.user_cache.async_set_cache(AuthObject.team("t1"), {"team_id": "t1"})

            assert not wired.user_cache.has_redis and wired.user_cache.redis_connection is None
            assert wired.user_cache.get_local(AuthObject.team("t1")) == {"team_id": "t1"}
            assert not wired.auth_store.attached

    @pytest.mark.parametrize(
        ("litellm_settings", "shared"),
        [({"enable_redis_auth_cache": True}, True), ({"enable_redis_auth_cache": False}, False), ({}, False)],
        ids=["flag-true", "flag-false", "flag-absent"],
    )
    def test_the_proxy_state_caches_share_redis_exactly_when_the_auth_cache_does(
        self, litellm_settings: dict[str, object], shared: bool
    ):
        """Hook state, MCP state, JWKS keys and UI settings used to live in user_api_key_cache, so they keep
        following its opt-in: shared across workers with the flag, per-worker without it."""
        with _patched_init_cache(litellm_settings, _REDIS_PARAMS) as wired:
            assert wired.user_cache.has_redis is shared
            assert [cache.redis_cache is wired.redis for cache in wired.state_caches] == [shared] * len(_STATE_CACHES)

    def test_spend_counter_store_always_gets_redis_regardless_of_flag(self):
        """The spend counter store must receive Redis regardless of the auth-cache flag."""
        for flag_value in (True, False, None):
            ls = {"enable_redis_auth_cache": flag_value} if flag_value is not None else {}
            with _patched_init_cache(ls, _REDIS_PARAMS) as wired:
                assert wired.spend_store.has_redis, (
                    f"spend_counter_store must always get Redis (enable_redis_auth_cache={flag_value!r})"
                )

    def test_flag_false_spend_gets_redis_but_user_cache_does_not(self):
        """Explicit False: spend cache wired, auth cache left in-memory."""
        with _patched_init_cache({"enable_redis_auth_cache": False}, _REDIS_PARAMS) as wired:
            assert wired.spend_store.has_redis
            assert not wired.user_cache.has_redis

    def test_cli_sso_session_cache_always_gets_redis_regardless_of_flag(self):
        """
        cli_sso_session_cache must receive Redis regardless of the auth-cache
        flag so that `lite login` works on multi-worker deployments without
        enable_redis_auth_cache (regression for the CLI SSO "Invalid CLI login
        session" bug)
        """
        for flag_value in (True, False, None):
            ls = {"enable_redis_auth_cache": flag_value} if flag_value is not None else {}
            with _patched_init_cache(ls, _REDIS_PARAMS) as wired:
                assert wired.cli_sso_cache.redis_cache is not None, (
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
        user_cache: Final = UserApiKeyCache(auth_cache=FakeAuthCache(attached=False).store)
        state_caches: Final = tuple(DualCache() for _ in _STATE_CACHES)
        touched_caches = (
            limiter_cache,
            ps.spend_counter_cache,
            ps.cli_sso_session_cache,
            ps.saml_sso_cache,
            ps.litellm_config_cache,
        )
        with ExitStack() as detached:
            detached.enter_context(patch.object(ps, "spend_counter_store", _spend_counter_store()))
            detached.enter_context(patch.object(ps, "user_api_key_cache", user_cache))
            for name, cache in zip(_STATE_CACHES, state_caches, strict=True):
                detached.enter_context(patch.object(ps, name, cache))
            for cache in touched_caches:
                detached.enter_context(patch.object(cache, "redis_cache", None))
            ps._attach_redis_usage_cache(fake_redis, enable_redis_auth_cache=False)
            assert limiter_cache.redis_cache is fake_redis
            assert not user_cache.has_redis
            assert all(cache.redis_cache is None for cache in state_caches)


@pytest.mark.parametrize(
    "litellm_settings",
    [{"enable_redis_auth_cache": True}, {"enable_redis_auth_cache": False}, {}],
    ids=["flag-true", "flag-false", "flag-absent"],
)
def test_saml_sso_cache_always_gets_redis_regardless_of_flag(litellm_settings: dict[str, bool]) -> None:
    with _patched_init_cache(litellm_settings, _REDIS_PARAMS):
        assert ps.saml_sso_cache.redis_cache is not None
