"""
Tests for the enable_redis_auth_cache litellm_settings flag.

Verifies that _init_cache attaches Redis to user_api_key_cache only when
the flag is explicitly set to True, and leaves it in-memory-only otherwise.
"""

from contextlib import contextmanager
import json
from unittest.mock import MagicMock, patch

import pytest

import litellm
import litellm.proxy.proxy_server as ps
from litellm.caching.caching import RedisCache
from litellm.caching.dual_cache import DualCache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeRedisCache(RedisCache):
    """
    Minimal RedisCache subclass that passes isinstance checks without
    requiring a real Redis connection.  __init__ is bypassed so no
    network calls are made.
    """

    def __init__(self):  # noqa: super().__init__ skipped intentionally
        self._store = {}

    def set_cache(self, key, value, **kwargs):  # type: ignore[override]
        # Enforce Redis JSON-serializable payload contract.
        self._store[key] = json.dumps(value)
        return True

    def get_cache(self, key, **kwargs):  # type: ignore[override]
        raw = self._store.get(key)
        if raw is None:
            return None
        return json.loads(raw)


@contextmanager
def _patched_init_cache(litellm_settings: dict, cache_params: dict):
    """
    Context manager that:
    1. Replaces the module-level globals with fresh DualCache instances.
    2. Patches ``litellm.Cache`` (locally imported inside _init_cache) so
       it returns a fake cache whose ``.cache`` attribute is a
       _FakeRedisCache (passes the isinstance guard in _init_cache).
    3. Extracts enable_redis_auth_cache from litellm_settings and passes it
       as the second argument to _init_cache (matching production behaviour).
    4. Yields (user_api_key_cache, spend_counter_cache) after calling
       _init_cache, then restores everything.
    """
    fake_redis = _FakeRedisCache()

    mock_litellm_cache = MagicMock()
    mock_litellm_cache.cache = fake_redis

    fresh_user_cache = DualCache()
    fresh_spend_cache = DualCache()

    enable_redis_auth_cache = litellm_settings.get("enable_redis_auth_cache", False)

    with (
        patch.object(ps, "user_api_key_cache", fresh_user_cache),
        patch.object(ps, "spend_counter_cache", fresh_spend_cache),
        patch.object(ps, "llm_router", None),
        # Cache is locally imported inside _init_cache: patch it at source.
        patch("litellm.Cache", return_value=mock_litellm_cache),
    ):
        litellm.cache = None
        ps.ProxyConfig()._init_cache(cache_params, enable_redis_auth_cache)
        yield fresh_user_cache, fresh_spend_cache


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRedisAuthCacheFlag:
    def test_flag_true_attaches_redis_to_user_api_key_cache(self):
        """When enable_redis_auth_cache=True, user_api_key_cache.redis_cache must be set."""
        with _patched_init_cache(
            litellm_settings={"enable_redis_auth_cache": True},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
        ) as (user_cache, _):
            assert user_cache.redis_cache is not None, (
                "Redis should be attached to user_api_key_cache when "
                "enable_redis_auth_cache=True"
            )

    def test_flag_false_leaves_user_api_key_cache_in_memory_only(self):
        """When enable_redis_auth_cache=False, user_api_key_cache must stay in-memory."""
        with _patched_init_cache(
            litellm_settings={"enable_redis_auth_cache": False},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
        ) as (user_cache, _):
            assert user_cache.redis_cache is None, (
                "user_api_key_cache must remain in-memory-only when "
                "enable_redis_auth_cache=False"
            )

    def test_flag_absent_leaves_user_api_key_cache_in_memory_only(self):
        """When enable_redis_auth_cache is not set at all, default is in-memory-only."""
        with _patched_init_cache(
            litellm_settings={},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
        ) as (user_cache, _):
            assert user_cache.redis_cache is None, (
                "user_api_key_cache must remain in-memory-only when "
                "enable_redis_auth_cache is absent from litellm_settings"
            )

    def test_spend_counter_cache_always_gets_redis_regardless_of_flag(self):
        """spend_counter_cache must receive Redis regardless of the auth-cache flag."""
        for flag_value in (True, False, None):
            ls = (
                {"enable_redis_auth_cache": flag_value}
                if flag_value is not None
                else {}
            )
            with _patched_init_cache(
                litellm_settings=ls,
                cache_params={"type": "redis", "host": "localhost", "port": 6379},
            ) as (_, spend_cache):
                assert spend_cache.redis_cache is not None, (
                    f"spend_counter_cache must always get Redis "
                    f"(enable_redis_auth_cache={flag_value!r})"
                )

    def test_flag_false_spend_gets_redis_but_user_cache_does_not(self):
        """Explicit False: spend cache wired, auth cache left in-memory."""
        with _patched_init_cache(
            litellm_settings={"enable_redis_auth_cache": False},
            cache_params={"type": "redis", "host": "localhost", "port": 6379},
        ) as (user_cache, spend_cache):
            assert spend_cache.redis_cache is not None
            assert user_cache.redis_cache is None
