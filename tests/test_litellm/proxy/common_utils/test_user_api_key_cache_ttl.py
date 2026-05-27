"""
Regression tests for LIT-3338 - general_settings.user_api_key_cache_ttl was
ignored for API key auth because auth_checks.py / handle_jwt.py /
mcp_server_manager.py explicitly pass
ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL (60s) on every management-
object cache write, which beat the cache instance's configured
default_in_memory_ttl.

The fix promotes the configured TTL when the caller passes the default sentinel
and the cache has a different configured default. Other explicit TTLs are
untouched.
"""
import asyncio
import time

from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


def _ttl_stored(cache: UserApiKeyCache, key: str) -> float:
    return cache.in_memory_cache.ttl_dict[key] - time.time()


def _make_configured_cache(configured_ttl: float) -> UserApiKeyCache:
    cache = UserApiKeyCache()
    cache.update_cache_ttl(
        default_in_memory_ttl=configured_ttl,
        default_redis_ttl=configured_ttl,
    )
    return cache


class TestUserApiKeyCacheTTLHonoured:
    def test_async_set_cache_promotes_configured_over_default_sentinel(self):
        cache = _make_configured_cache(300.0)
        asyncio.run(
            cache.async_set_cache(
                key="sk-hash-async",
                value={"token": "abc"},
                ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
            )
        )
        assert 295 <= _ttl_stored(cache, "sk-hash-async") <= 305

    def test_sync_set_cache_promotes_configured_over_default_sentinel(self):
        cache = _make_configured_cache(300.0)
        cache.set_cache(
            key="sk-hash-sync",
            value={"token": "abc"},
            ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
        )
        assert 295 <= _ttl_stored(cache, "sk-hash-sync") <= 305

    def test_no_configured_ttl_leaves_default_alone(self):
        cache = UserApiKeyCache()
        asyncio.run(
            cache.async_set_cache(
                key="k",
                value="v",
                ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
            )
        )
        assert 55 <= _ttl_stored(cache, "k") <= 65

    def test_explicit_non_default_ttl_is_passthrough(self):
        cache = _make_configured_cache(300.0)
        asyncio.run(cache.async_set_cache(key="k", value="v", ttl=120))
        assert 115 <= _ttl_stored(cache, "k") <= 125

    def test_configured_ttl_equal_to_sentinel_is_noop(self):
        cache = _make_configured_cache(
            float(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)
        )
        asyncio.run(
            cache.async_set_cache(
                key="k",
                value="v",
                ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
            )
        )
        assert 55 <= _ttl_stored(cache, "k") <= 65

    def test_ttl_none_in_kwargs_is_passthrough(self):
        cache = _make_configured_cache(300.0)
        kwargs = {"ttl": None}
        cache._resolve_management_object_ttl(kwargs)
        assert kwargs == {"ttl": None}

    def test_no_ttl_key_in_kwargs(self):
        cache = _make_configured_cache(300.0)
        kwargs: dict = {}
        cache._resolve_management_object_ttl(kwargs)
        assert kwargs == {}

    def test_invalid_ttl_string_is_passthrough(self):
        cache = _make_configured_cache(300.0)
        kwargs = {"ttl": "not-a-number"}
        cache._resolve_management_object_ttl(kwargs)
        assert kwargs == {"ttl": "not-a-number"}

    def test_default_in_memory_ttl_none_is_noop(self):
        cache = UserApiKeyCache()
        cache.default_in_memory_ttl = None  # type: ignore[assignment]
        kwargs = {"ttl": float(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)}
        cache._resolve_management_object_ttl(kwargs)
        assert kwargs == {"ttl": float(DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)}

    def test_model_type_path_still_honors_promotion(self):
        from pydantic import BaseModel

        class Dummy(BaseModel):
            foo: str = "bar"

        cache = _make_configured_cache(300.0)
        cache.set_cache(
            key="sk-hash-model",
            value=Dummy(),
            ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
            model_type=Dummy,
        )
        assert 295 <= _ttl_stored(cache, "sk-hash-model") <= 305
    def test_pipeline_promotes_configured_over_default_sentinel(self):
        cache = _make_configured_cache(300.0)
        asyncio.run(
            cache.async_set_cache_pipeline(
                cache_list=[("p1", "v1"), ("p2", "v2")],
                ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
            )
        )
        for k in ("p1", "p2"):
            assert 295 <= _ttl_stored(cache, k) <= 305

    def test_pipeline_explicit_non_default_ttl_is_passthrough(self):
        cache = _make_configured_cache(300.0)
        asyncio.run(
            cache.async_set_cache_pipeline(
                cache_list=[("p1", "v1")],
                ttl=120,
            )
        )
        assert 115 <= _ttl_stored(cache, "p1") <= 125
