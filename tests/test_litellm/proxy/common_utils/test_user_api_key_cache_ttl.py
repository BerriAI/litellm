"""Regression coverage for LIT-3338.

`general_settings.user_api_key_cache_ttl: 300` was silently capped at 60s
because management-object cache writes pass an explicit
`ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL` that shadowed
`UserApiKeyCache.default_in_memory_ttl` (the value populated from the user's
config at proxy startup). The fix is in `UserApiKeyCache.async_set_cache`:
when the explicit `ttl` matches the historical management-object default
and a different default is configured, promote the configured value.
"""
import time

import pytest

from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


def _make_configured_cache(ttl_seconds):
    """Mirror what proxy_server.py does when user_api_key_cache_ttl is set."""
    cache = UserApiKeyCache()
    cache.update_cache_ttl(
        default_in_memory_ttl=ttl_seconds,
        default_redis_ttl=ttl_seconds,
    )
    return cache


@pytest.mark.asyncio
async def test_management_ttl_promoted_to_configured_value():
    """When user_api_key_cache_ttl=300, a write with ttl=60 should land at ~300s."""
    cache = _make_configured_cache(300)
    t0 = time.time()
    await cache.async_set_cache(
        key="hashed-token",
        value={"x": 1},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = cache.in_memory_cache.ttl_dict["hashed-token"] - t0
    assert 295 <= observed <= 305, (
        f"expected ~300s TTL (configured user_api_key_cache_ttl), got {observed:.1f}s "
        f"(LIT-3338 regression)"
    )


@pytest.mark.asyncio
async def test_non_management_ttl_not_overridden():
    """An arbitrary explicit ttl (not the management default) must be respected verbatim."""
    cache = _make_configured_cache(300)
    t0 = time.time()
    await cache.async_set_cache(key="other-key", value={"x": 2}, ttl=10)
    observed = cache.in_memory_cache.ttl_dict["other-key"] - t0
    assert 8 <= observed <= 12, f"expected ~10s TTL (respected), got {observed:.1f}s"


@pytest.mark.asyncio
async def test_default_60s_preserved_when_unconfigured():
    """No user_api_key_cache_ttl set -> ttl=60 still wins (backward compat)."""
    cache = UserApiKeyCache()
    assert cache.default_in_memory_ttl is None
    t0 = time.time()
    await cache.async_set_cache(
        key="default-key",
        value={"x": 3},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = cache.in_memory_cache.ttl_dict["default-key"] - t0
    assert 55 <= observed <= 65, (
        f"unconfigured cache should keep the 60s default, got {observed:.1f}s"
    )


@pytest.mark.asyncio
async def test_no_promotion_when_configured_value_matches_management_default():
    """Edge case: if the user happens to configure user_api_key_cache_ttl=60,
    promotion is a no-op (still 60s, same as before)."""
    cache = _make_configured_cache(60)
    t0 = time.time()
    await cache.async_set_cache(
        key="match-key",
        value={"x": 4},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = cache.in_memory_cache.ttl_dict["match-key"] - t0
    assert 55 <= observed <= 65, f"got {observed:.1f}s, expected ~60s"
