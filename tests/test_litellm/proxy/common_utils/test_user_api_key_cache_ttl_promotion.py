"""Regression tests for LIT-3338: user_api_key_cache_ttl promotion.

The proxy applies ``general_settings.user_api_key_cache_ttl`` at startup via
``user_api_key_cache.update_cache_ttl(default_in_memory_ttl=ttl, default_redis_ttl=ttl)``.
However every management-object writer in ``auth_checks.py``, ``handle_jwt.py``,
and the MCP server manager calls
``user_api_key_cache.async_set_cache(..., ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL)``
explicitly. ``DualCache.async_set_cache`` only falls back to ``default_in_memory_ttl``
when ``ttl`` is absent from kwargs, so the operator setting is silently shadowed.

The fix is a one-time TTL promotion inside ``UserApiKeyCache.async_set_cache``
(and the rarely-used sync ``set_cache``) that only fires when the explicit ttl
*exactly* equals ``DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` AND the cache
has been configured to a different default. Any other explicit ttl is respected
verbatim, and when no operator setting was applied behaviour is identical to
before.

These tests pin all four corners of that contract.
"""

from __future__ import annotations

import asyncio
import math
import time

import pytest

from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


def _observed_ttl(cache: UserApiKeyCache, key: str) -> float:
    """Return the live TTL stamped on the in-memory entry."""
    return cache.in_memory_cache.ttl_dict[key] - time.time()


@pytest.mark.asyncio
async def test_async_set_cache_promotes_management_ttl_to_configured_default():
    """The reported regression: user_api_key_cache_ttl=300 -> 60 silently."""
    cache = UserApiKeyCache()
    # Operator configured general_settings.user_api_key_cache_ttl: 300
    cache.update_cache_ttl(default_in_memory_ttl=300, default_redis_ttl=300)

    await cache.async_set_cache(
        key="hashed-token",
        value={"user_id": "alice"},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = _observed_ttl(cache, "hashed-token")
    # Should now be ~300s, not 60s
    assert observed > 250, (
        f"expected ttl ~300, got {observed} — management constant still shadowing user_api_key_cache_ttl"
    )
    assert observed < 310


@pytest.mark.asyncio
async def test_async_set_cache_does_not_touch_non_management_ttl():
    """A caller that explicitly chooses a non-management ttl must be respected."""
    cache = UserApiKeyCache()
    cache.update_cache_ttl(default_in_memory_ttl=300, default_redis_ttl=300)

    # Caller picks ttl=10 for its own reasons. We must NOT promote that.
    await cache.async_set_cache(key="other-key", value={"x": 1}, ttl=10)
    observed = _observed_ttl(cache, "other-key")
    assert math.isclose(observed, 10, abs_tol=1.5), f"expected ttl ~10, got {observed}"


@pytest.mark.asyncio
async def test_async_set_cache_preserves_management_ttl_when_unconfigured():
    """When user_api_key_cache_ttl is NOT configured, the 60s constant still wins.

    This guards backward compatibility — operators who never set the general
    setting must see exactly the same behaviour as before the fix.
    """
    cache = UserApiKeyCache()  # no update_cache_ttl call
    await cache.async_set_cache(
        key="key-no-config",
        value={"x": 1},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = _observed_ttl(cache, "key-no-config")
    assert math.isclose(observed, DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL, abs_tol=2), (
        f"expected ttl ~{DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL}, got {observed}"
    )


@pytest.mark.asyncio
async def test_async_set_cache_no_promotion_when_configured_equals_management_constant():
    """When user_api_key_cache_ttl happens to equal 60, promotion is a no-op."""
    cache = UserApiKeyCache()
    cache.update_cache_ttl(
        default_in_memory_ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
        default_redis_ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    await cache.async_set_cache(
        key="key-eq",
        value={"x": 1},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = _observed_ttl(cache, "key-eq")
    assert math.isclose(observed, DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL, abs_tol=2)


def test_sync_set_cache_also_promotes_management_ttl():
    """Sync set_cache shares the same regression and must be patched in lockstep."""
    cache = UserApiKeyCache()
    cache.update_cache_ttl(default_in_memory_ttl=300, default_redis_ttl=300)
    cache.set_cache(
        key="sync-key",
        value={"x": 1},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = _observed_ttl(cache, "sync-key")
    assert observed > 250, f"expected ttl ~300, got {observed}"
    assert observed < 310


@pytest.mark.asyncio
async def test_async_set_cache_no_promotion_when_ttl_is_none():
    """If the caller omits ttl entirely, DualCache already applies the default itself."""
    cache = UserApiKeyCache()
    cache.update_cache_ttl(default_in_memory_ttl=300, default_redis_ttl=300)
    await cache.async_set_cache(key="no-ttl-key", value={"x": 1})
    observed = _observed_ttl(cache, "no-ttl-key")
    # DualCache stamps default_in_memory_ttl when ttl is absent, so this should be ~300.
    assert observed > 250, f"expected ttl ~300, got {observed}"
