"""LIT-3338: ``general_settings.user_api_key_cache_ttl`` must be honoured for the
management-object cache writes from ``auth_checks.py`` (key, team, user, budget,
vector store, permission).

Before the fix, those call sites passed an explicit
``ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL`` (60s) that silently capped
the user's configured ``user_api_key_cache_ttl`` at 60s, even though
``proxy_server.py`` had already pushed the user value into the cache via
``update_cache_ttl``.

The fix in ``UserApiKeyCache`` treats the constant as a sentinel meaning "use
the cache's configured default" and substitutes ``default_in_memory_ttl`` when
the user has overridden it.
"""

from __future__ import annotations

import pytest

from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


def _make_cache(configured_ttl):
    """Build a UserApiKeyCache the way ``proxy_server.py`` does at startup."""
    cache = UserApiKeyCache(default_in_memory_ttl=60)
    if configured_ttl is not None:
        cache.update_cache_ttl(
            default_in_memory_ttl=configured_ttl,
            default_redis_ttl=configured_ttl,
        )
    return cache


@pytest.mark.asyncio
async def test_configured_ttl_overrides_management_object_constant():
    """``user_api_key_cache_ttl: 300`` -> entries written with the management
    constant honour the configured 300s TTL."""
    import time

    cache = _make_cache(configured_ttl=300)

    before = time.time()
    await cache.async_set_cache(
        key="lit-3338-key-1",
        value={"ok": True},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    expiry = cache.in_memory_cache.ttl_dict.get("lit-3338-key-1")
    assert expiry is not None, "value was not cached in-memory"
    ttl_actual = expiry - before
    assert 295 <= ttl_actual <= 305, (
        f"expected ~300s TTL, got {ttl_actual:.1f}s (LIT-3338 regression)"
    )


@pytest.mark.asyncio
async def test_default_ttl_unchanged_when_unconfigured():
    """No ``user_api_key_cache_ttl`` set -> entries keep the historical 60s TTL.
    Regression guard against accidentally widening the cache for everyone."""
    import time

    cache = _make_cache(configured_ttl=None)

    before = time.time()
    await cache.async_set_cache(
        key="lit-3338-default-key",
        value={"ok": True},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    ttl_actual = cache.in_memory_cache.ttl_dict["lit-3338-default-key"] - before
    assert 55 <= ttl_actual <= 65, (
        f"default 60s TTL drifted to {ttl_actual:.1f}s (LIT-3338 regression)"
    )


@pytest.mark.asyncio
async def test_explicit_non_management_ttl_is_not_coerced():
    """Other callers passing an explicit ttl that is NOT the management constant
    must be passed through untouched."""
    import time

    cache = _make_cache(configured_ttl=300)

    before = time.time()
    await cache.async_set_cache(
        key="lit-3338-explicit-30",
        value={"ok": True},
        ttl=30,
    )
    ttl_actual = cache.in_memory_cache.ttl_dict["lit-3338-explicit-30"] - before
    assert 25 <= ttl_actual <= 35, (
        f"explicit ttl=30 was coerced to {ttl_actual:.1f}s -- caller intent broken"
    )


def test_sync_set_cache_also_coerces_management_object_constant():
    """Sync path must mirror async path so callers that use ``set_cache``
    receive the same TTL coercion."""
    import time

    cache = _make_cache(configured_ttl=300)

    before = time.time()
    cache.set_cache(
        key="lit-3338-sync-key",
        value={"ok": True},
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    ttl_actual = cache.in_memory_cache.ttl_dict["lit-3338-sync-key"] - before
    assert 295 <= ttl_actual <= 305, (
        f"sync path didn't honour configured TTL: got {ttl_actual:.1f}s"
    )
