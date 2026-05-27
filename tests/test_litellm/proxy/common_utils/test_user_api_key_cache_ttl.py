"""Regression coverage for LIT-3338.

`general_settings.user_api_key_cache_ttl: 300` was silently capped at 60s
because management-object cache writes pass an explicit
`ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL` that shadowed
`UserApiKeyCache.default_in_memory_ttl` (the value populated from the user's
config at proxy startup). The fix is in `UserApiKeyCache.async_set_cache`:
when the explicit `ttl` matches the historical management-object default,
a different default is configured, AND the write is Pydantic-typed (i.e.
``model_type=`` was passed), promote the configured value. The
``model_type`` gate is what keeps short-lived non-management writes (e.g.
single-use SSO login codes that also happen to use ``ttl=60``) on their
intended TTL.
"""
import time

import pytest

from litellm.constants import DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL
from litellm.proxy._types import UserAPIKeyAuth
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
    """user_api_key_cache_ttl=300 -> Pydantic-typed write with ttl=60 lands at ~300s."""
    cache = _make_configured_cache(300)
    t0 = time.time()
    await cache.async_set_cache(
        key="hashed-token",
        value=UserAPIKeyAuth(token="sk-x", api_key="sk-x"),
        model_type=UserAPIKeyAuth,
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = cache.in_memory_cache.ttl_dict["hashed-token"] - t0
    assert 295 <= observed <= 305, (
        f"expected ~300s TTL (configured user_api_key_cache_ttl), got {observed:.1f}s "
        f"(LIT-3338 regression)"
    )


@pytest.mark.asyncio
async def test_non_management_ttl_not_overridden():
    """Arbitrary explicit ttl (not the management default) is respected verbatim."""
    cache = _make_configured_cache(300)
    t0 = time.time()
    await cache.async_set_cache(
        key="other-key",
        value=UserAPIKeyAuth(token="sk-y", api_key="sk-y"),
        model_type=UserAPIKeyAuth,
        ttl=10,
    )
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
        value=UserAPIKeyAuth(token="sk-z", api_key="sk-z"),
        model_type=UserAPIKeyAuth,
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
        value=UserAPIKeyAuth(token="sk-a", api_key="sk-a"),
        model_type=UserAPIKeyAuth,
        ttl=DEFAULT_MANAGEMENT_OBJECT_IN_MEMORY_CACHE_TTL,
    )
    observed = cache.in_memory_cache.ttl_dict["match-key"] - t0
    assert 55 <= observed <= 65, f"got {observed:.1f}s, expected ~60s"


@pytest.mark.asyncio
async def test_no_promotion_for_writes_without_model_type():
    """LIT-3338 (Greptile follow-up): a non-management write that uses ttl=60
    as a bare literal MUST NOT be promoted to the configured cache TTL.

    Pins the security-sensitive case Greptile flagged: short-lived single-use
    login codes (`proxy_server.py:login_v3`, `ui_sso.py`) write to the same
    `user_api_key_cache` with `ttl=60` and advertise `expires_in: 60` to the
    client. Their lifetime must stay at 60s, even when the operator has
    bumped `user_api_key_cache_ttl` to 300s for the auth-cache hot path.
    """
    cache = _make_configured_cache(300)
    t0 = time.time()
    # Mirrors the proxy_server.login_v3 / ui_sso login-code write: no model_type.
    await cache.async_set_cache(
        key="login_code:xyz",
        value={"token": "jwt", "redirect_url": "/ui/?login=success"},
        ttl=60,
    )
    observed = cache.in_memory_cache.ttl_dict["login_code:xyz"] - t0
    assert 55 <= observed <= 65, (
        f"non-management ttl=60 write was unexpectedly promoted to {observed:.1f}s "
        f"(LIT-3338 over-match regression)"
    )
