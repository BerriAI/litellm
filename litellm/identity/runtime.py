"""Process-wide accessor for the shared ``IdentityCache``.

The proxy already owns one ``DualCache`` for caller identity at
``litellm.proxy.proxy_server.user_api_key_cache``. We layer an
``IdentityCache`` on top of it so the new identity load path shares a
single in-memory/Redis backend with the legacy caches. This avoids
double-caching on a single deploy and keeps invalidation surfaces
aligned.

Off-proxy callers (CLI, tests) can pass their own ``DualCache``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from litellm.identity.cache import IdentityCache

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache


_identity_cache: Optional[IdentityCache] = None


def get_identity_cache(
    dual_cache: Optional["DualCache"] = None,
) -> IdentityCache:
    """Return the shared ``IdentityCache``, building it on first call.

    When ``dual_cache`` is omitted, the proxy's module-level cache is
    used. The first call wins; subsequent calls ignore the argument so
    that every consumer in a process sees the same instance.
    """
    global _identity_cache
    if _identity_cache is not None:
        return _identity_cache

    if dual_cache is None:
        from litellm.proxy.proxy_server import user_api_key_cache as _proxy_cache

        dual_cache = _proxy_cache

    _identity_cache = IdentityCache(dual_cache=dual_cache)
    return _identity_cache


def reset_identity_cache_for_tests() -> None:
    global _identity_cache
    _identity_cache = None
