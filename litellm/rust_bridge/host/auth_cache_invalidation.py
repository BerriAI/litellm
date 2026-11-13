"""Applies one auth cache invalidation to the caches that are still Python's, after Rust applied it to its own."""

from __future__ import annotations

from litellm.redis.stores.auth_cache_invalidation import PythonCaches


def apply(caches: PythonCaches, cache_key: str, new_value: float | None, ttl: float | None) -> None:
    caches.apply(cache_key, new_value, ttl)
