"""
``FakeRedisCache`` — a drop-in ``redis_cache`` stand-in for unit tests.

Purpose
-------
``DualCache`` accepts any ``BaseCache`` as its ``redis_cache`` parameter.
In production that is a ``RedisCache`` backed by a live Redis server; in
tests we want cross-pod sharing without network I/O, so we normally pass
an ``InMemoryCache`` instead.

The problem with a plain ``InMemoryCache`` is that it happily stores
Pydantic ``BaseModel`` instances, masking a class of bugs that only
surface in production:

* Real Redis serializes every value to bytes via JSON.  Pydantic models
  **cannot** be stored directly — they must first be converted to a
  JSON-safe dict (``CacheCodec.serialize`` in ``UserApiKeyCache`` does
  this).
* If a test accidentally stores a ``BaseModel`` in the plain
  ``InMemoryCache`` stand-in it still works in the test, but the
  production Redis write would fail or return a corrupted payload on read.

``FakeRedisCache`` subclasses ``InMemoryCache`` and adds a write-time
guard that raises ``TypeError`` when a caller tries to store a Pydantic
``BaseModel`` directly.  This means:

1. Tests that exercise the ``UserApiKeyCache`` codec path correctly (the
   normal case) continue to pass — ``CacheCodec.serialize`` converts
   models to dicts *before* reaching this layer.
2. Tests that bypass the codec and write raw models fail immediately with
   a clear message rather than silently succeeding.

Usage
-----
Pass ``FakeRedisCache()`` wherever you would use a real ``RedisCache`` in
a test, typically as the ``redis_cache`` argument to ``UserApiKeyCache``::

    from tests.test_litellm.proxy.auth.fake_redis_cache import FakeRedisCache

    cache = UserApiKeyCache(
        in_memory_cache=InMemoryCache(),
        redis_cache=FakeRedisCache(),   # type: ignore[arg-type]
    )
"""

from __future__ import annotations

from typing import Any, List, Tuple

from pydantic import BaseModel

from litellm.caching.in_memory_cache import InMemoryCache


def _reject_pydantic(key: str, value: Any) -> None:
    """
    Raise ``TypeError`` if ``value`` is a Pydantic ``BaseModel``.

    Real Redis does not accept Python objects — callers must serialise to
    a JSON-safe dict first.  ``UserApiKeyCache`` does this via
    ``CacheCodec.serialize``; anything else reaching this layer is a bug.
    """
    if isinstance(value, BaseModel):
        raise TypeError(
            f"FakeRedisCache: attempted to store a Pydantic model directly. "
            f"Redis stores only JSON-serialised dicts, never BaseModel instances. "
            f"key={key!r}, value_type={type(value).__name__}. "
            f"Ensure the caller serialises via CacheCodec.serialize (UserApiKeyCache "
            f"does this automatically) before writing to the redis_cache layer."
        )


class FakeRedisCache(InMemoryCache):
    """
    ``InMemoryCache`` that enforces Redis storage constraints.

    Behaviour differences from ``InMemoryCache``:

    * **Write guard** — ``set_cache``, ``async_set_cache``, and
      ``async_set_cache_pipeline`` raise ``TypeError`` if the value being
      stored is a Pydantic ``BaseModel``.  Only plain dicts, lists, and
      scalars (the same types Redis can round-trip through JSON) are
      accepted.

    * **Shared across pods** — pass the *same* ``FakeRedisCache`` instance
      as ``redis_cache`` to multiple ``UserApiKeyCache`` objects to model a
      shared Redis tier.  A cache hit on one "pod" will be visible to all
      others, exactly as in production.

    Everything else (TTL, eviction, ``flush_cache``) behaves identically
    to ``InMemoryCache``.
    """

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        _reject_pydantic(key, value)
        super().set_cache(key, value, **kwargs)

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        _reject_pydantic(key, value)
        await super().async_set_cache(key, value, **kwargs)

    async def async_set_cache_pipeline(
        self,
        cache_list: List[Tuple[str, Any]],
        ttl: Any = None,
        **kwargs: Any,
    ) -> None:
        for key, value in cache_list:
            _reject_pydantic(key, value)
        await super().async_set_cache_pipeline(cache_list, ttl, **kwargs)
