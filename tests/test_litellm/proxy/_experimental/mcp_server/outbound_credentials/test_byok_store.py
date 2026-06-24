"""Tests for the v2-native BYOK credential cache (CachedByokStore)."""

from typing import Dict, List, Optional, Tuple

from litellm.proxy._experimental.mcp_server.outbound_credentials.byok_store import (
    CachedByokStore,
)


class _FakeStore:
    """A ByokCredentialStore that records calls and returns canned values."""

    def __init__(self, values: Dict[Tuple[str, str], Optional[str]]) -> None:
        self._values = values
        self.calls: List[Tuple[str, str]] = []

    async def fetch(self, user_id: str, server_id: str) -> Optional[str]:
        self.calls.append((user_id, server_id))
        return self._values.get((user_id, server_id))


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


async def test_returns_cached_value_within_ttl():
    inner = _FakeStore({("u", "s"): "key-1"})
    clock = _Clock()
    store = CachedByokStore(inner, ttl_seconds=60, clock=clock)

    assert await store.fetch("u", "s") == "key-1"
    clock.t = 59
    assert await store.fetch("u", "s") == "key-1"
    assert inner.calls == [("u", "s")]  # second read served from cache


async def test_refetches_after_ttl_expiry():
    inner = _FakeStore({("u", "s"): "key-1"})
    clock = _Clock()
    store = CachedByokStore(inner, ttl_seconds=60, clock=clock)

    await store.fetch("u", "s")
    clock.t = 61
    await store.fetch("u", "s")
    assert len(inner.calls) == 2  # expired entry re-fetched


async def test_caches_missing_credential():
    inner = _FakeStore({})  # no credential for ("u", "s")
    clock = _Clock()
    store = CachedByokStore(inner, ttl_seconds=60, clock=clock)

    assert await store.fetch("u", "s") is None
    assert await store.fetch("u", "s") is None
    assert inner.calls == [("u", "s")]  # the miss is cached, not re-hit


async def test_isolates_by_subject_and_server():
    inner = _FakeStore({("u1", "s"): "k1", ("u2", "s"): "k2"})
    store = CachedByokStore(inner, ttl_seconds=60, clock=_Clock())

    assert await store.fetch("u1", "s") == "k1"
    assert await store.fetch("u2", "s") == "k2"
    # caching u1 must never serve u1's key to u2
    assert await store.fetch("u2", "s") == "k2"


async def test_bounded_cache_clears_when_full():
    inner = _FakeStore({("u1", "s"): "k1", ("u2", "s"): "k2", ("u3", "s"): "k3"})
    store = CachedByokStore(inner, ttl_seconds=60, max_size=2, clock=_Clock())

    await store.fetch("u1", "s")
    await store.fetch("u2", "s")
    await store.fetch("u3", "s")  # len >= max_size -> clear before inserting u3
    await store.fetch("u1", "s")  # u1 was evicted -> re-fetch
    assert inner.calls.count(("u1", "s")) == 2
