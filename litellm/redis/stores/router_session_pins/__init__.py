from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm.caching.redis_batch import Batch
from litellm.redis.stores._route import ready, route
from litellm.redis.stores.router_session_pins.handle import RouterSessionPinsHandle
from litellm.redis.stores.router_session_pins.keys import (
    DeploymentPin,
    DeploymentSessionPin,
    PinKey,
    TierModelPin,
    TierSessionPin,
)

if TYPE_CHECKING:
    from litellm.redis.stores.connection import RedisStores

__all__ = [
    "ROUTER_SESSION_PINS_TARGET",
    "DeploymentPin",
    "DeploymentSessionPin",
    "PinKey",
    "RouterSessionPinStore",
    "TierModelPin",
    "TierSessionPin",
]

ROUTER_SESSION_PINS_TARGET: Final = "router_session_pins"


class RouterSessionPinStore:
    """The router's session affinity pins (deployment affinity, the complexity router's session and tier pins):
    memory per router, Redis shared by every pod. The store owns the claim rule, in Redis and in memory."""

    def __init__(self, handle: RouterSessionPinsHandle) -> None:
        self._handle: Final = handle

    @property
    def has_redis(self) -> bool:
        return self._handle.has_redis

    def attach_redis(self, redis: RedisStores) -> None:
        """Adds the Redis tier once, keeping this router's memory pins, as ``DualCache.attach_redis_cache`` does."""
        if redis.client is not None:
            self._handle.attach(redis.client)

    @with_service_target(ROUTER_SESSION_PINS_TARGET)
    async def get(self, key: PinKey) -> object:
        """``DualCache.async_get_cache``: memory, else Redis written back to memory."""
        if not self.has_redis:
            return self._handle.get_local(key)
        return await self._handle.get(key)

    def get_local(self, key: PinKey) -> object:
        return self._handle.get_local(key)

    def set_local(self, key: PinKey, value: object, ttl: float) -> None:
        """Replaces the memory entry, since ``InMemoryCache.set_cache`` keeps a live key's expiry."""
        self._handle.set_local(key, _plain(value), ttl)

    def delete_local(self, key: PinKey) -> None:
        self._handle.delete_local(key)

    @with_service_target(ROUTER_SESSION_PINS_TARGET)
    async def set(self, key: PinKey, value: object, ttl: float) -> None:
        """``DualCache.async_set_cache``: memory, then a ``SET`` of ``json.dumps(value)``."""
        if not self.has_redis:
            self.set_local(key, value, ttl)
            return
        await self._handle.set(key, _plain(value), ttl)

    @with_service_target(ROUTER_SESSION_PINS_TARGET)
    async def get_shared(self, key: PinKey) -> object:
        """``redis_cache.async_get_cache(raise_on_error=True)``: Redis alone, raising what it raises."""
        return await self._handle.get_shared(key)

    @with_service_target(ROUTER_SESSION_PINS_TARGET)
    def read(self, key: PinKey, ttl: float, *, batch: Batch) -> Awaitable[object]:
        """The read script on the batch ``batch`` picks: refreshes the pin's TTL, then memory takes what Redis holds
        or drops a missing pin. A Redis failure raises, and the caller keeps its memory pin."""
        if not self.has_redis:
            return ready(self._handle.get_local(key))
        target: Final = route(batch)
        return target.result(self._handle.read(key, ttl, target.send))

    @with_service_target(ROUTER_SESSION_PINS_TARGET)
    async def claim(
        self,
        key: PinKey,
        value: Mapping[str, str],
        ttl: float,
        *,
        eligible_values: Sequence[Mapping[str, str]] | None,
        preserve_legacy_model_ids: bool,
    ) -> object:
        """The claim script: the first writer keeps the key unless it is no longer eligible, and memory takes the
        winner. None when Redis answered no pin. A Redis failure raises, and the caller settles with
        ``claim_local``."""
        return await self._handle.claim(
            key, dict(value), ttl, *_eligibility(eligible_values, preserve_legacy_model_ids)
        )

    def claim_local(
        self,
        key: PinKey,
        value: Mapping[str, str],
        ttl: float,
        *,
        eligible_values: Sequence[Mapping[str, str]] | None,
        preserve_legacy_model_ids: bool,
    ) -> object:
        """``claim`` against this router's memory alone, for a router without Redis or one whose Redis claim
        failed. Nothing is awaited between its read and its write, so claims on one loop agree."""
        return self._handle.claim_local(
            key, dict(value), ttl, *_eligibility(eligible_values, preserve_legacy_model_ids)
        )


def _eligibility(
    eligible_values: Sequence[Mapping[str, str]] | None, preserve_legacy_model_ids: bool
) -> tuple[Sequence[object] | None, bool]:
    return (
        None if eligible_values is None else [dict(eligible) for eligible in eligible_values],
        preserve_legacy_model_ids and eligible_values is not None,
    )


def _plain(value: object) -> object:
    """A pin as the plain value the native store takes: a mapping becomes the dict it holds."""
    return dict(value) if isinstance(value, Mapping) else value
