"""
Wrapper around router cache. Meant to handle model cooldown logic
"""

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from litellm import verbose_logger
from litellm.caching.caching import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS
from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.redis.stores import router_cooldowns
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, DualCacheRouterCooldownStore

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any

__all__ = ("ROUTER_COOLDOWNS_TARGET", "CooldownCache", "CooldownCacheValue")


# Cap on the corrected in-memory TTL set in `_corrected_active_cooldown`: re-checks the
# real remaining cooldown against Redis at least this often, so an entry that later gets
# deleted or extended in Redis before its original deadline is still noticed promptly.
_MAX_CORRECTED_IN_MEMORY_TTL_SECONDS: Final = 60.0
ROUTER_COOLDOWNS_TARGET: Final = router_cooldowns.TARGET


class CooldownCache:
    def __init__(
        self,
        cache: DualCache,
        default_cooldown_time: float,
        redis_read_interval_seconds: float = DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS,
    ):
        self.cache = cache
        self.default_cooldown_time = default_cooldown_time
        self.in_memory_cache = InMemoryCache()
        self._cooldown_store = DualCache(
            in_memory_cache=self.in_memory_cache,
            default_redis_batch_cache_expiry=redis_read_interval_seconds,
        )
        # Initialize the masker with custom settings for exception strings
        self.exception_masker = SensitiveDataMasker(
            visible_prefix=50,  # Show first 50 characters
            visible_suffix=0,  # Show last 0 characters
            mask_char="*",  # Use * for masking
            mask_short_values=False,  # Truncate long messages only; keep short ones readable
        )
        self._store: Final = DualCacheRouterCooldownStore(memory=self.in_memory_cache, tiers=lambda: self.cooldown_store)

    @property
    def cooldown_store(self) -> DualCache:
        """
        The cache cooldown entries live in, with the router's Redis attached on first use.

        It is kept separate from the router-wide cache so that a key missing from memory is
        re-read from Redis every `redis_read_interval_seconds` rather than on the router
        cache's much longer batch interval, which is what lets a sibling replica see a
        cooldown another replica wrote, and so that unrelated router keys cannot evict a
        cooldown from the in-memory tier before it expires. Redis is attached lazily because
        the router builds its cooldown cache before it wires up the shared Redis client.
        """
        self._cooldown_store.attach_redis_cache(self.cache.redis_cache)
        return self._cooldown_store

    def _common_add_cooldown_logic(
        self, model_id: str, original_exception, exception_status, cooldown_time: float
    ) -> tuple[str, CooldownCacheValue]:
        try:
            current_time: Final = time.time()
            cooldown_key: Final = CooldownCache.get_cooldown_cache_key(model_id)

            # Store the cooldown information for the deployment separately
            cooldown_data: Final = CooldownCacheValue(
                exception_received=self.exception_masker.mask_value(str(original_exception)),
                status_code=str(exception_status),
                timestamp=current_time,
                cooldown_time=cooldown_time,
            )

            return cooldown_key, cooldown_data
        except Exception as e:
            verbose_logger.error("CooldownCache::_common_add_cooldown_logic - Exception occurred - %s", e)
            raise e

    def add_deployment_to_cooldown(
        self,
        model_id: str,
        original_exception: Exception,
        exception_status: int,
        cooldown_time: float | None,
    ):
        try:
            #########################################################
            # get cooldown time
            # 1. If dynamic cooldown time is set for the model/deployment, use that
            # 2. If no dynamic cooldown time is set, use the default cooldown time set on CooldownCache
            _cooldown_time = cooldown_time
            if _cooldown_time is None:
                _cooldown_time = self.default_cooldown_time
            #########################################################

            _, cooldown_data = self._common_add_cooldown_logic(
                model_id=model_id,
                original_exception=original_exception,
                exception_status=exception_status,
                cooldown_time=_cooldown_time,
            )

            self._store.put(model_id, cooldown_data, ttl=_cooldown_time)
        except Exception as e:
            verbose_logger.error("CooldownCache::add_deployment_to_cooldown - Exception occurred - %s", e)
            raise e

    @staticmethod
    def get_cooldown_cache_key(model_id: str) -> str:
        return router_cooldowns.cooldown_key(model_id)

    def _corrected_active_cooldown(
        self, model_id: str, entry: CooldownCacheValue, current_time: float
    ) -> CooldownCacheValue | None:
        """
        Return the entry if the cooldown is still active, or None if it has expired.

        Also corrects the in-memory TTL when DualCache promotes a Redis entry using the
        default 600s TTL instead of the true remaining cooldown time.
        """
        remaining: Final = (entry["timestamp"] + entry["cooldown_time"]) - current_time
        if remaining <= 0:
            self._store.drop_local(model_id)
            return None
        current_expiry: Final = self.in_memory_cache.ttl_dict.get(router_cooldowns.cooldown_key(model_id))
        if current_expiry is not None and current_expiry > current_time + remaining + 5:
            self._store.reset_local_ttl(model_id, entry, ttl=min(remaining, _MAX_CORRECTED_IN_MEMORY_TTL_SECONDS))
        return entry

    def _active(
        self, model_ids: Sequence[str], entries: Sequence[CooldownCacheValue | None]
    ) -> list[tuple[str, CooldownCacheValue]]:
        current_time: Final = time.time()
        corrected: Final = (
            (model_id, self._corrected_active_cooldown(model_id, entry, current_time))
            for model_id, entry in zip(model_ids, entries)
            if entry is not None
        )
        return [(model_id, entry) for model_id, entry in corrected if entry is not None]

    async def async_get_active_cooldowns(
        self, model_ids: list[str], parent_otel_span: Span | None
    ) -> list[tuple[str, CooldownCacheValue]]:
        return self._active(model_ids, await self._store.async_get_many(model_ids, parent_otel_span))

    def active_cooldowns_from_results(
        self, model_ids: list[str], results: Sequence[object] | None
    ) -> list[tuple[str, CooldownCacheValue]]:
        """The cooldowns still active in a `cooldown_store` batch read of `get_cooldown_cache_key(model_id)` per id."""
        return self._active(model_ids, [router_cooldowns.decode(raw) for raw in results or ()])

    def get_active_cooldowns(
        self, model_ids: list[str], parent_otel_span: Span | None
    ) -> list[tuple[str, CooldownCacheValue]]:
        return self._active(model_ids, self._store.get_many(model_ids, parent_otel_span))

    def get_min_cooldown(self, model_ids: list[str], parent_otel_span: Span | None) -> float:
        """Return min cooldown time required for a group of model id's."""
        cooldown_times: Final = [
            entry["cooldown_time"] for entry in self._store.get_many(model_ids, parent_otel_span) if entry is not None
        ]
        return min(cooldown_times, default=0.0) or self.default_cooldown_time
