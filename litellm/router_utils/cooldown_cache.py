from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from opentelemetry.trace import Span

from litellm.litellm_core_utils.sensitive_data_masker import SensitiveDataMasker
from litellm.redis.stores.router_cooldowns import (
    ROUTER_COOLDOWNS_TARGET,
    CooldownCacheValue,
    RouterCooldowns,
)

__all__: Final = ("ROUTER_COOLDOWNS_TARGET", "CooldownCache", "CooldownCacheValue")


class CooldownCache:
    def __init__(
        self, store: RouterCooldowns, default_cooldown_time: float, *, clock: Callable[[], float] | None = None
    ) -> None:
        self.cooldown_store: Final = store
        self._clock: Final = clock if clock is not None else lambda: time.time()
        self.default_cooldown_time: Final = default_cooldown_time
        self.exception_masker: Final = SensitiveDataMasker(
            visible_prefix=50, visible_suffix=0, mask_char="*", mask_short_values=False
        )

    def add_deployment_to_cooldown(
        self, model_id: str, original_exception: Exception, exception_status: int, cooldown_time: float | None
    ) -> None:
        ttl: Final = self.default_cooldown_time if cooldown_time is None else cooldown_time
        value: Final = CooldownCacheValue(
            exception_received=self.exception_masker.mask_value(str(original_exception)),
            status_code=str(exception_status),
            timestamp=self._clock(),
            cooldown_time=ttl,
        )
        self.cooldown_store.put(model_id, value, ttl)

    async def async_get_active_cooldowns(
        self, model_ids: Sequence[str], parent_otel_span: Span | None
    ) -> list[tuple[str, CooldownCacheValue]]:
        return list(await self.cooldown_store.active(model_ids, parent_otel_span))

    def get_min_cooldown(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> float:
        return self.cooldown_store.minimum(model_ids, self.default_cooldown_time, parent_otel_span)

    def get_active_cooldowns(
        self, model_ids: Sequence[str], parent_otel_span: Span | None
    ) -> list[tuple[str, CooldownCacheValue]]:
        return list(self.cooldown_store.active_sync(model_ids, parent_otel_span))
