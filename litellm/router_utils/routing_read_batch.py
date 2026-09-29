"""
One Redis round trip for the reads a request needs before a deployment can be picked.

The cooldown filter (`CooldownCache`, its own `DualCache`) and usage-based selection
(`LowestTPMLoggingHandler_v2`, the router cache) each issue their own MGET because they live in
different objects. `RoutingReadBatch` fetches both key sets in one
`DualCache.async_batch_get_cache_shared` while the healthy deployments are being resolved and hands
the usage slice to the strategy, so selection does not read again.
"""

from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2, PrefetchedUsage
from litellm.router_utils.cooldown_cache import CooldownCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.router import Router as _Router

    LitellmRouter = _Router
    Span = _Span
else:
    LitellmRouter = Any
    Span = Any


class RoutingReadBatch:
    def __init__(self, usage_selector: LowestTPMLoggingHandler_v2) -> None:
        self.usage_selector: Final = usage_selector
        self.prefetched_usage: PrefetchedUsage | None = None

    @staticmethod
    def for_strategy(strategy: str | None, selector: object) -> "RoutingReadBatch | None":
        if strategy == "usage-based-routing-v2" and isinstance(selector, LowestTPMLoggingHandler_v2):
            return RoutingReadBatch(usage_selector=selector)
        return None

    async def async_get_cooldown_deployments(
        self,
        litellm_router_instance: LitellmRouter,
        healthy_deployments: list,
        parent_otel_span: Span | None,
    ) -> list[str]:
        """
        `_async_get_cooldown_deployments`, with the strategy's tpm/rpm counters for
        `healthy_deployments` fetched in the same MGET and kept as `prefetched_usage`.
        """
        model_ids: Final = litellm_router_instance.get_model_ids()
        cooldown_keys: Final = [CooldownCache.get_cooldown_cache_key(model_id) for model_id in model_ids]
        tpm_keys, rpm_keys = self.usage_selector.usage_counter_keys(healthy_deployments)
        usage_keys: Final = tpm_keys + rpm_keys

        cooldown_results, usage_values = await DualCache.async_batch_get_cache_shared(
            [
                (litellm_router_instance.cooldown_cache.cooldown_store, cooldown_keys),
                (self.usage_selector.router_cache, usage_keys),
            ],
            parent_otel_span=parent_otel_span,
        )
        self.prefetched_usage = PrefetchedUsage(
            keys=frozenset(usage_keys),
            values=None if usage_values is None else dict(zip(usage_keys, usage_values)),
        )

        cooldown_models: Final = litellm_router_instance.cooldown_cache.active_cooldowns_from_results(
            model_ids, cooldown_results
        )
        verbose_router_logger.debug("retrieve cooldown models: %s", cooldown_models)
        return [model_id for model_id, _ in cooldown_models]
