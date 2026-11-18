import asyncio
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.redis.stores.router_usage import UsageCounter
from litellm.router_deprecated import renamed_router_param
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2, PrefetchedUsage
from litellm.router_utils import routing_prefetch

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.router_api import RoutingContext

__all__ = ("RoutingPrefetch", "RoutingReadBatch")


def _selector_counters(selector: LowestTPMLoggingHandler_v2, deployments: Sequence[object]) -> tuple[UsageCounter, ...]:
    tpm_counters, rpm_counters = selector.usage_counters(deployments)
    return tpm_counters + rpm_counters


def _deployment_ids(deployments: Sequence[object]) -> tuple[str, ...]:
    records: Final = (as_str_mapping(deployment) or {} for deployment in deployments)
    infos: Final = (as_str_mapping(record.get("model_info")) or {} for record in records)
    return tuple(model_id for info in infos if isinstance(model_id := info.get("id"), str))


class RoutingPrefetch:
    @staticmethod
    @renamed_router_param("litellm_router_instance")
    def arm(
        ctx: "RoutingContext", usage_selector: LowestTPMLoggingHandler_v2 | None, deployments: Sequence[object]
    ) -> None:
        RoutingPrefetch.arm_cooldowns(ctx, _deployment_ids(deployments))
        if usage_selector is not None:
            routing_prefetch.arm(usage_selector.usage, _selector_counters(usage_selector, deployments))

    @staticmethod
    def arm_cooldowns(ctx: "RoutingContext", model_ids: Sequence[str] | None = None) -> None:
        """Declare a scoped cooldown read; without known candidates, let routing read them later."""
        if model_ids:
            routing_prefetch.arm_cooldowns(ctx.health().cooldowns.cooldown_store, model_ids)

    @staticmethod
    def armed() -> bool:
        return routing_prefetch.armed()


_active_routing_read_batch: Final[ContextVar["RoutingReadBatch | None"]] = ContextVar(
    "routing_read_batch", default=None
)


class RoutingReadBatch:
    def __init__(self, usage_selector: LowestTPMLoggingHandler_v2 | None) -> None:
        self.usage_selector: Final = usage_selector
        self.prefetched_usage: PrefetchedUsage | None = None

    @staticmethod
    @contextmanager
    def scoped(batch: "RoutingReadBatch | None") -> Iterator[None]:
        token: Final = _active_routing_read_batch.set(batch)
        try:
            yield
        finally:
            _active_routing_read_batch.reset(token)

    @staticmethod
    def active() -> "RoutingReadBatch | None":
        return _active_routing_read_batch.get()

    @staticmethod
    def for_strategy(strategy: str | None, selector: object) -> "RoutingReadBatch | None":
        if strategy == "usage-based-routing-v2" and isinstance(selector, LowestTPMLoggingHandler_v2):
            return RoutingReadBatch(usage_selector=selector)
        return RoutingReadBatch(usage_selector=None) if RoutingPrefetch.armed() else None

    @renamed_router_param("litellm_router_instance")
    async def async_get_cooldown_deployments(
        self, ctx: "RoutingContext", healthy_deployments: Sequence[object], parent_otel_span: "Span | None"
    ) -> list[str]:
        cooldowns, _ = await asyncio.gather(
            routing_prefetch.read_cooldowns(
                ctx.health().cooldowns.cooldown_store, _deployment_ids(healthy_deployments), parent_otel_span
            ),
            self.read_usage(healthy_deployments, parent_otel_span),
        )
        verbose_router_logger.debug("retrieve cooldown models: %s", cooldowns)
        return [model_id for model_id, _ in cooldowns]

    async def read_usage(self, deployments: Sequence[object], parent_otel_span: "Span | None") -> None:
        """The selector's tpm/rpm counters for ``deployments``, kept as ``prefetched_usage``."""
        selector: Final = self.usage_selector
        if selector is None:
            return
        counters: Final = _selector_counters(selector, deployments)
        values: Final = await routing_prefetch.read(selector.usage, counters, parent_otel_span)
        self.prefetched_usage = PrefetchedUsage(counters=frozenset(counters), values=dict(zip(counters, values)))
