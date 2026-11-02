from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_router_logger
from litellm.redis.stores import _routing_prefetch
from litellm.redis.stores._routing_prefetch import TierRead, selector_read
from litellm.router_deprecated import renamed_router_param
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2, PrefetchedUsage

if TYPE_CHECKING:
    from opentelemetry.trace import Span

    from litellm.router_api import RoutingContext


ROUTER_USAGE_TARGET: Final = "router_usage"


def _selector_keys(selector: LowestTPMLoggingHandler_v2, deployments: list) -> list[str]:
    tpm_keys, rpm_keys = selector.usage_counter_keys(deployments)
    return tpm_keys + rpm_keys


def _selector_read(selector: LowestTPMLoggingHandler_v2 | None, deployments: list) -> TierRead | None:
    return None if selector is None else selector_read(selector.router_cache, _selector_keys(selector, deployments))


class RoutingPrefetch:
    @staticmethod
    @renamed_router_param("litellm_router_instance")
    def arm(ctx: "RoutingContext", usage_selector: LowestTPMLoggingHandler_v2 | None, deployments: list) -> None:
        ctx.health().cooldowns.cooldown_store.arm_routing_read(
            ctx.registry().names(kind="ids"), _selector_read(usage_selector, deployments)
        )

    @staticmethod
    def armed() -> bool:
        return _routing_prefetch.armed()


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
        self, ctx: "RoutingContext", healthy_deployments: list, parent_otel_span: "Span | None"
    ) -> list[str]:
        usage_read: Final = _selector_read(self.usage_selector, healthy_deployments)
        read: Final = await ctx.health().cooldowns.cooldown_store.routing_read(
            ctx.registry().names(kind="ids"), usage_read, parent_otel_span
        )
        if usage_read is not None:
            self.prefetched_usage = PrefetchedUsage(
                keys=frozenset(usage_read.keys),
                values=None if read.selector_values is None else dict(zip(usage_read.keys, read.selector_values)),
            )
        verbose_router_logger.debug("retrieve cooldown models: %s", read.cooldowns)
        return [model_id for model_id, _ in read.cooldowns]
