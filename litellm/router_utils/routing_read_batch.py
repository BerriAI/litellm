"""
One Redis round trip for the reads a request needs before a deployment can be picked.

The cooldown filter (`CooldownCache`, its own `DualCache`) and usage-based selection
(`LowestTPMLoggingHandler_v2`, the router cache) each issue their own MGET because they live in
different objects. `RoutingReadBatch` fetches both key sets in one
`DualCache.async_batch_get_cache_shared` while the healthy deployments are being resolved and hands
the usage slice to the strategy, so selection does not read again.

When a `RedisRequestPlan` is active (the proxy opens one per request), `declare_routing_prefetch`
declares the same MGET into the plan at auth time, so the shared read rides the request's single
pre-call pipeline instead of its own round trip.
"""

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache, SharedPendingBatchRead
from litellm.caching.redis_request_plan import RedisRequestPlan, active_redis_request_plan
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


@dataclass(frozen=True, slots=True)
class PrefetchedRoutingRead:
    """The routing shared batch read declared into the request's plan during auth."""

    cooldown_keys: tuple[str, ...]
    usage_keys: frozenset[str]
    pending: SharedPendingBatchRead
    future: "asyncio.Future[dict[str, object | None]]"
    plan: RedisRequestPlan


_prefetched_routing_read: Final[ContextVar["PrefetchedRoutingRead | None"]] = ContextVar(
    "prefetched_routing_read", default=None
)


def _consume_prefetch_failure(future: "asyncio.Future[dict[str, object | None]]") -> None:
    """A prefetch that was never consumed still resolves at plan flush; keep its failure from
    surfacing as an unretrieved exception warning."""
    if future.cancelled():
        return
    exc: Final = future.exception()
    if exc is not None:
        verbose_router_logger.debug("routing prefetch read failed, selection reads again: %s", exc)


async def declare_routing_prefetch(litellm_router_instance: LitellmRouter, model: str) -> None:
    """Declare the cooldown + usage MGET for ``model`` into the active request plan.

    Healthy deployments are not known yet at auth time, so the usage keys cover the model's
    full deployment list; the later read consumes the prefetch only when its own key set is
    a subset of what was declared.
    """
    plan: Final = active_redis_request_plan()
    if plan is None:
        return
    strategy, selector = litellm_router_instance._get_routing_context(model)  # pyright: ignore[reportPrivateUsage]  # same selector async_get_available_deployment would use
    read_batch: Final = RoutingReadBatch.for_strategy(strategy, selector)
    if read_batch is None:
        return
    usage_selector: Final = read_batch.usage_selector
    deployments: Final = litellm_router_instance.get_model_list(model_name=model)
    if not deployments:
        return
    model_ids: Final = litellm_router_instance.get_model_ids()
    cooldown_keys: Final = tuple(CooldownCache.get_cooldown_cache_key(model_id) for model_id in model_ids)
    tpm_keys, rpm_keys = usage_selector.usage_counter_keys(deployments)
    usage_keys: Final = tpm_keys + rpm_keys

    pending: Final = await DualCache.prepare_shared_batch_get(
        [
            (litellm_router_instance.cooldown_cache.cooldown_store, list(cooldown_keys)),
            (usage_selector.router_cache, usage_keys),
        ]
    )
    if pending.shared_redis is None or not pending.redis_keys:
        return
    future: Final = plan.batch_for(pending.shared_redis).mget(pending.redis_keys)
    future.add_done_callback(_consume_prefetch_failure)
    _prefetched_routing_read.set(
        PrefetchedRoutingRead(
            cooldown_keys=cooldown_keys,
            usage_keys=frozenset(usage_keys),
            pending=pending,
            future=future,
            plan=plan,
        )
    )


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

        prefetched: Final = _prefetched_routing_read.get()
        if prefetched is not None:
            _prefetched_routing_read.set(None)

        if (
            prefetched is not None
            and tuple(cooldown_keys) == prefetched.cooldown_keys
            and prefetched.usage_keys.issuperset(usage_keys)
        ):
            try:
                outcome: dict[str, object] | BaseException = await prefetched.plan.resolve(prefetched.future)
            except Exception as e:  # noqa: BLE001  # mapped like a failed shared read below
                outcome = e
            cooldown_results, usage_values = await DualCache.apply_shared_batch_get(prefetched.pending, outcome)
            ordered_usage_keys: Final = next(
                (pending.keys for index, _, pending in prefetched.pending.pendings if index == 1), None
            )
            self.prefetched_usage = PrefetchedUsage(
                keys=frozenset(ordered_usage_keys) if ordered_usage_keys is not None else prefetched.usage_keys,
                values=(
                    dict(zip(ordered_usage_keys, usage_values))
                    if ordered_usage_keys is not None and usage_values is not None
                    else None
                ),
            )
        else:
            if prefetched is not None:
                try:
                    orphan_outcome: dict[str, object] | BaseException = await prefetched.plan.resolve(prefetched.future)
                except Exception as e:  # noqa: BLE001  # still settles the auth-time reservations
                    orphan_outcome = e
                await DualCache.apply_shared_batch_get(prefetched.pending, orphan_outcome)
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
