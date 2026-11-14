import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

from pydantic import ValidationError

from litellm.proxy.spend_tracking.spend_counter_batch import bind_spend_counter_keys, read_batched_spend_counters
from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth


def customer_spend_counter(scope: CustomerScopeAuth) -> str:
    period: Final = scope.budget_reset_at.isoformat() if scope.budget_reset_at is not None else "lifetime"
    return f"spend:{scope.key}:{scope.entity_id}:{period}"


async def _seed_scope(scope: CustomerScopeAuth, store: SpendCounterStore) -> None:
    _ = await store.seed(customer_spend_counter(scope), scope.spend, increment_if_lost=False)


async def current_customer_spend(
    hierarchy: CustomerHierarchyAuth,
    read_spend: Callable[[str, float, float | None], Awaitable[float]],
) -> CustomerHierarchyAuth:
    bind_spend_counter_keys(frozenset(customer_spend_counter(scope) for scope in hierarchy.scopes))
    readings: Final = await asyncio.gather(
        *(read_spend(customer_spend_counter(scope), scope.spend, scope.max_budget) for scope in hierarchy.scopes)
    )
    scopes: Final = tuple(
        scope.model_copy(update={"spend": spend}) if spend > scope.spend else scope
        for scope, spend in zip(hierarchy.scopes, readings, strict=True)
    )
    return (
        hierarchy
        if all(a is b for a, b in zip(scopes, hierarchy.scopes, strict=True))
        else hierarchy.model_copy(update={"scopes": scopes})
    )


def customer_spend_hierarchy(metadata: Mapping[str, object]) -> CustomerHierarchyAuth | None:
    snapshot: Final = metadata.get("user_api_key_customer_hierarchy")
    if snapshot is None:
        return None
    try:
        return CustomerHierarchyAuth.model_validate(snapshot)
    except ValidationError:
        return None


def customer_spend_keys(metadata: Mapping[str, object]) -> frozenset[str]:
    hierarchy: Final = customer_spend_hierarchy(metadata)
    return (
        frozenset(customer_spend_counter(scope) for scope in hierarchy.scopes) if hierarchy is not None else frozenset()
    )


async def _settlement_read_many(counters: tuple[str, ...], store: SpendCounterStore) -> tuple[float | None, ...]:
    batched: Final = await read_batched_spend_counters(frozenset(counters))
    direct_keys: Final = tuple(counter for counter in counters if counter not in batched)
    if not direct_keys:
        return tuple(batched[counter] for counter in counters)
    direct: Final = await asyncio.gather(*(store.read(counter) for counter in direct_keys))
    fallback: Final = {key: value[0] for key, value in zip(direct_keys, direct, strict=True)}
    readings: Final = {**batched, **fallback}
    return tuple(readings[counter] for counter in counters)


async def settle_customer_spend(metadata: Mapping[str, object], cost: float, store: SpendCounterStore) -> None:
    hierarchy: Final = customer_spend_hierarchy(metadata)
    if hierarchy is None or cost == 0:
        return
    counters: Final = tuple(customer_spend_counter(scope) for scope in hierarchy.scopes)
    readings: Final = await _settlement_read_many(counters, store)
    _ = await asyncio.gather(
        *(
            _seed_scope(scope, store)
            for scope, reading in zip(hierarchy.scopes, readings, strict=True)
            if reading is None
        )
    )
    increments: Final = tuple((counter, cost) for counter in counters)
    if store.increment_many_post_call(increments) is None:
        _ = await store.increment_many(increments)
