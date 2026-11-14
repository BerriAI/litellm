import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Final

from pydantic import ValidationError

from litellm.redis.stores.spend_counters import SpendCounterStore
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerScopeAuth


def customer_spend_counter(scope: CustomerScopeAuth) -> str:
    period: Final = scope.budget_reset_at.isoformat() if scope.budget_reset_at is not None else "lifetime"
    return f"spend:{scope.key}:{scope.entity_id}:{period}"


async def _current_scope(
    scope: CustomerScopeAuth, current: float | None, store: SpendCounterStore
) -> CustomerScopeAuth:
    spend: Final = (
        await store.seed(customer_spend_counter(scope), scope.spend, increment_if_lost=False)
        if current is None
        else current
    )
    return scope.model_copy(update={"spend": max(scope.spend, spend)})


async def current_customer_spend(
    hierarchy: CustomerHierarchyAuth,
    read_spend: Callable[[str, float, float | None], Awaitable[float]],
) -> CustomerHierarchyAuth:
    readings: Final = await asyncio.gather(
        *(read_spend(customer_spend_counter(scope), scope.spend, scope.max_budget) for scope in hierarchy.scopes)
    )
    scopes: Final = tuple(
        scope.model_copy(update={"spend": max(scope.spend, spend)})
        for scope, spend in zip(hierarchy.scopes, readings, strict=True)
    )
    return hierarchy.model_copy(update={"scopes": tuple(scopes)})


async def settle_customer_spend(metadata: Mapping[str, object], cost: float, store: SpendCounterStore) -> None:
    snapshot: Final = metadata.get("user_api_key_customer_hierarchy")
    if snapshot is None or cost == 0:
        return
    try:
        hierarchy: Final = CustomerHierarchyAuth.model_validate(snapshot)
    except ValidationError:
        return
    readings: Final = await asyncio.gather(*(store.read(customer_spend_counter(scope)) for scope in hierarchy.scopes))
    _ = await asyncio.gather(
        *(_current_scope(scope, reading[0], store) for scope, reading in zip(hierarchy.scopes, readings, strict=True))
    )
    _ = await store.increment_many(tuple((customer_spend_counter(scope), cost) for scope in hierarchy.scopes))
