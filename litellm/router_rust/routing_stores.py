from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter
from typing_extensions import assert_never

from litellm._logging import verbose_router_logger
from litellm.constants import HEALTH_CHECKS_TARGET
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.redis.stores.router_cooldowns import ROUTER_COOLDOWNS_TARGET
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET, RouterUsage, UsageCounter
from litellm.redis.stores.router_usage.types import CounterKind
from litellm.router_utils import routing_prefetch
from litellm.router_utils.cooldown_cache import CooldownCache, CooldownCacheValue

__all__ = (
    "CACHE_STEP",
    "CacheStep",
    "CooldownsStep",
    "GetLocalStep",
    "IncrementLocalStep",
    "LocalStep",
    "RoutingStores",
    "StoreStep",
    "service_target_of",
)

_COOLDOWN_VALUE: Final = TypeAdapter(CooldownCacheValue)


def _cooldown_value(value: JsonValue) -> CooldownCacheValue:
    """The cooldown router-core wrote, its whole-second ``cooldown_time`` kept an ``int`` as it serialized it."""
    validated: Final = _COOLDOWN_VALUE.validate_python(value)
    written: Final = as_str_mapping(value)
    cooldown_time: Final = None if written is None else written.get("cooldown_time")
    if isinstance(cooldown_time, int) and not isinstance(cooldown_time, bool):
        return {**validated, "cooldown_time": cooldown_time}
    return validated


class _Step(BaseModel):
    model_config = ConfigDict(frozen=True)


class UsageWindow(_Step):
    deployment_id: str
    model: str
    minute: str

    def counter(self, kind: CounterKind) -> UsageCounter:
        return UsageCounter("global_router", self.deployment_id, self.model, self.minute, kind)


class CooldownsStep(_Step):
    op: Literal["cooldowns"]
    ids: tuple[str, ...]


class PutCooldownStep(_Step):
    op: Literal["put_cooldown"]
    id: str
    cooldown: JsonValue
    ttl: float


class IncrementAllowedFailsStep(_Step):
    op: Literal["increment_allowed_fails"]
    id: str
    suffix: str | None
    ttl: float


class AddUsageStep(_Step):
    op: Literal["add_usage"]
    window: UsageWindow
    tokens: int | float
    requests: int | float


class AddRequestStep(_Step):
    op: Literal["add_request"]
    window: UsageWindow
    ttl: float


class UsageStep(_Step):
    op: Literal["usage"]
    window: UsageWindow


class HealthStatesStep(_Step):
    op: Literal["health_states"]


class GetLocalStep(_Step):
    op: Literal["get_local"]
    key: str


class IncrementLocalStep(_Step):
    op: Literal["increment_local"]
    key: str
    amount: int | float
    ttl: float | None


StoreStep: TypeAlias = (
    CooldownsStep
    | PutCooldownStep
    | IncrementAllowedFailsStep
    | AddUsageStep
    | AddRequestStep
    | UsageStep
    | HealthStatesStep
)
LocalStep: TypeAlias = GetLocalStep | IncrementLocalStep
CacheStep: TypeAlias = Annotated[StoreStep | LocalStep, Field(discriminator="op")]
CACHE_STEP: Final[TypeAdapter[CacheStep]] = TypeAdapter(CacheStep)


def service_target_of(step: StoreStep | LocalStep) -> str:
    """The service target a cache step is attributed to, as the Python core's."""
    match step:
        case CooldownsStep() | PutCooldownStep() | IncrementAllowedFailsStep():
            return ROUTER_COOLDOWNS_TARGET
        case HealthStatesStep():
            return HEALTH_CHECKS_TARGET
        case AddUsageStep() | AddRequestStep() | UsageStep() | GetLocalStep() | IncrementLocalStep():
            return ROUTER_USAGE_TARGET
        case _:
            assert_never(step)


class HealthStates(Protocol):
    async def async_get_states(self) -> object: ...
    def get_states(self) -> object: ...


def _aligned(model_ids: Sequence[str], active: Sequence[tuple[str, CooldownCacheValue]]) -> tuple[object, ...]:
    found: Final = dict(active)
    return tuple(found.get(model_id) for model_id in model_ids)


def _usage_counters(window: UsageWindow) -> tuple[UsageCounter, UsageCounter]:
    return window.counter("tpm"), window.counter("rpm")


@dataclass(frozen=True, slots=True)
class RoutingStores:
    """Router-core's cache steps answered by the stores the Python core uses. The stores own every key."""

    cooldowns: CooldownCache
    usage: RouterUsage
    health: HealthStates

    async def answer(self, step: StoreStep) -> object:
        store: Final = self.cooldowns.cooldown_store
        match step:
            case CooldownsStep(ids=ids):
                return _aligned(ids, await routing_prefetch.read_cooldowns(store, ids, None))
            case PutCooldownStep(id=model_id, cooldown=cooldown, ttl=ttl):
                await store.async_put(model_id, _cooldown_value(cooldown), ttl)
                return None
            case IncrementAllowedFailsStep(id=model_id, suffix=suffix, ttl=ttl):
                return await store.async_increment_allowed_fails(model_id, suffix, ttl)
            case AddUsageStep(window=window, tokens=tokens, requests=requests):
                try:
                    await self.usage.increment_deployment(
                        window.deployment_id, window.model, window.minute, tokens, int(requests)
                    )
                except Exception as e:  # noqa: BLE001  # usage that cannot be written never fails the call, as on the Python router
                    verbose_router_logger.debug("_increment_deployment_usage: usage was not recorded. Got - %s", e)
                return None
            case AddRequestStep(window=window, ttl=ttl):
                await self.usage.increment(window.counter("rpm"), 1, int(ttl))
                return None
            case UsageStep(window=window):
                return tuple(await self.usage.get_many(_usage_counters(window)))
            case HealthStatesStep():
                return await self.health.async_get_states()
            case _:
                assert_never(step)

    def answer_sync(self, step: StoreStep) -> object:
        store: Final = self.cooldowns.cooldown_store
        match step:
            case CooldownsStep(ids=ids):
                return _aligned(ids, store.active_sync(ids))
            case PutCooldownStep(id=model_id, cooldown=cooldown, ttl=ttl):
                store.put(model_id, _cooldown_value(cooldown), ttl)
                return None
            case IncrementAllowedFailsStep(id=model_id, suffix=suffix, ttl=ttl):
                return store.increment_allowed_fails(model_id, suffix, ttl)
            case AddUsageStep(window=window, tokens=tokens, requests=requests):
                self.usage.increment_deployment_sync(
                    window.deployment_id, window.model, window.minute, tokens, int(requests)
                )
                return None
            case AddRequestStep(window=window, ttl=ttl):
                self.usage.increment_sync(window.counter("rpm"), 1, int(ttl))
                return None
            case UsageStep(window=window):
                return tuple(self.usage.get_many_sync(_usage_counters(window)))
            case HealthStatesStep():
                return self.health.get_states()
            case _:
                assert_never(step)
