from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from pydantic import TypeAdapter

from litellm._internal_context import service_target, with_service_target
from litellm.caching.redis_batch import Batch
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.redis.stores.router_cooldowns import ROUTER_COOLDOWNS_TARGET
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET, RouterUsage, UsageCounter
from litellm.router_utils import routing_prefetch
from litellm.router_utils.cooldown_cache import CooldownCache, CooldownCacheValue

__all__ = ("AllowedFailsKey", "CooldownKey", "Handled", "RoutingStores", "UsageKey", "parse_key")

_COOLDOWN_VALUE: Final = TypeAdapter(CooldownCacheValue)


def _cooldown_value(value: object) -> CooldownCacheValue:
    """The cooldown router-core wrote, its whole-second ``cooldown_time`` kept an ``int`` as it serialized it."""
    validated: Final = _COOLDOWN_VALUE.validate_python(value)
    written: Final = as_str_mapping(value)
    cooldown_time: Final = None if written is None else written.get("cooldown_time")
    if isinstance(cooldown_time, int) and not isinstance(cooldown_time, bool):
        return {**validated, "cooldown_time": cooldown_time}
    return validated


@dataclass(frozen=True, slots=True)
class CooldownKey:
    model_id: str


@dataclass(frozen=True, slots=True)
class AllowedFailsKey:
    model_id: str
    suffix: str | None


@dataclass(frozen=True, slots=True)
class UsageKey:
    counter: UsageCounter


RoutingKey: TypeAlias = CooldownKey | AllowedFailsKey | UsageKey


@dataclass(frozen=True, slots=True)
class Handled:
    value: object


def _deployment_key(rest: str) -> RoutingKey | None:
    if rest.endswith(":cooldown"):
        return CooldownKey(rest.removesuffix(":cooldown"))
    model_id, marker, suffix = rest.partition(":allowed_fails")
    if not marker or (suffix and not suffix.startswith(":")):
        return None
    return AllowedFailsKey(model_id, suffix.removeprefix(":") or None)


def _usage_key(rest: str) -> UsageKey | None:
    parts: Final = rest.rsplit(":", 2)
    if len(parts) != 3:
        return None
    head, kind, minute = parts
    model_id, separator, model = head.partition(":")
    if not separator:
        return None
    match kind:
        case "tpm" | "rpm":
            return UsageKey(UsageCounter(model_id, model, minute, kind))
        case _:
            return None


def parse_key(key: str) -> RoutingKey | None:
    """The store family a router-core key belongs to. Each store rebuilds the same key from what is parsed
    here, so an id or model holding a ``:`` still names the key router-core asked for."""
    if key.startswith("deployment:"):
        return _deployment_key(key.removeprefix("deployment:"))
    if key.startswith("global_router:"):
        return _usage_key(key.removeprefix("global_router:"))
    return None


def _service_target(key: RoutingKey | None) -> str | None:
    if isinstance(key, CooldownKey | AllowedFailsKey):
        return ROUTER_COOLDOWNS_TARGET
    if isinstance(key, UsageKey):
        return ROUTER_USAGE_TARGET
    return None


def _cooldown_ids(keys: Sequence[str]) -> tuple[str, ...] | None:
    ids: Final = tuple(parsed.model_id for parsed in map(parse_key, keys) if isinstance(parsed, CooldownKey))
    return ids if keys and len(ids) == len(keys) else None


def _usage_counters(keys: Sequence[object]) -> tuple[UsageCounter, ...] | None:
    counters: Final = tuple(
        parsed.counter for parsed in (parse_key(str(key)) for key in keys) if isinstance(parsed, UsageKey)
    )
    return counters if keys and len(counters) == len(keys) else None


def _aligned(model_ids: Sequence[str], active: Sequence[tuple[str, CooldownCacheValue]]) -> tuple[object, ...]:
    found: Final = dict(active)
    return tuple(found.get(model_id) for model_id in model_ids)


def _one_deployment(tpm: UsageCounter, rpm: UsageCounter) -> bool:
    return (tpm.kind, rpm.kind) == ("tpm", "rpm") and (tpm.deployment_id, tpm.model, tpm.minute) == (
        rpm.deployment_id,
        rpm.model,
        rpm.minute,
    )


def _deployment_increment(keys: Sequence[object], amounts: Sequence[object]) -> tuple[UsageCounter, float, int] | None:
    """``_increment_deployment_usage``'s tpm then rpm increment of one deployment."""
    counters: Final = _usage_counters(keys)
    if counters is None or len(counters) != 2 or len(amounts) != 2:
        return None
    tpm, rpm = counters
    tokens, requests = amounts
    if not _one_deployment(tpm, rpm) or not isinstance(tokens, int | float) or not isinstance(requests, int | float):
        return None
    return tpm, tokens, int(requests)


def _by(value: object) -> float:
    return value if isinstance(value, int | float) and not isinstance(value, bool) else 1


class RoutingStores:
    """Router-core's cache steps for cooldowns, allowed fails and tpm/rpm usage, answered by the native stores
    the Python core uses. ``None`` leaves the step to the routing cache."""

    def __init__(self, cooldowns: CooldownCache, usage: RouterUsage) -> None:
        self._cooldowns: Final = cooldowns
        self._usage: Final = usage

    def answer_sync(self, op: object, key: str, value: object, ttl: float | None) -> Handled | None:
        store: Final = self._cooldowns.cooldown_store
        parsed: Final = parse_key(key)
        target: Final = _service_target(parsed)
        if target is None:
            return None
        with service_target(target):
            match op, parsed:
                case "get", CooldownKey(model_id):
                    return Handled(_aligned((model_id,), store.active_sync((model_id,)))[0])
                case "set", CooldownKey(model_id):
                    store.put(model_id, _cooldown_value(value), ttl or 0.0)
                    return Handled(None)
                case "delete", CooldownKey(model_id):
                    store.evict_local(model_id)
                    return Handled(None)
                case "increment", AllowedFailsKey(model_id, suffix):
                    return Handled(store.increment_allowed_fails(model_id, suffix, ttl or 0.0))
                case "increment", UsageKey(counter):
                    return Handled(self._usage.increment_sync(counter, _by(value), int(ttl or 0)))
                case _:
                    return None

    async def answer(self, op: object, key: str, value: object, ttl: float | None) -> Handled | None:
        parsed: Final = parse_key(key)
        target: Final = _service_target(parsed)
        if target is None:
            return None
        store: Final = self._cooldowns.cooldown_store
        with service_target(target):
            match op, parsed:
                case "get", CooldownKey(model_id):
                    return Handled(_aligned((model_id,), await store.active((model_id,), batch=Batch.NONE))[0])
                case "set", CooldownKey(model_id):
                    await store.async_put(model_id, _cooldown_value(value), ttl or 0.0)
                    return Handled(None)
                case "increment", AllowedFailsKey(model_id, suffix):
                    return Handled(await store.async_increment_allowed_fails(model_id, suffix, ttl or 0.0))
                case "increment", UsageKey(counter):
                    by: Final = _by(value)
                    return Handled(await self._usage.increment(counter, by, int(ttl or 0)))
                case _:
                    return self.answer_sync(op, key, value, ttl)

    def batch_get_sync(self, keys: Sequence[str]) -> tuple[object, ...] | None:
        model_ids: Final = _cooldown_ids(keys)
        if model_ids is not None:
            with service_target(ROUTER_COOLDOWNS_TARGET):
                return _aligned(model_ids, self._cooldowns.cooldown_store.active_sync(model_ids))
        counters: Final = _usage_counters(keys)
        if counters is None:
            return None
        with service_target(ROUTER_USAGE_TARGET):
            return tuple(self._usage.get_many_sync(counters))

    async def batch_get(self, keys: Sequence[str]) -> tuple[object, ...] | None:
        model_ids: Final = _cooldown_ids(keys)
        if model_ids is not None:
            store: Final = self._cooldowns.cooldown_store
            with service_target(ROUTER_COOLDOWNS_TARGET):
                return _aligned(model_ids, await routing_prefetch.read_cooldowns(store, model_ids, None))
        counters: Final = _usage_counters(keys)
        if counters is None:
            return None
        with service_target(ROUTER_USAGE_TARGET):
            return tuple(await self._usage.get_many(counters))

    @with_service_target(ROUTER_USAGE_TARGET)
    def increment_usage_sync(self, keys: Sequence[object], amounts: Sequence[object]) -> bool:
        found: Final = _deployment_increment(keys, amounts)
        if found is None:
            return False
        tpm, tokens, requests = found
        self._usage.increment_deployment_sync(tpm.deployment_id, tpm.model, tpm.minute, tokens, requests)
        return True

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment_usage(self, keys: Sequence[object], amounts: Sequence[object]) -> bool:
        found: Final = _deployment_increment(keys, amounts)
        if found is None:
            return False
        tpm, tokens, requests = found
        await self._usage.increment_deployment(tpm.deployment_id, tpm.model, tpm.minute, tokens, requests)
        return True
