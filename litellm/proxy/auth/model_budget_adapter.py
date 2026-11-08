import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm._auth import model_budget_data
from litellm._auth.failure import NativeAuthFailure
from litellm._auth.model_budget_data import (
    ModelBudgetEntity,
    ModelBudgetEntry,
    ModelBudgetQueryFacts,
    ModelBudgetResolution,
    ModelBudgetResolveFacts,
    ModelBudgetSnapshot,
)
from litellm.caching.caching import DualCache
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.llms.bedrock.common_utils import get_bedrock_base_model
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.hooks.model_budget_shared_windows import SharedWindows
from litellm.proxy.hooks.model_max_budget_limiter import ModelAccessGroups, model_budget_spend_cache_key
from litellm.router_strategy.budget_limiter import budget_window_seconds
from litellm.types.utils import BudgetConfig

_FLOAT: Final = TypeAdapter(float)
_OBJECT: Final = TypeAdapter(object)
_ENTITIES: Final[Mapping[ModelBudgetEntity, Litellm_EntityType]] = {
    "Key": Litellm_EntityType.KEY,
    "User": Litellm_EntityType.USER,
    "Team": Litellm_EntityType.TEAM,
    "TeamMember": Litellm_EntityType.TEAM_MEMBER,
    "EndUser": Litellm_EntityType.END_USER,
}


def _entry(name: str, raw: object) -> ModelBudgetEntry:
    try:
        config: Final = BudgetConfig.model_validate(raw)
        seconds: Final = duration_in_seconds(config.budget_duration) if config.budget_duration is not None else None
    except NativeAuthFailure:
        raise
    except Exception:  # noqa: BLE001  # preserve malformed entry fallthrough
        return ModelBudgetEntry(name, None, None, None)
    return ModelBudgetEntry(name, config.max_budget, config.budget_duration, seconds)


def _spend(raw: object) -> float:
    try:
        return _FLOAT.validate_python(raw or 0.0)
    except NativeAuthFailure:
        raise
    except ValueError:
        return 0.0


@dataclass(frozen=True, slots=True)
class ModelBudgetReading:
    snapshot: ModelBudgetSnapshot
    budget_model: str | None
    named_model: bool


@dataclass(frozen=True, slots=True)
class ModelBudgetIO:
    cache: DualCache
    model_access_groups: ModelAccessGroups
    shared_windows: SharedWindows
    process_started_at: float
    clock: Callable[[], float] = time.time
    monotonic: Callable[[], float] = time.monotonic

    async def counter(self, key: str) -> float:
        redis: Final = self.cache.redis_cache
        raw: Final = _OBJECT.validate_python(
            await self.cache.async_get_cache(key=key) if redis is None else await redis.async_get_cache(key=key)  # pyright: ignore[reportUnknownMemberType]  # cache APIs remain legacy IO
        )
        return _spend(raw)


def resolve_model_budget(
    model: str,
    budgets: Mapping[str, object],
    io: ModelBudgetIO,
    team_id: str | None,
    key_budgets: Mapping[str, object] | None = None,
    *,
    team: bool = False,
) -> ModelBudgetResolution:
    base: Final = get_bedrock_base_model(model)
    raw_price: Final = as_str_mapping(_OBJECT.validate_python(litellm.model_cost.get(base))) or {}  # pyright: ignore[reportUnknownMemberType]  # project the legacy pricing map value
    facts: Final = ModelBudgetResolveFacts(
        model=model,
        bedrock_base=base,
        bedrock_provider=str(raw_price.get("litellm_provider") or ""),
        entries=tuple(_entry(name, config) for name, config in budgets.items()),
        key_entries=tuple(_entry(name, config) for name, config in (key_budgets or {}).items()),
        team=team,
    )
    initial: Final = model_budget_data.resolve(facts)
    if not initial.read_groups:
        return initial
    groups: Final = tuple(io.model_access_groups(model, team_id))
    return model_budget_data.resolve(replace(facts, access_groups=groups))


async def read_model_budget(
    io: ModelBudgetIO,
    entity: ModelBudgetEntity,
    entity_id: str | None,
    team_id: str | None,
    model: str,
    budgets: Mapping[str, object] | None,
    key_budgets: Mapping[str, object] | None = None,
) -> ModelBudgetReading:
    resolution: Final = resolve_model_budget(model, budgets or {}, io, team_id, key_budgets, team=entity == "Team")
    entry: Final = resolution.entry
    if entry is None:
        return ModelBudgetReading(ModelBudgetSnapshot(), None, False)
    spend_key: Final = model_budget_spend_cache_key(_ENTITIES[entity], entity_id, entry.name, entry.duration)
    legacy_key: Final = model_budget_spend_cache_key(_ENTITIES[entity], entity_id, model, entry.duration)
    facts: Final = ModelBudgetQueryFacts(
        entity=entity,
        snapshot=ModelBudgetSnapshot(max_budget=entry.max_budget),
        duration_seconds=entry.duration_seconds,
        process_uptime=io.monotonic() - io.process_started_at,
        same_counter=spend_key == legacy_key,
        redis_present=io.cache.redis_cache is not None,
    )
    snapshot: Final = await _read_snapshot(io, facts, spend_key, legacy_key, entry.duration)
    return ModelBudgetReading(snapshot, entry.name, resolution.named_model)


async def _read_snapshot(
    io: ModelBudgetIO,
    facts: ModelBudgetQueryFacts,
    key: str,
    legacy_key: str,
    duration: str | None,
) -> ModelBudgetSnapshot:
    decision: Final = model_budget_data.query(facts)
    match decision.read:
        case None:
            return facts.snapshot
        case "Local":
            snapshot: Final = replace(facts.snapshot, local_spend=await io.counter(key))
            return await _read_snapshot(io, replace(facts, snapshot=snapshot), key, legacy_key, duration)
        case "Legacy":
            legacy_snapshot: Final = replace(facts.snapshot, legacy_spend=await io.counter(legacy_key))
            return await _read_snapshot(io, replace(facts, snapshot=legacy_snapshot), key, legacy_key, duration)
        case "Shared":
            now: Final = io.clock()
            shared: Final = await io.shared_windows.read(key, duration) if duration is not None else None
            shared_snapshot: Final = replace(
                facts.snapshot,
                shared_spend=shared.spend if shared is not None else None,
                shared_started_at=shared.started_at if shared is not None else None,
                shared_window_seconds=budget_window_seconds(shared.started_at, duration)
                if shared is not None and duration is not None
                else None,
                now=now,
            )
            return await _read_snapshot(
                io, replace(facts, snapshot=shared_snapshot, shared_read=True), key, legacy_key, duration
            )


async def enforce_model_budget(
    io: ModelBudgetIO,
    entity: ModelBudgetEntity,
    entity_id: str | None,
    team_id: str | None,
    model: str,
    budgets: Mapping[str, object] | None,
    exceeded_message: str,
    key_budgets: Mapping[str, object] | None = None,
) -> bool:
    reading: Final = await read_model_budget(io, entity, entity_id, team_id, model, budgets, key_budgets)
    decision: Final = model_budget_data.check(reading.snapshot)
    if decision.denied:
        assert reading.snapshot.max_budget is not None
        raise litellm.BudgetExceededError(
            message=exceeded_message
            if reading.named_model
            else f"{exceeded_message}, budget_model={reading.budget_model}",
            current_cost=decision.current_spend,
            max_budget=reading.snapshot.max_budget,
            entity_type=_ENTITIES[entity].value,
            entity_id=entity_id,
        )
    return True
