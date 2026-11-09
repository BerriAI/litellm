import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm._auth import model_budget_aliases, model_budget_data
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
from litellm.litellm_core_utils.get_model_cost_map import AliasedCostEntry
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


def _price_entry(model: str) -> object:
    return _OBJECT.validate_python(litellm.model_cost.get(model))  # pyright: ignore[reportUnknownMemberType]  # raw catalog IO


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
    price_entry: Callable[[str], object] = _price_entry

    async def counter(self, key: str) -> float:
        redis: Final = self.cache.redis_cache
        raw: Final = _OBJECT.validate_python(
            await self.cache.async_get_cache(key=key) if redis is None else await redis.async_get_cache(key=key)  # pyright: ignore[reportUnknownMemberType]  # cache APIs remain legacy IO
        )
        return _spend(raw)


@dataclass(frozen=True, slots=True)
class AliasReading:
    row: model_budget_aliases.Row
    source: AliasedCostEntry | None


def read_alias_models(model: str, io: ModelBudgetIO, readings: tuple[AliasReading, ...] = ()) -> tuple[str, ...]:
    decision: Final = model_budget_aliases.decide(
        model_budget_aliases.Facts(model, tuple(reading.row for reading in readings))
    )
    match decision.read:
        case None:
            return decision.models
        case "Entry":
            raw: Final = io.price_entry(decision.key)
            source: Final = raw if isinstance(raw, AliasedCostEntry) else None
            entry: Final = AliasReading(
                model_budget_aliases.Row(decision.key, source.names if source is not None else ()), source
            )
            return read_alias_models(model, io, (*readings, entry))
        case "Prefix":
            prefix_known: Final = decision.key.partition("/")[0] in litellm.provider_list
            updated: Final = tuple(
                replace(reading, row=replace(reading.row, prefix_known=prefix_known))
                if reading.row.key == decision.key
                else reading
                for reading in readings
            )
            return read_alias_models(model, io, updated)
        case "Provider":
            selected: Final = next(reading for reading in readings if reading.row.key == decision.key)
            assert selected.source is not None
            provider: Final = selected.source.get("litellm_provider")
            provided: Final = replace(
                selected,
                row=replace(selected.row, provider_read=True, provider=provider if isinstance(provider, str) else None),
            )
            return read_alias_models(
                model, io, tuple(provided if reading.row.key == decision.key else reading for reading in readings)
            )


def _resolve_facts(
    model: str,
    budgets: Mapping[str, object],
    io: ModelBudgetIO,
    key_budgets: Mapping[str, object] | None = None,
    *,
    team: bool = False,
    excluded_entries: tuple[str, ...] = (),
) -> ModelBudgetResolveFacts:
    base: Final = get_bedrock_base_model(model)
    raw_price: Final = as_str_mapping(io.price_entry(base)) or {}
    return ModelBudgetResolveFacts(
        model=model,
        bedrock_base=base,
        bedrock_provider=str(raw_price.get("litellm_provider") or ""),
        entries=tuple(_entry(name, config) for name, config in budgets.items()),
        key_entries=tuple(_entry(name, config) for name, config in (key_budgets or {}).items()),
        team=team,
        excluded_entries=excluded_entries,
    )


def resolve_model_budget(
    model: str,
    budgets: Mapping[str, object],
    io: ModelBudgetIO,
    team_id: str | None,
    key_budgets: Mapping[str, object] | None = None,
    *,
    team: bool = False,
    excluded_entries: tuple[str, ...] = (),
) -> ModelBudgetResolution:
    facts: Final = _resolve_facts(model, budgets, io, key_budgets, team=team, excluded_entries=excluded_entries)
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
    models: Final = read_alias_models(model, io)
    exclusions: Final = (
        model_budget_data.team_exclusions(
            tuple(_resolve_facts(candidate, {}, io, key_budgets, team=True) for candidate in models)
        )
        if entity == "Team"
        else ()
    )
    scope_models: Final = read_alias_models(model, io) if entity == "Team" else models
    return await _read_scope(io, entity, entity_id, team_id, scope_models, budgets, exclusions)


async def _read_scope(
    io: ModelBudgetIO,
    entity: ModelBudgetEntity,
    entity_id: str | None,
    team_id: str | None,
    models: tuple[str, ...],
    budgets: Mapping[str, object] | None,
    excluded_entries: tuple[str, ...],
    readings: tuple[ModelBudgetReading, ...] = (),
) -> ModelBudgetReading:
    decision: Final = model_budget_aliases.scope(
        model_budget_aliases.ScopeFacts(models, tuple(reading.snapshot for reading in readings), bool(budgets))
    )
    if decision.read_index is not None:
        reading: Final = await _read_single(
            io, entity, entity_id, team_id, models[decision.read_index], budgets, excluded_entries
        )
        return await _read_scope(
            io, entity, entity_id, team_id, models, budgets, excluded_entries, (*readings, reading)
        )
    return (
        readings[decision.result_index]
        if decision.result_index is not None
        else ModelBudgetReading(ModelBudgetSnapshot(), None, False)
    )


async def _read_single(
    io: ModelBudgetIO,
    entity: ModelBudgetEntity,
    entity_id: str | None,
    team_id: str | None,
    model: str,
    budgets: Mapping[str, object] | None,
    excluded_entries: tuple[str, ...],
) -> ModelBudgetReading:
    resolution: Final = resolve_model_budget(model, budgets or {}, io, team_id, excluded_entries=excluded_entries)
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
            shared: Final = (
                await io.shared_windows.read(spend_key=key, budget_duration=duration) if duration is not None else None
            )
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
