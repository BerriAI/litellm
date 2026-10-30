import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.proxy._types import Litellm_EntityType
from litellm.redis.transport.contract import RedisFailure, RedisTransport

VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX: Final = "virtual_key_spend"
END_USER_SPEND_CACHE_KEY_PREFIX: Final = "end_user_model_spend"
USER_SPEND_CACHE_KEY_PREFIX: Final = "user_model_spend"
TEAM_SPEND_CACHE_KEY_PREFIX: Final = "team_model_spend"
TEAM_MEMBER_SPEND_CACHE_KEY_PREFIX: Final = "team_member_model_spend"

_SPEND_PREFIXES: Final = MappingProxyType(
    {
        Litellm_EntityType.KEY: VIRTUAL_KEY_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.USER: USER_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.END_USER: END_USER_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.TEAM: TEAM_SPEND_CACHE_KEY_PREFIX,
        Litellm_EntityType.TEAM_MEMBER: TEAM_MEMBER_SPEND_CACHE_KEY_PREFIX,
    }
)
_START_PREFIXES: Final = MappingProxyType(
    {
        Litellm_EntityType.KEY: "virtual_key_budget_start_time",
        Litellm_EntityType.USER: "user_model_budget_start_time",
        Litellm_EntityType.END_USER: "end_user_budget_start_time",
        Litellm_EntityType.TEAM: "team_model_budget_start_time",
        Litellm_EntityType.TEAM_MEMBER: "team_member_model_budget_start_time",
    }
)
_NUMBER: Final = TypeAdapter[float | None](float | None)


@dataclass(frozen=True, slots=True)
class ModelBudgetWindow:
    entity_type: Litellm_EntityType
    entity_id: str | None
    budget_model: str
    budget_duration: str | None


@dataclass(frozen=True, slots=True)
class ModelBudgetDelta:
    window: ModelBudgetWindow
    amount: float
    ttl: int


def team_member_entity_id(user_id: str, team_id: str) -> str:
    return f"{len(user_id)}:{user_id}:{team_id}"


def spend_key(window: ModelBudgetWindow) -> str:
    return f"{_SPEND_PREFIXES[window.entity_type]}:{window.entity_id}:{window.budget_model}:{window.budget_duration}"


def start_key(window: ModelBudgetWindow) -> str:
    return f"{_START_PREFIXES[window.entity_type]}:{window.entity_id}:{window.budget_model}:{window.budget_duration}"


def batch_marker_key(spend_cache_key: str, batch_id: str) -> str:
    return f"{spend_cache_key}:batch:{batch_id}"


def _number(raw: bytes | None | RedisFailure) -> float | None | RedisFailure:
    if raw is None or isinstance(raw, RedisFailure):
        return raw
    try:
        return _NUMBER.validate_json(raw)
    except ValidationError:
        return RedisFailure(kind="decode", message="model budget value is not a number")


class ModelBudgetWindowStore(Protocol):
    async def spend(self, window: ModelBudgetWindow) -> float | None | RedisFailure: ...

    async def started_at(self, window: ModelBudgetWindow) -> float | None | RedisFailure: ...

    async def readings(
        self, windows: Sequence[ModelBudgetWindow]
    ) -> tuple[float | None | RedisFailure, ...] | RedisFailure: ...

    async def set_start(self, window: ModelBudgetWindow, value: float, *, ttl: int) -> bool | RedisFailure: ...

    async def set_spend(self, window: ModelBudgetWindow, value: float, *, ttl: int) -> bool | RedisFailure: ...

    async def add_spend(self, deltas: Sequence[ModelBudgetDelta]) -> None | RedisFailure: ...

    async def claim_batch(self, window: ModelBudgetWindow, batch_id: str, *, ttl: int) -> float | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportModelBudgetWindowStore:
    transport: RedisTransport

    async def spend(self, window: ModelBudgetWindow) -> float | None | RedisFailure:
        return _number(await self.transport.get(spend_key(window)))

    async def started_at(self, window: ModelBudgetWindow) -> float | None | RedisFailure:
        return _number(await self.transport.get(start_key(window)))

    async def readings(
        self, windows: Sequence[ModelBudgetWindow]
    ) -> tuple[float | None | RedisFailure, ...] | RedisFailure:
        keys: Final = (*tuple(spend_key(window) for window in windows), *tuple(start_key(window) for window in windows))
        batch: Final = self.transport.request_batch()
        raw: Final = await (batch.mget(keys) if batch is not None else self.transport.mget(keys))
        return raw if isinstance(raw, RedisFailure) else tuple(_number(value) for value in raw)

    async def set_start(self, window: ModelBudgetWindow, value: float, *, ttl: int) -> bool | RedisFailure:
        if ttl <= 0:
            return RedisFailure(kind="server", message="invalid expire time in set")
        return await self.transport.set(start_key(window), json.dumps(value), ttl=timedelta(seconds=ttl))

    async def set_spend(self, window: ModelBudgetWindow, value: float, *, ttl: int) -> bool | RedisFailure:
        if ttl <= 0:
            return RedisFailure(kind="server", message="invalid expire time in set")
        return await self.transport.set(spend_key(window), json.dumps(value), ttl=timedelta(seconds=ttl))

    async def add_spend(self, deltas: Sequence[ModelBudgetDelta]) -> None | RedisFailure:
        batch: Final = self.transport.post_call_batch() or self.transport.batch()
        replies: Final = tuple(
            (
                batch.incrbyfloat(spend_key(delta.window), delta.amount),
                batch.expire(spend_key(delta.window), timedelta(seconds=delta.ttl))
                if delta.ttl > 0
                else batch.delete(spend_key(delta.window)),
            )
            for delta in deltas
        )
        results: Final = tuple([(await increment, await expiry) for increment, expiry in replies])
        return next((failure for failure in chain.from_iterable(results) if isinstance(failure, RedisFailure)), None)

    async def claim_batch(self, window: ModelBudgetWindow, batch_id: str, *, ttl: int) -> float | RedisFailure:
        key: Final = batch_marker_key(spend_key(window), batch_id)
        batch: Final = self.transport.post_call_batch() or self.transport.batch()
        increment: Final = batch.incrbyfloat(key, 1)
        expiry: Final = batch.expire(key, timedelta(seconds=ttl)) if ttl > 0 else batch.delete(key)
        polls: Final = await increment
        applied_expiry: Final = await expiry
        return applied_expiry if isinstance(applied_expiry, RedisFailure) else polls
