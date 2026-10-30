import ast
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal, Protocol

from pydantic import JsonValue, TypeAdapter

from litellm.redis.transport.contract import RedisFailure, RedisTransport

TPM_KEY_TEMPLATE: Final = "global_router:{id}:{model}:tpm:{current_minute}"
RPM_KEY_TEMPLATE: Final = "global_router:{id}:{model}:rpm:{current_minute}"
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)


@dataclass(frozen=True, slots=True)
class GlobalUsageCounter:
    deployment_id: str
    model: str | None
    minute: str
    metric: Literal["tpm", "rpm"]


def usage_key(counter: GlobalUsageCounter) -> str:
    template: Final = TPM_KEY_TEMPLATE if counter.metric == "tpm" else RPM_KEY_TEMPLATE
    return template.format(id=counter.deployment_id, model=counter.model, current_minute=counter.minute)


def _decode(raw: bytes | None) -> JsonValue:
    if raw is None:
        return None
    text: Final = raw.decode("utf-8")
    try:
        return _JSON.validate_json(text)
    except ValueError:
        return _JSON.validate_python(ast.literal_eval(text))


@dataclass(frozen=True, slots=True)
class GlobalUsageRead:
    values: tuple[JsonValue | bytes, ...]
    failure: RedisFailure | None = None


class RouterGlobalUsageStore(Protocol):
    async def read_global_usage(self, counters: Sequence[GlobalUsageCounter]) -> GlobalUsageRead | RedisFailure: ...

    async def record_global_usage(
        self, deployment_id: str, model: str | None, minute: str, *, tokens: float, requests: int, ttl: timedelta
    ) -> tuple[float, float] | RedisFailure: ...

    async def record_global_failure(
        self, deployment_id: str, model: str | None, minute: str, *, ttl: timedelta
    ) -> float | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportRouterGlobalUsageStore:
    transport: RedisTransport

    async def read_global_usage(self, counters: Sequence[GlobalUsageCounter]) -> GlobalUsageRead | RedisFailure:
        raw: Final = await self.transport.mget(tuple(usage_key(counter) for counter in counters))
        if isinstance(raw, RedisFailure):
            return raw
        try:
            return GlobalUsageRead(tuple(_decode(value) for value in raw))
        except (ValueError, SyntaxError, UnicodeError) as error:
            return GlobalUsageRead(raw, RedisFailure(kind="decode", message=str(error)))

    async def record_global_usage(
        self, deployment_id: str, model: str | None, minute: str, *, tokens: float, requests: int, ttl: timedelta
    ) -> tuple[float, float] | RedisFailure:
        batch: Final = self.transport.post_call_batch() or self.transport.batch()
        tpm: Final = usage_key(GlobalUsageCounter(deployment_id, model, minute, "tpm"))
        rpm: Final = usage_key(GlobalUsageCounter(deployment_id, model, minute, "rpm"))
        token_result: Final = batch.incrbyfloat(tpm, tokens)
        token_expiry: Final = batch.expire_seconds(tpm, ttl)
        request_result: Final = batch.incrbyfloat(rpm, requests)
        request_expiry: Final = batch.expire_seconds(rpm, ttl)
        token_value: Final = await token_result
        token_expired: Final = await token_expiry
        request_value: Final = await request_result
        request_expired: Final = await request_expiry
        if isinstance(token_value, RedisFailure):
            return token_value
        if isinstance(token_expired, RedisFailure):
            return token_expired
        if isinstance(request_value, RedisFailure):
            return request_value
        if isinstance(request_expired, RedisFailure):
            return request_expired
        return token_value, request_value

    async def record_global_failure(
        self, deployment_id: str, model: str | None, minute: str, *, ttl: timedelta
    ) -> float | RedisFailure:
        key: Final = usage_key(GlobalUsageCounter(deployment_id, model, minute, "rpm"))
        batch: Final = self.transport.batch()
        increment: Final = batch.incrbyfloat(key, 1)
        expiry: Final = batch.ttl(key)
        value: Final = await increment
        remaining: Final = await expiry
        if isinstance(value, RedisFailure):
            return value
        if isinstance(remaining, RedisFailure):
            return remaining
        if remaining == -1:
            expired: Final = await self.transport.expire_seconds(key, ttl)
            if isinstance(expired, RedisFailure):
                return expired
        return value
