import ast
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol, TypeAlias

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport

QUEUE_PREFIX: Final = "scheduler:queue"
QueueEntry: TypeAlias = tuple[int, str]
_ENTRIES: Final = TypeAdapter(list[QueueEntry])
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_OBJECT: Final = TypeAdapter(object)


def queue_key(model_name: str) -> str:
    return f"{QUEUE_PREFIX}:{model_name}"


def queue_entries(value: object) -> Sequence[QueueEntry]:
    return _ENTRIES.validate_python(value) if isinstance(value, list) else []


def _decode(raw: bytes) -> object:
    decoded: Final = raw.decode("utf-8")
    try:
        return _JSON.validate_json(decoded)
    except ValueError:
        return _OBJECT.validate_python(ast.literal_eval(decoded))


@dataclass(frozen=True, slots=True)
class InvalidQueue:
    error: ValidationError


class SchedulerQueueStore(Protocol):
    async def read_queue(self, model_name: str) -> tuple[QueueEntry, ...] | None | InvalidQueue | RedisFailure: ...

    async def replace_queue(
        self, model_name: str, queue: Sequence[QueueEntry], *, ttl: timedelta | None
    ) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportSchedulerQueueStore:
    transport: RedisTransport

    async def read_queue(self, model_name: str) -> tuple[QueueEntry, ...] | None | InvalidQueue | RedisFailure:
        raw: Final = await self.transport.get(queue_key(model_name))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            value: Final = _decode(raw)
        except (ValueError, SyntaxError, UnicodeError) as error:
            return RedisFailure(kind="decode", message=str(error))
        if value is None:
            return None
        try:
            return tuple(queue_entries(value))
        except ValidationError as error:
            return InvalidQueue(error)

    async def replace_queue(
        self, model_name: str, queue: Sequence[QueueEntry], *, ttl: timedelta | None
    ) -> None | RedisFailure:
        reply: Final = await self.transport.set_seconds(queue_key(model_name), json.dumps(list(queue)), ttl=ttl)
        return reply if isinstance(reply, RedisFailure) else None
