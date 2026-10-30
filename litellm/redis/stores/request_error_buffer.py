import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.constants import REDIS_REQUEST_ERRORS_BUFFER_KEY
from litellm.redis.transport.contract import RedisFailure, RedisTransport
from litellm.types.proxy.request_errors import RequestErrorKey, RequestErrorSnapshot

_BufferedRow: TypeAlias = tuple[str, str, str, str, str, int, int]
_BUFFERED_ROWS: Final = TypeAdapter(tuple[_BufferedRow, ...])
RequestErrorEntries: TypeAlias = tuple[str | bytes, ...]
RequestErrorItems: TypeAlias = tuple[tuple[RequestErrorKey, int], ...]


class RequestErrorBufferStore(Protocol):
    async def append(self, snapshot: RequestErrorSnapshot) -> None | RedisFailure: ...

    async def restore(self, entries: RequestErrorEntries) -> None | RedisFailure: ...

    async def take(self, *, count: int) -> RequestErrorEntries | RedisFailure: ...

    def decode(self, entries: RequestErrorEntries) -> RequestErrorItems | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportRequestErrorBufferStore:
    transport: RedisTransport

    async def append(self, snapshot: RequestErrorSnapshot) -> None | RedisFailure:
        if not snapshot:
            return None
        rows: Final = tuple(
            (key.date, key.api_key, key.team_id, key.user_id, key.model_group, key.status_code, count)
            for key, count in snapshot.items()
        )
        return await self.restore((json.dumps(rows),))

    async def restore(self, entries: RequestErrorEntries) -> None | RedisFailure:
        if not entries:
            return None
        reply: Final = await self.transport.rpush(REDIS_REQUEST_ERRORS_BUFFER_KEY, *entries)
        return reply if isinstance(reply, RedisFailure) else None

    async def take(self, *, count: int) -> RequestErrorEntries | RedisFailure:
        reply: Final = await self.transport.lpop(REDIS_REQUEST_ERRORS_BUFFER_KEY, count)
        if isinstance(reply, RedisFailure):
            return reply
        return () if reply is None else (reply,) if isinstance(reply, bytes) else reply

    def decode(self, entries: RequestErrorEntries) -> RequestErrorItems | RedisFailure:
        try:
            return tuple(_decode_entries(entries))
        except ValidationError as exc:
            return RedisFailure(kind="decode", message=str(exc))


def _decode_entries(entries: RequestErrorEntries) -> Iterator[tuple[RequestErrorKey, int]]:
    for entry in entries:
        for date, api_key, team_id, user_id, model_group, status_code, failed in _BUFFERED_ROWS.validate_json(entry):
            yield (
                RequestErrorKey(
                    date=date,
                    api_key=api_key,
                    team_id=team_id,
                    user_id=user_id,
                    model_group=model_group,
                    status_code=status_code,
                ),
                failed,
            )
