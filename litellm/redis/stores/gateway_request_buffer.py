import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import TypeAdapter, ValidationError

from litellm.constants import REDIS_GATEWAY_REQUESTS_BUFFER_KEY, REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2
from litellm.redis.transport.contract import RedisFailure, RedisTransport
from litellm.types.proxy.gateway_requests import GatewayRequestCounts, GatewayRequestKey, GatewayRequestSnapshot

BufferVersion: TypeAlias = Literal["v2", "legacy"]
_BufferedRow: TypeAlias = tuple[str, str, str, int | None, int, int]
_LegacyBufferedRow: TypeAlias = tuple[str, str, str, int, int]
_BUFFERED_ROWS: Final = TypeAdapter(tuple[_BufferedRow, ...])
_LEGACY_BUFFERED_ROWS: Final = TypeAdapter(tuple[_LegacyBufferedRow, ...])
GatewayRequestItems: TypeAlias = tuple[tuple[GatewayRequestKey, GatewayRequestCounts], ...]


@dataclass(frozen=True, slots=True)
class GatewayRequestBatch:
    entries: tuple[bytes, ...]
    version: BufferVersion


class GatewayRequestBufferStore(Protocol):
    async def append(self, snapshot: GatewayRequestSnapshot) -> None | RedisFailure: ...

    async def take(self, version: BufferVersion, *, count: int) -> GatewayRequestBatch | RedisFailure: ...

    def decode(self, batches: tuple[GatewayRequestBatch, ...]) -> GatewayRequestItems | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportGatewayRequestBufferStore:
    transport: RedisTransport

    async def append(self, snapshot: GatewayRequestSnapshot) -> None | RedisFailure:
        if not snapshot:
            return None
        rows: Final = tuple(
            (key.date, key.category, key.route, key.status_code, counts.successful_requests, counts.failed_requests)
            for key, counts in snapshot.items()
        )
        reply: Final = await self.transport.rpush(REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2, json.dumps(rows))
        return reply if isinstance(reply, RedisFailure) else None

    async def take(self, version: BufferVersion, *, count: int) -> GatewayRequestBatch | RedisFailure:
        key: Final = REDIS_GATEWAY_REQUESTS_BUFFER_KEY_V2 if version == "v2" else REDIS_GATEWAY_REQUESTS_BUFFER_KEY
        reply: Final = await self.transport.lpop(key, count)
        if isinstance(reply, RedisFailure):
            return reply
        entries: Final = () if reply is None else (reply,) if isinstance(reply, bytes) else reply
        return GatewayRequestBatch(entries, version)

    def decode(self, batches: tuple[GatewayRequestBatch, ...]) -> GatewayRequestItems | RedisFailure:
        try:
            return tuple(_decode_batches(batches))
        except ValidationError as exc:
            return RedisFailure(kind="decode", message=str(exc))


def _decode_rows(batch: GatewayRequestBatch) -> Iterator[_BufferedRow]:
    for entry in batch.entries:
        if batch.version == "v2":
            yield from _BUFFERED_ROWS.validate_json(entry)
        else:
            for date, category, route, succeeded, failed in _LEGACY_BUFFERED_ROWS.validate_json(entry):
                yield date, category, route, None, succeeded, failed


def _decode_batches(
    batches: tuple[GatewayRequestBatch, ...],
) -> Iterator[tuple[GatewayRequestKey, GatewayRequestCounts]]:
    for batch in batches:
        for date, category, route, status_code, succeeded, failed in _decode_rows(batch):
            yield (
                GatewayRequestKey(date=date, category=category, route=route, status_code=status_code),
                GatewayRequestCounts(successful_requests=succeeded, failed_requests=failed),
            )
