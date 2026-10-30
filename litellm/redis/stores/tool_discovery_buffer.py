from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from pydantic import TypeAdapter, ValidationError

from litellm.constants import REDIS_TOOL_DISCOVERY_BUFFER_KEY
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.proxy._types import ToolDiscoveryQueueItem
from litellm.redis.transport.contract import RedisFailure, RedisTransport

_ITEMS: Final = TypeAdapter(list[ToolDiscoveryQueueItem])


class ToolDiscoveryBufferStore(Protocol):
    async def push(self, items: Sequence[ToolDiscoveryQueueItem]) -> int | RedisFailure: ...

    async def drain(self, *, count: int) -> tuple[tuple[ToolDiscoveryQueueItem, ...], ...] | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportToolDiscoveryBufferStore:
    transport: RedisTransport

    async def push(self, items: Sequence[ToolDiscoveryQueueItem]) -> int | RedisFailure:
        return await self.transport.rpush(REDIS_TOOL_DISCOVERY_BUFFER_KEY, safe_dumps(list(items)))

    async def drain(self, *, count: int) -> tuple[tuple[ToolDiscoveryQueueItem, ...], ...] | RedisFailure:
        popped: Final = await self.transport.lpop(REDIS_TOOL_DISCOVERY_BUFFER_KEY, count)
        if isinstance(popped, RedisFailure):
            return popped
        entries: Final = () if popped is None else (popped,) if isinstance(popped, bytes) else popped
        try:
            return tuple(tuple(_ITEMS.validate_json(entry)) for entry in entries)
        except ValidationError as error:
            return RedisFailure(kind="decode", message="Invalid tool discovery buffer entry", exception=error)
