import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.redis.transport.contract import RedisFailure, RedisTransport
from litellm.types.llms.openai import ResponsesAPIResponse

KEY_PREFIX: Final = "litellm:polling:response:"

_DOCUMENT: Final = TypeAdapter(str)
_STATE: Final = TypeAdapter(dict[str, JsonValue])


def state_key(polling_id: str) -> str:
    return f"{KEY_PREFIX}{polling_id}"


class ResponsePollingStore(Protocol):
    async def create(
        self, polling_id: str, response: ResponsesAPIResponse, *, ttl: timedelta
    ) -> None | RedisFailure: ...

    async def update(self, polling_id: str, state: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure: ...

    async def get(self, polling_id: str) -> Mapping[str, JsonValue] | None | RedisFailure: ...

    async def delete(self, polling_id: str) -> None | RedisFailure: ...


@dataclass(frozen=True, slots=True)
class TransportResponsePollingStore:
    transport: RedisTransport

    async def create(self, polling_id: str, response: ResponsesAPIResponse, *, ttl: timedelta) -> None | RedisFailure:
        return await self._write(polling_id, json.dumps(response.model_dump_json()), ttl)

    async def update(self, polling_id: str, state: Mapping[str, object], *, ttl: timedelta) -> None | RedisFailure:
        return await self._write(polling_id, json.dumps(json.dumps(state)), ttl)

    async def get(self, polling_id: str) -> Mapping[str, JsonValue] | None | RedisFailure:
        raw: Final = await self.transport.get(state_key(polling_id))
        if raw is None or isinstance(raw, RedisFailure):
            return raw
        try:
            return _STATE.validate_json(_DOCUMENT.validate_json(raw))
        except ValidationError as e:
            return RedisFailure(kind="decode", message=f"polling state is not a JSON object: {e}")

    async def delete(self, polling_id: str) -> None | RedisFailure:
        deleted: Final = await self.transport.delete(state_key(polling_id))
        return deleted if isinstance(deleted, RedisFailure) else None

    async def _write(self, polling_id: str, document: str, ttl: timedelta) -> None | RedisFailure:
        written: Final = await self.transport.set(state_key(polling_id), document, ttl=ttl)
        return written if isinstance(written, RedisFailure) else None
