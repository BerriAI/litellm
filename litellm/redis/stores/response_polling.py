import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Protocol

from pydantic import JsonValue, TypeAdapter

from litellm.caching.redis_cache import RedisCache
from litellm.types.llms.openai import ResponsesAPIResponse

KEY_PREFIX: Final = "litellm:polling:response:"

_DOCUMENT: Final = TypeAdapter(str)
_STATE: Final = TypeAdapter(dict[str, JsonValue])


def state_key(polling_id: str) -> str:
    return f"{KEY_PREFIX}{polling_id}"


class ResponsePollingStore(Protocol):
    async def create(self, polling_id: str, response: ResponsesAPIResponse, *, ttl: timedelta) -> None: ...

    async def update(self, polling_id: str, state: Mapping[str, object], *, ttl: timedelta) -> None: ...

    async def get(self, polling_id: str) -> Mapping[str, JsonValue] | None: ...

    async def delete(self, polling_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class RedisCacheResponsePollingStore:
    cache: RedisCache

    async def create(self, polling_id: str, response: ResponsesAPIResponse, *, ttl: timedelta) -> None:
        await self._write(polling_id, response.model_dump_json(), ttl)

    async def update(self, polling_id: str, state: Mapping[str, object], *, ttl: timedelta) -> None:
        await self._write(polling_id, json.dumps(state), ttl)

    async def get(self, polling_id: str) -> Mapping[str, JsonValue] | None:
        raw: Final = await self.cache.async_get_cache(state_key(polling_id))
        if not raw:
            return None
        return _STATE.validate_json(_DOCUMENT.validate_python(raw))

    async def delete(self, polling_id: str) -> None:
        await self.cache.async_delete_cache(state_key(polling_id))

    async def _write(self, polling_id: str, document: str, ttl: timedelta) -> None:
        await self.cache.async_set_cache(key=state_key(polling_id), value=document, ttl=int(ttl.total_seconds()))
