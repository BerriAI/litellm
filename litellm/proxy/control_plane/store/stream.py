from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from pydantic import SecretStr, TypeAdapter
from redis.asyncio import Redis

from litellm.proxy.control_plane.store.models import Change, Entry

_READ: Final = TypeAdapter(tuple[tuple[str, tuple[tuple[str, dict[str, str]], ...]], ...])
_ID: Final = TypeAdapter(str)


class ConfigStream(Protocol):
    async def append(self, change: Change) -> str: ...
    async def read(self, after: str) -> tuple[Entry, ...]: ...
    async def close(self) -> None: ...


@runtime_checkable
class RedisCommands(Protocol):
    async def xadd(self, name: str, fields: Mapping[str, str], *, maxlen: int, approximate: bool) -> object: ...
    async def xread(self, streams: Mapping[str, str], *, count: int, block: int) -> object: ...
    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class RedisStream:
    client: RedisCommands
    name: str
    maxlen: int

    @classmethod
    def build(cls, url: SecretStr, name: str, maxlen: int) -> "RedisStream":
        client: Final = Redis.from_url(
            url.get_secret_value(),
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
        if not isinstance(client, RedisCommands):
            raise TypeError("The Redis async client must support stream commands and aclose.")
        return cls(client, name, maxlen)

    async def append(self, change: Change) -> str:
        return _ID.validate_python(
            await self.client.xadd(
                self.name,
                {"event": change.model_dump_json()},
                maxlen=self.maxlen,
                approximate=True,
            )
        )

    async def read(self, after: str) -> tuple[Entry, ...]:
        result: Final = _READ.validate_python(await self.client.xread({self.name: after}, count=100, block=1000))
        return tuple(
            Entry(revision=revision, change=Change.model_validate_json(fields["event"]))
            for _, entries in result
            for revision, fields in entries
        )  # comprehension-ok: flatten Redis stream batches

    async def close(self) -> None:
        await self.client.aclose()
