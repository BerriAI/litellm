import asyncio
import time
from collections.abc import Awaitable, Callable, Hashable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Generic, TypeVar

from litellm.proxy.agent_endpoints.external_agents.models import HookFailure, MappedUser, Policy, Source

T: Final = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Entry(Generic[T]):
    value: T
    expires_at: float


class MetadataCache(Generic[T]):
    def __init__(self, *, ttl: float = 20, capacity: int = 10000, clock: Callable[[], float] = time.monotonic) -> None:
        self.ttl = ttl
        self.capacity = capacity
        self.clock = clock
        self.entries: Mapping[Hashable, Entry[T]] = MappingProxyType({})
        self.pending: Mapping[tuple[int, Hashable], asyncio.Task[T]] = MappingProxyType({})
        self.generation = 0

    def invalidate(self) -> None:
        self.generation += 1
        self.entries = MappingProxyType({})

    def put(self, key: Hashable, value: T) -> None:
        now: Final = self.clock()
        retained: Final = tuple((k, v) for k, v in self.entries.items() if k != key and v.expires_at > now)
        bounded: Final = retained[-(self.capacity - 1) :] if self.capacity > 1 else ()
        self.entries = MappingProxyType({**dict(bounded), key: Entry(value, now + self.ttl)})

    async def get(self, key: Hashable, load: Callable[[], Awaitable[T]]) -> T:
        entry: Final = self.entries.get(key)
        if entry is not None and entry.expires_at > self.clock():
            return entry.value
        identity: Final = (self.generation, key)
        task: Final = self.pending.get(identity)
        if task is not None:
            return await asyncio.shield(task)
        started: Final = asyncio.create_task(self._load(identity, load))
        self.pending = MappingProxyType({**self.pending, identity: started})
        return await asyncio.shield(started)

    async def _load(self, identity: tuple[int, Hashable], load: Callable[[], Awaitable[T]]) -> T:
        try:
            value: Final = await load()
            if identity[0] == self.generation:
                self.put(identity[1], value)
            return value
        finally:
            self.pending = MappingProxyType({k: v for k, v in self.pending.items() if k != identity})


class GovernanceMetadata:
    def __init__(self) -> None:
        self.sources = MetadataCache[Source | None]()
        self.actors = MetadataCache[MappedUser | None]()
        self.policies = MetadataCache[Policy | None]()
        self.bindings = MetadataCache[HookFailure | None]()

    def invalidate(self) -> None:
        self.sources.invalidate()
        self.actors.invalidate()
        self.policies.invalidate()
        self.bindings.invalidate()


metadata: Final = GovernanceMetadata()
