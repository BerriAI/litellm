"""A stand-in for ``NativeAuthCache`` with the native store's semantics: memory in front of a Redis dict, every
direct round trip logged in ``commands`` and every pre-call declaration logged on the request's ``events``."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Final, TypeVar

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.auth_cache import AuthCacheSettings, AuthCacheStore, AuthObject, WriteBack
from litellm.rust_bridge._native import LocalSizes, NativeRedisClient, NativeRedisStoreError

T = TypeVar("T")


@dataclass(frozen=True)
class _Roster:
    row: object

    def member(self, user_id: str) -> object:
        if not isinstance(self.row, dict):
            return None
        return deepcopy(
            next((item for item in self.row.get("members_with_roles", ()) if item.get("user_id") == user_id), None)
        )


@dataclass
class DeclaredOp:
    """Settles only once its request's pre-call batch was flushed, like a native declaration."""

    request: object
    answer: Callable[[], object]

    async def resolve(self) -> object:
        events: Final = getattr(self.request, "events", None)
        if events is not None and "flush_pre" not in events:
            raise ConnectionError("resolved before its batch was flushed")
        return self.answer()


def _on_request(send: object) -> bool:
    """A fake request stands in for a request's ``NativeSend``; anything else is ``NativeSend.Now()``."""
    return getattr(send, "events", None) is not None


@dataclass
class FakeAuthCache:
    """``redis`` holds what each wire key decodes to; ``fail`` makes every direct Redis round trip raise. Two doubles
    built on one ``redis`` and one ``generations`` are two workers on one Redis. Without ``attached`` it reports no
    Redis but still answers from ``redis``. ``requests`` logs the request each
    request-aware call was handed, since the native request owns which keys it found absent."""

    redis: dict[str, object] = field(default_factory=dict)
    generations: dict[str, str] = field(default_factory=dict)
    memory: dict[str, tuple[object, float | None]] = field(default_factory=dict)
    commands: list[tuple[object, ...]] = field(default_factory=list)
    settings: list[AuthCacheSettings] = field(default_factory=list)
    requests: list[tuple[str, object]] = field(default_factory=list)
    fail: BaseException | None = None
    attached: bool = True
    clock: Callable[[], float] | None = None
    expires_at: dict[str, float] = field(default_factory=dict)
    _local_generation: int = 0
    _local_epoch: int = 0
    _local_generations: dict[str, int] = field(default_factory=dict)

    def _round_trip(self, *command: object) -> None:
        self.commands.append(command)
        if self.fail is not None:
            raise self.fail

    @staticmethod
    def _declare(request: object, event: str) -> None:
        events: Final = getattr(request, "events", None)
        if events is not None:
            events.append(event)

    def attach(self, _client: NativeRedisClient) -> bool:
        attached: Final = not self.attached
        self.attached = True
        return attached

    @property
    def has_redis(self) -> bool:
        return self.attached

    async def get(self, key: AuthObject, request: object) -> object:
        self.requests.append(("get", request))
        return self.get_sync(key)

    def get_sync(self, key: AuthObject) -> object:
        cached: Final = self.get_local(key)
        if cached is not None:
            return cached
        self._round_trip("GET", key.key)
        value: Final = self.redis.get(key.key)
        if value is not None:
            self.set_local(key, value)
        return value

    async def set(self, key: AuthObject, value: object, ttl_seconds: float | None, request: object) -> None:
        self.requests.append(("set", request))
        self.set_local(key, value, ttl_seconds)
        self._round_trip("SET", key.key, ttl_seconds)
        self.redis[key.key] = value

    def set_sync(self, key: AuthObject, value: object, ttl_seconds: float | None, request: object) -> None:
        self.requests.append(("set_sync", request))
        self.set_local(key, value, ttl_seconds)
        self._round_trip("SET_STR", key.key, ttl_seconds)
        self.redis[key.key] = value

    async def lease(self, key: AuthObject) -> str | None:
        try:
            self._round_trip("GET_GENERATION", key.key)
        except NativeRedisStoreError:
            return None
        return self.generations.get(key.key)

    async def set_leased(
        self, key: AuthObject, lease: str | None, value: object, ttl_seconds: float | None, request: object
    ) -> bool:
        self.requests.append(("set_leased", request))
        try:
            self._round_trip("SET_LEASED", key.key, ttl_seconds)
        except NativeRedisStoreError:
            return True
        if self.generations.get(key.key) != lease:
            return False
        self.redis[key.key] = value
        return True

    def _void_leases(self, keys: Sequence[str]) -> None:
        self.generations.update((key, uuid.uuid4().hex) for key in keys)

    def declare_set(self, key: AuthObject, value: object, ttl_seconds: float | None, send: object) -> DeclaredOp:
        self.set_local(key, value, ttl_seconds)
        self._declare(send, f"declare set {key.key} ttl={ttl_seconds}")
        self.redis[key.key] = value
        return DeclaredOp(send, lambda: None)

    def declare_delete(self, key: str, send: object) -> DeclaredOp:
        self.delete_local(key)
        self._declare(send, f"declare del {key}")
        self._void_leases((key,))
        self.redis.pop(key, None)
        return DeclaredOp(send, lambda: None)

    async def set_many(
        self, entries: Sequence[tuple[AuthObject, object]], ttl_seconds: float | None, request: object
    ) -> None:
        self.requests.append(("set_many", request))
        for key, value in entries:
            self.set_local(key, value, ttl_seconds)
        self._round_trip("SET_MANY", tuple(key.key for key, _ in entries), ttl_seconds)
        self.redis.update((key.key, value) for key, value in entries)

    async def write_back(self, entries: Sequence[WriteBack], send: object) -> None:
        for key, value, ttl in entries:
            self.set_local(key, value, ttl)
            self.redis[key.key] = value
        written: Final = tuple((key.key, ttl) for key, _value, ttl in entries)
        if not _on_request(send):
            self._round_trip("WRITE_BACK", written)
            return
        self._declare(send, f"declare write_back {written}")

    async def delete(self, keys: Sequence[str]) -> None:
        self.delete_sync(keys)

    def delete_sync(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.delete_local(key)
        self._round_trip("DEL", tuple(keys))
        self._void_leases(keys)
        for key in keys:
            self.redis.pop(key, None)

    def get_many(self, keys: Sequence[AuthObject], send: object) -> DeclaredOp:
        on_request: Final = _on_request(send)
        wire: Final = tuple(key.key for key in keys)
        self._declare(send, f"declare auth_cache {','.join(wire)}")

        def answer() -> list[object]:
            if on_request and self.fail is not None:
                raise self.fail
            if not on_request:
                self._round_trip("MGET", wire)
            return [self.redis.get(key) for key in wire]

        return DeclaredOp(send, answer)

    async def increment_if_present(self, key: AuthObject, amount: float) -> float | None:
        self._round_trip("INCRBYFLOAT_IF_PRESENT", key.key, amount)
        current: Final = self.redis.get(key.key)
        if not isinstance(current, (int, float)):
            return None
        total: Final = float(current) + amount
        self.redis[key.key] = total
        return total

    def get_local(self, key: AuthObject) -> object:
        expires_at: Final = self.expires_at.get(key.key)
        if self.clock is not None and expires_at is not None and self.clock() >= expires_at:
            self.memory.pop(key.key, None)
        entry: Final = self.memory.get(key.key)
        return None if entry is None else entry[0]

    def get_local_by_key(self, key: str) -> object:
        entry: Final = self.memory.get(key)
        return None if entry is None else entry[0]

    @staticmethod
    def _team_fields(value: object, fields: Sequence[str] | None, user_id: str | None) -> object:
        if not isinstance(value, dict):
            return value
        if fields is not None:
            return deepcopy({name: item for name, item in value.items() if name in fields})
        members: Final = value.get("members_with_roles", ())
        if not isinstance(members, (list, tuple)) or any(not isinstance(member, dict) for member in members):
            return deepcopy(value)
        member: Final = next((member for member in members if member.get("user_id") == user_id), None)
        projected: Final = {
            name: item for name, item in value.items() if name not in ("members_with_roles", "members", "admins")
        }
        return deepcopy(
            {**projected, "members_with_roles": [member] if user_id is not None and member is not None else []}
        )

    async def team_fields(
        self, key: AuthObject, fields: Sequence[str] | None, user_id: str | None, request: object
    ) -> object:
        return self._team_fields(await self.get(key, request), fields, user_id)

    async def team_snapshot(self, key: AuthObject, request: object) -> tuple[object, _Roster | None] | None:
        row: Final = await self.get(key, request)
        return None if row is None else (self._team_fields(row, None, None), _Roster(row))

    def local_team_fields(self, key: AuthObject, fields: Sequence[str] | None, user_id: str | None) -> object:
        return self._team_fields(self.get_local(key), fields, user_id)

    def add_local_team_spend(self, key: AuthObject, delta: float) -> bool:
        row: Final = self.get_local(key)
        if not isinstance(row, dict):
            return False
        self.memory[key.key] = ({**row, "spend": (row.get("spend") or 0.0) + delta}, self.memory[key.key][1])
        return True

    def set_local(self, key: AuthObject, value: object, ttl_seconds: float | None = None) -> None:
        self.memory[key.key] = (value, ttl_seconds)
        if self.clock is not None and ttl_seconds is not None:
            self.expires_at[key.key] = self.clock() + ttl_seconds

    def delete_local(self, key: str) -> None:
        self._local_generation += 1
        self._local_generations[key] = self._local_generation
        self.memory.pop(key, None)

    def flush_local(self) -> None:
        self._local_generation += 1
        self._local_epoch = self._local_generation
        self._local_generations.clear()
        self.memory.clear()

    def local_generation(self, key: AuthObject) -> int:
        return self._local_generations.get(key.key, self._local_epoch)

    def set_local_if_current(self, generation: int, key: AuthObject, value: object, ttl_seconds: float | None) -> bool:
        if generation != self.local_generation(key):
            return False
        self.set_local(key, value, ttl_seconds)
        return True

    def len_local(self) -> LocalSizes:
        return LocalSizes(objects=len(self.memory), key_objects=0)

    def reconfigure(self, *, memory_entries: int, memory_ttl_seconds: float | None = None) -> None:
        self.settings.append(AuthCacheSettings(memory_entries, memory_ttl_seconds))

    def store(self, settings: AuthCacheSettings) -> AuthCacheStore:
        self.settings.append(settings)
        return AuthCacheStore(self)


def shared_cache(
    fake: FakeAuthCache | None = None, default_in_memory_ttl: float | None = None
) -> tuple[UserApiKeyCache, FakeAuthCache]:
    """A ``UserApiKeyCache`` with ``enable_redis_auth_cache`` on, its store backed by ``fake``."""
    double: Final = fake or FakeAuthCache()
    cache: Final = UserApiKeyCache(default_in_memory_ttl=default_in_memory_ttl, auth_cache=double.store)
    return cache, double
