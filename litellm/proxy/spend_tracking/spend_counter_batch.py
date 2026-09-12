"""One Redis MGET per phase (admission, reservation, post-call) for the spend counters it reads, not one GET each."""

import asyncio
from collections.abc import Iterator, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Final

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import model_access_group_spend_counter_key

_CounterValues: Final = TypeAdapter(dict[str, float | None])
_NO_VALUES: Final[Mapping[str, float | None]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class PendingSpendIncrement:
    counter_key: str
    increment: float


class SpendCounterBatch:
    """Bound counters are read with one MGET on first use; counters bound later join the next MGET.
    ``async_batch_get_cache`` maps a clean miss to ``None`` and drops keys only when Redis failed, so an absent
    key means "read it yourself" and a present ``None`` is an authoritative miss."""

    __slots__ = ("_fetched", "_keys", "_loaded", "_lock", "_open", "_redis_cache")

    def __init__(self, redis_cache: RedisCache) -> None:
        self._redis_cache: Final = redis_cache
        self._lock: Final = asyncio.Lock()
        self._open = True
        self._keys: frozenset[str] = frozenset()
        self._fetched: frozenset[str] = frozenset()
        self._loaded: Mapping[str, float | None] = _NO_VALUES

    @property
    def counter_keys(self) -> frozenset[str]:
        return self._keys

    @property
    def is_open(self) -> bool:
        return self._open

    def bind(self, counter_keys: frozenset[str]) -> None:
        if self._open:
            self._keys = self._keys | counter_keys

    def close(self) -> None:
        """Later reads go to Redis directly; call before any read-then-write on the counters."""
        self._open = False

    async def read(self, counter_key: str) -> tuple[float | None, bool] | None:
        """(value, authoritative) for a bound counter, None when the caller must read Redis itself."""
        if not self._open or counter_key not in self._keys:
            return None
        loaded: Final = await self._load()
        if counter_key not in loaded:
            return None
        return loaded[counter_key], True

    def record(self, counter_key: str, value: float) -> None:
        """A write returned the counter's new value; later reads in this scope see it instead of the MGET value."""
        if not self._open:
            return
        key: Final = frozenset((counter_key,))
        self._keys = self._keys | key
        self._fetched = self._fetched | key
        self._loaded = MappingProxyType({**self._loaded, counter_key: value})

    def forget(self, counter_key: str) -> None:
        """A write left the counter's value unknown; later reads in this scope go to Redis."""
        key: Final = frozenset((counter_key,))
        self._keys = self._keys - key
        self._fetched = self._fetched - key
        self._loaded = MappingProxyType({k: v for k, v in self._loaded.items() if k != counter_key})

    async def _load(self) -> Mapping[str, float | None]:
        async with self._lock:
            pending: Final = self._keys - self._fetched
            if pending:
                self._fetched = self._fetched | pending
                fetched: Final = await self._fetch(pending)
                self._loaded = MappingProxyType({**fetched, **self._loaded})
            return self._loaded

    async def _fetch(self, keys: frozenset[str]) -> Mapping[str, float | None]:
        try:
            return _CounterValues.validate_python(
                await self._redis_cache.async_batch_get_cache(key_list=sorted(keys))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # untyped cache API
            )
        except Exception as e:  # noqa: BLE001  # per-key reads take over and apply their own Redis fallback
            verbose_proxy_logger.debug("spend counter batch read failed, falling back to per-key reads: %s", e)
            return _NO_VALUES


_active_batch: Final[ContextVar[SpendCounterBatch | None]] = ContextVar("spend_counter_batch", default=None)


def active_spend_counter_batch() -> SpendCounterBatch | None:
    return _active_batch.get()


class spend_counter_batch_scope:
    """Reads inside the scope share one MGET for the keys bound here or by ``bind_*`` calls inside it.
    Opened inside a scope whose batch is still open, it binds into that batch so both phases share the MGET."""

    __slots__ = ("_counter_keys", "_redis_cache", "_token")

    def __init__(self, redis_cache: RedisCache | None, counter_keys: frozenset[str] = frozenset()) -> None:
        self._redis_cache: Final = redis_cache
        self._counter_keys: Final = counter_keys
        self._token: Token[SpendCounterBatch | None] | None = None

    def __enter__(self) -> None:
        if self._redis_cache is None:
            return
        outer: Final = _active_batch.get()
        if outer is not None and outer.is_open:
            outer.bind(self._counter_keys)
            return
        batch: Final = SpendCounterBatch(self._redis_cache)
        batch.bind(self._counter_keys)
        self._token = _active_batch.set(batch)

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        if self._token is not None:
            _active_batch.reset(self._token)


def release_spend_counter_batch() -> None:
    batch: Final = _active_batch.get()
    if batch is not None:
        batch.close()


def _iter_admission_counter_keys(token: UserAPIKeyAuth, end_user_id: str | None) -> Iterator[str]:
    if token.token is not None:
        yield f"spend:key:{token.token}"
    if token.team_id is not None:
        yield f"spend:team:{token.team_id}"
        if token.user_id is not None:
            yield f"spend:team_member:{token.user_id}:{token.team_id}"
    if token.user_id is not None:
        yield f"spend:user:{token.user_id}"
    if end_user_id is not None:
        yield f"spend:end_user:{end_user_id}"
    if token.org_id is not None:
        yield f"spend:org:{token.org_id}"


def admission_counter_keys(token: UserAPIKeyAuth, end_user_id: str | None) -> frozenset[str]:
    return frozenset(_iter_admission_counter_keys(token, end_user_id))


def post_call_counter_keys(
    token: str | None,
    team_id: str | None,
    user_id: str | None,
    org_id: str | None,
    end_user_id: str | None,
    tags: Sequence[object] | None,
    model_access_groups: Sequence[object] | None,
) -> frozenset[str]:
    """Every counter ``increment_spend_counters`` warm-checks, except budget windows which bind on read."""
    entity_keys: Final = admission_counter_keys(
        UserAPIKeyAuth(token=token, team_id=team_id, user_id=user_id, org_id=org_id), end_user_id
    )
    tag_keys: Final = frozenset(f"spend:tag:{tag}" for tag in tags or () if tag and isinstance(tag, str))
    group_keys: Final = frozenset(
        model_access_group_spend_counter_key(group)
        for group in model_access_groups or ()
        if group and isinstance(group, str)
    )
    return entity_keys | tag_keys | group_keys


def bind_admission_counter_keys(token: UserAPIKeyAuth, end_user_id: str | None) -> None:
    """Idempotent: call again after the token gains ids (end user, team org) so those counters join the MGET."""
    bind_spend_counter_keys(admission_counter_keys(token, end_user_id))


def bind_spend_counter_keys(counter_keys: frozenset[str]) -> None:
    batch: Final = _active_batch.get()
    if batch is None:
        return
    batch.bind(counter_keys)


def record_spend_counter_value(counter_key: str, value: float) -> None:
    batch: Final = _active_batch.get()
    if batch is None:
        return
    batch.record(counter_key, value)


def forget_spend_counter(counter_key: str) -> None:
    batch: Final = _active_batch.get()
    if batch is None:
        return
    batch.forget(counter_key)


async def read_batched_spend_counter(counter_key: str) -> tuple[float | None, bool] | None:
    """Bind-on-read for counters only known at read time (budget windows); the first reader pays the MGET."""
    batch: Final = _active_batch.get()
    if batch is None:
        return None
    batch.bind(frozenset((counter_key,)))
    return await batch.read(counter_key)
