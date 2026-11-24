"""One Redis MGET per phase (admission, reservation, post-call) for the spend counters it reads, not one GET each."""

import asyncio
from collections.abc import Awaitable, Iterator, Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import dataclass
from types import MappingProxyType, TracebackType
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import Batch, RequestRedisBatches, active_request_redis_batches
from litellm.proxy._types import UserAPIKeyAuth
from litellm.redis.stores.spend_counters import (
    SpendCounterStore,
    end_user_counter,
    key_counter,
    model_access_group_counter,
    org_counter,
    project_counter,
    tag_counter,
    team_counter,
    team_member_counter,
    user_counter,
)

_NO_VALUES: Final[Mapping[str, float | None]] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class _InflightRead:
    result: Awaitable[Mapping[str, float | None] | None]


@dataclass(frozen=True, slots=True)
class PendingSpendIncrement:
    counter_key: str
    increment: float


class SpendCounterBatch:
    """Bound counters are read with one MGET on first use; counters bound later join the next MGET.
    ``async_batch_get_cache`` maps a clean miss to ``None`` and drops keys only when Redis failed, so an absent
    key means "read it yourself" and a present ``None`` is an authoritative miss.

    Inside a ``request_redis_batch_scope`` the read rides the request's pre-call pipeline instead: the pre-call
    flush hook declares whatever is bound but unread, so whichever pre-call read the request awaits first (the
    armed cooldown read, or this batch's own) carries the spend counters in the same round trip."""

    __slots__ = ("_fetched", "_inflight", "_keys", "_loaded", "_lock", "_open", "_request", "_store")

    def __init__(self, store: SpendCounterStore) -> None:
        self._store: Final = store
        self._lock: Final = asyncio.Lock()
        self._open = True
        self._keys: frozenset[str] = frozenset()
        self._fetched: frozenset[str] = frozenset()
        self._loaded: Mapping[str, float | None] = _NO_VALUES
        self._inflight: Final[list[_InflightRead]] = []  # mutable-ok: drained by _load
        self._request: Final[RequestRedisBatches | None] = active_request_redis_batches()
        if self._request is not None:
            self._request.add_pre_call_flush_hook(self._declare_pending)

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
            if self._request is not None:
                self._declare_pending()
                await self._collect_inflight()
                return self._loaded
            pending: Final = self._keys - self._fetched
            if pending:
                self._fetched = self._fetched | pending
                fetched: Final = await self._fetch(pending)
                self._loaded = MappingProxyType({**fetched, **self._loaded})
            return self._loaded

    def _declare_pending(self) -> None:
        """Pre-call flush hook: put every bound-but-unread counter on the pre-call pipeline about to go out."""
        if self._request is None or not self._open:
            return
        pending: Final = tuple(sorted(self._keys - self._fetched))
        if not pending or not self._store.has_redis:
            return
        self._fetched = self._fetched | frozenset(pending)
        self._inflight.append(_InflightRead(result=self._store.read_many(pending, batch=Batch.PRE_CALL)))

    async def _collect_inflight(self) -> None:
        reads: Final = tuple(self._inflight)
        self._inflight.clear()
        for read in reads:
            try:
                values: Mapping[str, float | None] | None = await read.result
            except Exception as e:  # noqa: BLE001  # per-key reads take over and apply their own Redis fallback
                verbose_proxy_logger.debug("spend counter batch read failed, falling back to per-key reads: %s", e)
                continue
            if values is not None:
                self._loaded = MappingProxyType({**values, **self._loaded})

    async def _fetch(self, keys: frozenset[str]) -> Mapping[str, float | None]:
        fetched: Final = await self._store.read_many(sorted(keys), batch=Batch.NONE)
        return _NO_VALUES if fetched is None else fetched


_active_batch: Final[ContextVar[SpendCounterBatch | None]] = ContextVar("spend_counter_batch", default=None)


def active_spend_counter_batch() -> SpendCounterBatch | None:
    return _active_batch.get()


class spend_counter_batch_scope:
    """Reads inside the scope share one MGET for the keys bound here or by ``bind_*`` calls inside it.
    Opened inside a scope whose batch is still open, it binds into that batch so both phases share the MGET."""

    __slots__ = ("_counter_keys", "_store", "_token")

    def __init__(self, store: SpendCounterStore, counter_keys: frozenset[str] = frozenset()) -> None:
        self._store: Final = store
        self._counter_keys: Final = counter_keys
        self._token: Token[SpendCounterBatch | None] | None = None

    def __enter__(self) -> None:
        if not self._store.has_redis:
            return
        outer: Final = _active_batch.get()
        if outer is not None and outer.is_open:
            outer.bind(self._counter_keys)
            return
        batch: Final = SpendCounterBatch(self._store)
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


def _iter_entity_counter_keys(
    token: object,
    team_id: object,
    user_id: object,
    org_id: object,
    project_id: object,
    end_user_id: object,
) -> Iterator[str]:
    """Only string ids name a counter; anything else (None, or an unresolved placeholder in synthetic
    logging payloads) simply has no counter to bind."""
    if isinstance(token, str):
        yield key_counter(token)
    if isinstance(team_id, str):
        yield team_counter(team_id)
        if isinstance(user_id, str):
            yield team_member_counter(user_id, team_id)
    if isinstance(user_id, str):
        yield user_counter(user_id)
    if isinstance(end_user_id, str):
        yield end_user_counter(end_user_id)
    if isinstance(org_id, str):
        yield org_counter(org_id)
    if isinstance(project_id, str):
        yield project_counter(project_id)


def admission_counter_keys(token: UserAPIKeyAuth, end_user_id: str | None) -> frozenset[str]:
    return frozenset(
        _iter_entity_counter_keys(
            token=token.token,
            team_id=token.team_id,
            user_id=token.user_id,
            org_id=token.org_id,
            project_id=token.project_id,
            end_user_id=end_user_id,
        )
    )


def post_call_counter_keys(
    token: str | None,
    team_id: str | None,
    user_id: str | None,
    org_id: str | None,
    end_user_id: str | None,
    tags: Sequence[object] | None,
    model_access_groups: Sequence[object] | None,
    project_id: str | None = None,
) -> frozenset[str]:
    """Every counter ``increment_spend_counters`` warm-checks, except budget windows which bind on read."""
    entity_keys: Final = frozenset(
        _iter_entity_counter_keys(
            token=token,
            team_id=team_id,
            user_id=user_id,
            org_id=org_id,
            project_id=project_id,
            end_user_id=end_user_id,
        )
    )
    tag_keys: Final = frozenset(tag_counter(tag) for tag in tags or () if tag and isinstance(tag, str))
    group_keys: Final = frozenset(
        model_access_group_counter(group) for group in model_access_groups or () if group and isinstance(group, str)
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
