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
from litellm.caching.redis_request_plan import RedisRequestPlan, active_redis_request_plan
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import (
    model_access_group_spend_counter_key,
    project_spend_counter_key,
)

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

    __slots__ = ("_declared", "_declared_keys", "_fetched", "_keys", "_loaded", "_lock", "_open", "_redis_cache")

    def __init__(self, redis_cache: RedisCache) -> None:
        self._redis_cache: Final = redis_cache
        self._lock: Final = asyncio.Lock()
        self._open = True
        self._keys: frozenset[str] = frozenset()
        self._fetched: frozenset[str] = frozenset()
        self._loaded: Mapping[str, float | None] = _NO_VALUES
        self._declared: list[tuple[frozenset[str], asyncio.Future[dict[str, object | None]], RedisRequestPlan]] = []
        self._declared_keys: frozenset[str] = frozenset()

    @property
    def counter_keys(self) -> frozenset[str]:
        return self._keys

    @property
    def is_open(self) -> bool:
        return self._open

    def bind(self, counter_keys: frozenset[str]) -> None:
        if not self._open:
            return
        self._keys = self._keys | counter_keys
        plan: Final = active_redis_request_plan()
        if plan is None:
            return
        declarable: Final = counter_keys - self._fetched - self._declared_keys
        if not declarable:
            return
        future: Final = plan.batch_for(self._redis_cache).mget(sorted(declarable))
        self._declared.append((declarable, future, plan))
        self._declared_keys = self._declared_keys | declarable

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
            if not pending:
                return self._loaded
            self._fetched = self._fetched | pending
            declared: Final = self._declared
            self._declared = []
            self._declared_keys = frozenset()
            values: dict[str, float | None] = {}
            remaining = pending  # rebind-ok: shrinks by each resolved declared key set
            for key_set, future, plan in declared:
                if not key_set & pending:
                    continue
                remaining = remaining - key_set
                try:
                    values.update(_CounterValues.validate_python(await plan.resolve(future)))
                except Exception as e:  # noqa: BLE001  # per-key reads take over and apply their own Redis fallback
                    self._fetched = self._fetched - key_set
                    verbose_proxy_logger.debug("spend counter batch read failed, falling back to per-key reads: %s", e)
            if remaining:
                values.update(await self._fetch(remaining))
            self._loaded = MappingProxyType({**values, **self._loaded})
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
        yield f"spend:key:{token}"
    if isinstance(team_id, str):
        yield f"spend:team:{team_id}"
        if isinstance(user_id, str):
            yield f"spend:team_member:{user_id}:{team_id}"
    if isinstance(user_id, str):
        yield f"spend:user:{user_id}"
    if isinstance(end_user_id, str):
        yield f"spend:end_user:{end_user_id}"
    if isinstance(org_id, str):
        yield f"spend:org:{org_id}"
    if isinstance(project_id, str):
        yield project_spend_counter_key(project_id)


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
