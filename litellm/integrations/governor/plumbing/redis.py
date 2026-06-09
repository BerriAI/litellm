"""L2 Redis counter store: the cross-pod consensus authority.

Every primitive is one ``EVALSHA`` of a vendored Lua script. The wrapper raises
:class:`TierDegraded` on any Redis fault; it never catches and returns 0, never
deletes a key, and never reseeds from a snapshot. Per R6, writes never retry on
an ambiguous timeout (a timeout can mean "ran but reply lost"); reads may retry
once.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, List, Protocol, Sequence

from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.integrations.governor.model.errors import TierDegraded

_LUA_DIR = Path(__file__).parent / "lua"
_SAFE_EVICTION_POLICIES = frozenset(
    {"noeviction", "volatile-lru", "volatile-lfu", "volatile-ttl", "volatile-random"}
)


@dataclass(frozen=True)
class CheckIncrementResult:
    admitted: bool
    value: float
    limit: float
    ttl_seconds: int


@dataclass(frozen=True)
class ReconcileResult:
    applied: bool
    duplicate: bool
    value: float | None


@dataclass(frozen=True)
class GcraResult:
    limited: bool
    remaining: int
    retry_after_s: float
    reset_after_s: float


class RedisClient(Protocol):
    """The subset of ``redis.asyncio.Redis`` the store needs, kept as a Protocol
    so tests inject a fake without a live server."""

    def script_load(self, script: str) -> Awaitable[str]: ...

    def evalsha(
        self, sha: str, numkeys: int, *keys_and_args: Any
    ) -> Awaitable[Any]: ...

    def get(self, key: str) -> Awaitable[Any]: ...

    def config_get(self, parameter: str) -> Awaitable[dict[str, str]]: ...


class L2Store(Protocol):
    async def check_and_increment(
        self,
        counter_key: str,
        window_key: str | None,
        *,
        limit: float,
        increment: float,
        window_period_s: int,
        ttl_s: int,
    ) -> CheckIncrementResult: ...

    async def reconcile_actual(
        self, reconciled_key: str, counter_key: str, *, delta: float, dedup_ttl_s: int
    ) -> ReconcileResult: ...

    async def gcra_admit(
        self, gcra_key: str, *, period_s: int, capacity: int, burst: int, cost: int
    ) -> GcraResult: ...

    async def read_value(self, counter_key: str) -> float | None: ...

    async def assert_safe_eviction_policy(self, *, has_fail_closed: bool) -> str: ...


def _decode(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _read_lua(name: str) -> str:
    return (_LUA_DIR / name).read_text(encoding="utf-8")


class RedisCounterStore:
    def __init__(
        self,
        client: RedisClient,
        *,
        check_increment_sha: str,
        reconcile_sha: str,
        gcra_sha: str,
    ) -> None:
        self._client = client
        self._check_increment_sha = check_increment_sha
        self._reconcile_sha = reconcile_sha
        self._gcra_sha = gcra_sha

    @classmethod
    async def create(cls, client: RedisClient) -> "RedisCounterStore":
        try:
            check_increment_sha = await client.script_load(
                _read_lua("check_and_increment.lua")
            )
            reconcile_sha = await client.script_load(_read_lua("reconcile_actual.lua"))
            gcra_sha = await client.script_load(_read_lua("gcra_admit.lua"))
        except (RedisError, OSError) as exc:
            raise TierDegraded(tier="L2", reason="script_load_failed") from exc
        return cls(
            client,
            check_increment_sha=check_increment_sha,
            reconcile_sha=reconcile_sha,
            gcra_sha=gcra_sha,
        )

    async def _eval_write(
        self, sha: str, keys: Sequence[str], args: Sequence[Any]
    ) -> Any:
        try:
            return await self._client.evalsha(sha, len(keys), *keys, *args)
        except (RedisError, TimeoutError) as exc:
            raise TierDegraded(tier="L2", reason="redis_timeout_ambiguous") from exc

    async def check_and_increment(
        self,
        counter_key: str,
        window_key: str | None,
        *,
        limit: float,
        increment: float,
        window_period_s: int,
        ttl_s: int,
    ) -> CheckIncrementResult:
        keys: List[str] = [counter_key]
        if window_key is not None:
            keys.append(window_key)
        reply = await self._eval_write(
            self._check_increment_sha,
            keys,
            [limit, increment, window_period_s, ttl_s],
        )
        return _parse_check_increment(reply)

    async def reconcile_actual(
        self, reconciled_key: str, counter_key: str, *, delta: float, dedup_ttl_s: int
    ) -> ReconcileResult:
        reply = await self._eval_write(
            self._reconcile_sha, [reconciled_key, counter_key], [delta, dedup_ttl_s]
        )
        return _parse_reconcile(reply)

    async def gcra_admit(
        self, gcra_key: str, *, period_s: int, capacity: int, burst: int, cost: int
    ) -> GcraResult:
        reply = await self._eval_write(
            self._gcra_sha, [gcra_key], [period_s, capacity, burst, cost]
        )
        return _parse_gcra(reply)

    async def read_value(self, counter_key: str) -> float | None:
        raw = await self._get_with_retry(counter_key)
        if raw is None:
            return None
        return float(_decode(raw))

    async def _get_with_retry(self, key: str) -> Any:
        try:
            return await self._client.get(key)
        except (RedisConnectionError, RedisTimeoutError):
            try:
                return await self._client.get(key)
            except (RedisError, TimeoutError) as exc:
                raise TierDegraded(tier="L2", reason="redis_unavailable") from exc
        except RedisError as exc:
            raise TierDegraded(tier="L2", reason="redis_unavailable") from exc

    async def assert_safe_eviction_policy(self, *, has_fail_closed: bool) -> str:
        """Refuse to run fail-closed policies under an unsafe maxmemory-policy.

        An ``allkeys-*`` eviction can drop a live budget counter while its window
        survives, which the R1 sentinel only catches if the counter genuinely
        has no safe re-seed. Failing closed budgets under such a policy is a
        silent-leak risk, so the engine surfaces this at startup.
        """
        try:
            result = await self._client.config_get("maxmemory-policy")
        except (RedisError, TimeoutError) as exc:
            raise TierDegraded(tier="L2", reason="config_get_failed") from exc
        policy = result.get("maxmemory-policy", "")
        if has_fail_closed and policy not in _SAFE_EVICTION_POLICIES:
            raise TierDegraded(tier="L2", reason=f"unsafe_eviction_policy:{policy}")
        return policy


def _parse_check_increment(reply: Any) -> CheckIncrementResult:
    code = int(reply[0])
    if code == 0:
        return CheckIncrementResult(
            admitted=True,
            value=float(_decode(reply[1])),
            limit=float(_decode(reply[2])),
            ttl_seconds=int(reply[3]),
        )
    if code == 1:
        return CheckIncrementResult(
            admitted=False,
            value=float(_decode(reply[1])),
            limit=float(_decode(reply[2])),
            ttl_seconds=int(reply[3]),
        )
    if code == 3:
        raise TierDegraded(tier="L2", reason="evicted_mid_window")
    raise TierDegraded(tier="L2", reason=f"lua_error:{_decode(reply[1])}")


def _parse_reconcile(reply: Any) -> ReconcileResult:
    code = int(reply[0])
    if code == 0:
        return ReconcileResult(
            applied=True, duplicate=False, value=float(_decode(reply[1]))
        )
    if code == 1:
        return ReconcileResult(applied=False, duplicate=True, value=None)
    raise TierDegraded(tier="L2", reason="reconcile_against_evicted")


def _parse_gcra(reply: Any) -> GcraResult:
    return GcraResult(
        limited=int(reply[0]) == 1,
        remaining=int(reply[1]),
        retry_after_s=float(_decode(reply[2])),
        reset_after_s=float(_decode(reply[3])),
    )
