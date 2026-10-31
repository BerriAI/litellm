"""
Coalesced reseed of spend counters from the authoritative DB.

When a Redis spend counter expires (or is missing on a fresh pod), enforcement
must read the current spend from somewhere. The in-process management cache
(`user_api_key_cache.team_membership.spend`, etc.) is per-pod and lags DB
writes from other pods, so trusting it allows budget bypass in multi-pod
deployments. This module reseeds from the authoritative DB instead.

A per-counter singleflight lock collapses concurrent reseeds on the same pod
to one DB query per cold-cache window. The lock dict is bounded LRU to cap
memory in long-lived deployments.
"""

import asyncio
import functools
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, Optional

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.constants import SPEND_COUNTER_COLD_SEED_JOIN_SECONDS, SPEND_COUNTER_RESEED_LOCKS_MAX_SIZE
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.db.db_lookup_gate import bounded_db_lookup, db_lookup_gate
from litellm.proxy.spend_tracking.spend_counter_batch import (
    SPEND_COUNTERS_TARGET,
    SpendCounterRead,
    forget_spend_counter,
    read_batched_spend_counter,
    record_spend_counter_value,
    spend_counter_tick,
)
from litellm.repositories.organization_repository import OrganizationRepository
from litellm.repositories.project_repository import ProjectRepository
from litellm.repositories.table_repositories import (
    BudgetWindowSpendRepository,
    EndUserRepository,
    SpendLogsRepository,
    TeamMembershipRepository,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import (
    VerificationTokenRepository,
)

if TYPE_CHECKING:
    from prisma.types import LiteLLM_EndUserTableWhereUniqueInput

    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.utils import PrismaClient


_WINDOW_SPEND_ENTITY_TYPES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Key": Litellm_EntityType.KEY.value,
        "Team": Litellm_EntityType.TEAM.value,
    }
)

END_USER_COUNTER_PREFIX: Final = "spend:end_user:"

_WINDOW_SPEND_LOG_FIELDS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Key": "api_key",
        "Team": "team_id",
    }
)


@dataclass(slots=True)
class _CachedSpendSeed:
    """This process adding its cached spend to one cold counter: the add in flight, the tick it started at, and the
    tick the last one landed at."""

    started_at: int = 0
    landed_at: int = 0
    in_flight: asyncio.Task[float | None] | None = None

    def on_this_loop(self) -> bool:
        return self.in_flight is None or self.in_flight.get_loop() is asyncio.get_running_loop()

    def start(self, add: Callable[[], Awaitable[float | None]]) -> asyncio.Task[float | None]:
        self.started_at = spend_counter_tick()
        in_flight: Final = asyncio.ensure_future(add())
        in_flight.add_done_callback(self._settle)
        self.in_flight = in_flight
        return in_flight

    def _settle(self, done: asyncio.Future[float | None]) -> None:
        if self.in_flight is done:
            self.in_flight = None
        if not done.cancelled() and done.exception() is None:
            self.landed_at = spend_counter_tick()


@functools.lru_cache(maxsize=SPEND_COUNTER_RESEED_LOCKS_MAX_SIZE)
def _cached_spend_seed(spend_counter_cache: "DualCache", counter_key: str) -> _CachedSpendSeed:
    return _CachedSpendSeed()


def _redis_breaker_closed(spend_counter_cache: "DualCache") -> bool:
    redis_cache: Final = spend_counter_cache.redis_cache
    return redis_cache is None or redis_cache.circuit_breaker_closed


def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


class SpendCounterReseed:
    """
    Reseeds spend counters from the authoritative DB and warms the cache,
    coalesced via per-counter singleflight locks.

    Counter key prefixes map to DB tables:
        spend:key:{token}                 -> LiteLLM_VerificationToken.spend
        spend:team:{team_id}              -> LiteLLM_TeamTable.spend
        spend:team_member:{uid}:{tid}     -> LiteLLM_TeamMembership.spend
        spend:user:{user_id}              -> LiteLLM_UserTable.spend
        spend:org:{org_id}                -> LiteLLM_OrganizationTable.spend
        spend:project:{project_id}        -> LiteLLM_ProjectTable.spend

    End-user and tag spend counters intentionally do not reseed here. Their
    auth paths already load the corresponding objects via get_end_user_object()
    and get_tag_objects_batch(); callers pass those values as fallback_spend.
    end_user_from_db is the one end-user read, used only as the budget floor when
    a counter sits below that cached spend: a worker that did not run the budget
    reset still caches the pre-reset end-user object, and LiteLLM_EndUserTable
    is the row the reset zeroed.
    """

    _locks: ClassVar["OrderedDict[str, asyncio.Lock]"] = OrderedDict()
    _registry_lock: ClassVar[asyncio.Lock | None] = None

    @staticmethod
    async def _get_lock(counter_key: str) -> asyncio.Lock:
        if SpendCounterReseed._registry_lock is None:
            SpendCounterReseed._registry_lock = asyncio.Lock()
        async with SpendCounterReseed._registry_lock:
            lock = SpendCounterReseed._locks.get(counter_key)
            if lock is not None:
                SpendCounterReseed._locks.move_to_end(counter_key)
                return lock
            lock = asyncio.Lock()
            SpendCounterReseed._locks[counter_key] = lock
            if len(SpendCounterReseed._locks) > SPEND_COUNTER_RESEED_LOCKS_MAX_SIZE:
                SpendCounterReseed._locks.popitem(last=False)
            return lock

    @staticmethod
    async def add_cached_spend_once(
        spend_counter_cache: "DualCache",
        counter_key: str,
        read: SpendCounterRead,
        add: Callable[[], Awaitable[float | None]],
        is_warm: Callable[[], Awaitable[bool]],
    ) -> None:
        """Run ``add``, this process adding its cached spend to a cold counter, once for all of this process's
        requests that read the counter cold before that add landed. A request joins an add that started after its read
        came back and gets its result, and starts or joins another add when that one fails. When an add landed after
        its read went out, or one that started before its read came back finishes, the request re-reads the counter
        and adds again only if the counter is gone. A request waits on other requests' adds for at most
        ``SPEND_COUNTER_COLD_SEED_JOIN_SECONDS`` in all and then adds its own. A request on another event loop than
        the add in flight, one that finds the Redis circuit breaker not closed, and each other process, still add their
        own cached spend."""
        seed: Final = _cached_spend_seed(spend_counter_cache, counter_key)
        loop: Final = asyncio.get_running_loop()
        join_by: Final = loop.time() + SPEND_COUNTER_COLD_SEED_JOIN_SECONDS

        async def add_unless_covered(cold: SpendCounterRead) -> float | None:
            in_flight: Final = seed.in_flight
            if not seed.on_this_loop() or not _redis_breaker_closed(spend_counter_cache):
                return await add()
            if in_flight is None:
                if seed.landed_at < cold.sent_at:
                    return await asyncio.shield(seed.start(add))
                forget_spend_counter(counter_key)
                sent_at: Final = spend_counter_tick()
                if await is_warm():
                    return None
                return await add_unless_covered(SpendCounterRead(sent_at=sent_at, returned_at=spend_counter_tick()))
            covers: Final = seed.started_at > cold.returned_at
            await asyncio.wait((in_flight,), timeout=max(0.0, join_by - loop.time()))
            if not in_flight.done():
                return await asyncio.shield(seed.start(add))
            if covers and not in_flight.cancelled() and in_flight.exception() is None:
                return in_flight.result()
            return await add_unless_covered(cold)

        added: Final = await add_unless_covered(read)
        if added is not None:
            record_spend_counter_value(counter_key, added)

    @staticmethod
    @with_service_target(SPEND_COUNTERS_TARGET)
    async def increment_in_memory(spend_counter_cache: "DualCache", counter_key: str, increment: float) -> float | None:
        """Apply local deltas after an in-flight reseed establishes the spend balance."""
        lock: Final = await SpendCounterReseed._get_lock(counter_key)
        async with lock:
            return await spend_counter_cache.async_increment_cache(
                key=counter_key, value=increment, local_only=True, refresh_ttl=True
            )

    @staticmethod
    async def from_db(prisma_client: Optional["PrismaClient"], counter_key: str) -> float | None:
        """
        Read the authoritative spend for a counter from the DB.

        Returns the spend value (including 0.0) when the DB is reachable
        and the row exists. Returns None when prisma is unavailable, the
        row is missing, the key format is unrecognized, or the query
        raises. Callers use None to fall back to a caller-supplied source.
        """
        if prisma_client is None:
            return None
        # Per-window key/team counters share prefixes with primary counters
        # but don't correspond to a DB row. Do not reject arbitrary entity IDs
        # or tag names that merely contain ":window:".
        if SpendCounterReseed._is_key_or_team_window_counter(counter_key):
            return None
        try:
            row: Final = await bounded_db_lookup(
                SpendCounterReseed._counter_row(prisma_client, counter_key), name="spend_counter"
            )
        except Exception:
            verbose_proxy_logger.exception("SpendCounterReseed.from_db: failed for %s", counter_key)
            return None
        if row is None:
            return None
        return float(getattr(row, "spend", 0.0) or 0.0)

    @staticmethod
    async def _counter_row(prisma_client: "PrismaClient", counter_key: str) -> object | None:
        async with db_lookup_gate.current():
            if counter_key.startswith("spend:key:"):
                token: Final = counter_key[len("spend:key:") :]
                return await VerificationTokenRepository(prisma_client).table.find_unique(where={"token": token})
            if counter_key.startswith("spend:team_member:"):
                suffix: Final = counter_key[len("spend:team_member:") :]
                if ":" not in suffix:
                    return None
                user_id, team_id = suffix.rsplit(":", 1)
                return await TeamMembershipRepository(prisma_client).table.find_unique(
                    where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}}
                )
            if counter_key.startswith("spend:team:"):
                return await TeamRepository(prisma_client).table.find_unique(
                    where={"team_id": counter_key[len("spend:team:") :]}
                )
            if counter_key.startswith("spend:user:"):
                return await UserRepository(prisma_client).table.find_unique(
                    where={"user_id": counter_key[len("spend:user:") :]}
                )
            if counter_key.startswith("spend:org:"):
                return await OrganizationRepository(prisma_client).table.find_unique(
                    where={"organization_id": counter_key[len("spend:org:") :]}
                )
            if counter_key.startswith("spend:project:"):
                return await ProjectRepository(prisma_client).table.find_unique(
                    where={"project_id": counter_key[len("spend:project:") :]}
                )
            return None

    @staticmethod
    async def end_user_from_db(prisma_client: Optional["PrismaClient"], counter_key: str) -> float | None:
        if prisma_client is None or not counter_key.startswith(END_USER_COUNTER_PREFIX):
            return None
        where: Final[LiteLLM_EndUserTableWhereUniqueInput] = {"user_id": counter_key[len(END_USER_COUNTER_PREFIX) :]}
        try:
            row: Final = await bounded_db_lookup(
                EndUserRepository(prisma_client).table.find_unique(where=where), name="end_user_spend"
            )
        except Exception:  # noqa: BLE001  # a failed floor read falls back to the cached spend, like from_db
            verbose_proxy_logger.exception("SpendCounterReseed.end_user_from_db: failed for %s", counter_key)
            return None
        if row is None:
            return None
        return float(row.spend or 0.0)

    @staticmethod
    def _is_key_or_team_window_counter(counter_key: str) -> bool:
        for prefix in ("spend:key:", "spend:team:"):
            if not counter_key.startswith(prefix):
                continue
            _, separator, duration = counter_key.rpartition(":window:")
            if not separator or not duration:
                return False
            try:
                duration_in_seconds(duration)
            except Exception:
                return False
            return True
        return False

    @staticmethod
    async def _read_active_batch(counter_key: str) -> tuple[float | None, bool] | None:
        """The request's MGET answers for this counter; a Redis miss there is authoritative."""
        return await read_batched_spend_counter(counter_key)

    @staticmethod
    @with_service_target(SPEND_COUNTERS_TARGET)
    async def coalesced(
        prisma_client: Optional["PrismaClient"],
        spend_counter_cache: "DualCache",
        counter_key: str,
        require_cache_warm: bool = False,
    ) -> float | None:
        """
        Reseed a cold spend counter from the DB and warm the cache,
        coalesced via a per-counter lock so concurrent callers (read path
        + write path) collapse to one DB query per cold-cache window.

        Returns the spend value (including 0.0 from a fresh budget reset)
        when the DB read succeeds, or None when the DB is unavailable.
        """
        lock: Final = await SpendCounterReseed._get_lock(counter_key)
        async with lock:
            batched: Final = await SpendCounterReseed._read_active_batch(counter_key)
            if batched is not None and batched[0] is not None:
                return batched[0]
            # Re-check after acquiring the lock. Skip in-memory on a clean
            # Redis miss - in-memory is per-pod-stale.
            redis_clean_miss = batched is not None
            if spend_counter_cache.redis_cache is not None and not redis_clean_miss:
                try:
                    val = await spend_counter_cache.redis_cache.async_get_cache(key=counter_key)
                    if val is not None:
                        return float(val)
                    redis_clean_miss = True
                except Exception:
                    pass
            if not redis_clean_miss:
                val = spend_counter_cache.in_memory_cache.get_cache(key=counter_key)
                if val is not None:
                    return float(val)

            db_spend: Final = await SpendCounterReseed.from_db(prisma_client, counter_key)
            if db_spend is None:
                return None
            # Warm even when 0 so subsequent reads hit cache, not DB.
            #
            # Seed via SET NX (cross-pod safe): only one pod initializes the
            # Redis key with db_spend; concurrent seeders read the winner's
            # value. INCRBYFLOAT-of-db_spend from N pods would multiply the
            # counter (N x db_spend) and trigger spurious budget alerts.
            current_value: float = float(db_spend)
            try:
                if spend_counter_cache.redis_cache is not None:
                    seeded: Final = await spend_counter_cache.redis_cache.async_set_cache(
                        key=counter_key,
                        value=db_spend,
                        nx=True,
                    )
                    if seeded:
                        current_value = float(db_spend)
                    else:
                        cached: Final = await spend_counter_cache.redis_cache.async_get_cache(key=counter_key)
                        current_value = float(cached) if cached is not None else float(db_spend)
                    spend_counter_cache.in_memory_cache.set_cache(
                        key=counter_key,
                        value=current_value,
                    )
                    record_spend_counter_value(counter_key, current_value)
                else:
                    cached_spend: Final = spend_counter_cache.in_memory_cache.get_cache(key=counter_key)
                    seeded_spend: Final = max(db_spend, float(cached_spend)) if cached_spend is not None else db_spend
                    spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=seeded_spend)
                    return seeded_spend
            except Exception:
                verbose_proxy_logger.exception(
                    "SpendCounterReseed.coalesced: failed to warm counter %s",
                    counter_key,
                )
                if require_cache_warm:
                    raise
            return current_value

    @staticmethod
    async def window_from_table(
        prisma_client: Optional["PrismaClient"],
        entity_type: str,
        entity_id: str,
        window_duration: str,
        expected_window_start: datetime,
    ) -> float | None:
        """
        Read the maintained per-window spend row by primary key.

        Returns the row's spend only when the row belongs to the window the
        caller is enforcing, i.e. ``row.window_start >= expected_window_start``.
        A row at or past the expected start was rolled by a pod whose reset_at
        was at least as fresh as this caller's, so it is trusted; an older row
        means the window boundary was crossed and nothing has rolled the row
        yet, so its spend belongs to a previous window.

        Returns None for a missing, stale or unreadable row so the caller falls
        back to the spend-logs aggregate. ``entity_type`` is the counter-facing
        label ("Key"/"Team"); anything else has no row and returns None.
        """
        if prisma_client is None:
            return None
        row_entity_type: Final = _WINDOW_SPEND_ENTITY_TYPES.get(entity_type)
        if row_entity_type is None:
            return None

        try:
            row: Final = await BudgetWindowSpendRepository(prisma_client).table.find_unique(
                where={
                    "entity_type_entity_id_window_duration": {
                        "entity_type": row_entity_type,
                        "entity_id": entity_id,
                        "window_duration": window_duration,
                    }
                }
            )
        except Exception:  # noqa: BLE001  # any read failure (DB, stale prisma client) must degrade to the aggregate path
            verbose_proxy_logger.exception(
                "SpendCounterReseed.window_from_table: failed for %s=%s window=%s",
                entity_type,
                entity_id,
                window_duration,
            )
            return None

        if row is None:
            return None
        if _as_utc(row.window_start) < _as_utc(expected_window_start):
            return None
        return float(row.spend or 0.0)

    @staticmethod
    async def window_from_db(
        prisma_client: Optional["PrismaClient"],
        entity_type: str,
        entity_id: str,
        window_duration: str | None,
        window_start: datetime,
    ) -> float | None:
        """
        Authoritative window spend: the maintained row first, falling back to
        the spend-logs aggregate only when no current row exists.

        The aggregate range-scans an unindexed table, so it must stay a
        transitional path (window configured before the row existed) rather
        than a steady-state read.
        """
        if window_duration is not None:
            from_table: Final = await SpendCounterReseed.window_from_table(
                prisma_client=prisma_client,
                entity_type=entity_type,
                entity_id=entity_id,
                window_duration=window_duration,
                expected_window_start=window_start,
            )
            if from_table is not None:
                return from_table
        return await SpendCounterReseed.window_from_spend_logs(
            prisma_client=prisma_client,
            entity_type=entity_type,
            entity_id=entity_id,
            window_start=window_start,
        )

    @staticmethod
    async def window_from_spend_logs(
        prisma_client: Optional["PrismaClient"],
        entity_type: str,
        entity_id: str,
        window_start: datetime,
    ) -> float | None:
        if prisma_client is None:
            return None

        group_field: Final = _WINDOW_SPEND_LOG_FIELDS.get(entity_type)
        if group_field is None:
            return None
        where: Final = {
            group_field: entity_id,
            "startTime": {"gte": window_start},
        }

        try:
            response: Final = await SpendLogsRepository(prisma_client).table.group_by(
                by=[group_field],
                where=where,
                sum={"spend": True},
            )
        except Exception:
            verbose_proxy_logger.exception(
                "SpendCounterReseed.window_from_spend_logs: failed for %s=%s",
                entity_type,
                entity_id,
            )
            return None

        if not response:
            return 0.0
        first_row: Final = response[0]
        sum_row: Final = first_row.get("_sum") if isinstance(first_row, dict) else getattr(first_row, "_sum", None)
        spend: Final = sum_row.get("spend") if isinstance(sum_row, dict) else getattr(sum_row, "spend", None)
        return float(spend or 0.0)

    @staticmethod
    @with_service_target(SPEND_COUNTERS_TARGET)
    async def coalesced_window(
        prisma_client: Optional["PrismaClient"],
        spend_counter_cache: "DualCache",
        counter_key: str,
        entity_type: str,
        entity_id: str,
        window_duration: str | None,
        window_start: datetime,
    ) -> float | None:
        lock: Final = await SpendCounterReseed._get_lock(counter_key)
        async with lock:
            batched: Final = await SpendCounterReseed._read_active_batch(counter_key)
            if batched is not None and batched[0] is not None:
                return batched[0]
            redis_clean_miss = batched is not None
            if spend_counter_cache.redis_cache is not None and not redis_clean_miss:
                try:
                    val = await spend_counter_cache.redis_cache.async_get_cache(key=counter_key)
                    if val is not None:
                        return float(val)
                    redis_clean_miss = True
                except Exception:
                    pass
            if not redis_clean_miss:
                val = spend_counter_cache.in_memory_cache.get_cache(key=counter_key)
                if val is not None:
                    return float(val)

            window_spend: Final = await SpendCounterReseed.window_from_db(
                prisma_client=prisma_client,
                entity_type=entity_type,
                entity_id=entity_id,
                window_duration=window_duration,
                window_start=window_start,
            )
            if window_spend is None:
                return None
            try:
                if spend_counter_cache.redis_cache is not None:
                    seeded: Final = await spend_counter_cache.redis_cache.async_set_cache(
                        key=counter_key,
                        value=window_spend,
                        nx=True,
                    )
                    if seeded:
                        current_value = window_spend
                    else:
                        current_cached_value = await spend_counter_cache.redis_cache.async_get_cache(key=counter_key)
                        if current_cached_value is None:
                            current_value = await spend_counter_cache.redis_cache.async_increment(
                                key=counter_key,
                                value=window_spend,
                            )
                        else:
                            current_value = float(current_cached_value)
                    spend_counter_cache.in_memory_cache.set_cache(
                        key=counter_key,
                        value=current_value,
                    )
                    record_spend_counter_value(counter_key, float(current_value))
                else:
                    cached_spend: Final = spend_counter_cache.in_memory_cache.get_cache(key=counter_key)
                    seeded_spend: Final = (
                        max(window_spend, float(cached_spend)) if cached_spend is not None else window_spend
                    )
                    spend_counter_cache.in_memory_cache.set_cache(key=counter_key, value=seeded_spend)
                    return seeded_spend
            except Exception:
                verbose_proxy_logger.exception(
                    "SpendCounterReseed.coalesced_window: failed to warm counter %s",
                    counter_key,
                )
                raise
            return current_value
