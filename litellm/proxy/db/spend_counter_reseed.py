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
from collections import OrderedDict
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Optional

from litellm._logging import verbose_proxy_logger
from litellm.constants import SPEND_COUNTER_RESEED_LOCKS_MAX_SIZE
from litellm.litellm_core_utils.duration_parser import duration_in_seconds

if TYPE_CHECKING:
    from litellm.caching.dual_cache import DualCache
    from litellm.proxy.utils import PrismaClient


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

    End-user and tag spend counters intentionally do not reseed here. Their
    auth paths already load the corresponding objects via get_end_user_object()
    and get_tag_objects_batch(); callers pass those values as fallback_spend.
    """

    _locks: ClassVar["OrderedDict[str, asyncio.Lock]"] = OrderedDict()
    _registry_lock: ClassVar[Optional[asyncio.Lock]] = None

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
    async def from_db(
        prisma_client: Optional["PrismaClient"], counter_key: str
    ) -> Optional[float]:
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
            if counter_key.startswith("spend:key:"):
                token = counter_key[len("spend:key:") :]
                row = await prisma_client.db.litellm_verificationtoken.find_unique(
                    where={"token": token}
                )
            elif counter_key.startswith("spend:team_member:"):
                suffix = counter_key[len("spend:team_member:") :]
                if ":" not in suffix:
                    return None
                user_id, team_id = suffix.rsplit(":", 1)
                row = await prisma_client.db.litellm_teammembership.find_unique(
                    where={"user_id_team_id": {"user_id": user_id, "team_id": team_id}}
                )
            elif counter_key.startswith("spend:team:"):
                team_id = counter_key[len("spend:team:") :]
                row = await prisma_client.db.litellm_teamtable.find_unique(
                    where={"team_id": team_id}
                )
            elif counter_key.startswith("spend:user:"):
                user_id = counter_key[len("spend:user:") :]
                row = await prisma_client.db.litellm_usertable.find_unique(
                    where={"user_id": user_id}
                )
            elif counter_key.startswith("spend:end_user:"):
                return None
            elif counter_key.startswith("spend:tag:"):
                return None
            elif counter_key.startswith("spend:org:"):
                org_id = counter_key[len("spend:org:") :]
                row = await prisma_client.db.litellm_organizationtable.find_unique(
                    where={"organization_id": org_id}
                )
            else:
                return None
        except Exception:
            verbose_proxy_logger.exception(
                "SpendCounterReseed.from_db: failed for %s", counter_key
            )
            return None
        if row is None:
            return None
        return float(getattr(row, "spend", 0.0) or 0.0)

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
    async def coalesced(
        prisma_client: Optional["PrismaClient"],
        spend_counter_cache: "DualCache",
        counter_key: str,
        require_cache_warm: bool = False,
    ) -> Optional[float]:
        """
        Reseed a cold spend counter from the DB and warm the cache,
        coalesced via a per-counter lock so concurrent callers (read path
        + write path) collapse to one DB query per cold-cache window.

        Returns the spend value (including 0.0 from a fresh budget reset)
        when the DB read succeeds, or None when the DB is unavailable.
        """
        lock = await SpendCounterReseed._get_lock(counter_key)
        async with lock:
            # Re-check after acquiring the lock. Skip in-memory on a clean
            # Redis miss - in-memory is per-pod-stale.
            redis_clean_miss = False
            if spend_counter_cache.redis_cache is not None:
                try:
                    val = await spend_counter_cache.redis_cache.async_get_cache(
                        key=counter_key
                    )
                    if val is not None:
                        return float(val)
                    redis_clean_miss = True
                except Exception:
                    pass
            if not redis_clean_miss:
                val = spend_counter_cache.in_memory_cache.get_cache(key=counter_key)
                if val is not None:
                    return float(val)

            db_spend = await SpendCounterReseed.from_db(prisma_client, counter_key)
            if db_spend is None:
                return None
            # Warm even when 0 so subsequent reads hit cache, not DB.
            try:
                if spend_counter_cache.redis_cache is not None:
                    current_value = (
                        await spend_counter_cache.redis_cache.async_increment(
                            key=counter_key,
                            value=db_spend,
                            refresh_ttl=True,
                        )
                    )
                    spend_counter_cache.in_memory_cache.set_cache(
                        key=counter_key,
                        value=current_value,
                    )
                else:
                    await spend_counter_cache.async_increment_cache(
                        key=counter_key, value=db_spend, refresh_ttl=True
                    )
            except Exception:
                verbose_proxy_logger.exception(
                    "SpendCounterReseed.coalesced: failed to warm counter %s",
                    counter_key,
                )
                if require_cache_warm:
                    raise
            return db_spend

    @staticmethod
    async def window_from_spend_logs(
        prisma_client: Optional["PrismaClient"],
        entity_type: str,
        entity_id: str,
        window_start: datetime,
    ) -> Optional[float]:
        if prisma_client is None:
            return None

        if entity_type == "Key":
            group_field = "api_key"
            where = {
                "api_key": entity_id,
                "startTime": {"gte": window_start},
            }
        elif entity_type == "Team":
            group_field = "team_id"
            where = {
                "team_id": entity_id,
                "startTime": {"gte": window_start},
            }
        else:
            return None

        try:
            response = await prisma_client.db.litellm_spendlogs.group_by(
                by=[group_field],
                where=where,  # type: ignore[arg-type]
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
        first_row = response[0]
        sum_row = (
            first_row.get("_sum")
            if isinstance(first_row, dict)
            else getattr(first_row, "_sum", None)
        )
        spend = (
            sum_row.get("spend")
            if isinstance(sum_row, dict)
            else getattr(sum_row, "spend", None)
        )
        return float(spend or 0.0)

    @staticmethod
    async def coalesced_window(
        prisma_client: Optional["PrismaClient"],
        spend_counter_cache: "DualCache",
        counter_key: str,
        entity_type: str,
        entity_id: str,
        window_start: datetime,
    ) -> Optional[float]:
        lock = await SpendCounterReseed._get_lock(counter_key)
        async with lock:
            redis_clean_miss = False
            if spend_counter_cache.redis_cache is not None:
                try:
                    val = await spend_counter_cache.redis_cache.async_get_cache(
                        key=counter_key
                    )
                    if val is not None:
                        return float(val)
                    redis_clean_miss = True
                except Exception:
                    pass
            if not redis_clean_miss:
                val = spend_counter_cache.in_memory_cache.get_cache(key=counter_key)
                if val is not None:
                    return float(val)

            window_spend = await SpendCounterReseed.window_from_spend_logs(
                prisma_client=prisma_client,
                entity_type=entity_type,
                entity_id=entity_id,
                window_start=window_start,
            )
            if window_spend is None:
                return None
            try:
                if spend_counter_cache.redis_cache is not None:
                    seeded = await spend_counter_cache.redis_cache.async_set_cache(
                        key=counter_key,
                        value=window_spend,
                        nx=True,
                    )
                    if seeded:
                        current_value = window_spend
                    else:
                        current_cached_value = (
                            await spend_counter_cache.redis_cache.async_get_cache(
                                key=counter_key
                            )
                        )
                        if current_cached_value is None:
                            current_value = (
                                await spend_counter_cache.redis_cache.async_increment(
                                    key=counter_key,
                                    value=window_spend,
                                )
                            )
                        else:
                            current_value = float(current_cached_value)
                    spend_counter_cache.in_memory_cache.set_cache(
                        key=counter_key,
                        value=current_value,
                    )
                else:
                    await spend_counter_cache.async_increment_cache(
                        key=counter_key, value=window_spend
                    )
            except Exception:
                verbose_proxy_logger.exception(
                    "SpendCounterReseed.coalesced_window: failed to warm counter %s",
                    counter_key,
                )
                raise
            return window_spend
