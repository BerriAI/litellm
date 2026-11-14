import logging
import math
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._internal_context import with_service_target
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.constants import (
    KEY_EXPIRY_ALERT_JOB_NAME,
    KEY_EXPIRY_ALERT_MAX_DAYS,
    LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS,
)
from litellm.proxy.common_utils.key_expiry_alerts import key_expiry_alert_days
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.proxy.utils import PrismaClient
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.user_repository import UserRepository

_METADATA: Final[TypeAdapter[Mapping[str, object] | None]] = TypeAdapter(Mapping[str, object] | None)
_NO_METADATA: Final[Mapping[str, object]] = MappingProxyType({})
_ROWS: Final = TypeAdapter(tuple[object, ...])
_EXPIRING_KEYS_SQL: Final = """
SELECT token, key_alias, key_name, user_id, expires, metadata
FROM "LiteLLM_VerificationToken"
WHERE expires IS NOT NULL AND expires > $1::timestamp AND expires <= $2::timestamp
  AND (metadata ? 'key_expiry_alert_days' OR expires <= $3::timestamp)
  AND user_id IS NOT NULL AND (blocked = FALSE OR blocked IS NULL)
"""
logger: Final = logging.getLogger(__name__)


class _ExpiryAlertKey(BaseModel):
    model_config = ConfigDict(frozen=True)

    token: str
    key_alias: str | None = None
    key_name: str | None = None
    user_id: str | None
    expires: datetime | None
    metadata: object


def _key_or_none(row: object) -> _ExpiryAlertKey | None:
    try:
        return _ExpiryAlertKey.model_validate(row)
    except ValidationError:
        logger.exception("Invalid key expiry alert row")
        return None


class ExpiryAlertCache(Protocol):
    async def async_increment_cache(self, *, key: str, value: float, ttl: int) -> float | None: ...

    async def async_delete_cache(self, *, key: str) -> None: ...


def _metadata(value: object) -> Mapping[str, object]:
    """Key metadata as a mapping. A key created without metadata stores JSON null, which reads as empty so the
    global default still applies to it."""
    parsed: Final = _METADATA.validate_json(value) if isinstance(value, str) else _METADATA.validate_python(value)
    return _NO_METADATA if parsed is None else parsed


@runtime_checkable
class KeyExpiryEmailLogger(Protocol):
    async def send_key_expiring_email(
        self, *, user_id: str, user_email: str, key_display: str, expires: datetime, days_remaining: int
    ) -> None: ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class KeyExpiryAlertManager:
    def __init__(
        self,
        prisma_client: PrismaClient,
        cache: ExpiryAlertCache,
        email_logger: KeyExpiryEmailLogger,
        clock: Callable[[], datetime] = utc_now,
        default_days: int | None = None,
        pod_lock_manager: PodLockManager | None = None,
    ) -> None:
        self.prisma_client = prisma_client
        self.default_days = default_days
        self.cache = cache
        self.email_logger = email_logger
        self.clock = clock
        self.pod_lock_manager = pod_lock_manager

    async def process_alerts(self) -> None:
        """Entry point for the scheduled job: one pod runs it per clock window of one interval. Pods' timers fire at
        their own boot time plus jitter, so the lock is named after the window the run falls in (the same
        int(now // window) * window bucketing as file_usage_caps) and held for the whole window: the first pod to
        fire in a window runs, every later pod in that window skips, and the next window's lock is a new key. If
        Redis cannot answer, this pod runs anyway: a skipped window can let a key expire unwarned, and the per-key
        claim still guards each email."""
        lock: Final = self.pod_lock_manager if self.pod_lock_manager and self.pod_lock_manager.redis_cache else None
        if lock is None or lock.redis_cache is None:
            await self._send_due_alerts()
            return
        window_start: Final = (
            int(self.clock().timestamp() // LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS)
            * LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS
        )
        window_job: Final = f"{KEY_EXPIRY_ALERT_JOB_NAME}:{window_start}"
        if await lock.acquire_lock(
            cronjob_id=window_job, ttl=LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS, allow_reentrant=False
        ):
            try:
                await self._send_due_alerts()
            except Exception:
                await lock.release_lock(cronjob_id=window_job)
                raise
            return
        try:
            holder: Final = await lock.redis_cache.async_get_cache(  # pyright: ignore[reportUnknownMemberType]  # RedisCache.async_get_cache leaves its key parameter untyped
                PodLockManager.get_redis_lock_key(window_job)
            )
        except RedisCircuitBreakerOpenError:
            logger.warning("Key expiry alerts: Redis circuit breaker is open, running without the job lock.")
            await self._send_due_alerts()
            return
        if holder is not None:
            logger.debug("Key expiry alerts: another pod already ran this window, skipping.")
            return
        logger.warning("Key expiry alerts: could not take the job lock from Redis, running without it.")
        await self._send_due_alerts()

    async def _send_due_alerts(self) -> None:
        try:
            from litellm_enterprise.enterprise_callbacks.send_emails.endpoints import (
                get_email_settings,  # pyright: ignore[reportUnknownVariableType]  # litellm_enterprise is absent from the type-check venv
            )
        except ImportError:
            logger.debug("Enterprise package not installed, skipping key expiry alerts")
            return

        if not (await get_email_settings(self.prisma_client)).get("Virtual Key Expiring", True):  # pyright: ignore[reportUnknownMemberType]  # litellm_enterprise is absent from the type-check venv
            return
        now: Final = self.clock().astimezone(timezone.utc)
        rows: Final = _ROWS.validate_python(
            await self.prisma_client.db.query_raw(
                _EXPIRING_KEYS_SQL,
                now,
                now + timedelta(days=KEY_EXPIRY_ALERT_MAX_DAYS),
                now + timedelta(days=self.default_days or 0),
            )
        )
        keys: Final = tuple(key for row in rows if (key := _key_or_none(row)) is not None)
        user_ids: Final = sorted(frozenset(key.user_id for key in keys if key.user_id))
        if not user_ids:
            return
        owners: Final = await find_many_in(UserRepository(self.prisma_client).table, "user_id", user_ids)
        owner_emails: Final = {owner.user_id: owner.user_email for owner in owners if owner.user_email}
        for key in keys:
            try:
                await self._process_key(key, now, owner_emails)
            except Exception:
                logger.exception("Failed to process a key expiry alert")

    @with_service_target("key_expiry_alerts")
    async def _process_key(self, key: _ExpiryAlertKey, now: datetime, owner_emails: Mapping[str, str]) -> None:
        if not key.user_id or key.expires is None:
            return
        expires: Final = key.expires.replace(tzinfo=timezone.utc) if key.expires.tzinfo is None else key.expires
        remaining: Final = expires - now
        if remaining <= timedelta(0):
            return
        metadata: Final = _metadata(key.metadata)
        days: Final = (
            key_expiry_alert_days(metadata["key_expiry_alert_days"])
            if "key_expiry_alert_days" in metadata
            else self.default_days
        )
        if days is None or remaining > timedelta(days=days):
            return
        owner_email: Final = owner_emails.get(key.user_id)
        if not owner_email:
            return
        claim: Final = f"key_expiry_alert:{key.token}:{expires.isoformat()}:{days}"
        count: Final = await self.cache.async_increment_cache(
            key=claim, value=1, ttl=math.ceil(remaining.total_seconds()) + 60
        )
        if count != 1:
            return
        try:
            await self.email_logger.send_key_expiring_email(
                user_id=key.user_id,
                user_email=owner_email,
                key_display=key.key_alias
                or (f"{key.key_name[:3]}...{key.key_name[-4:]}" if key.key_name else "Virtual key"),
                expires=expires,
                days_remaining=math.ceil(remaining.total_seconds() / 86400),
            )
        except Exception:
            try:
                await self.cache.async_delete_cache(key=claim)
            except Exception:
                logger.exception("Failed to release a key expiry alert claim")
            logger.exception("Failed to send a key expiry alert")
