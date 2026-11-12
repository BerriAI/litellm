import logging
import math
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from importlib import import_module
from typing import Final, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._internal_context import with_service_target
from litellm.constants import KEY_EXPIRY_ALERT_MAX_DAYS
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.common_utils.key_expiry_alerts import key_expiry_alert_days
from litellm.proxy.utils import PrismaClient
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.user_repository import UserRepository

_METADATA: Final = TypeAdapter(dict[str, object])
_ROWS: Final = TypeAdapter(tuple[object, ...])
_SETTINGS: Final = TypeAdapter(dict[str, bool])
_EXPIRING_KEYS_SQL: Final = """
SELECT token, key_alias, key_name, user_id, expires, metadata
FROM "LiteLLM_VerificationToken"
WHERE metadata ? 'key_expiry_alert_days'
  AND expires IS NOT NULL AND expires > $1::timestamp AND expires <= $2::timestamp
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


@runtime_checkable
class _EmailSettingsModule(Protocol):
    async def get_email_settings(self, prisma_client: PrismaClient) -> object: ...


@runtime_checkable
class _EmailLoggerModule(Protocol):
    @property
    def BaseEmailLogger(self) -> type[CustomLogger]: ...


_EMAIL_SETTINGS_MODULE: Final = TypeAdapter(_EmailSettingsModule, config=ConfigDict(arbitrary_types_allowed=True))
_EMAIL_LOGGER_MODULE: Final = TypeAdapter(_EmailLoggerModule, config=ConfigDict(arbitrary_types_allowed=True))


def key_expiry_email_logger_type() -> type[CustomLogger]:
    return _EMAIL_LOGGER_MODULE.validate_python(
        import_module("litellm_enterprise.enterprise_callbacks.send_emails.base_email")
    ).BaseEmailLogger


class ExpiryAlertCache(Protocol):
    async def async_increment_cache(self, *, key: str, value: float, ttl: int) -> float | None: ...

    async def async_delete_cache(self, *, key: str) -> None: ...


def _metadata(value: object) -> Mapping[str, object]:
    return _METADATA.validate_json(value) if isinstance(value, str) else _METADATA.validate_python(value)


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
    ) -> None:
        self.prisma_client = prisma_client
        self.cache = cache
        self.email_logger = email_logger
        self.clock = clock

    async def process_alerts(self) -> None:
        email_settings_module: Final = _EMAIL_SETTINGS_MODULE.validate_python(
            import_module("litellm_enterprise.enterprise_callbacks.send_emails.endpoints")
        )
        settings: Final = _SETTINGS.validate_python(await email_settings_module.get_email_settings(self.prisma_client))
        if not settings.get("Virtual Key Expiring", True):
            return
        now: Final = self.clock().astimezone(timezone.utc)
        rows: Final = _ROWS.validate_python(
            await self.prisma_client.db.query_raw(
                _EXPIRING_KEYS_SQL, now, now + timedelta(days=KEY_EXPIRY_ALERT_MAX_DAYS)
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
        thresholds: Final = key_expiry_alert_days(metadata.get("key_expiry_alert_days"))
        due: Final = min((day for day in thresholds or () if remaining <= timedelta(days=day)), default=None)
        if due is None:
            return
        owner_email: Final = owner_emails.get(key.user_id)
        if not owner_email:
            return
        claim: Final = f"key_expiry_alert:{key.token}:{expires.isoformat()}:{due}"
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
