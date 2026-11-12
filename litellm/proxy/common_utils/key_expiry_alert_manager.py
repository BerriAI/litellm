import logging
import math
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Protocol

from pydantic import TypeAdapter

from litellm.constants import KEY_EXPIRY_ALERT_MAX_DAYS
from litellm.proxy.common_utils.key_expiry_alerts import key_expiry_alert_days
from litellm.proxy.utils import PrismaClient
from litellm.repositories.user_repository import UserRepository
from litellm.repositories.verification_token_repository import VerificationTokenRepository

if TYPE_CHECKING:
    from prisma.models import LiteLLM_VerificationToken

_METADATA: Final = TypeAdapter(dict[str, object])
logger: Final = logging.getLogger(__name__)


class ExpiryAlertCache(Protocol):
    async def async_increment_cache(self, *, key: str, value: float, ttl: int) -> float | None: ...

    async def async_delete_cache(self, *, key: str) -> None: ...


def _metadata(value: object) -> Mapping[str, object]:
    return _METADATA.validate_json(value) if isinstance(value, str) else _METADATA.validate_python(value)


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
        if TYPE_CHECKING:
            from litellm.enterprise.litellm_enterprise.enterprise_callbacks.send_emails.endpoints import (
                get_email_settings,
            )
            from litellm.enterprise.litellm_enterprise.types.enterprise_callbacks.send_emails import EmailEvent
        else:
            from litellm_enterprise.enterprise_callbacks.send_emails.endpoints import get_email_settings
            from litellm_enterprise.types.enterprise_callbacks.send_emails import EmailEvent

        settings: Final = await get_email_settings(self.prisma_client)
        if not settings.get(EmailEvent.virtual_key_expiring.value, True):
            return
        now: Final = self.clock()
        keys: Final = await VerificationTokenRepository(self.prisma_client).table.find_many(
            where={
                "expires": {"not": None, "gt": now, "lte": now + timedelta(days=KEY_EXPIRY_ALERT_MAX_DAYS)},
                "user_id": {"not": None},
                "OR": [{"blocked": False}, {"blocked": None}],
            }
        )
        for key in keys:
            try:
                await self._process_key(key, now)
            except Exception:
                logger.exception("Failed to process a key expiry alert")

    async def _process_key(self, key: "LiteLLM_VerificationToken", now: datetime) -> None:
        if not key.user_id or key.blocked or key.expires is None:
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
        owner: Final = await UserRepository(self.prisma_client).table.find_unique(where={"user_id": key.user_id})
        if owner is None or not owner.user_email:
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
                user_email=owner.user_email,
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
