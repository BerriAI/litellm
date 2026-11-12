import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.models import LiteLLM_VerificationToken

from litellm.caching.dual_cache import DualCache
from litellm.proxy.common_utils.key_expiry_alert_manager import KeyExpiryAlertManager

if TYPE_CHECKING:
    from litellm.enterprise.litellm_enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger
else:
    from litellm_enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger

NOW: Final = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _key(**updates: object) -> LiteLLM_VerificationToken:
    return LiteLLM_VerificationToken.model_validate(
        {
            "token": "secret-token-hash",
            "user_id": "owner",
            "key_alias": "build key",
            "expires": NOW + timedelta(days=6),
            "metadata": json.dumps({"key_expiry_alert_days": [7, 1]}),
            "soft_budget_cooldown": False,
            "spend": 0,
            "total_spend": 0,
            "models": [],
            "aliases": "{}",
            "config": "{}",
            "permissions": "{}",
            "allowed_cache_controls": [],
            "allowed_routes": [],
            "policies": [],
            "access_group_ids": [],
            "model_spend": "{}",
            "model_max_budget": "{}",
            "budget_fallbacks": "{}",
            **updates,
        }
    )


def _database(keys: tuple[LiteLLM_VerificationToken, ...], email: str | None = "owner@example.com") -> MagicMock:
    db: Final = MagicMock()
    db.db.litellm_verificationtoken.find_many = AsyncMock(return_value=keys)
    db.db.litellm_usertable.find_unique = AsyncMock(return_value=SimpleNamespace(user_email=email))
    db.db.litellm_config.find_unique = AsyncMock(return_value=None)
    return db


@pytest.mark.asyncio
async def test_shared_claim_sends_once_per_due_threshold_and_new_expiry_rearms() -> None:
    db: Final = _database((_key(),))
    cache: Final = DualCache()
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    first: Final = KeyExpiryAlertManager(db, cache, email, lambda: NOW)
    second: Final = KeyExpiryAlertManager(db, cache, email, lambda: NOW)
    await asyncio.gather(first.process_alerts(), second.process_alerts())
    await first.process_alerts()
    assert email.send_key_expiring_email.await_count == 1
    later: Final = KeyExpiryAlertManager(db, cache, email, lambda: NOW + timedelta(days=5))
    await later.process_alerts()
    await later.process_alerts()
    assert email.send_key_expiring_email.await_count == 2
    db.db.litellm_verificationtoken.find_many.return_value = (_key(expires=NOW + timedelta(days=7)),)
    await first.process_alerts()
    assert email.send_key_expiring_email.await_count == 3
    assert email.send_key_expiring_email.call_args.kwargs == {
        "user_id": "owner",
        "user_email": "owner@example.com",
        "key_display": "build key",
        "expires": NOW + timedelta(days=7),
        "days_remaining": 7,
    }


@pytest.mark.asyncio
async def test_only_smallest_due_threshold_is_claimed() -> None:
    db: Final = _database((_key(expires=NOW + timedelta(hours=12)),))
    cache: Final = DualCache()
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    manager: Final = KeyExpiryAlertManager(db, cache, email, lambda: NOW)
    await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 1
    assert email.send_key_expiring_email.call_args.kwargs["days_remaining"] == 1
    seven_day_db: Final = _database(
        (_key(expires=NOW + timedelta(hours=12), metadata=json.dumps({"key_expiry_alert_days": [7]})),)
    )
    await KeyExpiryAlertManager(seven_day_db, cache, email, lambda: NOW).process_alerts()
    assert email.send_key_expiring_email.await_count == 2


@pytest.mark.parametrize(
    "updates",
    [
        {"user_id": None},
        {"blocked": True},
        {"expires": NOW},
        {"expires": NOW - timedelta(seconds=1)},
        {"metadata": "{}"},
        {"metadata": '{"key_expiry_alert_days": []}'},
        {"metadata": '{"key_expiry_alert_days": [true]}'},
        {"expires": NOW + timedelta(days=8)},
    ],
)
@pytest.mark.asyncio
async def test_ineligible_keys_do_not_send(updates: dict[str, object]) -> None:
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    manager: Final = KeyExpiryAlertManager(_database((_key(**updates),)), DualCache(), email, lambda: NOW)
    await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 0


@pytest.mark.asyncio
async def test_no_email_and_disabled_event_do_not_send() -> None:
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    db: Final = _database((_key(),), email=None)
    manager: Final = KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW)
    await manager.process_alerts()
    db.db.litellm_config.find_unique.return_value = SimpleNamespace(
        param_value=json.dumps({"email_settings": {"Virtual Key Expiring": False}})
    )
    db.db.litellm_usertable.find_unique.return_value = SimpleNamespace(user_email="owner@example.com")
    await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 0


@pytest.mark.asyncio
async def test_send_failure_releases_claim_and_does_not_stop_other_keys() -> None:
    db: Final = _database((_key(), _key(token="second-token")))
    email: Final = MagicMock(send_key_expiring_email=AsyncMock(side_effect=[RuntimeError("transport"), None, None]))
    manager: Final = KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW)
    await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 2
    await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 3


class _EmailTransport(BaseEmailLogger):
    def __init__(self) -> None:
        super().__init__()
        self.bodies: tuple[str, ...] = ()

    async def send_email(self, from_email: str, to_email: list[str], subject: str, html_body: str) -> None:
        self.bodies = (*self.bodies, html_body)


@pytest.mark.asyncio
async def test_job_email_uses_abbreviated_name_and_never_includes_token() -> None:
    key: Final = _key(token="secret-token-hash", key_alias=None, key_name="sk-raw-secret")
    email: Final = _EmailTransport()
    manager: Final = KeyExpiryAlertManager(_database((key,)), DualCache(), email, lambda: NOW)
    await manager.process_alerts()
    assert len(email.bodies) == 1
    assert "sk-...cret" in email.bodies[0]
    assert "2026-01-07 12:00:00 UTC" in email.bodies[0]
    assert key.token not in email.bodies[0]
    assert "sk-raw-secret" not in email.bodies[0]


@pytest.mark.asyncio
async def test_bad_metadata_does_not_block_another_key() -> None:
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    manager: Final = KeyExpiryAlertManager(
        _database((_key(metadata="[]"), _key(token="second-token"))), DualCache(), email, lambda: NOW
    )
    await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 1
