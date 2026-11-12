import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest
from litellm_enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger

from litellm.caching.dual_cache import DualCache
from litellm.constants import KEY_EXPIRY_ALERT_MAX_DAYS
from litellm.proxy.common_utils.key_expiry_alert_manager import KeyExpiryAlertManager
from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE

NOW: Final = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _key(**updates: object) -> dict[str, object]:
    return {
        "token": "secret-token-hash",
        "user_id": "owner",
        "key_alias": "build key",
        "expires": NOW + timedelta(days=6),
        "metadata": json.dumps({"key_expiry_alert_days": [7, 1]}),
        **updates,
    }


def _database(keys: tuple[dict[str, object], ...], email: str | None = "owner@example.com") -> SimpleNamespace:
    return SimpleNamespace(
        db=SimpleNamespace(
            query_raw=AsyncMock(return_value=keys),
            litellm_usertable=SimpleNamespace(
                find_many=AsyncMock(return_value=(SimpleNamespace(user_id="owner", user_email=email),))
            ),
            litellm_config=SimpleNamespace(find_unique=AsyncMock(return_value=None)),
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("clock_time", [NOW, NOW.astimezone(timezone(timedelta(hours=5)))])
async def test_database_query_filters_opt_in_and_batches_distinct_owners(clock_time: datetime) -> None:
    db: Final = _database((_key(), _key(token="second-token"), _key(token="third-token", user_id="second-owner")))
    db.db.litellm_usertable.find_many.return_value = (
        SimpleNamespace(user_id="owner", user_email="owner@example.com"),
        SimpleNamespace(user_id="second-owner", user_email="second@example.com"),
    )
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    await KeyExpiryAlertManager(db, DualCache(), email, lambda: clock_time).process_alerts()
    db.db.query_raw.assert_awaited_once()
    query, start, end = db.db.query_raw.call_args.args
    assert "SELECT token, key_alias, key_name, user_id, expires, metadata" in query
    assert "metadata ? 'key_expiry_alert_days'" in query
    assert "expires IS NOT NULL AND expires > $1::timestamp AND expires <= $2::timestamp" in query
    assert "user_id IS NOT NULL AND (blocked = FALSE OR blocked IS NULL)" in query
    assert (start, end) == (NOW, NOW + timedelta(days=KEY_EXPIRY_ALERT_MAX_DAYS))
    assert start.tzinfo is timezone.utc
    assert end.tzinfo is timezone.utc
    db.db.litellm_usertable.find_many.assert_awaited_once_with(where={"user_id": {"in": ["owner", "second-owner"]}})
    assert tuple(call.kwargs["user_email"] for call in email.send_key_expiring_email.await_args_list) == (
        "owner@example.com",
        "owner@example.com",
        "second@example.com",
    )


@pytest.mark.asyncio
async def test_empty_query_does_not_fetch_owners_or_send() -> None:
    db: Final = _database(())
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    await KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW).process_alerts()
    db.db.litellm_usertable.find_many.assert_not_awaited()
    email.send_key_expiring_email.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_owner_does_not_send() -> None:
    db: Final = _database((_key(),))
    db.db.litellm_usertable.find_many.return_value = ()
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    await KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW).process_alerts()
    email.send_key_expiring_email.assert_not_awaited()


@pytest.mark.asyncio
async def test_bad_raw_row_does_not_block_valid_row() -> None:
    db: Final = _database((_key(expires="invalid"), _key(token="second-token")))
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    await KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW).process_alerts()
    assert email.send_key_expiring_email.await_count == 1


@pytest.mark.asyncio
async def test_owner_lookup_over_one_chunk_still_notifies_owners_in_each_chunk() -> None:
    owner_ids: Final = tuple(f"owner-{index:05d}" for index in range(IN_LIST_CHUNK_SIZE + 1))
    recipients: Final = frozenset((owner_ids[0], owner_ids[-1]))
    db: Final = _database(tuple(_key(token=owner, user_id=owner) for owner in owner_ids))

    async def bounded_owners(where: dict[str, dict[str, list[str]]]) -> tuple[SimpleNamespace, ...]:
        ids: Final = where["user_id"]["in"]
        if len(ids) > IN_LIST_CHUNK_SIZE:
            raise RuntimeError("owner lookup exceeds the database parameter budget")
        return tuple(
            SimpleNamespace(user_id=owner, user_email=f"{owner}@example.com") for owner in ids if owner in recipients
        )

    db.db.litellm_usertable.find_many.side_effect = bounded_owners
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    await KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW).process_alerts()
    assert email.send_key_expiring_email.await_count == len(recipients)
    assert frozenset(call.kwargs["user_id"] for call in email.send_key_expiring_email.await_args_list) == recipients


@pytest.mark.asyncio
async def test_query_failure_does_not_send_or_fetch_owners() -> None:
    db: Final = _database((_key(),))
    db.db.query_raw.side_effect = RuntimeError("database unavailable")
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    with pytest.raises(RuntimeError, match="database unavailable"):
        await KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW).process_alerts()
    db.db.litellm_usertable.find_many.assert_not_awaited()
    email.send_key_expiring_email.assert_not_awaited()


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
    db.db.query_raw.return_value = (_key(expires=NOW + timedelta(days=7)),)
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
    db.db.litellm_usertable.find_many.return_value = (SimpleNamespace(user_id="owner", user_email="owner@example.com"),)
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
    assert str(key["token"]) not in email.bodies[0]
    assert "sk-raw-secret" not in email.bodies[0]


@pytest.mark.asyncio
async def test_bad_metadata_does_not_block_another_key() -> None:
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    manager: Final = KeyExpiryAlertManager(
        _database((_key(metadata="[]"), _key(token="second-token"))), DualCache(), email, lambda: NOW
    )
    await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 1
