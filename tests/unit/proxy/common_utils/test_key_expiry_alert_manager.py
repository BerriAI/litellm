import asyncio
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from litellm_enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    KEY_EXPIRY_ALERT_JOB_NAME,
    KEY_EXPIRY_ALERT_MAX_DAYS,
    LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS,
)
from litellm.proxy.common_utils.key_expiry_alert_manager import KeyExpiryAlertManager
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.repositories.chunked_in import IN_LIST_CHUNK_SIZE

NOW: Final = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)


def _key(**updates: object) -> dict[str, object]:
    return {
        "token": "secret-token-hash",
        "user_id": "owner",
        "key_alias": "build key",
        "expires": NOW + timedelta(days=6),
        "metadata": json.dumps({"key_expiry_alert_days": 7}),
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
    query, start, end, default_end = db.db.query_raw.call_args.args
    assert "SELECT token, key_alias, key_name, user_id, expires, metadata" in query
    assert "expires IS NOT NULL AND expires > $1::timestamp AND expires <= $2::timestamp" in query
    assert "(metadata ? 'key_expiry_alert_days' OR expires <= $3::timestamp)" in query
    assert "user_id IS NOT NULL AND (blocked = FALSE OR blocked IS NULL)" in query
    assert (start, end, default_end) == (NOW, NOW + timedelta(days=KEY_EXPIRY_ALERT_MAX_DAYS), NOW)
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
async def test_shared_claim_sends_once_per_expiry_and_new_expiry_rearms() -> None:
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
    assert email.send_key_expiring_email.await_count == 1
    db.db.query_raw.return_value = (_key(expires=NOW + timedelta(days=7)),)
    await first.process_alerts()
    assert email.send_key_expiring_email.await_count == 2
    assert email.send_key_expiring_email.call_args.kwargs == {
        "user_id": "owner",
        "user_email": "owner@example.com",
        "key_display": "build key",
        "expires": NOW + timedelta(days=7),
        "days_remaining": 7,
    }


@pytest.mark.parametrize(
    ("metadata", "default_days", "sends"),
    [
        ({}, 7, True),
        ({}, 5, False),
        ({}, None, False),
        ({"key_expiry_alert_days": 1}, 7, False),
        ({"key_expiry_alert_days": 7}, 1, True),
        ({"key_expiry_alert_days": 7}, None, True),
    ],
)
@pytest.mark.asyncio
async def test_key_metadata_overrides_global_default(
    metadata: dict[str, int], default_days: int | None, sends: bool
) -> None:
    db: Final = _database((_key(metadata=json.dumps(metadata)),))
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    await KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW, default_days).process_alerts()
    assert email.send_key_expiring_email.await_count == int(sends)


@pytest.mark.parametrize("stored_metadata", [None, "null"])
@pytest.mark.parametrize(("default_days", "sends"), [(7, 1), (None, 0)])
@pytest.mark.asyncio
async def test_key_created_without_metadata_follows_global_default(
    stored_metadata: str | None, default_days: int | None, sends: int
) -> None:
    db: Final = _database((_key(metadata=stored_metadata),))
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    await KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW, default_days).process_alerts()
    assert email.send_key_expiring_email.await_count == sends


def _window_job(at: datetime) -> str:
    """The job lock name for the clock window `at` falls in."""
    window_start: Final = (
        int(at.timestamp() // LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS)
        * LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS
    )
    return f"{KEY_EXPIRY_ALERT_JOB_NAME}:{window_start}"


def _pod(db: SimpleNamespace, redis: RedisCache | None, clock: Callable[[], datetime]) -> KeyExpiryAlertManager:
    return KeyExpiryAlertManager(
        db,
        DualCache(),
        MagicMock(send_key_expiring_email=AsyncMock()),
        clock,
        pod_lock_manager=PodLockManager(redis_cache=redis),
    )


@pytest.mark.asyncio
async def test_another_pod_holding_this_windows_lock_skips_the_scan(router_redis: tuple[RedisCache, object]) -> None:
    redis, _ = router_redis
    assert await PodLockManager(redis_cache=redis).acquire_lock(cronjob_id=_window_job(NOW))
    db: Final = _database((_key(),))
    await _pod(db, redis, lambda: NOW).process_alerts()
    db.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_only_the_first_pod_to_fire_in_a_window_runs_it(router_redis: tuple[RedisCache, object]) -> None:
    redis, _ = router_redis
    first_db: Final = _database((_key(),))
    later_db: Final = _database((_key(),))
    first: Final = _pod(first_db, redis, lambda: NOW)
    later: Final = _pod(later_db, redis, lambda: NOW + timedelta(hours=5))
    await first.process_alerts()
    await later.process_alerts()
    await first.process_alerts()
    assert first_db.db.query_raw.await_count == 1
    later_db.db.query_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_the_same_pod_runs_again_in_the_next_window(router_redis: tuple[RedisCache, object]) -> None:
    redis, _ = router_redis
    db: Final = _database((_key(),))
    times: Final = iter((NOW, NOW + timedelta(seconds=LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS)))
    current: Final = [NOW]

    def clock() -> datetime:
        return current[0]

    pod: Final = _pod(db, redis, clock)
    for at in times:
        current[0] = at
        await pod.process_alerts()
    assert db.db.query_raw.await_count == 2


@pytest.mark.asyncio
async def test_a_failed_scan_frees_the_window_for_another_pod(router_redis: tuple[RedisCache, object]) -> None:
    redis, _ = router_redis
    failing_db: Final = _database((_key(),))
    failing_db.db.query_raw.side_effect = RuntimeError("database unavailable")
    healthy_db: Final = _database((_key(),))
    with pytest.raises(RuntimeError, match="database unavailable"):
        await _pod(failing_db, redis, lambda: NOW).process_alerts()
    await _pod(healthy_db, redis, lambda: NOW + timedelta(minutes=10)).process_alerts()
    healthy_db.db.query_raw.assert_awaited_once()


@asynccontextmanager
async def _disconnected_redis(url: str) -> AsyncIterator[RedisCache]:
    """A RedisCache whose in-process fakeredis server refuses every command with a ConnectionError, wired the same
    way as the router_redis fixture."""
    from fakeredis import FakeRedis, FakeServer
    from fakeredis.aioredis import FakeConnection
    from redis.asyncio import BlockingConnectionPool

    server: Final = FakeServer()
    server.connected = False
    client: Final = FakeRedis(server=server)
    pool: Final = BlockingConnectionPool(connection_class=FakeConnection, server=server)
    try:
        with (
            patch("redis.Redis.from_url", return_value=client),
            patch("redis.asyncio.BlockingConnectionPool.from_url", return_value=pool),
        ):
            yield RedisCache(url=url)
    finally:
        await pool.aclose()
        client.close()


@pytest.mark.asyncio
async def test_open_redis_circuit_breaker_still_scans_and_sends(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.redis import store_settings

    monkeypatch.setattr(store_settings, "REDIS_CIRCUIT_BREAKER_ENABLED", True)
    monkeypatch.setattr(store_settings, "REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD", 1)
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    async with _disconnected_redis("redis://expiry-breaker.example:6379/7") as redis:
        manager: Final = KeyExpiryAlertManager(
            _database((_key(),)), DualCache(), email, lambda: NOW, pod_lock_manager=PodLockManager(redis_cache=redis)
        )
        await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 1


@pytest.mark.asyncio
async def test_unreachable_redis_still_scans_and_sends() -> None:
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    async with _disconnected_redis("redis://expiry-down.example:6379/0") as redis:
        manager: Final = KeyExpiryAlertManager(
            _database((_key(),)), DualCache(), email, lambda: NOW, pod_lock_manager=PodLockManager(redis_cache=redis)
        )
        await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 1


@pytest.mark.asyncio
async def test_without_redis_the_job_runs_unlocked() -> None:
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    manager: Final = KeyExpiryAlertManager(
        _database((_key(),)), DualCache(), email, lambda: NOW, pod_lock_manager=PodLockManager(redis_cache=None)
    )
    await manager.process_alerts()
    assert email.send_key_expiring_email.await_count == 1


@pytest.mark.asyncio
async def test_global_default_widens_query_to_keys_without_metadata() -> None:
    db: Final = _database(())
    email: Final = MagicMock(send_key_expiring_email=AsyncMock())
    await KeyExpiryAlertManager(db, DualCache(), email, lambda: NOW, 7).process_alerts()
    assert db.db.query_raw.call_args.args[3] == NOW + timedelta(days=7)


@pytest.mark.parametrize(
    "updates",
    [
        {"user_id": None},
        {"expires": NOW},
        {"expires": NOW - timedelta(seconds=1)},
        {"metadata": "{}"},
        {"metadata": '{"key_expiry_alert_days": [7]}'},
        {"metadata": '{"key_expiry_alert_days": true}'},
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


def _email_enabled_proxy(monkeypatch: pytest.MonkeyPatch, settings: dict[str, object]) -> None:
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "premium_user", True)
    monkeypatch.setattr(proxy_server, "prisma_client", _database(()))
    monkeypatch.setattr(proxy_server, "general_settings", {"alerting": ["email"], **settings})
    monkeypatch.setattr(proxy_server.proxy_logging_obj, "email_logging_instance", _EmailTransport())


@pytest.mark.asyncio
async def test_startup_built_job_waits_for_the_proxy_pod_lock_then_scans_with_the_global_default(
    monkeypatch: pytest.MonkeyPatch, router_redis: tuple[RedisCache, object]
) -> None:
    from litellm.constants import LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS
    from litellm.proxy import proxy_server
    from litellm.proxy.proxy_server import ProxyStartupEvent

    redis, _ = router_redis
    _email_enabled_proxy(monkeypatch, {"key_expiry_alert_days": 7})
    db: Final = _database(())
    monkeypatch.setattr(proxy_server, "prisma_client", db)
    monkeypatch.setattr(
        proxy_server.proxy_logging_obj.db_spend_update_writer, "pod_lock_manager", PodLockManager(redis_cache=redis)
    )
    manager: Final = ProxyStartupEvent._build_key_expiry_alert_manager()
    assert manager is not None
    manager.clock = lambda: NOW
    other_pod: Final = PodLockManager(redis_cache=redis)
    window_job: Final = _window_job(NOW)
    assert await other_pod.acquire_lock(cronjob_id=window_job)
    await manager.process_alerts()
    db.db.query_raw.assert_not_awaited()
    await other_pod.release_lock(cronjob_id=window_job)
    await manager.process_alerts()
    _, start, _, default_end = db.db.query_raw.call_args.args
    assert default_end - start == timedelta(days=7)
    assert LITELLM_KEY_EXPIRY_ALERT_CHECK_INTERVAL_SECONDS == 12 * 60 * 60


@pytest.mark.parametrize("value", [0, [7], "7"])
def test_startup_skips_expiry_job_for_invalid_global_default(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    from litellm.proxy.proxy_server import ProxyStartupEvent

    _email_enabled_proxy(monkeypatch, {"key_expiry_alert_days": value})
    assert ProxyStartupEvent._build_key_expiry_alert_manager() is None


def test_startup_expiry_setup_failure_returns_none_instead_of_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy.proxy_server import ProxyStartupEvent

    _email_enabled_proxy(monkeypatch, {})
    monkeypatch.setitem(sys.modules, "litellm_enterprise.enterprise_callbacks.send_emails.base_email", None)
    assert ProxyStartupEvent._build_key_expiry_alert_manager() is None
