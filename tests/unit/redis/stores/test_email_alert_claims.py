import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import timedelta
from typing import Final, Literal
from unittest.mock import patch

import fakeredis
import pytest
import pytest_asyncio
from litellm_enterprise.enterprise_callbacks.send_emails.base_email import BaseEmailLogger
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import EMAIL_BUDGET_ALERT_TTL
from litellm.proxy._types import CallInfo, Litellm_EntityType
from litellm.redis.stores.email_alert_claims import SoftBudgetAlert, TransportEmailAlertClaimsStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import async_client, fake_redis_cache, legacy_increment, local_get


@dataclass(frozen=True, slots=True)
class SentEmail:
    recipients: tuple[str, ...]
    subject: str


class EmailSink(BaseEmailLogger):
    def __init__(self, cache: DualCache, on_send: Callable[[], Awaitable[None]] | None = None) -> None:
        super().__init__(internal_usage_cache=cache)  # pyright: ignore[reportUnknownMemberType]  # legacy logger init is untyped
        self.sent: tuple[SentEmail, ...] = ()
        self.on_send = on_send

    async def send_email(self, from_email: str, to_email: list[str], subject: str, html_body: str) -> None:
        self.sent += (SentEmail(tuple(sorted(to_email)), subject),)
        if self.on_send is not None:
            await self.on_send()


@pytest_asyncio.fixture(loop_scope="function")
async def redis_env() -> AsyncIterator[tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer]]:
    with patch("time.time", return_value=1700000000.0), patch.dict("os.environ", {"PROXY_BASE_URL": "http://test.com"}):
        server: Final = fakeredis.FakeServer()
        cache: Final = fake_redis_cache(server, namespace="alerts")
        raw: Final = fakeredis.FakeAsyncRedis(server=server)
        try:
            yield cache, raw, server
        finally:
            await raw.aclose()  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]  # fakeredis stubs predate aclose


_AlertType = Literal[
    "token_budget", "soft_budget", "max_budget_alert", "user_budget", "team_budget", "organization_budget"
]
_ALERTS: Final[tuple[tuple[_AlertType, CallInfo, str], ...]] = (
    (
        "soft_budget",
        CallInfo(
            user_id="user:雪",
            user_email="owner@example.com",
            spend=60,
            soft_budget=50,
            event_group=Litellm_EntityType.USER,
        ),
        "email_budget_alerts:soft_budget_crossed:user:雪",
    ),
    (
        "max_budget_alert",
        CallInfo(
            token="key:雪", user_email="owner@example.com", spend=85, max_budget=100, event_group=Litellm_EntityType.KEY
        ),
        "email_budget_alerts:max_budget_alert:key:雪",
    ),
    (
        "max_budget_alert",
        CallInfo(
            user_id="member:雪",
            team_id="team:a",
            user_email="owner@example.com",
            spend=85,
            max_budget=100,
            max_budget_alert_emails={"50": ["finance@example.com"]},
            event_group=Litellm_EntityType.TEAM_MEMBER,
        ),
        "email_budget_alerts:max_budget_alert:50:team_member:member:雪:team:a",
    ),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("alert_type,info,key", _ALERTS)
async def test_old_and_new_claims_share_numeric_bytes_without_refreshing_expiry(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
    alert_type: Literal["soft_budget", "max_budget_alert"],
    info: CallInfo,
    key: str,
) -> None:
    cache, raw, _ = redis_env
    logger: Final = EmailSink(DualCache(redis_cache=cache))
    assert await legacy_increment(cache, key, 0.25, ttl=37) == 0.25
    await logger.budget_alerts(type=alert_type, user_info=info)
    assert logger.sent == ()
    assert await raw.get(f"alerts:{key}") == b"1.25"
    assert await raw.pttl(f"alerts:{key}") == 37000
    await raw.delete(f"alerts:{key}")
    await logger.budget_alerts(type=alert_type, user_info=info)
    assert len(logger.sent) == 1
    assert await raw.get(f"alerts:{key}") == b"1"
    assert await raw.pttl(f"alerts:{key}") == EMAIL_BUDGET_ALERT_TTL * 1000
    assert await legacy_increment(cache, key, 1, ttl=2) == 2
    assert await raw.get(f"alerts:{key}") == b"2"
    assert await raw.pttl(f"alerts:{key}") == EMAIL_BUDGET_ALERT_TTL * 1000
    assert await raw.keys("*") == [f"alerts:{key}".encode()]


@pytest.mark.asyncio
@pytest.mark.parametrize("alert_type,info,key", _ALERTS)
async def test_concurrent_loggers_claim_once_and_failed_send_deletes_both_tiers(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
    alert_type: Literal["soft_budget", "max_budget_alert"],
    info: CallInfo,
    key: str,
) -> None:
    cache, raw, _ = redis_env
    first: Final = EmailSink(DualCache(redis_cache=cache))
    second: Final = EmailSink(DualCache(redis_cache=cache))
    await asyncio.gather(
        first.budget_alerts(type=alert_type, user_info=info), second.budget_alerts(type=alert_type, user_info=info)
    )
    assert len(first.sent) + len(second.sent) == 1
    assert await raw.get(f"alerts:{key}") == b"2"
    await raw.delete(f"alerts:{key}")

    async def fail_send() -> None:
        raise ValueError("email backend failed")

    failed: Final = EmailSink(DualCache(redis_cache=cache), fail_send)
    await failed.budget_alerts(type=alert_type, user_info=info)
    assert len(failed.sent) == 1
    assert await raw.get(f"alerts:{key}") is None
    assert local_get(failed.internal_usage_cache, key) is None
    retry: Final = EmailSink(failed.internal_usage_cache)
    await retry.budget_alerts(type=alert_type, user_info=info)
    assert len(retry.sent) == 1
    assert await raw.get(f"alerts:{key}") == b"1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seed,sends,stored", [(b"-1", 1, b"0"), (b"0", 1, b"1"), (b"1", 0, b"2"), (b"broken", 1, b"broken")]
)
async def test_counter_result_and_server_error_keep_existing_send_policy(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
    seed: bytes,
    sends: int,
    stored: bytes,
) -> None:
    cache, raw, _ = redis_env
    logger: Final = EmailSink(DualCache(redis_cache=cache))
    alert_type, info, key = _ALERTS[0]
    await raw.set(f"alerts:{key}", seed)
    await logger.budget_alerts(type=alert_type, user_info=info)
    assert len(logger.sent) == sends
    assert await raw.get(f"alerts:{key}") == stored
    assert await cache.transport().ping() is True
    assert await raw.pttl(f"alerts:{key}") == (-1 if seed == b"broken" else EMAIL_BUDGET_ALERT_TTL * 1000)


@pytest.mark.asyncio
@pytest.mark.parametrize("local_cache", [True, False])
async def test_outage_and_open_breaker_use_local_count_or_none(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
    local_cache: bool,
) -> None:
    cache, raw, server = redis_env
    dual: Final = DualCache(redis_cache=cache)
    if not local_cache:
        dual.in_memory_cache = None  # pyright: ignore[reportAttributeAccessIssue]  # the logger tolerates a cache without a local layer
    logger: Final = EmailSink(dual)
    alert_type, info, key = _ALERTS[0]
    server.connected = False
    for _ in range(6):
        await logger.budget_alerts(type=alert_type, user_info=info)
    assert len(logger.sent) == (1 if local_cache else 6)
    result: Final = await cache.transport().get(key)
    assert isinstance(result, RedisFailure)
    assert result.kind == "breaker_open"
    server.connected = True
    assert await raw.get(f"alerts:{key}") is None


@pytest.mark.asyncio
async def test_applied_increment_with_lost_reply_falls_back_without_retrying_or_rolling_back(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
) -> None:
    cache, raw, _ = redis_env
    logger: Final = EmailSink(DualCache(redis_cache=cache))
    alert_type, info, key = _ALERTS[0]
    client: Final = async_client(cache)
    pipeline: Final = client.pipeline(transaction=False)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # fakeredis pipeline is untyped
    execute: Final = pipeline.execute

    async def lose_reply(raise_on_error: bool = True) -> list[object]:
        await execute(raise_on_error=raise_on_error)
        assert local_get(logger.internal_usage_cache, key) == 1
        raise RedisConnectionError("reply lost after write applied")

    with (
        patch.object(client, "pipeline", return_value=pipeline),
        patch.object(pipeline, "execute", side_effect=lose_reply),
    ):
        await logger.budget_alerts(type=alert_type, user_info=info)
    assert len(logger.sent) == 1
    assert await raw.get(f"alerts:{key}") == b"1"
    assert await raw.pttl(f"alerts:{key}") == -1
    await logger.budget_alerts(type=alert_type, user_info=info)
    assert len(logger.sent) == 1
    assert await raw.get(f"alerts:{key}") == b"2"
    assert await raw.pttl(f"alerts:{key}") == EMAIL_BUDGET_ALERT_TTL * 1000


@pytest.mark.asyncio
async def test_expiry_failure_leaves_the_increment_and_uses_local_fallback(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
) -> None:
    cache, raw, _ = redis_env
    logger: Final = EmailSink(DualCache(redis_cache=cache))
    alert_type, info, key = _ALERTS[0]
    client: Final = async_client(cache)
    with patch.object(client, "expire", side_effect=RedisConnectionError("lost before expiry")):
        await logger.budget_alerts(type=alert_type, user_info=info)
    assert len(logger.sent) == 1
    assert await raw.get(f"alerts:{key}") == b"1"
    assert await raw.pttl(f"alerts:{key}") == -1


@pytest.mark.asyncio
async def test_rejected_expiry_uses_local_result_instead_of_applied_redis_count(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
) -> None:
    cache, raw, _ = redis_env
    logger: Final = EmailSink(DualCache(redis_cache=cache))
    alert_type, info, key = _ALERTS[0]
    await raw.set(f"alerts:{key}", b"1")
    with patch.object(cache.init_async_client(), "expire", side_effect=ResponseError("NOPERM expire denied")):
        await logger.budget_alerts(type=alert_type, user_info=info)
    assert len(logger.sent) == 1
    assert await raw.get(f"alerts:{key}") == b"2"
    assert await raw.pttl(f"alerts:{key}") == -1


@pytest.mark.asyncio
async def test_unknown_applied_release_deletes_locally_and_keeps_retry_policy(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
) -> None:
    cache, raw, _ = redis_env

    async def fail_send() -> None:
        raise ValueError("email backend failed")

    logger: Final = EmailSink(DualCache(redis_cache=cache), fail_send)
    alert_type, info, key = _ALERTS[0]
    client: Final = async_client(cache)
    delete: Final = client.delete

    async def lose_reply(*keys: str) -> int:
        await delete(*keys)
        raise RedisConnectionError("release reply lost after deletion")

    with patch.object(client, "delete", side_effect=lose_reply):
        await logger.budget_alerts(type=alert_type, user_info=info)
    assert len(logger.sent) == 1
    assert await raw.get(f"alerts:{key}") is None
    assert local_get(logger.internal_usage_cache, key) is None
    retry: Final = EmailSink(logger.internal_usage_cache)
    await retry.budget_alerts(type=alert_type, user_info=info)
    assert len(retry.sent) == 1
    assert await raw.get(f"alerts:{key}") == b"1"


@pytest.mark.asyncio
@pytest.mark.parametrize("seconds", [0, -1])
async def test_nonpositive_expiry_deletes_the_counter_like_legacy_expire(
    redis_env: tuple[RedisCache, fakeredis.FakeAsyncRedis, fakeredis.FakeServer],
    seconds: int,
) -> None:
    cache, raw, _ = redis_env
    store: Final = TransportEmailAlertClaimsStore(cache.transport())
    assert await store.claim(SoftBudgetAlert("expired"), ttl=timedelta(seconds=seconds)) == 1
    with patch("time.time", return_value=1700000000.001):
        assert await raw.get("alerts:email_budget_alerts:soft_budget_crossed:expired") is None
