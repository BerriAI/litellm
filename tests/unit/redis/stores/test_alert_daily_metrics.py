import datetime
import time
from typing import Final

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.redis.stores.alert_daily_metrics import DeploymentMetric, TransportAlertDailyMetricsStore
from litellm.redis.transport.contract import RedisFailure
from litellm.types.integrations.slack_alerting import DeploymentMetrics
from tests._support.fake_redis import fake_redis_cache


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 1800000000.0)


def _metrics() -> DeploymentMetrics:
    return DeploymentMetrics(
        id="deployment",
        failed_request=True,
        latency_per_output_token=0.25,
        updated_at=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
    )


def _router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "model",
                "litellm_params": {"model": "openai/gpt-5-mini", "api_key": "unused"},
                "model_info": {"id": "deployment"},
            }
        ]
    )


@pytest.mark.asyncio
async def test_new_daily_metrics_keep_legacy_wire_and_first_increment_ttl() -> None:
    server: Final = fakeredis.FakeServer()
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    cache: Final = DualCache(redis_cache=redis, default_in_memory_ttl=31)
    slack: Final = SlackAlerting(internal_usage_cache=cache, alerting=[])
    assert await slack.async_update_daily_reports(_metrics()) == 2
    assert await client.mget(
        "alerts:deployment:failed_requests_daily_metrics", "alerts:deployment:latency_daily_metrics"
    ) == [b"1", b"0.25"]
    assert await client.pttl("alerts:deployment:failed_requests_daily_metrics") == redis.default_ttl * 1000
    await client.pexpire("alerts:deployment:failed_requests_daily_metrics", 12345)
    assert await slack.async_update_daily_reports(_metrics()) == 2
    assert await redis.async_get_cache("deployment:failed_requests_daily_metrics") == 2
    assert await client.pttl("alerts:deployment:failed_requests_daily_metrics") == 12345
    assert await slack.send_daily_reports(_router()) is True
    assert await client.mget(
        "alerts:deployment:failed_requests_daily_metrics", "alerts:deployment:latency_daily_metrics"
    ) == [b"0", b"0"]
    assert await client.pttl("alerts:deployment:failed_requests_daily_metrics") == 31000
    assert (
        await client.exists(
            "deployment:failed_requests_daily_metrics", "alerts:alerts:deployment:failed_requests_daily_metrics"
        )
        == 0
    )


@pytest.mark.asyncio
async def test_legacy_metrics_are_reported_then_reset_by_new_caller() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, "alerts")
    await redis.async_increment("deployment:failed_requests_daily_metrics", 3)
    await redis.async_set_cache("deployment:latency_daily_metrics", 1.5)
    slack: Final = SlackAlerting(internal_usage_cache=DualCache(redis_cache=redis), alerting=[])
    assert await slack.send_daily_reports(_router()) is True
    assert await redis.async_get_cache("deployment:failed_requests_daily_metrics") == 0
    assert await redis.async_get_cache("deployment:latency_daily_metrics") == 0


@pytest.mark.asyncio
async def test_disconnected_redis_keeps_local_metrics_and_report_reset() -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, "alerts")
    cache: Final = DualCache(redis_cache=redis)
    slack: Final = SlackAlerting(internal_usage_cache=cache, alerting=[])
    server.connected = False
    assert await slack.async_update_daily_reports(_metrics()) == 2
    assert await slack.send_daily_reports(_router()) is True
    assert await cache.async_get_cache("deployment:failed_requests_daily_metrics", local_only=True) == 0
    failure: Final = await TransportAlertDailyMetricsStore(redis.transport()).read_deployment_metrics(
        (DeploymentMetric("deployment", "latency"),)
    )
    assert isinstance(failure, RedisFailure)
    assert failure.kind in ("transport", "breaker_open")


@pytest.mark.asyncio
async def test_malformed_batch_metric_does_not_become_a_report_or_reset() -> None:
    server: Final = fakeredis.FakeServer()
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    await client.set("alerts:deployment:failed_requests_daily_metrics", b"not-a-literal")
    await client.set("alerts:deployment:latency_daily_metrics", b"3")
    slack: Final = SlackAlerting(internal_usage_cache=DualCache(redis_cache=redis), alerting=[])
    assert await slack.send_daily_reports(_router()) is False
    assert await client.get("alerts:deployment:latency_daily_metrics") == b"3"
    await client.set("alerts:deployment:failed_requests_daily_metrics", b"2")
    assert await slack.send_daily_reports(_router()) is False


def _lost_increment_reply(reply: object, **kwargs: object) -> object:
    raise RedisTimeoutError("increment applied before reply was lost")


@pytest.mark.asyncio
async def test_ambiguous_increment_is_not_retried() -> None:
    server: Final = fakeredis.FakeServer()
    client: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    redis.init_async_client().set_response_callback("INCRBYFLOAT", _lost_increment_reply)
    slack: Final = SlackAlerting(internal_usage_cache=DualCache(redis_cache=redis), alerting=[])
    assert await slack.async_update_daily_reports(_metrics()) == 2
    assert await client.mget(
        "alerts:deployment:failed_requests_daily_metrics", "alerts:deployment:latency_daily_metrics"
    ) == [b"1", b"0.25"]
    assert await client.pttl("alerts:deployment:failed_requests_daily_metrics") == -1
