import time
from typing import Final

import fakeredis
import pytest
import respx
from redis.exceptions import TimeoutError as RedisTimeoutError

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.redis.stores.alert_budget_markers import BudgetAlertMarker, TransportAlertBudgetMarkersStore
from litellm.redis.transport.contract import RedisFailure
from litellm.types.integrations.slack_alerting import AlertType, CallInfo, Litellm_EntityType
from tests._support.fake_redis import fake_redis_cache

_WEBHOOK: Final = "https://slack.example/budget-alerts"


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 1800000000.0)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def _slack(cache: DualCache) -> SlackAlerting:
    return SlackAlerting(
        internal_usage_cache=cache,
        alerting=["slack"],
        alert_types=[AlertType.failed_tracking_spend, AlertType.budget_alerts],
        default_webhook_url=_WEBHOOK,
        alerting_args={"budget_alert_ttl": 57},
    )


@pytest.mark.asyncio
async def test_both_budget_marker_families_keep_wire_ttl_and_deduplicate_delivery(respx_mock: respx.MockRouter) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    slack: Final = _slack(DualCache(redis_cache=redis))
    route: Final = respx_mock.post(_WEBHOOK).respond(200, text="ok")
    info: Final = CallInfo(token="key-é", spend=11, max_budget=10, event_group=Litellm_EntityType.KEY)
    await slack.failed_tracking_alert("unpriced", "model-é")
    await slack.budget_alerts(type="token_budget", user_info=info)
    await slack.failed_tracking_alert("unpriced", "model-é")
    await slack.budget_alerts(type="token_budget", user_info=info)
    await slack.flush_queue()
    assert route.call_count == 2
    for key in ("budget_alerts:failed_tracking:model-é", "budget_alerts:budget_crossed:key-é"):
        assert await raw.get(f"alerts:{key}") == b'"SENT"'
        assert await redis.async_get_cache(key) == "SENT"
        assert await raw.pttl(f"alerts:{key}") == 57000
        assert await raw.exists(key, f"alerts:alerts:{key}") == 0
    assert len(slack.log_queue) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("wire", "sent"),
    [
        (b'"SENT"', True),
        (b"'SENT'", True),
        (b"0", True),
        (b"false", True),
        (b"null", False),
        (b"None", False),
        (b"invalid", False),
    ],
)
async def test_old_marker_decoding_keeps_presence_not_truthiness(wire: bytes, sent: bool) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    await raw.set("alerts:budget_alerts:failed_tracking:model", wire)
    slack: Final = _slack(DualCache(redis_cache=redis))
    assert await slack._was_budget_alert_sent(BudgetAlertMarker("failed_tracking", "model")) is sent


@pytest.mark.asyncio
async def test_legacy_marker_blocks_new_delivery_and_pending_delete_stays_missing(respx_mock: respx.MockRouter) -> None:
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, "alerts")
    await redis.async_set_cache("budget_alerts:failed_tracking:model", "SENT", ttl=91)
    cache: Final = DualCache(redis_cache=redis)
    slack: Final = _slack(cache)
    route: Final = respx_mock.post(_WEBHOOK).respond(200, text="ok")
    await slack.failed_tracking_alert("unpriced", "model")
    await slack.flush_queue()
    assert route.call_count == 0
    with request_redis_batch_scope():
        pending: Final = await cache.async_delete_cache_pre_call("budget_alerts:failed_tracking:model")
        assert pending is not None
        assert await redis.async_get_cache("budget_alerts:failed_tracking:model") == "SENT"
        assert await slack._was_budget_alert_sent(BudgetAlertMarker("failed_tracking", "model")) is False
        await pending


def _lost_set_reply(reply: object, **kwargs: object) -> object:
    raise RedisTimeoutError("write applied before reply was lost")


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_failure_keeps_local_delivery_marker_without_retry(ambiguous: bool, respx_mock: respx.MockRouter) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    cache: Final = DualCache(redis_cache=redis)
    slack: Final = _slack(cache)
    route: Final = respx_mock.post(_WEBHOOK).respond(200, text="ok")
    if ambiguous:
        redis.init_async_client().set_response_callback("SET", _lost_set_reply)
    else:
        server.connected = False
    await slack.failed_tracking_alert("unpriced", "model")
    await slack.failed_tracking_alert("unpriced", "model")
    await slack.flush_queue()
    assert route.call_count == 1
    assert await cache.async_get_cache("budget_alerts:failed_tracking:model", local_only=True) == "SENT"
    server.connected = True
    assert await raw.get("alerts:budget_alerts:failed_tracking:model") == (b'"SENT"' if ambiguous else None)
    server.connected = False
    failure: Final = await TransportAlertBudgetMarkersStore(redis.transport()).was_budget_alert_sent(
        BudgetAlertMarker("failed_tracking", "model")
    )
    assert isinstance(failure, RedisFailure)
