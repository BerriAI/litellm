import json
import time
from typing import Final

import fakeredis
import pytest
import respx
from redis.exceptions import TimeoutError as RedisTimeoutError

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.redis.stores.alert_deployment_outages import TransportAlertDeploymentOutagesStore
from litellm.redis.transport.contract import RedisFailure
from litellm.types.integrations.slack_alerting import AlertType
from tests._support.fake_redis import fake_redis_cache

_WEBHOOK: Final = "https://slack.example/deployment-outages"
_ID: Final = "model-é"


@pytest.fixture(autouse=True)
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "time", lambda: 1800000000.0)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def _slack(cache: DualCache) -> SlackAlerting:
    slack: Final = SlackAlerting(
        internal_usage_cache=cache,
        alerting=["slack"],
        alert_types=[AlertType.outage_alerts],
        default_webhook_url=_WEBHOOK,
        alerting_args={"outage_alert_ttl": 73, "minor_outage_alert_threshold": 2, "major_outage_alert_threshold": 3},
    )
    slack.update_values(
        llm_router=Router(
            model_list=[
                {
                    "model_name": "model",
                    "litellm_params": {"model": "openai/gpt-5-mini", "api_key": "unused"},
                    "model_info": {"id": _ID},
                }
            ]
        )
    )
    return slack


def _error() -> litellm.Timeout:
    return litellm.Timeout(message="unavailable", model="model", llm_provider="openai")


@pytest.mark.asyncio
async def test_deployment_record_wire_threshold_delivery_and_asymmetric_ttl(respx_mock: respx.MockRouter) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    slack: Final = _slack(DualCache(redis_cache=redis, default_in_memory_ttl=19))
    route: Final = respx_mock.post(_WEBHOOK).respond(200, text="ok")
    expected: Final = {
        "model_id": _ID,
        "alerts": [408],
        "minor_alert_sent": False,
        "major_alert_sent": False,
        "last_updated_at": 1800000000.0,
    }
    await slack.outage_alerts(_error(), _ID)
    assert await raw.get(f"alerts:{_ID}") == json.dumps(expected).encode()
    assert await redis.async_get_cache(_ID) == expected
    assert await raw.pttl(f"alerts:{_ID}") == 73000
    await slack.outage_alerts(_error(), _ID)
    await slack.outage_alerts(_error(), _ID)
    await slack.flush_queue()
    assert route.call_count == 2
    assert await redis.async_get_cache(_ID) == {
        **expected,
        "alerts": [408, 408, 408],
        "minor_alert_sent": True,
        "major_alert_sent": True,
    }
    assert await raw.pttl(f"alerts:{_ID}") == 19000
    assert await raw.exists(_ID, f"alerts:alerts:{_ID}") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_repr", [False, True])
async def test_legacy_deployment_record_keeps_fields_and_order(legacy_repr: bool, respx_mock: respx.MockRouter) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    value: Final = {
        "model_id": _ID,
        "alerts": [408],
        "minor_alert_sent": True,
        "major_alert_sent": False,
        "last_updated_at": 1.0,
        "future": "é",
    }
    if legacy_repr:
        await raw.set(f"alerts:{_ID}", str(value), ex=73)
    else:
        await redis.async_set_cache(_ID, value, ttl=73)
    slack: Final = _slack(DualCache(redis_cache=redis))
    respx_mock.post(_WEBHOOK).respond(200, text="ok")
    await slack.outage_alerts(_error(), _ID)
    assert (
        await raw.get(f"alerts:{_ID}")
        == json.dumps({**value, "alerts": [408, 408], "last_updated_at": 1800000000.0}).encode()
    )
    assert await raw.pttl(f"alerts:{_ID}") == redis.default_ttl * 1000


@pytest.mark.asyncio
async def test_invalid_record_is_swallowed_without_reset_and_tombstone_stays_pending() -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    await raw.set(f"alerts:{_ID}", b"{}")
    cache: Final = DualCache(redis_cache=redis)
    slack: Final = _slack(cache)
    await slack.outage_alerts(_error(), _ID)
    assert await raw.get(f"alerts:{_ID}") == b"{}"
    with request_redis_batch_scope():
        pending: Final = await cache.async_delete_cache_pre_call(_ID)
        assert pending is not None
        assert await slack._read_deployment_outage(_ID) is None
        assert await raw.get(f"alerts:{_ID}") == b"{}"
        await pending
    await raw.set(f"alerts:{_ID}", b"invalid")
    await slack.outage_alerts(_error(), _ID)
    assert await raw.pttl(f"alerts:{_ID}") == 73000


def _lost_reply(reply: object, **kwargs: object) -> object:
    raise RedisTimeoutError("applied before reply was lost")


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_failure_retains_local_deployment_state(ambiguous: bool) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    slack: Final = _slack(DualCache(redis_cache=redis))
    if ambiguous:
        redis.init_async_client().set_response_callback("SET", _lost_reply)
    else:
        server.connected = False
    await slack.outage_alerts(_error(), _ID)
    local: Final = await slack._read_deployment_outage(_ID)
    assert local is not None and local.outage["alerts"] == [408]
    server.connected = True
    assert await raw.exists(f"alerts:{_ID}") == int(ambiguous)
    server.connected = False
    assert isinstance(
        await TransportAlertDeploymentOutagesStore(redis.transport()).read_deployment_outage(_ID), RedisFailure
    )
