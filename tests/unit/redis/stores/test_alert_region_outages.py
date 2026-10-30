import json
import time
from typing import Final

import fakeredis
import pytest
import respx
from pydantic import ValidationError
from redis.exceptions import TimeoutError as RedisTimeoutError

import litellm
from litellm import Router
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_batch import request_redis_batch_scope
from litellm.integrations.SlackAlerting.slack_alerting import SlackAlerting
from litellm.redis.stores.alert_region_outages import RegionOutage, TransportAlertRegionOutagesStore
from litellm.redis.transport.contract import RedisFailure
from litellm.types.integrations.slack_alerting import AlertType
from tests._support.fake_redis import fake_redis_cache

_WEBHOOK: Final = "https://slack.example/region-outages"
_KEY: Final = "openairégion"


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
        alerting_args={
            "region_outage_alert_ttl": 73,
            "minor_outage_alert_threshold": 2,
            "major_outage_alert_threshold": 3,
        },
    )
    slack.update_values(
        llm_router=Router(
            model_list=[
                {
                    "model_name": "model",
                    "litellm_params": {
                        "model": "openai/gpt-5-mini",
                        "custom_llm_provider": "openai",
                        "region_name": "région",
                        "api_key": "unused",
                    },
                    "model_info": {"id": identity},
                }
                for identity in ("one", "two")
            ]
        )
    )
    return slack


def _error() -> litellm.Timeout:
    return litellm.Timeout(message="unavailable", model="model", llm_provider="openai")


@pytest.mark.asyncio
async def test_new_region_record_matches_legacy_bytes_then_uses_default_update_ttl(
    respx_mock: respx.MockRouter,
) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    slack: Final = _slack(DualCache(redis_cache=redis, default_in_memory_ttl=19))
    route: Final = respx_mock.post(_WEBHOOK).respond(200, text="ok")
    expected: Final = {
        "provider_region_id": _KEY,
        "alerts": [408],
        "minor_alert_sent": False,
        "major_alert_sent": False,
        "last_updated_at": 1800000000.0,
        "deployment_ids": ["one"],
    }
    await slack.region_outage_alerts(_error(), "one")
    assert await raw.get(f"alerts:{_KEY}") == json.dumps(expected).encode()
    assert await redis.async_get_cache(_KEY) == expected
    assert await raw.pttl(f"alerts:{_KEY}") == 73000
    await slack.region_outage_alerts(_error(), "two")
    await slack.region_outage_alerts(_error(), "two")
    await slack.flush_queue()
    assert route.call_count == 2
    result: Final = await slack._read_region_outage(RegionOutage("openai", "région"))
    assert result is not None and result.outage == {
        **expected,
        "alerts": [408, 408, 408],
        "deployment_ids": {"one", "two"},
        "minor_alert_sent": True,
        "major_alert_sent": True,
    }
    assert await raw.pttl(f"alerts:{_KEY}") == 19000
    assert await raw.exists(_KEY, f"alerts:alerts:{_KEY}") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_repr", [False, True])
async def test_old_region_writer_is_read_without_losing_order_or_extra_fields(legacy_repr: bool) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    value: Final = {
        "provider_region_id": _KEY,
        "alerts": [408],
        "minor_alert_sent": False,
        "major_alert_sent": False,
        "last_updated_at": 1.0,
        "deployment_ids": ["one"],
        "future": "é",
    }
    if legacy_repr:
        await raw.set(f"alerts:{_KEY}", str(value), ex=73)
    else:
        await redis.async_set_cache(_KEY, value, ttl=73)
    slack: Final = _slack(DualCache(redis_cache=redis))
    await slack.region_outage_alerts(_error(), "one")
    expected: Final = {**value, "alerts": [408, 408], "last_updated_at": 1800000000.0}
    assert await raw.get(f"alerts:{_KEY}") == json.dumps(expected).encode()
    assert await raw.pttl(f"alerts:{_KEY}") == redis.default_ttl * 1000


@pytest.mark.asyncio
async def test_invalid_record_is_not_overwritten_and_tombstone_does_not_flush() -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    await raw.set(f"alerts:{_KEY}", b"{}")
    cache: Final = DualCache(redis_cache=redis)
    slack: Final = _slack(cache)
    with pytest.raises(ValidationError):
        await slack.region_outage_alerts(_error(), "one")
    assert await raw.get(f"alerts:{_KEY}") == b"{}"
    with request_redis_batch_scope():
        pending: Final = await cache.async_delete_cache_pre_call(_KEY)
        assert pending is not None
        assert await slack._read_region_outage(RegionOutage("openai", "région")) is None
        assert await raw.get(f"alerts:{_KEY}") == b"{}"
        await pending
    await raw.set(f"alerts:{_KEY}", b"invalid")
    await slack.region_outage_alerts(_error(), "one")
    assert await raw.pttl(f"alerts:{_KEY}") == 73000


def _lost_reply(reply: object, **kwargs: object) -> object:
    raise RedisTimeoutError("applied before reply was lost")


@pytest.mark.asyncio
@pytest.mark.parametrize("ambiguous", [False, True])
async def test_outage_or_ambiguous_write_retains_local_region_state(ambiguous: bool) -> None:
    server: Final = fakeredis.FakeServer()
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    redis: Final = fake_redis_cache(server, "alerts")
    slack: Final = _slack(DualCache(redis_cache=redis))
    if ambiguous:
        redis.init_async_client().set_response_callback("SET", _lost_reply)
    else:
        server.connected = False
    await slack.region_outage_alerts(_error(), "one")
    await slack.region_outage_alerts(_error(), "one")
    region: Final = RegionOutage("openai", "région")
    local: Final = await slack._read_region_outage(region)
    assert local is not None and local.outage["alerts"] == [408, 408]
    server.connected = True
    assert await raw.exists(f"alerts:{_KEY}") == int(ambiguous)
    server.connected = False
    assert isinstance(
        await TransportAlertRegionOutagesStore(redis.transport()).read_region_outage(region), RedisFailure
    )
