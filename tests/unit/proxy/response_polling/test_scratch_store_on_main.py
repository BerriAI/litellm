import fakeredis
import pytest

import litellm
from litellm._service_logger import ServiceLogging
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.response_polling.polling_handler import ResponsePollingHandler


def _fake_cache(server: fakeredis.FakeServer) -> RedisCache:
    cache = RedisCache(host="127.0.0.1", port=1, namespace="ns", service_logger_obj=ServiceLogging())
    litellm.in_memory_llm_clients_cache.set_cache(
        key=cache._get_async_client_cache_key(), value=fakeredis.FakeAsyncRedis(server=server)
    )
    return cache


@pytest.mark.asyncio
async def test_round_trip_through_store_on_real_redis_semantics():
    server = fakeredis.FakeServer()
    handler = ResponsePollingHandler(redis_cache=_fake_cache(server), ttl=7200)

    await handler.create_initial_state("litellm_poll_x", {"model": "m"})
    await handler.update_state("litellm_poll_x", error={"code": "boom"})

    state = await handler.get_state("litellm_poll_x")
    assert state is not None and state["status"] == "failed" and state["error"] == {"code": "boom"}
    raw = fakeredis.FakeAsyncRedis(server=server)
    assert 7_190_000 < await raw.pttl("ns:litellm:polling:response:litellm_poll_x") <= 7_200_000
    assert await handler.delete_polling("litellm_poll_x") and await handler.get_state("litellm_poll_x") is None


def test_cooldown_lands_in_redis_as_json_and_legacy_repr_still_reads():
    import json

    from litellm.caching.dual_cache import DualCache
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.router_utils.cooldown_cache import CooldownCache

    server = fakeredis.FakeServer()

    def replica() -> CooldownCache:
        cache = _fake_cache(server)
        cache.redis_client = fakeredis.FakeRedis(server=server)
        return CooldownCache(cache=DualCache(in_memory_cache=InMemoryCache(), redis_cache=cache), default_cooldown_time=60)

    replica().add_deployment_to_cooldown("new", Exception("boom"), 429, cooldown_time=60)
    raw = fakeredis.FakeRedis(server=server)
    assert json.loads(raw.get("ns:deployment:new:cooldown"))["status_code"] == "429"

    legacy = {"exception_received": "x", "status_code": "500", "timestamp": __import__("time").time(), "cooldown_time": 60.0}
    raw.set("ns:deployment:old:cooldown", str(legacy))
    assert [m for m, _ in replica().get_active_cooldowns(["new", "old"], parent_otel_span=None)] == ["new", "old"]
