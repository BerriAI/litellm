import pytest

from litellm.caching.caching import DualCache
from litellm.router_utils.cooldown_cache import CooldownCache


def test_get_active_cooldowns_ignores_expired_cache_payload(monkeypatch):
    cache = DualCache()
    cooldown_cache = CooldownCache(cache=cache, default_cooldown_time=120)
    model_id = "deployment-1"

    monkeypatch.setattr("litellm.router_utils.cooldown_cache.time.time", lambda: 300)
    cache.set_cache(
        key=CooldownCache.get_cooldown_cache_key(model_id),
        value={
            "exception_received": "rate limit",
            "status_code": "429",
            "timestamp": 100,
            "cooldown_time": 120,
        },
        ttl=600,
    )

    assert (
        cooldown_cache.get_active_cooldowns(
            model_ids=[model_id],
            parent_otel_span=None,
        )
        == []
    )


@pytest.mark.asyncio
async def test_async_get_active_cooldowns_ignores_expired_cache_payload(monkeypatch):
    cache = DualCache()
    cooldown_cache = CooldownCache(cache=cache, default_cooldown_time=120)
    model_id = "deployment-1"

    monkeypatch.setattr("litellm.router_utils.cooldown_cache.time.time", lambda: 300)
    cache.set_cache(
        key=CooldownCache.get_cooldown_cache_key(model_id),
        value={
            "exception_received": "rate limit",
            "status_code": "429",
            "timestamp": 100,
            "cooldown_time": 120,
        },
        ttl=600,
    )

    assert (
        await cooldown_cache.async_get_active_cooldowns(
            model_ids=[model_id],
            parent_otel_span=None,
        )
        == []
    )


def test_get_min_cooldown_ignores_expired_cache_payload(monkeypatch):
    cache = DualCache()
    cooldown_cache = CooldownCache(cache=cache, default_cooldown_time=120)
    expired_model_id = "expired-deployment"
    active_model_id = "active-deployment"

    monkeypatch.setattr("litellm.router_utils.cooldown_cache.time.time", lambda: 300)
    cache.set_cache(
        key=CooldownCache.get_cooldown_cache_key(expired_model_id),
        value={
            "exception_received": "rate limit",
            "status_code": "429",
            "timestamp": 100,
            "cooldown_time": 10,
        },
        ttl=600,
    )
    cache.set_cache(
        key=CooldownCache.get_cooldown_cache_key(active_model_id),
        value={
            "exception_received": "rate limit",
            "status_code": "429",
            "timestamp": 250,
            "cooldown_time": 120,
        },
        ttl=600,
    )

    assert (
        cooldown_cache.get_min_cooldown(
            model_ids=[expired_model_id, active_model_id],
            parent_otel_span=None,
        )
        == 120
    )
