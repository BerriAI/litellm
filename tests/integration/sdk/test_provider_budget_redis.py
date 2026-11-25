import asyncio
import os
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Final

import litellm
import pytest
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig
from redis import Redis

WINDOWS: Final = {"openai": ("1d", 86400), "vertex_ai": ("1h", 3600)}
SPEND_KEYS: Final = {provider: f"provider_spend:{provider}:{window}" for provider, (window, _) in WINDOWS.items()}
START_KEYS: Final = tuple(f"provider_budget_start_time:{provider}" for provider in WINDOWS)


@pytest.fixture
def redis_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[Redis]:
    monkeypatch.setattr(litellm, "callbacks", [])
    with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as client:
        client.delete(*SPEND_KEYS.values(), *START_KEYS)
        yield client
        client.delete(*SPEND_KEYS.values(), *START_KEYS)


def _limiter() -> RouterBudgetLimiting:
    return RouterBudgetLimiting(
        dual_cache=DualCache(redis_cache=RedisCache(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))),
        provider_budget_config={
            provider: BudgetConfig(budget_duration=window, max_budget=100) for provider, (window, _) in WINDOWS.items()
        },
    )


async def _windows_opened(redis_client: Redis) -> bool:
    for _ in range(100):
        if all(int(redis_client.ttl(key)) > 0 for key in SPEND_KEYS.values()):
            return True
        await asyncio.sleep(0.1)
    return False


@pytest.mark.asyncio
async def test_spend_written_to_redis_by_another_instance_is_pulled_into_memory(redis_client: Redis) -> None:
    limiter: Final = _limiter()
    assert await _windows_opened(redis_client)
    elsewhere: Final = {SPEND_KEYS["openai"]: 50.0, SPEND_KEYS["vertex_ai"]: 75.0}
    for key, value in elsewhere.items():
        redis_client.set(key, str(value), keepttl=True)
    await limiter._sync_in_memory_spend_with_redis()
    in_memory: Final = {key: await limiter.dual_cache.in_memory_cache.async_get_cache(key) for key in elsewhere}
    assert in_memory == elsewhere
    assert await limiter._get_current_provider_spend("openai") == 50.0


@pytest.mark.asyncio
async def test_budget_reset_time_follows_the_redis_window_expiry(redis_client: Redis) -> None:
    limiter: Final = _limiter()
    assert await _windows_opened(redis_client)
    assert await limiter._get_current_provider_budget_reset_at("anthropic") is None
    reset_times: Final = {
        provider: await limiter._get_current_provider_budget_reset_at(provider) for provider in WINDOWS
    }
    now: Final = datetime.now(timezone.utc)
    drift: Final = {
        provider: abs(
            (datetime.fromisoformat(str(reset_times[provider])) - (now + timedelta(seconds=seconds))).total_seconds()
        )
        for provider, (_, seconds) in WINDOWS.items()
    }
    assert all(seconds < 5 for seconds in drift.values()), (reset_times, drift)
