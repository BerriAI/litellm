import os
import uuid
from typing import Final

import pytest

import litellm
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import GLOBAL_PROXY_SPEND_CACHE_KEY
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.proxy_server import update_cache


async def _add_request_cost(cache: UserApiKeyCache, response_cost: float) -> None:
    await update_cache(
        token=None,
        user_id=None,
        end_user_id=None,
        team_id=None,
        response_cost=response_cost,
        parent_otel_span=None,
        cache=cache,
    )


@pytest.mark.asyncio
async def test_global_proxy_spend_adds_every_instance_cost_and_stays_cleared_after_a_budget_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "max_budget", 100.0)
    redis_cache: Final = RedisCache(
        host=os.environ["REDIS_HOST"],
        port=int(os.environ["REDIS_PORT"]),
        namespace=f"global-proxy-spend-{uuid.uuid4().hex}",
    )
    gateway: Final = UserApiKeyCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)
    peer: Final = UserApiKeyCache(in_memory_cache=InMemoryCache(), redis_cache=redis_cache)

    try:
        await gateway.async_set_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, value=10.0, ttl=60)
        await _add_request_cost(peer, 0.5)
        await _add_request_cost(gateway, 0.25)
        before_reset: Final = (
            gateway.get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY, local_only=True),
            await redis_cache.async_get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY),
        )
        ttl_before_reset: Final = await redis_cache.async_get_ttl(key=GLOBAL_PROXY_SPEND_CACHE_KEY)

        await gateway.async_delete_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY)
        await _add_request_cost(peer, 0.5)
        await _add_request_cost(gateway, 0.25)

        assert before_reset == (10.75, 10.75)
        assert ttl_before_reset is not None and 0 < ttl_before_reset <= 60
        assert await redis_cache.async_get_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY) is None
    finally:
        await redis_cache.async_delete_cache(key=GLOBAL_PROXY_SPEND_CACHE_KEY)
