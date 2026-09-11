"""Regression for #39309: parallel rejects must not burn RPM quota."""

from pathlib import Path

import pytest


def test_should_rate_limit_refunds_on_parallel_over_limit():
    source = Path("litellm/proxy/hooks/parallel_request_limiter_v3.py").read_text()
    assert "_refund_windowed_increments" in source
    assert 'gauge_response["overall_code"] == "OVER_LIMIT"' in source
    gauge_idx = source.index("await self._check_parallel_request_gauges")
    refund_idx = source.index("await self._refund_windowed_increments", gauge_idx)
    return_idx = source.index(
        "return RateLimitResponse(\n            overall_code=gauge_response",
        gauge_idx,
    )
    assert gauge_idx < refund_idx < return_idx


@pytest.mark.asyncio
async def test_refund_windowed_increments_decrements_counter():
    from unittest.mock import AsyncMock, MagicMock

    from litellm.proxy.hooks import parallel_request_limiter_v3 as mod

    stub = MagicMock()
    stub.window_size = 60
    stub.internal_usage_cache = MagicMock()
    stub.internal_usage_cache.dual_cache.redis_cache = None
    store = {"{api_key:k}:requests": 2}

    async def get_cache(key, litellm_parent_otel_span=None, local_only=False):
        return store.get(key)

    async def set_cache(key, value, ttl=None, litellm_parent_otel_span=None, local_only=False):
        store[key] = value

    stub.internal_usage_cache.async_get_cache = AsyncMock(side_effect=get_cache)
    stub.internal_usage_cache.async_set_cache = AsyncMock(side_effect=set_cache)

    await mod._PROXY_MaxParallelRequestsHandler_v3._refund_windowed_increments(
        stub,
        keys_to_fetch=["{api_key:k}:window", "{api_key:k}:requests"],
        parent_otel_span=None,
    )
    assert store["{api_key:k}:requests"] == 1
