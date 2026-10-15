"""
Unit tests for PrometheusLogger._assemble_key_object DB access.

The post-request budget metrics run for every LLM API request. Auth has
already cached the key object for any real key in the same request, so the
metrics path must read the cache only. Falling through to the DB turns every
request whose token has no DB row (e.g. master-key requests, whose token is
an alias hash that never matches a stored key) into per-request
LiteLLM_VerificationToken and LiteLLM_DeprecatedVerificationToken queries.
"""

import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def prometheus_logger():
    return PrometheusLogger()


@pytest.mark.asyncio
async def test_assemble_key_object_does_not_query_db_on_cache_miss(prometheus_logger):
    mock_prisma = MagicMock()
    mock_prisma.get_data = AsyncMock()
    cache = DualCache(in_memory_cache=InMemoryCache())

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", cache),
    ):
        result = await prometheus_logger._assemble_key_object(
            user_api_key="hashed-token-not-in-cache",
            user_api_key_alias="",
            key_max_budget=None,
            key_spend=1.0,
            response_cost=0.5,
        )

    mock_prisma.get_data.assert_not_called()
    assert result.spend == 1.5
    assert result.budget_reset_at is None


@pytest.mark.asyncio
async def test_assemble_key_object_reads_budget_reset_at_from_cache(prometheus_logger):
    hashed_token = "hashed-token-in-cache"
    reset_at = datetime.datetime(2026, 8, 1, tzinfo=datetime.timezone.utc)
    cached_key = UserAPIKeyAuth(token=hashed_token, budget_reset_at=reset_at)

    mock_prisma = MagicMock()
    mock_prisma.get_data = AsyncMock()
    cache = DualCache(in_memory_cache=InMemoryCache())
    await cache.async_set_cache(key=hashed_token, value=cached_key)

    with (
        patch("litellm.proxy.proxy_server.prisma_client", mock_prisma),
        patch("litellm.proxy.proxy_server.user_api_key_cache", cache),
    ):
        result = await prometheus_logger._assemble_key_object(
            user_api_key=hashed_token,
            user_api_key_alias="alias",
            key_max_budget=10.0,
            key_spend=1.0,
            response_cost=0.5,
        )

    mock_prisma.get_data.assert_not_called()
    assert result.budget_reset_at == reset_at
