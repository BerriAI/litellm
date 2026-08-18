"""Pin behavior of ``InternalUsageCache``: a thin adapter over ``DualCache``.

Each method should pass-through to the underlying ``DualCache`` with
exactly the same arguments, mapping ``litellm_parent_otel_span`` to the
``DualCache`` kw it expects.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.caching.caching import DualCache
from litellm.proxy.utils import InternalUsageCache


def _kwargs_snapshot(call):
    return dict(call.kwargs)


def test_internal_usage_cache_init_stores_dual_cache():
    inner = DualCache(default_in_memory_ttl=1)
    cache = InternalUsageCache(dual_cache=inner)
    snapshot = {
        "is_internal_usage_cache": isinstance(cache, InternalUsageCache),
        "dual_cache_is_inner": cache.dual_cache is inner,
        "ttl_is_one": inner.default_in_memory_ttl == 1,
    }
    assert snapshot == {
        "is_internal_usage_cache": True,
        "dual_cache_is_inner": True,
        "ttl_is_one": True,
    }


def test_internal_usage_cache_init_error_requires_dual_cache():
    with pytest.raises(TypeError):
        InternalUsageCache()  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_async_get_cache_forwards_args():
    inner = MagicMock()
    inner.async_get_cache = AsyncMock(return_value={"hit": True, "value": 42, "source": "redis"})
    cache = InternalUsageCache(dual_cache=inner)

    result = await cache.async_get_cache(key="k", litellm_parent_otel_span="span", local_only=True, extra="x")
    forwarded = _kwargs_snapshot(inner.async_get_cache.call_args)
    assert forwarded == {"key": "k", "local_only": True, "parent_otel_span": "span", "extra": "x"}
    assert result == {"hit": True, "value": 42, "source": "redis"}


@pytest.mark.asyncio
async def test_async_get_cache_propagates_underlying_error_raises():
    inner = MagicMock()
    inner.async_get_cache = AsyncMock(side_effect=RuntimeError("redis down"))
    cache = InternalUsageCache(dual_cache=inner)
    with pytest.raises(RuntimeError, match="redis down"):
        await cache.async_get_cache(key="k", litellm_parent_otel_span=None)


@pytest.mark.asyncio
async def test_async_set_cache_forwards_args():
    inner = MagicMock()
    inner.async_set_cache = AsyncMock()
    cache = InternalUsageCache(dual_cache=inner)

    await cache.async_set_cache(key="k", value="v", litellm_parent_otel_span="span", local_only=False, ttl=60)
    forwarded = _kwargs_snapshot(inner.async_set_cache.call_args)
    assert forwarded == {
        "key": "k",
        "value": "v",
        "local_only": False,
        "litellm_parent_otel_span": "span",
        "ttl": 60,
    }


@pytest.mark.asyncio
async def test_async_set_cache_propagates_error_raises():
    inner = MagicMock()
    inner.async_set_cache = AsyncMock(side_effect=ValueError("bad value"))
    cache = InternalUsageCache(dual_cache=inner)
    with pytest.raises(ValueError, match="bad value"):
        await cache.async_set_cache(key="k", value="v", litellm_parent_otel_span=None)


@pytest.mark.asyncio
async def test_async_batch_set_cache_forwards_pipeline():
    inner = MagicMock()
    inner.async_set_cache_pipeline = AsyncMock()
    cache = InternalUsageCache(dual_cache=inner)

    pairs = [("a", 1), ("b", 2)]
    await cache.async_batch_set_cache(cache_list=pairs, litellm_parent_otel_span=None, local_only=True, ttl=10)
    forwarded = _kwargs_snapshot(inner.async_set_cache_pipeline.call_args)
    assert forwarded == {
        "cache_list": pairs,
        "local_only": True,
        "litellm_parent_otel_span": None,
        "ttl": 10,
    }


@pytest.mark.asyncio
async def test_async_batch_set_cache_propagates_error_raises():
    inner = MagicMock()
    inner.async_set_cache_pipeline = AsyncMock(side_effect=ConnectionError("network"))
    cache = InternalUsageCache(dual_cache=inner)
    with pytest.raises(ConnectionError):
        await cache.async_batch_set_cache(cache_list=[], litellm_parent_otel_span=None)


@pytest.mark.asyncio
async def test_async_batch_get_cache_forwards_args():
    inner = MagicMock()
    inner.async_batch_get_cache = AsyncMock(return_value=[1, 2, 3])
    cache = InternalUsageCache(dual_cache=inner)
    result = await cache.async_batch_get_cache(keys=["a", "b", "c"], parent_otel_span="span", local_only=False)
    forwarded = _kwargs_snapshot(inner.async_batch_get_cache.call_args)
    assert forwarded == {"keys": ["a", "b", "c"], "parent_otel_span": "span", "local_only": False}
    assert result == [1, 2, 3]


@pytest.mark.asyncio
async def test_async_batch_get_cache_invalid_input_raises():
    inner = MagicMock()
    inner.async_batch_get_cache = AsyncMock(side_effect=TypeError("not a list"))
    cache = InternalUsageCache(dual_cache=inner)
    with pytest.raises(TypeError):
        await cache.async_batch_get_cache(keys=None)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_async_increment_cache_forwards_args():
    inner = MagicMock()
    inner.async_increment_cache = AsyncMock(return_value=5.0)
    cache = InternalUsageCache(dual_cache=inner)
    result = await cache.async_increment_cache(key="counter", value=1.5, litellm_parent_otel_span="span")
    forwarded = _kwargs_snapshot(inner.async_increment_cache.call_args)
    assert forwarded == {"key": "counter", "value": 1.5, "local_only": False, "parent_otel_span": "span"}
    assert result == 5.0


@pytest.mark.asyncio
async def test_async_increment_cache_propagates_error_raises():
    inner = MagicMock()
    inner.async_increment_cache = AsyncMock(side_effect=OverflowError())
    cache = InternalUsageCache(dual_cache=inner)
    with pytest.raises(OverflowError):
        await cache.async_increment_cache(key="x", value=1.0, litellm_parent_otel_span=None)


def test_set_cache_forwards_args():
    inner = MagicMock()
    cache = InternalUsageCache(dual_cache=inner)
    cache.set_cache(key="k", value="v", local_only=True, ttl=30)
    forwarded = _kwargs_snapshot(inner.set_cache.call_args)
    assert forwarded == {"key": "k", "value": "v", "local_only": True, "ttl": 30}


def test_set_cache_propagates_error_raises():
    inner = MagicMock()
    inner.set_cache = MagicMock(side_effect=RuntimeError("no redis"))
    cache = InternalUsageCache(dual_cache=inner)
    with pytest.raises(RuntimeError):
        cache.set_cache(key="k", value="v")


def test_get_cache_forwards_args_and_returns_inner_result():
    inner = MagicMock()
    inner.get_cache = MagicMock(return_value={"k": "v", "ttl": 60, "source": "mem"})
    cache = InternalUsageCache(dual_cache=inner)
    result = cache.get_cache(key="k", local_only=False)
    forwarded = _kwargs_snapshot(inner.get_cache.call_args)
    assert forwarded == {"key": "k", "local_only": False}
    assert result == {"k": "v", "ttl": 60, "source": "mem"}


def test_get_cache_propagates_error_raises():
    inner = MagicMock()
    inner.get_cache = MagicMock(side_effect=KeyError("missing"))
    cache = InternalUsageCache(dual_cache=inner)
    with pytest.raises(KeyError):
        cache.get_cache(key="missing")
