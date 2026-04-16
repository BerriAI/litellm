"""
Tests that RedisCache writers tolerate `datetime.timedelta` values in the payload,
serializing them as `float` seconds (not stringified). Defence-in-depth against
the latency-cache regression class — keeps the cache numeric for the
`_get_available_deployments` read path.
"""

import json
import os
import sys
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.caching.redis_cache import RedisCache


@pytest.fixture
def redis_no_ping():
    """Patch RedisCache initialization to prevent async ping tasks from being created."""
    with patch("asyncio.get_running_loop") as mock_get_loop:
        mock_get_loop.side_effect = RuntimeError("No running event loop")
        yield


@pytest.mark.asyncio
async def test_async_set_cache_serializes_timedelta_as_float(
    monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()
    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    payload = {"latency": [timedelta(seconds=1.5), 2.0]}

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_set_cache(key="embed-group_map", value=payload)

    mock_redis_instance.set.assert_called_once()
    call_kwargs = mock_redis_instance.set.call_args.kwargs
    assert call_kwargs["name"] == "embed-group_map"
    decoded = json.loads(call_kwargs["value"])
    assert decoded == {
        "latency": [1.5, 2.0]
    }, f"timedelta did not round-trip as float: {decoded!r}"


@pytest.mark.asyncio
async def test_async_set_cache_pipeline_serializes_timedelta_as_float(
    monkeypatch, redis_no_ping
):
    monkeypatch.setenv("REDIS_HOST", "https://my-test-host")
    redis_cache = RedisCache()

    mock_redis_instance = AsyncMock()
    mock_redis_instance.__aenter__.return_value = mock_redis_instance
    mock_redis_instance.__aexit__.return_value = None

    mock_pipe = MagicMock()
    mock_pipe.set = MagicMock()
    mock_pipe.execute = AsyncMock(return_value=[True])

    pipe_ctx = MagicMock()
    pipe_ctx.__aenter__ = AsyncMock(return_value=mock_pipe)
    pipe_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_redis_instance.pipeline = MagicMock(return_value=pipe_ctx)

    payload = {"latency": [timedelta(seconds=0.25)]}

    with patch.object(
        redis_cache, "init_async_client", return_value=mock_redis_instance
    ):
        await redis_cache.async_set_cache_pipeline(
            cache_list=[("embed-group_map", payload)]
        )

    mock_pipe.set.assert_called_once()
    call_kwargs = mock_pipe.set.call_args.kwargs
    decoded = json.loads(call_kwargs["value"])
    assert decoded == {
        "latency": [0.25]
    }, f"pipeline timedelta did not round-trip as float: {decoded!r}"
