"""
Regression tests for https://github.com/BerriAI/litellm/issues/16060

When the tpm/rpm usage cache read fails (DualCache.[async_]batch_get_cache
returns None, e.g. on a transient Redis error), usage-based-routing-v2 must
fail open and still return a healthy deployment, instead of raising
"No deployments available" (RateLimitError / 429).
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))  # Adds the parent directory to the system path

from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2

MODEL_GROUP = "gpt-5"

HEALTHY_DEPLOYMENTS = [
    {
        "model_name": MODEL_GROUP,
        "litellm_params": {"model": "azure/gpt-5", "tpm": 5_000_000},
        "model_info": {"id": "deployment-1"},
    },
    {
        "model_name": MODEL_GROUP,
        "litellm_params": {"model": "azure/gpt-5-eu", "tpm": 5_000_000},
        "model_info": {"id": "deployment-2"},
    },
]


def _handler() -> LowestTPMLoggingHandler_v2:
    return LowestTPMLoggingHandler_v2(router_cache=DualCache())


@pytest.mark.asyncio
async def test_async_cache_read_failure_fails_open():
    """Batch cache read returning None must not fail the request."""
    handler = _handler()
    with patch.object(
        handler.router_cache,
        "async_batch_get_cache",
        new=AsyncMock(return_value=None),
    ):
        deployment = await handler.async_get_available_deployments(
            model_group=MODEL_GROUP,
            healthy_deployments=HEALTHY_DEPLOYMENTS,
            messages=[{"role": "user", "content": "hey"}],
        )
    assert deployment is not None
    assert deployment["model_info"]["id"] in {"deployment-1", "deployment-2"}


def test_sync_cache_read_failure_fails_open():
    """Sync path: batch cache read returning None must not fail the request."""
    handler = _handler()
    with patch.object(handler.router_cache, "batch_get_cache", return_value=None):
        deployment = handler.get_available_deployments(
            model_group=MODEL_GROUP,
            healthy_deployments=HEALTHY_DEPLOYMENTS,
            messages=[{"role": "user", "content": "hey"}],
        )
    assert deployment is not None
    assert deployment["model_info"]["id"] in {"deployment-1", "deployment-2"}


@pytest.mark.asyncio
async def test_async_over_limit_deployments_still_excluded():
    """Fail-open must not weaken normal limit enforcement when cache reads work."""
    handler = _handler()
    cache_values = [5_000_001, 100, 1, 1]
    with patch.object(
        handler.router_cache,
        "async_batch_get_cache",
        new=AsyncMock(return_value=cache_values),
    ):
        deployment = await handler.async_get_available_deployments(
            model_group=MODEL_GROUP,
            healthy_deployments=HEALTHY_DEPLOYMENTS,
            messages=[{"role": "user", "content": "hey"}],
        )
    assert deployment is not None
    assert deployment["model_info"]["id"] == "deployment-2"


@pytest.mark.asyncio
async def test_async_all_over_limit_still_raises():
    """When usage data is present and all deployments are over limit, keep raising."""
    import litellm

    handler = _handler()
    cache_values = [5_000_001, 5_000_001, 1, 1]
    with patch.object(
        handler.router_cache,
        "async_batch_get_cache",
        new=AsyncMock(return_value=cache_values),
    ):
        with pytest.raises(litellm.RateLimitError):
            await handler.async_get_available_deployments(
                model_group=MODEL_GROUP,
                healthy_deployments=HEALTHY_DEPLOYMENTS,
                messages=[{"role": "user", "content": "hey"}],
            )
