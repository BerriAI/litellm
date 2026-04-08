"""
Test that LeastBusyLoggingHandler clamps request counters to zero.

When success/failure callbacks fire before the pre-call callback
(race condition) or fire multiple times, the counter can go negative.
A negative count causes that deployment to attract ALL traffic while
others starve.

Regression test for https://github.com/BerriAI/litellm/issues/25323
"""

import pytest

from litellm.router_strategy.least_busy import LeastBusyLoggingHandler


def _make_kwargs(model_group: str, deployment_id: str):
    """Build kwargs dict matching the structure log_success_event expects."""
    return {
        "litellm_params": {
            "metadata": {"model_group": model_group},
            "model_info": {"id": deployment_id},
        },
    }


class FakeCache:
    """Minimal DualCache stand-in that stores values in a plain dict."""

    def __init__(self):
        self.store: dict = {}

    def get_cache(self, key, **kwargs):
        return self.store.get(key)

    def set_cache(self, key, value, **kwargs):
        self.store[key] = value

    async def async_get_cache(self, key, **kwargs):
        return self.store.get(key)

    async def async_set_cache(self, key, value, **kwargs):
        self.store[key] = value


def test_sync_success_counter_never_goes_negative():
    """log_success_event should clamp the counter at 0, not go negative."""
    cache = FakeCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)

    model_group = "test-group"
    deploy_id = "deploy-1"
    cache_key = f"{model_group}_request_count"

    # Simulate counter already at 0 (no in-flight requests)
    cache.store[cache_key] = {deploy_id: 0}

    # Fire success callback — would decrement to -1 without the clamp
    kwargs = _make_kwargs(model_group, deploy_id)
    handler.log_success_event(kwargs, None, None, None)

    assert cache.store[cache_key][deploy_id] == 0, "Counter went negative"


def test_sync_failure_counter_never_goes_negative():
    """log_failure_event should clamp the counter at 0."""
    cache = FakeCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)

    model_group = "test-group"
    deploy_id = "deploy-1"
    cache_key = f"{model_group}_request_count"

    cache.store[cache_key] = {deploy_id: 0}

    kwargs = _make_kwargs(model_group, deploy_id)
    handler.log_failure_event(kwargs, None, None, None)

    assert cache.store[cache_key][deploy_id] == 0, "Counter went negative"


@pytest.mark.asyncio
async def test_async_success_counter_never_goes_negative():
    """async_log_success_event should clamp the counter at 0."""
    cache = FakeCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)

    model_group = "test-group"
    deploy_id = "deploy-1"
    cache_key = f"{model_group}_request_count"
    cache.store[cache_key] = {deploy_id: 0}

    kwargs = _make_kwargs(model_group, deploy_id)
    await handler.async_log_success_event(kwargs, None, None, None)

    assert cache.store[cache_key][deploy_id] == 0, "Async success counter went negative"


@pytest.mark.asyncio
async def test_async_failure_counter_never_goes_negative():
    """async_log_failure_event should clamp the counter at 0."""
    cache = FakeCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)

    model_group = "test-group"
    deploy_id = "deploy-1"
    cache_key = f"{model_group}_request_count"
    cache.store[cache_key] = {deploy_id: 0}

    kwargs = _make_kwargs(model_group, deploy_id)
    await handler.async_log_failure_event(kwargs, None, None, None)

    assert cache.store[cache_key][deploy_id] == 0, "Async failure counter went negative"


def test_counter_clamp_with_multiple_decrements():
    """Multiple success callbacks should never push counter below 0."""
    cache = FakeCache()
    handler = LeastBusyLoggingHandler(router_cache=cache)

    model_group = "test-group"
    deploy_id = "deploy-1"
    cache_key = f"{model_group}_request_count"

    # Start with 1 in-flight request
    cache.store[cache_key] = {deploy_id: 1}

    kwargs = _make_kwargs(model_group, deploy_id)

    # First decrement: 1 -> 0
    handler.log_success_event(kwargs, None, None, None)
    assert cache.store[cache_key][deploy_id] == 0

    # Second decrement (duplicate callback): should stay at 0, not go to -1
    handler.log_success_event(kwargs, None, None, None)
    assert cache.store[cache_key][deploy_id] == 0

    # Third decrement: still 0
    handler.log_failure_event(kwargs, None, None, None)
    assert cache.store[cache_key][deploy_id] == 0
