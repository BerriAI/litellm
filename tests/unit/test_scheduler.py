# What is this?
## Unit tests for the Scheduler.py (workload prioritization scheduler)

import asyncio
import importlib
import os

import pytest

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.scheduler import FlowItem, Scheduler, SchedulerCacheKeys
from litellm.utils import _invalidate_model_cost_lowercase_map
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.asyncio
async def test_scheduler_diff_model_names():
    """
    Assert 2 requests to 2 diff model groups are top of their respective queue's
    """
    scheduler = Scheduler()

    item1 = FlowItem(priority=0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=0, request_id="11", model_name="gpt-4")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    assert await scheduler.poll(id="10", model_name="gpt-3.5-turbo", health_deployments=[{"key": "value"}]) == True
    assert await scheduler.poll(id="11", model_name="gpt-4", health_deployments=[{"key": "value"}]) == True


@pytest.mark.asyncio
async def test_scheduler_poll_persists_queue_to_cache():
    class StubRedisCache:
        def __init__(self):
            self.store = {}

        async def async_get_cache(self, key, **kwargs):
            return self.store.get(key)

        async def async_set_cache(self, key, value, **kwargs):
            self.store[key] = value

    redis_cache = StubRedisCache()
    scheduler = Scheduler(redis_cache=redis_cache)

    item1 = FlowItem(priority=0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=0, request_id="11", model_name="gpt-3.5-turbo")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    await scheduler.poll(id="10", model_name="gpt-3.5-turbo", health_deployments=[])

    queue_key = f"{SchedulerCacheKeys.queue.value}:{item1.model_name}"
    updated_queue = redis_cache.store[queue_key]
    assert updated_queue[0][1] == "11"


@pytest.mark.parametrize("p0, p1", [(0, 0), (0, 1), (1, 0)])
@pytest.mark.parametrize("healthy_deployments", [[{"key": "value"}], []])
@pytest.mark.asyncio
async def test_scheduler_prioritized_requests(p0, p1, healthy_deployments):
    """
    2 requests for same model group
    """
    scheduler = Scheduler()

    item1 = FlowItem(priority=p0, request_id="10", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=p1, request_id="11", model_name="gpt-3.5-turbo")
    await scheduler.add_request(item1)
    await scheduler.add_request(item2)

    if p0 == 0:
        assert (
            await scheduler.peek(
                id="10",
                model_name="gpt-3.5-turbo",
                health_deployments=healthy_deployments,
            )
            == True
        ), "queue={}".format(await scheduler.get_queue(model_name="gpt-3.5-turbo"))
        assert (
            await scheduler.peek(
                id="11",
                model_name="gpt-3.5-turbo",
                health_deployments=healthy_deployments,
            )
            == False
        )
    else:
        assert (
            await scheduler.peek(
                id="11",
                model_name="gpt-3.5-turbo",
                health_deployments=healthy_deployments,
            )
            == True
        )
        assert (
            await scheduler.peek(
                id="10",
                model_name="gpt-3.5-turbo",
                health_deployments=healthy_deployments,
            )
            == False
        )


@pytest.mark.asyncio
async def test_scheduler_queue_cleanup_on_timeout():
    """
    Test that a timed-out request is properly removed from the queue.
    This prevents memory leaks from accumulating timed-out requests.
    """
    scheduler = Scheduler()

    # Add multiple requests with different priorities
    item1 = FlowItem(priority=0, request_id="req-0", model_name="gpt-3.5-turbo")
    item2 = FlowItem(priority=1, request_id="req-1", model_name="gpt-3.5-turbo")
    item3 = FlowItem(priority=2, request_id="req-2", model_name="gpt-3.5-turbo")

    await scheduler.add_request(item1)
    await scheduler.add_request(item2)
    await scheduler.add_request(item3)

    # Verify initial queue size
    queue_before = await scheduler.get_queue(model_name="gpt-3.5-turbo")
    assert len(queue_before) == 3, f"Expected 3 items in queue, got {len(queue_before)}"

    # Simulate timeout cleanup - remove a non-front request (item2)
    await scheduler.remove_request(request_id="req-1", model_name="gpt-3.5-turbo")

    # Verify queue was cleaned up
    queue_after = await scheduler.get_queue(model_name="gpt-3.5-turbo")
    assert len(queue_after) == 2, f"Expected 2 items after cleanup, got {len(queue_after)}"

    # Verify the correct request was removed
    remaining_ids = [item[1] for item in queue_after]
    assert "req-1" not in remaining_ids, "Expected req-1 to be removed"
    assert "req-0" in remaining_ids, "Expected req-0 to remain"
    assert "req-2" in remaining_ids, "Expected req-2 to remain"

    # Verify remaining items are in correct priority order (0 should be first)
    assert queue_after[0][1] == "req-0", "Expected req-0 (priority 0) to be at front"


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def isolate_litellm_state():
    """
    Per-function isolation fixture.

    Resets litellm globals to their true defaults before each test and
    restores them afterward, so tests don't leak side effects.
    Works safely under pytest-xdist parallel execution.
    """
    original_state = {}
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in ("pre_call_rules", "post_call_rules"):
        if hasattr(litellm, attr):
            val = getattr(litellm, attr)
            original_state[attr] = val.copy() if val else []
    for attr in _SCALAR_DEFAULTS:
        if hasattr(litellm, attr):
            original_state[attr] = getattr(litellm, attr)
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
        "pre_call_rules",
        "post_call_rules",
    ):
        if hasattr(litellm, attr):
            setattr(litellm, attr, [])
    for attr, default_val in _SCALAR_DEFAULTS.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, default_val)
    yield
    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    if hasattr(litellm, "in_memory_llm_clients_cache"):
        litellm.in_memory_llm_clients_cache.flush_cache()
    for attr, original_value in original_state.items():
        if hasattr(litellm, attr):
            setattr(litellm, attr, original_value)
    _invalidate_model_cost_lowercase_map()


_SCALAR_DEFAULTS = {
    "num_retries": getattr(litellm, "num_retries", None),
    "num_retries_per_request": getattr(litellm, "num_retries_per_request", None),
    "request_timeout": getattr(litellm, "request_timeout", None),
    "set_verbose": getattr(litellm, "set_verbose", False),
    "cache": getattr(litellm, "cache", None),
    "allowed_fails": getattr(litellm, "allowed_fails", 3),
    "default_fallbacks": getattr(litellm, "default_fallbacks", None),
    "enable_azure_ad_token_refresh": getattr(litellm, "enable_azure_ad_token_refresh", None),
    "tag_budget_config": getattr(litellm, "tag_budget_config", None),
    "model_cost": getattr(litellm, "model_cost", None),
    "token_counter": getattr(litellm, "token_counter", None),
    "disable_aiohttp_transport": getattr(litellm, "disable_aiohttp_transport", False),
    "force_ipv4": getattr(litellm, "force_ipv4", False),
    "drop_params": getattr(litellm, "drop_params", None),
    "modify_params": getattr(litellm, "modify_params", False),
    "api_base": getattr(litellm, "api_base", None),
    "api_key": getattr(litellm, "api_key", None),
}


@pytest.fixture(scope="module", autouse=True)
def setup_and_teardown():
    """
    Module-scoped setup. Reloads litellm only in single-process mode
    (skipped under xdist to avoid cross-worker interference).
    """
    import litellm

    worker_id = os.environ.get("PYTEST_XDIST_WORKER", None)
    if worker_id is None:
        importlib.reload(litellm)
        try:
            if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
                import litellm.proxy.proxy_server

                importlib.reload(litellm.proxy.proxy_server)
        except Exception:
            pass
        if hasattr(litellm, "in_memory_llm_clients_cache"):
            litellm.in_memory_llm_clients_cache.flush_cache()
    yield
