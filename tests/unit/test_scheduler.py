# What is this?
## Unit tests for the Scheduler.py (workload prioritization scheduler)

import asyncio
import importlib
import json
import os
from collections.abc import Sequence
from typing import Final

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

    assert await scheduler.poll(request=item1, health_deployments=[{"key": "value"}]) == True
    assert await scheduler.poll(request=item2, health_deployments=[{"key": "value"}]) == True


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

    await scheduler.poll(request=item1, health_deployments=[])

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


@pytest.mark.asyncio
async def test_poll_admits_request_missing_from_queue_while_a_deployment_is_healthy():
    scheduler: Final = Scheduler()

    assert await scheduler.poll(
        request=FlowItem(priority=1, request_id="erased-by-concurrent-write", model_name="sched-model"),
        health_deployments=[{"model_info": {"id": "a"}}],
    )


@pytest.mark.asyncio
async def test_poll_during_cooldown_admits_only_the_head_of_the_queue():
    scheduler: Final = Scheduler()
    later: Final = FlowItem(priority=2, request_id="later", model_name="sched-model")
    head: Final = FlowItem(priority=1, request_id="head", model_name="sched-model")
    await scheduler.add_request(later)
    await scheduler.add_request(head)

    assert not await scheduler.poll(request=later, health_deployments=[])
    assert await scheduler.poll(request=head, health_deployments=[])
    assert await scheduler.get_queue("sched-model") == [(2, "later")]


@pytest.mark.asyncio
async def test_poll_during_cooldown_re_enqueues_a_request_a_concurrent_writer_erased_behind_the_head():
    scheduler: Final = Scheduler()
    still_queued: Final = FlowItem(priority=0, request_id="still-queued", model_name="sched-model")
    erased: Final = FlowItem(priority=1, request_id="erased-by-concurrent-write", model_name="sched-model")
    await scheduler.add_request(still_queued)

    assert not await scheduler.poll(request=erased, health_deployments=[])
    assert await scheduler.get_queue("sched-model") == [(0, "still-queued"), (1, "erased-by-concurrent-write")]
    assert await scheduler.poll(request=still_queued, health_deployments=[])
    assert await scheduler.poll(request=erased, health_deployments=[])
    assert await scheduler.get_queue("sched-model") == []


@pytest.mark.asyncio
async def test_poll_during_cooldown_admits_an_erased_request_that_outranks_the_queue():
    scheduler: Final = Scheduler()
    await scheduler.add_request(FlowItem(priority=2, request_id="still-queued", model_name="sched-model"))

    assert await scheduler.poll(
        request=FlowItem(priority=0, request_id="erased-urgent", model_name="sched-model"), health_deployments=[]
    )
    assert await scheduler.get_queue("sched-model") == [(2, "still-queued")]


class _ExpiringCache:
    def __init__(self) -> None:
        self.store: dict[str, object] = {}

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        return self.store.get(key)

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.store[key] = value


@pytest.mark.asyncio
async def test_wait_for_turn_during_cooldown_survives_the_queue_key_expiring():
    cache: Final = _ExpiringCache()
    scheduler: Final = Scheduler(redis_cache=cache)
    scheduler.cache.in_memory_cache.cache_dict.clear()

    async def no_healthy_deployments_after_the_key_expired() -> Sequence[object]:
        cache.store.clear()
        scheduler.cache.in_memory_cache.cache_dict.clear()
        return ()

    await scheduler.wait_for_turn(
        request=FlowItem(priority=1, request_id="sole-waiter", model_name="sched-model"),
        timeout=5,
        get_healthy_deployments=no_healthy_deployments_after_the_key_expired,
    )

    assert await scheduler.get_queue("sched-model") == []



class _JsonRoundTripRedisCache:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}

    async def async_get_cache(self, key: str, **kwargs: object) -> object:
        raw: Final = self.store.get(key)
        return None if raw is None else json.loads(raw)

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.store[key] = json.dumps(value)


@pytest.mark.asyncio
async def test_second_replica_enqueues_behind_a_queue_decoded_from_redis():
    redis_cache: Final = _JsonRoundTripRedisCache()
    replica_a: Final = Scheduler(redis_cache=redis_cache)
    replica_b: Final = Scheduler(redis_cache=redis_cache)
    waiting_on_a: Final = FlowItem(priority=1, request_id="waiting-on-a", model_name="sched-model")
    urgent_on_b: Final = FlowItem(priority=0, request_id="urgent-on-b", model_name="sched-model")
    await replica_a.add_request(waiting_on_a)

    await replica_b.add_request(urgent_on_b)

    assert await replica_b.get_queue("sched-model") == [(0, "urgent-on-b"), (1, "waiting-on-a")]
    assert await replica_b.poll(request=urgent_on_b, health_deployments=[])
    assert await replica_b.poll(request=waiting_on_a, health_deployments=[])
    await replica_b.remove_request(request_id="waiting-on-a", model_name="sched-model")
    assert await replica_b.get_queue("sched-model") == []

class _PausesAfterEnqueueScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()
        self.enqueued: Final = asyncio.Event()

    async def add_request(self, request: FlowItem) -> None:
        await super().add_request(request)
        self.enqueued.set()
        await asyncio.Event().wait()


async def _no_healthy_deployments() -> Sequence[object]:
    return ()


@pytest.mark.asyncio
async def test_wait_for_turn_removes_entry_when_cancelled_mid_enqueue():
    scheduler: Final = _PausesAfterEnqueueScheduler()
    waiting: Final = asyncio.create_task(
        scheduler.wait_for_turn(
            request=FlowItem(priority=1, request_id="cancelled", model_name="sched-model"),
            timeout=5,
            get_healthy_deployments=_no_healthy_deployments,
        )
    )
    await scheduler.enqueued.wait()
    assert await scheduler.get_queue("sched-model") == [(1, "cancelled")]

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert await scheduler.get_queue("sched-model") == []


class _HeldRemovalScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__()
        self.removing: Final = asyncio.Event()
        self.finish_removal: Final = asyncio.Event()

    async def remove_request(self, request_id: str, model_name: str) -> None:
        self.removing.set()
        await self.finish_removal.wait()
        await super().remove_request(request_id=request_id, model_name=model_name)


@pytest.mark.asyncio
async def test_wait_for_turn_finishes_removal_when_cancelled_again_during_cleanup():
    scheduler: Final = _HeldRemovalScheduler()
    await scheduler.add_request(FlowItem(priority=0, request_id="head", model_name="sched-model"))
    polling: Final = asyncio.Event()

    async def no_healthy_deployments() -> Sequence[object]:
        polling.set()
        return ()

    waiting: Final = asyncio.create_task(
        scheduler.wait_for_turn(
            request=FlowItem(priority=1, request_id="cancelled", model_name="sched-model"),
            timeout=5,
            get_healthy_deployments=no_healthy_deployments,
        )
    )
    await polling.wait()
    waiting.cancel()
    await scheduler.removing.wait()
    waiting.cancel()
    scheduler.finish_removal.set()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert await scheduler.get_queue("sched-model") == [(0, "head")]


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
