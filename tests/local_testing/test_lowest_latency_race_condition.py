"""
Tests for race condition fix in latency-based routing.

The lost-update race condition (https://github.com/BerriAI/litellm/issues/24720)
occurs when concurrent requests read-modify-write the same cache key without
synchronization, causing the last writer to overwrite all previous updates.

This results in latency data being constantly lost, making latency-based routing
degrade to random selection proportional to deployment count.
"""

import asyncio
import copy
import os
import sys
import time
from datetime import timedelta

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm.caching.caching import DualCache
from litellm.router_strategy.lowest_latency import LowestLatencyLoggingHandler


def _make_kwargs(model_group: str, deployment_id: str):
    return {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                "deployment": f"provider/{deployment_id}",
            },
            "model_info": {"id": deployment_id},
        }
    }


def _make_response(total_tokens: int = 50, completion_tokens: int = 25):
    return litellm.ModelResponse(
        usage=litellm.Usage(
            prompt_tokens=total_tokens - completion_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    )


@pytest.mark.asyncio
async def test_concurrent_async_log_success_no_lost_updates():
    """
    Reproduce the lost-update race condition from #24720.

    Without the fix: concurrent coroutines read the same stale snapshot,
    each appends its own latency, then writes back — the last writer wins,
    earlier updates are lost.

    With the fix: all updates are preserved because the read-modify-write
    is serialized per cache key via asyncio.Lock.
    """
    test_cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=test_cache)

    model_group = "medium"
    num_deployments = 4
    num_concurrent_per_deployment = 5
    total_expected = num_deployments * num_concurrent_per_deployment

    tasks = []
    for deploy_idx in range(num_deployments):
        deployment_id = f"deploy_{deploy_idx}"
        for req_idx in range(num_concurrent_per_deployment):
            kwargs = _make_kwargs(model_group, deployment_id)
            response_obj = _make_response()
            start_time = time.time() - 0.5  # simulate 500ms response
            end_time = time.time()
            tasks.append(
                handler.async_log_success_event(
                    kwargs=kwargs,
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

    # Fire all concurrently — this is the scenario that triggers the race
    await asyncio.gather(*tasks)

    # Verify: every deployment should have exactly num_concurrent_per_deployment
    # latency entries. Without the lock, many entries would be lost.
    latency_key = f"{model_group}_map"
    request_count_dict = test_cache.get_cache(key=latency_key)
    assert request_count_dict is not None, "Cache should contain latency data"

    total_latency_entries = 0
    for deploy_idx in range(num_deployments):
        deployment_id = f"deploy_{deploy_idx}"
        assert deployment_id in request_count_dict, (
            f"Deployment {deployment_id} should be in cache"
        )
        latency_list = request_count_dict[deployment_id].get("latency", [])
        total_latency_entries += len(latency_list)
        assert len(latency_list) == num_concurrent_per_deployment, (
            f"Deployment {deployment_id} should have {num_concurrent_per_deployment} "
            f"latency entries, got {len(latency_list)}"
        )

    assert total_latency_entries == total_expected, (
        f"Total latency entries should be {total_expected}, got {total_latency_entries}. "
        f"Lost updates indicate the race condition is not fixed."
    )


@pytest.mark.asyncio
async def test_concurrent_async_log_success_different_model_groups_independent():
    """
    Verify that locks are per-model-group: concurrent updates to different
    model groups should not block each other and should all succeed.
    """
    test_cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=test_cache)

    model_groups = ["fast", "medium", "slow"]
    num_requests = 10

    tasks = []
    for mg in model_groups:
        for i in range(num_requests):
            kwargs = _make_kwargs(mg, f"deploy_{mg}_{i % 2}")
            response_obj = _make_response()
            start_time = time.time() - 0.1
            end_time = time.time()
            tasks.append(
                handler.async_log_success_event(
                    kwargs=kwargs,
                    response_obj=response_obj,
                    start_time=start_time,
                    end_time=end_time,
                )
            )

    await asyncio.gather(*tasks)

    for mg in model_groups:
        latency_key = f"{mg}_map"
        request_count_dict = test_cache.get_cache(key=latency_key)
        assert request_count_dict is not None, (
            f"Model group '{mg}' should have cached data"
        )
        total_entries = sum(
            len(v.get("latency", []))
            for v in request_count_dict.values()
        )
        assert total_entries == num_requests, (
            f"Model group '{mg}' should have {num_requests} total latency entries, "
            f"got {total_entries}"
        )


@pytest.mark.asyncio
async def test_concurrent_failure_and_success_no_lost_updates():
    """
    Mix of async_log_success_event and async_log_failure_event (timeout)
    running concurrently on the same model group should not lose updates.
    """
    test_cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=test_cache)

    model_group = "mixed"
    deployment_id = "deploy_mixed"
    num_success = 8
    num_failure = 4

    tasks = []

    # Success events
    for _ in range(num_success):
        kwargs = _make_kwargs(model_group, deployment_id)
        response_obj = _make_response()
        start_time = time.time() - 0.3
        end_time = time.time()
        tasks.append(
            handler.async_log_success_event(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )
        )

    # Failure (timeout) events
    for _ in range(num_failure):
        kwargs = _make_kwargs(model_group, deployment_id)
        kwargs["exception"] = litellm.Timeout(
            message="timeout", model="test", llm_provider="test"
        )
        tasks.append(
            handler.async_log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=time.time() - 5,
                end_time=time.time(),
            )
        )

    await asyncio.gather(*tasks)

    latency_key = f"{model_group}_map"
    request_count_dict = test_cache.get_cache(key=latency_key)
    assert request_count_dict is not None
    assert deployment_id in request_count_dict

    latency_list = request_count_dict[deployment_id].get("latency", [])
    # max_latency_list_size defaults to 10, and we have 12 total events
    # so the list should be capped at 10
    expected_count = min(num_success + num_failure, 10)
    assert len(latency_list) == expected_count, (
        f"Expected {expected_count} latency entries (capped at max_latency_list_size), "
        f"got {len(latency_list)}"
    )


def test_sync_log_success_no_lost_updates():
    """
    Verify the sync path also preserves all updates when called sequentially.
    (Sync path uses threading.Lock for thread safety.)
    """
    test_cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=test_cache)

    model_group = "sync_test"
    num_deployments = 3
    calls_per_deployment = 5

    for deploy_idx in range(num_deployments):
        deployment_id = f"sync_deploy_{deploy_idx}"
        for _ in range(calls_per_deployment):
            kwargs = _make_kwargs(model_group, deployment_id)
            response_obj = _make_response()
            start_time = time.time() - 0.2
            end_time = time.time()
            handler.log_success_event(
                kwargs=kwargs,
                response_obj=response_obj,
                start_time=start_time,
                end_time=end_time,
            )

    latency_key = f"{model_group}_map"
    request_count_dict = test_cache.get_cache(key=latency_key)
    assert request_count_dict is not None

    total_entries = 0
    for deploy_idx in range(num_deployments):
        deployment_id = f"sync_deploy_{deploy_idx}"
        latency_list = request_count_dict[deployment_id].get("latency", [])
        total_entries += len(latency_list)
        assert len(latency_list) == calls_per_deployment

    assert total_entries == num_deployments * calls_per_deployment


@pytest.mark.asyncio
async def test_async_locks_are_per_key():
    """
    Verify that different model groups use different locks —
    acquiring a lock for one group should not block another.
    """
    test_cache = DualCache()
    handler = LowestLatencyLoggingHandler(router_cache=test_cache)

    # Acquire lock for "group_a"
    lock_a = handler._async_locks["group_a_map"]
    lock_b = handler._async_locks["group_b_map"]

    # They should be different lock instances
    assert lock_a is not lock_b, "Locks for different keys should be independent"

    # Same key should return the same lock
    lock_a2 = handler._async_locks["group_a_map"]
    assert lock_a is lock_a2, "Same key should return the same lock instance"
