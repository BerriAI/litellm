"""
Regression tests for multi-instance TPM sync in lowest_tpm_rpm_v2.

These guard against the regression described in issue #27736:
deployment-level TPM enforcement was per-pod (not cross-pod), so the
effective limit became `tpm_limit * N_replica`.

The fix routes TPM increments through `_increment_value_in_current_window`
(same path RPM uses since #9357), which registers the key in
`redis_increment_operation_queue` and `in_memory_keys_to_update` so the
periodic `_sync_in_memory_spend_with_redis` task can pull cross-pod sums
back into the local in-memory cache.
"""

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.router_strategy.lowest_tpm_rpm_v2 import LowestTPMLoggingHandler_v2


def _logging_payload(model_id: str, model: str, total_tokens: int) -> dict:
    return {
        "standard_logging_object": {
            "model_group": "test-model",
            "hidden_params": {"litellm_model_name": model},
            "model_id": model_id,
            "total_tokens": total_tokens,
        },
        "litellm_params": {"metadata": {}, "model_info": {"id": model_id}},
    }


@pytest.mark.asyncio
async def test_tpm_key_registered_for_cross_pod_sync():
    """After `async_log_success_event`, the TPM key must be in
    `in_memory_keys_to_update` so the periodic sync task pulls cross-pod
    sums back from Redis. Without this, the per-pod in-memory counter
    only reflects locally-processed responses (issue #27736)."""
    cache = DualCache()
    handler = LowestTPMLoggingHandler_v2(router_cache=cache, routing_args={})

    await handler.async_log_success_event(
        _logging_payload("deploy-1", "test-model", 100), None, None, None
    )

    keys = handler.get_in_memory_keys_to_update()
    assert any(
        "tpm" in k for k in keys
    ), f"TPM key not registered for cross-pod sync. Got: {keys}"


@pytest.mark.asyncio
async def test_tpm_increment_queued_for_redis_push():
    """The increment must land in `redis_increment_operation_queue` so it
    is batched to Redis by the periodic sync, instead of being written to
    Redis directly (the pre-fix behavior bypassed the sync mechanism)."""
    cache = DualCache()
    handler = LowestTPMLoggingHandler_v2(router_cache=cache, routing_args={})

    await handler.async_log_success_event(
        _logging_payload("deploy-1", "test-model", 100), None, None, None
    )

    assert any(
        "tpm" in op["key"] and op["increment_value"] == 100
        for op in handler.redis_increment_operation_queue
    ), f"TPM increment not queued. Queue: {handler.redis_increment_operation_queue}"


@pytest.mark.asyncio
async def test_tpm_local_in_memory_increment_still_immediate():
    """The local in-memory counter must still be incremented immediately
    so the same pod sees its own usage right away. Only the Redis write
    is deferred to the periodic sync task."""
    cache = DualCache()
    handler = LowestTPMLoggingHandler_v2(router_cache=cache, routing_args={})

    await handler.async_log_success_event(
        _logging_payload("deploy-1", "test-model", 100), None, None, None
    )

    tpm_keys = [k for k in handler.get_in_memory_keys_to_update() if "tpm" in k]
    assert tpm_keys, "no TPM key registered"
    value = await cache.in_memory_cache.async_get_cache(key=tpm_keys[0])
    assert value == 100, f"in-memory TPM counter not incremented. Got: {value}"
