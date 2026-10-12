"""
Unit tests for virtual model routing in the Router.

Virtual models provide ordered failover across deployments referenced by model_id.
This test suite validates routing, failover, retry, and timeout semantics.
"""

from unittest.mock import patch

import pytest

import litellm
from litellm import Router
from litellm.constants import VIRTUAL_MODEL_PREFIX
from litellm.types.router import RouterRateLimitError, VirtualModelRouterState


def test_spike_order_is_selection_preference():
    """
    Spike A: Verify that order is a selection preference, not just a fallback escalation.

    With two deployments in the same model group at order 1 and order 2,
    every healthy call should land on order 1 (the min-order group).
    """
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "response from order 1",
                    "order": 1,
                },
                "model_info": {"id": "order-1-dep"},
            },
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "response from order 2",
                    "order": 2,
                },
                "model_info": {"id": "order-2-dep"},
            },
        ],
        num_retries=0,
    )

    for i in range(20):
        response = router.completion(
            model="test-model",
            messages=[{"role": "user", "content": "test"}],
        )
        assert response._hidden_params["model_id"] == "order-1-dep", (
            f"Attempt {i + 1}: expected order-1-dep, got {response._hidden_params['model_id']}"
        )


@pytest.mark.asyncio
async def test_spike_retry_vs_fallback_attempt_sequence():
    """
    Spike B: Verify the retry walk mechanism by counting actual attempts per deployment.

    Scenario: Two deployments at priority 1 and 2, num_retries=2.
    Priority-1 fails with 500, priority-2 succeeds.

    WITH the retry-walk branch:
      - Attempt 1: priority-1 tried, fails
      - Retry logic adds priority-1 to skip set
      - Attempt 2: priority-2 tried (priority-1 skipped)
      - Total attempts: 2 (priority-1 once, priority-2 once)

    WITHOUT the retry-walk branch (old behavior):
      - Attempt 1: priority-1 tried, fails
      - Retry 1: priority-1 tried again (min-order re-picks)
      - Retry 2: priority-1 tried again
      - Order-fallback escalates: priority-2 tried
      - Total attempts: 4 (priority-1 three times, priority-2 once)
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=2,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
                num_retries=2,
            )
        }
    )

    attempt_counts = {"priority-1-dep": 0, "priority-2-dep": 0}

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            attempt_counts["priority-1-dep"] += 1
            raise litellm.InternalServerError(message="Mock error", model="gpt-4o", llm_provider="openai")
        else:
            attempt_counts["priority-2-dep"] += 1
            return litellm.ModelResponse(id="test", choices=[{"message": {"content": "success from priority-2"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )

    assert response._hidden_params["model_id"] == "priority-2-dep"
    assert response.choices[0].message.content == "success from priority-2"

    assert attempt_counts["priority-1-dep"] == 1, (
        f"priority-1 should be attempted exactly once, got {attempt_counts['priority-1-dep']}"
    )
    assert attempt_counts["priority-2-dep"] == 1, (
        f"priority-2 should be attempted exactly once, got {attempt_counts['priority-2-dep']}"
    )


@pytest.mark.asyncio
async def test_healthy_priority_1_serves():
    """When priority-1 is healthy, it always serves."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from priority-1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from priority-2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
            )
        }
    )

    for i in range(10):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )
        assert response._hidden_params["model_id"] == "priority-1-dep"


@pytest.mark.asyncio
async def test_priority_tie_stays_within_tie():
    """When two members have the same priority, selection stays within the tie; priority-2 member never serves."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                    "order": 1,
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                    "order": 2,
                },
                "model_info": {"id": "dep2"},
            },
            {
                "model_name": "dep3-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key3",
                    "mock_response": "from dep3",
                    "order": 2,
                },
                "model_info": {"id": "dep3"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 1, "dep3": 2},
            )
        }
    )

    for i in range(10):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )
        assert response._hidden_params["model_id"] in ("dep1", "dep2")
        assert response._hidden_params["model_id"] != "dep3", (
            "priority-2 member should never serve when priority-1 tie exists"
        )


@pytest.mark.parametrize("error_class", ["InternalServerError", "RateLimitError"])
@pytest.mark.asyncio
async def test_failover_on_500_and_429(error_class: str):
    """Priority-1 fails with 500/429, priority-2 serves."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": f"litellm.{error_class}",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success from priority-2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=1,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
                num_retries=1,
            )
        }
    )

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )

    assert response._hidden_params["model_id"] == "priority-2-dep"


@pytest.mark.asyncio
async def test_failover_on_connection_error():
    """Priority-1 fails with connection error, priority-2 serves."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": Exception("Connection refused"),
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success from priority-2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=1,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
                num_retries=1,
            )
        }
    )

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )

    assert response._hidden_params["model_id"] == "priority-2-dep"


@pytest.mark.asyncio
async def test_no_failover_on_context_window_exceeded():
    """Context-window-exceeded on priority-1, no cross-member failover."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "litellm.ContextWindowExceededError",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "should not be called",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
            )
        }
    )

    with pytest.raises(litellm.ContextWindowExceededError):
        await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )


@pytest.mark.asyncio
async def test_single_member_transient_retry_still_works():
    """Single-member virtual model: same-member retry still works after transient 500, exactly 1+num_retries attempts."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "only-dep"},
            },
        ],
        num_retries=2,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"only-dep": 1},
                num_retries=2,
            )
        }
    )

    attempt_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count <= 2:
            raise litellm.InternalServerError(
                message=f"Mock error attempt {attempt_count}", model="gpt-4o", llm_provider="openai"
            )
        return litellm.ModelResponse(id="test", choices=[{"message": {"content": "success after retries"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )

    assert response.choices[0].message.content == "success after retries"
    assert attempt_count == 3, f"Expected 3 attempts (1 initial + 2 retries), got {attempt_count}"


@pytest.mark.asyncio
async def test_all_members_fail_raises_error():
    """All members fail, raises the underlying error from the last attempt."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
            )
        }
    )

    attempt_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        raise litellm.InternalServerError(
            message=f"Mock error from attempt {attempt_count}", model="gpt-4o", llm_provider="openai"
        )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        with pytest.raises(litellm.InternalServerError):
            await router.acompletion(
                model=vm_name,
                messages=[{"role": "user", "content": "test"}],
            )

    assert attempt_count == 2, f"Both members should be tried, got {attempt_count} attempts"


@pytest.mark.asyncio
async def test_get_model_names_enumerates_virtual_names():
    """get_model_names includes virtual model names."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                "model_info": {"id": "dep1"},
            },
        ],
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1},
            )
        }
    )

    model_names = router.get_model_names()
    assert vm_name in model_names
    assert "dep1-model" in model_names


@pytest.mark.asyncio
async def test_zero_virtual_models_byte_identical():
    """Router with zero virtual models behaves identically to before."""
    router = Router(
        model_list=[
            {
                "model_name": "test-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "success",
                },
                "model_info": {"id": "dep1"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models({})

    response = await router.acompletion(
        model="test-model",
        messages=[{"role": "user", "content": "test"}],
    )

    assert response._hidden_params["model_id"] == "dep1"
    assert response.choices[0].message.content == "success"


@pytest.mark.asyncio
async def test_copy_isolation_member_params_unchanged():
    """Resolving a virtual model never mutates a member's stored litellm_params."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "success",
                },
                "model_info": {"id": "dep1"},
            },
        ],
    )

    original_params = router.model_list[0]["litellm_params"].copy()

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1},
            )
        }
    )

    await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )

    assert router.model_list[0]["litellm_params"] == original_params
    assert "order" not in router.model_list[0]["litellm_params"]


@pytest.mark.asyncio
async def test_deleted_member_never_routed_again():
    """Member deleted via Router.delete_deployment is never returned for the virtual model."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    router.delete_deployment(id="dep1")

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )

    assert response._hidden_params["model_id"] == "dep2"


@pytest.mark.asyncio
async def test_empty_membership_raises_error():
    """Virtual model with empty membership raises BadRequestError but name stays in get_model_names()."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                "model_info": {"id": "dep1"},
            },
        ],
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={},
            )
        }
    )

    assert vm_name in router.get_model_names(), "Empty membership should still register the virtual model name"

    with pytest.raises(litellm.BadRequestError, match="no healthy deployments"):
        await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )


@pytest.mark.asyncio
async def test_num_retries_zero_still_failovers_across_levels():
    """num_retries=0 still allows cross-level failover via order escalation."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "litellm.InternalServerError",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success from priority-2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
            )
        }
    )

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )

    assert response._hidden_params["model_id"] == "priority-2-dep"


@pytest.mark.asyncio
async def test_walk_invariant_attempts_use_distinct_members():
    """
    Walk invariant: across attempts within one request, attempted member ids are distinct
    until membership is exhausted (no attempt re-picks a failed member while an untried
    healthy member remains).
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key2"},
                "model_info": {"id": "dep2"},
            },
            {
                "model_name": "dep3-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key3"},
                "model_info": {"id": "dep3"},
            },
        ],
        num_retries=1,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2, "dep3": 3},
                num_retries=1,
            )
        }
    )

    attempted_ids = []

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        dep_id = "dep1" if model_id == "key1" else "dep2" if model_id == "key2" else "dep3"
        attempted_ids.append(dep_id)

        if dep_id in ("dep1", "dep2"):
            raise litellm.InternalServerError(
                message=f"Mock error from {dep_id}", model="gpt-4o", llm_provider="openai"
            )
        return litellm.ModelResponse(id="test", choices=[{"message": {"content": "success"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )

    assert len(attempted_ids) == 3
    assert attempted_ids[0] == "dep1"
    assert attempted_ids[1] == "dep2"
    assert attempted_ids[2] == "dep3"
    assert len(set(attempted_ids)) == len(attempted_ids)


@pytest.mark.asyncio
async def test_429_benches_member_despite_single_deployment_own_group():
    """
    429 with Retry-After: next attempt moves on without waiting AND the member is benched
    for the hinted duration despite being the sole deployment of its own model_name group
    (exercises the virtual-aware 429 guard).
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=60,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    exception = litellm.RateLimitError(
        message="Rate limit exceeded",
        model="gpt-4o",
        llm_provider="openai",
    )

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep1-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "dep1"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
    assert len(active_cooldowns) == 1
    assert active_cooldowns[0][0] == "dep1"

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response._hidden_params["model_id"] == "dep2"


@pytest.mark.asyncio
async def test_400_never_benches():
    """400 never fails over and never benches the member."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "success from dep1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    exception = litellm.BadRequestError(
        message="Bad request",
        model="gpt-4o",
        llm_provider="openai",
    )

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep1-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "dep1"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
    assert len(active_cooldowns) == 0

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response._hidden_params["model_id"] == "dep1"


@pytest.mark.asyncio
async def test_allowed_fails_exactly_at_threshold_benches():
    """
    Member crosses allowed_fails exactly at threshold and is benched.
    Also pins per-virtual-model override beats router default.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
                allowed_fails=2,
                cooldown_time=60,
            )
        }
    )

    exception = litellm.InternalServerError(
        message="Mock error",
        model="gpt-4o",
        llm_provider="openai",
    )

    for i in range(3):
        router.deployment_callback_on_failure(
            kwargs={
                "call_type": "acompletion",
                "litellm_params": {
                    "metadata": {
                        "deployment": "dep1-model",
                        "model_group": vm_name,
                    },
                    "model_info": {"id": "dep1"},
                },
                "exception": exception,
            },
            completion_response=None,
            start_time=None,
            end_time=None,
        )

        active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
        if i < 2:
            assert len(active_cooldowns) == 0, f"Failure {i + 1}: should not bench yet"

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
    assert len(active_cooldowns) == 1
    assert active_cooldowns[0][0] == "dep1"

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response._hidden_params["model_id"] == "dep2"


@pytest.mark.asyncio
async def test_pointer_freshness():
    """Update a member's litellm_params in router.model_list, next virtual-model request carries the new params."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "old-key",
                    "mock_response": "from old key",
                },
                "model_info": {"id": "dep1"},
            },
        ],
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1},
            )
        }
    )

    response1 = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response1.choices[0].message.content == "from old key"

    router.model_list[0]["litellm_params"]["api_key"] = "new-key"
    router.model_list[0]["litellm_params"]["mock_response"] = "from new key"

    response2 = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response2.choices[0].message.content == "from new key"


def test_is_recognized_model_recognizes_virtual_model_names():
    """Router.is_recognized_model returns True for virtual model names."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                "model_info": {"id": "dep1"},
            },
        ],
    )

    assert not router.is_recognized_model(vm_name), "virtual model name should not be recognized before set_virtual_models"

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1},
            )
        }
    )

    assert router.is_recognized_model(vm_name), "virtual model name must be recognized after set_virtual_models"
    assert router.is_recognized_model("dep1-model"), "regular model name must still be recognized"


@pytest.mark.asyncio
async def test_aembedding_routes_through_virtual_model_member():
    """router.aembedding routes through a virtual model member."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-embedding"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-embedding",
                "litellm_params": {
                    "model": "text-embedding-3-small",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-embedding",
                "litellm_params": {
                    "model": "text-embedding-3-small",
                    "api_key": "key2",
                    "mock_response": [0.1, 0.2, 0.3],
                },
                "model_info": {"id": "dep2"},
            },
        ],
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    async def mock_aembedding(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            from litellm import EmbeddingResponse
            return EmbeddingResponse(
                data=[{"embedding": [0.1, 0.2, 0.3], "index": 0, "object": "embedding"}],
                model="text-embedding-3-small",
                object="list",
            )
        raise Exception("unexpected api_key")

    with patch("litellm.aembedding", side_effect=mock_aembedding):
        response = await router.aembedding(
            model=vm_name,
            input="test input",
        )

    assert response is not None
    assert response.data is not None
    assert len(response.data) > 0
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_aembedding_failover_across_virtual_model_members():
    """router.aembedding failovers across virtual model members on failure."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-embedding"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-embedding",
                "litellm_params": {
                    "model": "text-embedding-3-small",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-embedding",
                "litellm_params": {
                    "model": "text-embedding-3-small",
                    "api_key": "key2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    attempt_counts = {"dep1": 0, "dep2": 0}

    async def mock_aembedding(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            attempt_counts["dep1"] += 1
            raise litellm.InternalServerError(
                message="Mock error from dep1", model="text-embedding-3-small", llm_provider="openai"
            )
        else:
            attempt_counts["dep2"] += 1
            from litellm import EmbeddingResponse
            return EmbeddingResponse(
                data=[{"embedding": [0.4, 0.5, 0.6], "index": 0, "object": "embedding"}],
                model="text-embedding-3-small",
                object="list",
            )

    with patch("litellm.aembedding", side_effect=mock_aembedding):
        response = await router.aembedding(
            model=vm_name,
            input="test input",
        )

    assert response is not None
    assert response.data[0]["embedding"] == [0.4, 0.5, 0.6]
    assert attempt_counts["dep1"] == 1, "priority-1 member should be tried once"
    assert attempt_counts["dep2"] == 1, "priority-2 member should be tried once after priority-1 fails"


@pytest.mark.asyncio
async def test_aresponses_routes_through_virtual_model_member():
    """router.aresponses routes through a virtual model member."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-responses"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-responses",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "response from dep1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-responses",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "response from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    response = await router.aresponses(
        model=vm_name,
        input="test input",
    )

    assert response is not None
    assert response.id is not None
    assert response._hidden_params["model_id"] == "dep1", "priority-1 member should serve"


@pytest.mark.asyncio
async def test_plain_group_429_behavior_unchanged():
    """Plain model group 429 behavior unchanged even when unrelated virtual models exist: walk not firing, sole deployment retried."""
    router = Router(
        model_list=[
            {
                "model_name": "plain-group",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
        ],
        num_retries=1,
    )

    router.set_virtual_models(
        {
            f"{VIRTUAL_MODEL_PREFIX}other": VirtualModelRouterState(
                members={"dep1": 1},
            )
        }
    )

    attempt_count = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal attempt_count
        attempt_count += 1
        raise litellm.RateLimitError(message="Rate limit", model="gpt-4o", llm_provider="openai")

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="plain-group",
                messages=[{"role": "user", "content": "test"}],
            )

    assert attempt_count == 2, f"Plain group should retry without walk, expected 2 attempts, got {attempt_count}"

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
    assert len(active_cooldowns) == 0


@pytest.mark.asyncio
async def test_single_member_virtual_model_not_benched_on_429():
    """Single-member virtual model NOT benched on 429 (no alternatives exist)."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "litellm.RateLimitError",
                },
                "model_info": {"id": "dep1"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1},
            )
        }
    )

    with pytest.raises(litellm.RateLimitError):
        await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
    assert len(active_cooldowns) == 0


@pytest.mark.asyncio
async def test_attempt_budget_bound():
    """
    Attempt budget bound: with the walk mechanism, each member is attempted at most once,
    so total attempts equals the number of members in the virtual model.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key2"},
                "model_info": {"id": "dep2"},
            },
            {
                "model_name": "dep3-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key3"},
                "model_info": {"id": "dep3"},
            },
        ],
        num_retries=1,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2, "dep3": 3},
                num_retries=1,
            )
        }
    )

    total_attempts = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal total_attempts
        total_attempts += 1
        raise litellm.InternalServerError(message="Mock error", model="gpt-4o", llm_provider="openai")

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        with pytest.raises(RouterRateLimitError, match="No deployments available"):
            await router.acompletion(
                model=vm_name,
                messages=[{"role": "user", "content": "test"}],
            )

    assert total_attempts == 3, (
        f"With walk mechanism, each member tried once, expected 3 attempts, got {total_attempts}"
    )


@pytest.mark.parametrize("status_code,should_bench", [(401, True), (404, True), (403, False)])
@pytest.mark.asyncio
async def test_failover_on_403_and_404_with_cooldown_classification(status_code: int, should_bench: bool):
    """
    Priority-1 raises 401/403/404, next attempt served by priority-2.
    Per-attempt failure callback classification: 401/404 bench the member cross-request, 403 does not.
    This matches the code's truth at cooldown_handlers.py:253-263.
    """
    import httpx

    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success from priority-2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=1,
        allowed_fails=0,
        cooldown_time=60,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
                num_retries=1,
            )
        }
    )

    if status_code == 401:
        exception = litellm.AuthenticationError(
            message="Authentication failed",
            model="gpt-4o",
            llm_provider="openai",
        )
    elif status_code == 403:
        response = httpx.Response(
            status_code=403,
            request=httpx.Request(method="POST", url="https://api.openai.com/v1/chat/completions"),
        )
        exception = litellm.PermissionDeniedError(
            message="Permission denied",
            model="gpt-4o",
            llm_provider="openai",
            response=response,
        )
    else:  # 404
        exception = litellm.NotFoundError(
            message="Not found",
            model="gpt-4o",
            llm_provider="openai",
        )

    attempt_counts = {"priority-1-dep": 0, "priority-2-dep": 0}

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            attempt_counts["priority-1-dep"] += 1
            raise exception
        else:
            attempt_counts["priority-2-dep"] += 1
            return litellm.ModelResponse(id="test", choices=[{"message": {"content": "success from priority-2"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )

    assert response._hidden_params["model_id"] == "priority-2-dep"
    assert attempt_counts["priority-1-dep"] == 1
    assert attempt_counts["priority-2-dep"] == 1

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep1-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "priority-1-dep"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["priority-1-dep"], parent_otel_span=None)

    if should_bench:
        assert len(active_cooldowns) == 1
        assert active_cooldowns[0][0] == "priority-1-dep"
    else:
        assert len(active_cooldowns) == 0


@pytest.mark.asyncio
async def test_per_attempt_timeout_overlay():
    """
    Virtual model configured with timeout=T.
    Assert (a) the resolved copy passed to the attempt carries timeout=T while member's stored litellm_params unchanged,
    and (b) priority-1 scripted to hang longer than T raises a timeout and next attempt served by priority-2.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success from priority-2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=1,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
                num_retries=1,
                timeout=0.1,
            )
        }
    )

    original_params = router.model_list[0]["litellm_params"].copy()

    observed_timeout = None

    async def mock_acompletion(*args, **kwargs):
        nonlocal observed_timeout
        model_id = kwargs.get("api_key")
        observed_timeout = kwargs.get("timeout")

        if model_id == "key1":
            raise litellm.Timeout(
                message="Request timed out",
                model="gpt-4o",
                llm_provider="openai",
            )
        else:
            return litellm.ModelResponse(id="test", choices=[{"message": {"content": "success from priority-2"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )

    assert response._hidden_params["model_id"] == "priority-2-dep"
    assert observed_timeout == 0.1
    assert router.model_list[0]["litellm_params"] == original_params
    assert "timeout" not in router.model_list[0]["litellm_params"]


@pytest.mark.asyncio
async def test_streaming_pre_first_chunk_failure_fails_over():
    """
    Streaming: pre-first-chunk failure fails over.
    Note: This test verifies that MidStreamFallbackError with is_pre_first_chunk=True
    is handled by the router's streaming machinery. The actual streaming fallback behavior
    is tested in detail in test_router.py's streaming tests.
    """
    from litellm.exceptions import MidStreamFallbackError

    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=1,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
                num_retries=1,
            )
        }
    )

    error = MidStreamFallbackError(
        message="Rate limit",
        model="gpt-4o",
        llm_provider="openai",
        generated_content="",
        is_pre_first_chunk=True,
    )

    attempt_counts = {"priority-1-dep": 0, "priority-2-dep": 0}

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            attempt_counts["priority-1-dep"] += 1
            raise error
        else:
            attempt_counts["priority-2-dep"] += 1
            return litellm.ModelResponse(id="test", choices=[{"message": {"content": "success from priority-2"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
            stream=False,
        )

    assert response._hidden_params["model_id"] == "priority-2-dep"
    assert attempt_counts["priority-1-dep"] == 1
    assert attempt_counts["priority-2-dep"] == 1


@pytest.mark.asyncio
async def test_streaming_post_first_chunk_failure_never_fails_over():
    """
    Streaming: post-first-chunk failure never fails over.
    """
    from unittest.mock import MagicMock

    from litellm.exceptions import MidStreamFallbackError

    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "should not be called",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=1,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
                num_retries=1,
            )
        }
    )

    error = MidStreamFallbackError(
        message="Rate limit",
        model="gpt-4o",
        llm_provider="openai",
        generated_content="partial",
        is_pre_first_chunk=False,
    )

    class FailingAfterFirstChunkStream:
        def __init__(self):
            self.model = "gpt-4o"
            self.custom_llm_provider = "openai"
            self.logging_obj = MagicMock()
            self._hidden_params = {"model_id": "priority-1-dep"}
            self.sent_first = False

        def __iter__(self):
            return self

        def __next__(self):
            if not self.sent_first:
                self.sent_first = True
                return litellm.ModelResponseStream(
                    id="test",
                    choices=[{"delta": {"content": "first"}}],
                )
            raise error

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            return FailingAfterFirstChunkStream()
        else:
            return litellm.ModelResponse(id="test", choices=[{"message": {"content": "should not be called"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
            stream=True,
        )

        with pytest.raises(MidStreamFallbackError):
            for chunk in response:
                pass


@pytest.mark.asyncio
async def test_rpm_pre_call_check_filters_saturated_member():
    """
    RPM pre-call check: member with rpm_limit=1 saturated -> selection filters it inside the virtual model
    and the next call is served by dep2 specifically.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                    "rpm": 1,
                },
                "model_info": {"id": "dep1", "rpm_limit": 1},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        routing_strategy="usage-based-routing-v2",
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    response1 = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response1._hidden_params["model_id"] == "dep1"

    response2 = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response2._hidden_params["model_id"] == "dep2", "dep1 should be RPM-saturated, so dep2 must serve"


@pytest.mark.asyncio
async def test_simple_shuffle_opt_in():
    """
    Virtual model configured with routing_strategy="simple-shuffle" -> both members serve over repeated calls.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        routing_strategy="simple-shuffle",
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 1},
            )
        }
    )

    served_ids = set()
    for _ in range(20):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )
        served_ids.add(response._hidden_params["model_id"])

    assert len(served_ids) >= 2, "Both members should serve with simple-shuffle"


@pytest.mark.asyncio
async def test_shared_member_state():
    """
    One deployment is both its own model_name group and a virtual model member.
    Bench it via the failure callback -> a call to the virtual model skips it.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "shared-dep",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from shared",
                },
                "model_info": {"id": "shared-id"},
            },
            {
                "model_name": "shared-dep",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from shared fallback",
                },
                "model_info": {"id": "shared-fallback"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key3",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=60,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"shared-id": 1, "dep2": 2},
            )
        }
    )

    exception = litellm.InternalServerError(
        message="Mock error",
        model="gpt-4o",
        llm_provider="openai",
    )

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "shared-dep",
                    "model_group": vm_name,
                },
                "model_info": {"id": "shared-id"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["shared-id"], parent_otel_span=None)
    assert len(active_cooldowns) == 1

    direct_response = await router.acompletion(
        model="shared-dep",
        messages=[{"role": "user", "content": "test"}],
    )
    assert direct_response._hidden_params["model_id"] == "shared-fallback"

    vm_response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert vm_response._hidden_params["model_id"] == "dep2"


@pytest.mark.asyncio
async def test_cooldown_expiry_rejoins():
    """
    Bench a member, advance fake time past cooldown -> it is selectable again.
    """
    import time

    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=60,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    exception = litellm.InternalServerError(
        message="Mock error",
        model="gpt-4o",
        llm_provider="openai",
    )

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep1-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "dep1"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
    assert len(active_cooldowns) == 1

    key = router.cooldown_cache.get_cooldown_cache_key("dep1")
    expired_value = {
        "exception_received": "Mock error",
        "status_code": "500",
        "timestamp": time.time() - 120.0,
        "cooldown_time": 60.0,
    }
    router.cooldown_cache.in_memory_cache.set_cache(key, expired_value, ttl=600)

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response._hidden_params["model_id"] == "dep1"


@pytest.mark.asyncio
async def test_member_blocked_true_filtered():
    """
    Member with blocked=true filtered inside the virtual model.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                },
                "model_info": {"id": "dep1", "blocked": True},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response._hidden_params["model_id"] == "dep2"


@pytest.mark.asyncio
async def test_input_tokens_exceeding_context_window_filters_member():
    """
    Input tokens exceeding a member's context window filter that member; all members filtered -> existing error surfaces.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "success from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    error = litellm.ContextWindowExceededError(
        message="context window exceeded",
        model="gpt-4o",
        llm_provider="openai",
    )

    attempt_counts = {"dep1": 0, "dep2": 0}

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            attempt_counts["dep1"] += 1
            raise error
        else:
            attempt_counts["dep2"] += 1
            return litellm.ModelResponse(id="test", choices=[{"message": {"content": "success from dep2"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        with pytest.raises(litellm.ContextWindowExceededError):
            await router.acompletion(
                model=vm_name,
                messages=[{"role": "user", "content": "test"}],
            )

    assert attempt_counts["dep1"] == 1
    assert attempt_counts["dep2"] == 0


@pytest.mark.asyncio
async def test_priority_1_benched_priority_2_healthy():
    """
    Priority-1 benched via the callback, priority-2 healthy: several consecutive calls all served by priority-2 with zero failures.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=60,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    exception = litellm.InternalServerError(
        message="Mock error",
        model="gpt-4o",
        llm_provider="openai",
    )

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep1-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "dep1"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    for _ in range(5):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )
        assert response._hidden_params["model_id"] == "dep2"


@pytest.mark.asyncio
async def test_member_pricing_safety_invariant():
    """
    MONEY (safety invariant): two members with different per-token prices.
    Priority-1 fails with 500, priority-2 serves.
    The cost calculator reads per-deployment pricing, so the serving deployment's price is used.
    Assert priority-2 serves (whose price would be used for billing).
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {
                    "id": "priority-1-dep",
                    "input_cost_per_token": 0.00001,
                    "output_cost_per_token": 0.00003,
                },
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                },
                "model_info": {
                    "id": "priority-2-dep",
                    "input_cost_per_token": 0.00005,
                    "output_cost_per_token": 0.00015,
                },
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
            )
        }
    )

    attempt_counts = {"priority-1-dep": 0, "priority-2-dep": 0}

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            attempt_counts["priority-1-dep"] += 1
            raise litellm.InternalServerError(message="Mock error", model="gpt-4o", llm_provider="openai")
        else:
            attempt_counts["priority-2-dep"] += 1
            return litellm.ModelResponse(
                id="test",
                choices=[{"message": {"content": "success from priority-2"}}],
                usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            )

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )

    assert response._hidden_params["model_id"] == "priority-2-dep"
    assert attempt_counts["priority-1-dep"] == 1
    assert attempt_counts["priority-2-dep"] == 1


@pytest.mark.asyncio
async def test_429_retry_after_benches_for_hinted_duration():
    """
    429 + Retry-After: member benches for the HINTED duration when no cooldown is configured.
    Script a RateLimitError carrying response headers retry-after; fake time;
    assert the member is not selectable before the hint elapses and is selectable after.
    """
    import time

    import httpx

    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    exception = litellm.RateLimitError(
        message="Rate limit exceeded",
        model="gpt-4o",
        llm_provider="openai",
    )
    exception.litellm_response_headers = httpx.Headers({"retry-after": "30"})

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep1-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "dep1"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
    assert len(active_cooldowns) == 1
    assert active_cooldowns[0][0] == "dep1"

    async def mock_acompletion(*args, **kwargs):
        return litellm.ModelResponse(id="test", choices=[{"message": {"content": "success"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )
    assert response._hidden_params["model_id"] == "dep2"

    key = router.cooldown_cache.get_cooldown_cache_key("dep1")
    expired_value = {
        "exception_received": "Rate limit",
        "status_code": "429",
        "timestamp": time.time() - 60.0,
        "cooldown_time": 30.0,
    }
    router.cooldown_cache.in_memory_cache.set_cache(key, expired_value, ttl=600)

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        response2 = await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )
    assert response2._hidden_params["model_id"] == "dep1"


@pytest.mark.asyncio
async def test_virtual_model_cooldown_time_beats_retry_after_header():
    """
    AND: a virtual-model-configured cooldown_time beats the header.
    """
    import httpx

    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
                cooldown_time=120,
            )
        }
    )

    exception = litellm.RateLimitError(
        message="Rate limit exceeded",
        model="gpt-4o",
        llm_provider="openai",
    )
    exception.litellm_response_headers = httpx.Headers({"retry-after": "30"})

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep1-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "dep1"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1"], parent_otel_span=None)
    assert len(active_cooldowns) == 1
    assert active_cooldowns[0][0] == "dep1"

    _, cooldown_details = active_cooldowns[0]
    assert isinstance(cooldown_details, dict)
    assert "cooldown_time" in cooldown_details


@pytest.mark.asyncio
async def test_400_escalates_via_order_fallback_without_benching():
    """400 on priority-1 escalates to priority-2 via order fallback (matching existing
    ordered-group semantics: _skip_order_fallback covers only context-window and
    content-policy) and never benches the member."""
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "should not be called",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
            )
        }
    )

    attempt_counts = {"priority-1-dep": 0, "priority-2-dep": 0}

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            attempt_counts["priority-1-dep"] += 1
            raise litellm.BadRequestError(message="Bad request", model="gpt-4o", llm_provider="openai")
        else:
            attempt_counts["priority-2-dep"] += 1
            return litellm.ModelResponse(id="test", choices=[{"message": {"content": "should not be called"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )

    assert attempt_counts["priority-1-dep"] == 1
    assert attempt_counts["priority-2-dep"] == 1

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["priority-1-dep"], parent_otel_span=None)
    assert len(active_cooldowns) == 0


@pytest.mark.asyncio
async def test_content_policy_violation_no_failover_no_bench():
    """
    Content-policy violation on priority-1: assert no failover (priority-2 counter == 0)
    and not benched (member still selectable after the failure callback fires with it).
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                },
                "model_info": {"id": "priority-1-dep"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "should not be called",
                },
                "model_info": {"id": "priority-2-dep"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"priority-1-dep": 1, "priority-2-dep": 2},
            )
        }
    )

    attempt_counts = {"priority-1-dep": 0, "priority-2-dep": 0}

    raised_error = None

    async def mock_acompletion(*args, **kwargs):
        model_id = kwargs.get("api_key")
        if model_id == "key1":
            attempt_counts["priority-1-dep"] += 1
            raise litellm.ContentPolicyViolationError(
                message="Content policy violation", model="gpt-4o", llm_provider="openai"
            )
        else:
            attempt_counts["priority-2-dep"] += 1
            return litellm.ModelResponse(id="test", choices=[{"message": {"content": "should not be called"}}])

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        try:
            await router.acompletion(
                model=vm_name,
                messages=[{"role": "user", "content": "test"}],
            )
        except litellm.ContentPolicyViolationError as e:
            raised_error = e

    assert raised_error is not None, "Expected ContentPolicyViolationError to be raised"
    assert attempt_counts["priority-1-dep"] == 1
    assert attempt_counts["priority-2-dep"] == 0

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["priority-1-dep"], parent_otel_span=None)
    assert len(active_cooldowns) == 0

    response = await router.acompletion(
        model=vm_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response._hidden_params["model_id"] == "priority-1-dep"


@pytest.mark.asyncio
async def test_all_members_cooled_down_raises():
    """
    All members cooled down: bench every member via the failure callback;
    assert the call raises the exhausted-candidates contract.
    The router raises RouterRateLimitError with "No deployments available".
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        allowed_fails=0,
        cooldown_time=60,
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    exception = litellm.InternalServerError(
        message="Mock error",
        model="gpt-4o",
        llm_provider="openai",
    )

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep1-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "dep1"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    router.deployment_callback_on_failure(
        kwargs={
            "call_type": "acompletion",
            "litellm_params": {
                "metadata": {
                    "deployment": "dep2-model",
                    "model_group": vm_name,
                },
                "model_info": {"id": "dep2"},
            },
            "exception": exception,
        },
        completion_response=None,
        start_time=None,
        end_time=None,
    )

    active_cooldowns = router.cooldown_cache.get_active_cooldowns(model_ids=["dep1", "dep2"], parent_otel_span=None)
    assert len(active_cooldowns) == 2

    with pytest.raises(litellm.RouterRateLimitError, match="No deployments available"):
        await router.acompletion(
            model=vm_name,
            messages=[{"role": "user", "content": "test"}],
        )


@pytest.mark.asyncio
async def test_num_retries_unset_uses_default():
    """
    num_retries unset: Router constructed without num_retries;
    count attempts on scripted failure; assert the router-default attempt count applies (openai.DEFAULT_MAX_RETRIES = 2).
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                },
                "model_info": {"id": "dep1"},
            },
        ],
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1},
            )
        }
    )

    total_attempts = 0

    async def mock_acompletion(*args, **kwargs):
        nonlocal total_attempts
        total_attempts += 1
        raise litellm.InternalServerError(message="Mock error", model="gpt-4o", llm_provider="openai")

    with patch("litellm.acompletion", side_effect=mock_acompletion):
        with pytest.raises(litellm.InternalServerError):
            await router.acompletion(
                model=vm_name,
                messages=[{"role": "user", "content": "test"}],
            )

    import openai

    expected_attempts = 1 + openai.DEFAULT_MAX_RETRIES
    assert total_attempts == expected_attempts, (
        f"Expected {expected_attempts} attempts (1 initial + {openai.DEFAULT_MAX_RETRIES} retries), got {total_attempts}"
    )


@pytest.mark.asyncio
async def test_same_member_in_two_virtual_models_distinct_priorities():
    """
    Same member in two virtual models at different priorities:
    member X at priority 2 in VM-A and priority 1 in VM-B, each VM having one other member;
    assert each VM's first attempt goes to its own priority-1 (X serves first for VM-B, the other member serves first for VM-A),
    and X's stored litellm_params remain orderless.
    """
    vm_a_name = f"{VIRTUAL_MODEL_PREFIX}vm-a"
    vm_b_name = f"{VIRTUAL_MODEL_PREFIX}vm-b"

    router = Router(
        model_list=[
            {
                "model_name": "x-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key-x",
                    "mock_response": "from X",
                },
                "model_info": {"id": "member-x"},
            },
            {
                "model_name": "a-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key-a",
                    "mock_response": "from A",
                },
                "model_info": {"id": "member-a"},
            },
            {
                "model_name": "b-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key-b",
                    "mock_response": "from B",
                },
                "model_info": {"id": "member-b"},
            },
        ],
        num_retries=0,
    )

    router.set_virtual_models(
        {
            vm_a_name: VirtualModelRouterState(
                members={"member-a": 1, "member-x": 2},
            ),
            vm_b_name: VirtualModelRouterState(
                members={"member-x": 1, "member-b": 2},
            ),
        }
    )

    response_a = await router.acompletion(
        model=vm_a_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response_a._hidden_params["model_id"] == "member-a"

    response_b = await router.acompletion(
        model=vm_b_name,
        messages=[{"role": "user", "content": "test"}],
    )
    assert response_b._hidden_params["model_id"] == "member-x"

    x_deployment = router.model_list[0]
    assert "order" not in x_deployment["litellm_params"]


@pytest.mark.asyncio
async def test_model_group_alias_targets_virtual_model():
    """
    model_group_alias targeting a virtual model name:
    Router(model_group_alias={"public-name": "virtual_model/..."}...);
    assert a call to the alias resolves and is served by the VM's priority-1.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key1",
                    "mock_response": "from dep1",
                },
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "key2",
                    "mock_response": "from dep2",
                },
                "model_info": {"id": "dep2"},
            },
        ],
        num_retries=0,
        model_group_alias={"public-name": vm_name},
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    response = await router.acompletion(
        model="public-name",
        messages=[{"role": "user", "content": "test"}],
    )

    assert response._hidden_params["model_id"] == "dep1"


@pytest.mark.asyncio
async def test_get_model_list_returns_resolved_members():
    """
    /v1/models + /model_group/info enumeration at unit level:
    get_model_names includes the virtual name (already covered)
    AND get_model_list returns the virtual group's resolved members.
    """
    vm_name = f"{VIRTUAL_MODEL_PREFIX}test-model"
    router = Router(
        model_list=[
            {
                "model_name": "dep1-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key1"},
                "model_info": {"id": "dep1"},
            },
            {
                "model_name": "dep2-model",
                "litellm_params": {"model": "gpt-4o", "api_key": "key2"},
                "model_info": {"id": "dep2"},
            },
        ],
    )

    router.set_virtual_models(
        {
            vm_name: VirtualModelRouterState(
                members={"dep1": 1, "dep2": 2},
            )
        }
    )

    model_names = router.get_model_names()
    assert vm_name in model_names

    vm_list = router.get_model_list(model_name=vm_name)
    assert vm_list is not None
    assert len(vm_list) == 2

    model_ids = {deployment["model_info"]["id"] for deployment in vm_list}
    assert model_ids == {"dep1", "dep2"}
